import json
import time
import itertools
import numpy as np
import pandas as pd
import torch
import os
from model import get_model
from dataset import get_dataset
from utils.seed_utils import set_seed
from config import nan_inf_clip_factor

# transforms
from transforms import *

# 搜索空间
from utils.param_utils import get_params_space_and_org

# ========= 基本配置 =========
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42

MODEL_NAME = "TimerXL"
DATA_NAME  = "Exchange"    # 可选：ETTh1,ETTh2,ETTm1,ETTm2,Exchange,Weather,Electricity,Traffic
EVAL_BATCH_SIZE = 64

TARGET_COLUMN = "OT"
MAX_SEQ_LEN   = 96 * 7     # 需满足: MAX_SEQ_LEN <= train_len/2
PRED_LEN      = 96

# 搜索控制
PRIMARY_METRIC = "mse"
MAX_TRIALS = None                # None=全空间；否则随机抽样
RESULT_CSV = "new_dp_search_results.csv"

USE_GPUS = [0,1,2,3,4,5,6,7]    # 用2个GPU，整体一次处理8×2=16个数据
os.environ['MPLBACKEND'] = 'Agg'  # 解决matplotlib多进程问题
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 避免GPU内存溢出

# ========= 指标 =========
def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def smape(y_true, y_pred, eps=1e-8):
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return float(200.0 * np.mean(np.abs(y_pred - y_true) / denom))

# ========= 数据批生成=========
def get_eval_batches(dataset, mode="test", batch_size=32,
                     target=TARGET_COLUMN, max_seq_len=MAX_SEQ_LEN, pred_len=PRED_LEN):
    assert mode in ["train", "val", "test"]
    idx_list = dataset.get_available_idx_list(mode, max_seq_len, pred_len)
    N = len(idx_list)
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        batch_idx = idx_list[s:e]
        Xb = np.zeros((len(batch_idx), max_seq_len, 1), dtype=np.float32)
        Yb = np.zeros((len(batch_idx), pred_len,    1), dtype=np.float32)
        for i, real_idx in enumerate(batch_idx):
            hwl = dataset.get_history_with_label(
                target=target, flag=mode, real_idx=real_idx,
                max_seq_len=max_seq_len, pred_len=pred_len
            )
            hist = hwl[:max_seq_len].reshape(-1, 1)
            lab  = hwl[max_seq_len:].reshape(-1, 1)
            Xb[i] = hist
            Yb[i] = lab
        yield Xb, Yb

# ========= 模型推理=========
@torch.no_grad()
def model_infer(model, x_np):
    x = torch.from_numpy(x_np).float().to(DEVICE)
    y_pred = model.forcast(x, PRED_LEN)
    return y_pred

# ========= 构建处理器=========
def build_pipeline_fns(cfg, dataset, input_data, history_data):
    """
    返回 pre_fn, post_fn 两个函数。
    只对 cfg 指定的算子实例化并按固定次序串接：
      Pre 顺序：Sampler -> Trimmer -> Aligner -> Inputer -> Denoiser -> Warper -> Normalizer -> Differentiator
      Post顺序：               Differentiator -> Normalizer -> Warper -> (其余为恒等，且 Sampler 不做 post)
    注：
      - Sampler 的 post_process 在你实现里会改变长度，不适合作为预测后还原，这里明确**不调用**。
      - Trimmer.post_process 只会裁到 pred_len；若模型已输出 pred_len，等价恒等。
      - Aligner/Inputer/Denoiser post 均为恒等，故无需纳入 post_fn。
      - Normalizer 需要 dataset_scaler（当 mode='dataset' 时）→ 用 train 段拟合更合理。
    """
    # —— clip 因子（传给会用到的算子）——
    clip_factor = cfg.get("clip_factor", "none")

    # —— Sampler（可选）——
    sampler = None
    if int(cfg.get("sampler_factor", 1)) != 1:
        sampler = Sampler(factor=int(cfg["sampler_factor"]))

    # —— Trimmer（可选）——
    trimmer = None
    tr_seq = int(cfg.get("trimmer_seq_len", MAX_SEQ_LEN))
    if tr_seq != MAX_SEQ_LEN:
        trimmer = Trimmer(seq_l=tr_seq, pred_l=PRED_LEN)

    # —— Aligner（可选）——
    aligner = None
    if cfg.get("aligner_mode", "none") != "none" and cfg.get("aligner_method", "none") != "none":
        # 这里 data_patch_len/model_patch_len 用输入长度对齐；若你的模型有特定 patch 长度，可替换
        aligner = Aligner(mode=cfg["aligner_mode"],
                          method=cfg["aligner_method"],
                          data_patch_len=MAX_SEQ_LEN,
                          model_patch_len=MAX_SEQ_LEN)

    # —— Inputer（可选）——
    inputer = None
    if cfg.get("inputer_detect_method", "none") != "none" and cfg.get("inputer_fill_method", "none") != "none":
        inputer = Inputer(detect_method=cfg["inputer_detect_method"],
                          fill_method=cfg["inputer_fill_method"],
                          history_seq=history_data)

    # —— Denoiser（可选）——
    denoiser = None
    if cfg.get("denoiser_method", "none") != "none":
        denoiser = Denoiser(method=cfg["denoiser_method"])

    # —— Warper（可选，可逆）——
    warper = None
    if cfg.get("warper_method", "none") != "none":
        warper = Warper(method=cfg["warper_method"], clip_factor=clip_factor)

    # —— Normalizer（可选，可逆）——
    normalizer = None
    if cfg.get("normalizer_method", "none") != "none" and cfg.get("normalizer_mode", "none") != "none":
        mode = cfg["normalizer_mode"]
        dataset_scaler = None
        if mode == "dataset":
            # 用 train 段拟合更合理
            dataset_scaler = dataset.get_scaler(method=cfg["normalizer_method"], target=TARGET_COLUMN)
        normalizer = Normalizer(
            method=cfg["normalizer_method"], mode=mode,
            input_data=input_data, history_data=history_data,
            dataset_scaler=dataset_scaler,
            ratio=cfg.get("normalizer_ratio", 1),
            clip_factor=clip_factor
        )

    # —— Differentiator（可逆）——
    differentiator = None
    if int(cfg.get("differentiator_n", 0)) > 0:
        differentiator = Differentiator(n=int(cfg["differentiator_n"]), clip_factor=clip_factor)

    # # —— Decomposer（如需启用，注意其开销/可逆性）——
    # decomposer = None
    # if cfg.get("decomposer_components","none")!="none" and cfg.get("decomposer_period","none")!="none":
    #     decomposer = Decomposer(period=int(cfg["decomposer_period"]),
    #                             component_for_model=cfg["decomposer_components"])

    def pre_fn(x):
        if sampler is not None:       x = sampler.pre_process(x)
        if trimmer is not None:       x = trimmer.pre_process(x)
        if aligner is not None:       x = aligner.pre_process(x)
        if inputer is not None:       x = inputer.pre_process(x)
        if denoiser is not None:      x = denoiser.pre_process(x)
        if warper is not None:        x = warper.pre_process(x)
        if normalizer is not None:    x = normalizer.pre_process(x)
        if differentiator is not None:x = differentiator.pre_process(x)
        # if decomposer is not None: x = decomposer.pre_process(x)
        return x

    def post_fn(y):
        # 只做“有意义的反变换”，且严格逆序
        if differentiator is not None: y = differentiator.post_process(y)
        if normalizer   is not None:   y = normalizer.post_process(y)
        if warper       is not None:   y = warper.post_process(y)
        # denoiser/inputer/aligner 都是恒等 post；sampler 的 post 会改变长度，这里**有意不调用**
        if trimmer      is not None:   y = trimmer.post_process(y)  # 裁成 pred_len（若已是则不改）
        # if decomposer is not None:   y = decomposer.post_process(y)
        return y

    return pre_fn, post_fn


# ========= 从 param_utils 生成搜索组合 =========
def get_search_space():
    params_space, origin = get_params_space_and_org()
    # 本脚本已接线的键：
    used_keys = [
        "sampler_factor",
        "trimmer_seq_len",
        "aligner_mode", "aligner_method",
        "inputer_detect_method", "inputer_fill_method",
        "denoiser_method",
        "warper_method",
        "normalizer_method", "normalizer_mode", "normalizer_ratio",
        "differentiator_n",
        "clip_factor",
        # "decomposer_period","decomposer_components",  # 如需启用自行加入
    ]
    space = {}
    for k in used_keys:
        if k in params_space:
            space[k] = list(params_space[k]["values"])
        else:
            # 用 origin 兜底为单值
            space[k] = [origin.get(k)]
    baseline = {k: origin.get(k) for k in used_keys}
    return used_keys, space, baseline


def enumerate_combinations(space, max_trials=None, seed=SEED, ensure_include=None):
    keys = list(space.keys())
    grid = list(itertools.product(*[space[k] for k in keys]))
    combos = [{k: v for k, v in zip(keys, tpl)} for tpl in grid]
    if ensure_include is not None and ensure_include not in combos:
        combos.insert(0, ensure_include)
    if max_trials is None or max_trials >= len(combos):
        return combos
    rng = np.random.default_rng(seed)
    picked = set(rng.choice(len(combos), size=max_trials, replace=False).tolist())
    if ensure_include is not None:
        try:
            base_idx = combos.index(ensure_include)
            if base_idx not in picked:
                picked.pop()
                picked.add(base_idx)
        except ValueError:
            pass
    return [combos[i] for i in sorted(picked)]

# ========= 评测一个组合 =========
@torch.no_grad()
def evaluate_with_multi_gpu(model, dataset, cfg, batch_size, gpu_ids):
    """用多GPU并行处理数据批，一次处理 batch_size×len(gpu_ids) 个数据"""
    mse_list, mae_list, smape_list = [], [], []
    t_start = time.time()
    
    # 遍历数据批（每个批会自动分给多个GPU）
    for x_np, y_np in get_eval_batches(dataset, mode="val", batch_size=batch_size):
        # 1. 数据预处理
        pre_fn, post_fn = build_pipeline_fns(cfg, dataset, x_np, x_np)
        x_proc = pre_fn(x_np.copy())
        
        # 2. 多GPU并行推理（DP自动拆分数据到多个GPU）
        x_tensor = torch.from_numpy(x_proc).float().cuda(gpu_ids[0])  # 先移到主GPU
        y_pred_tensor = model.module.forcast(x_tensor, PRED_LEN)  # 用model.module调用原模型方法
        y_pred = y_pred_tensor.cpu().numpy()  # 结果移回CPU
        
        # 3. 数据后处理与指标计算
        y_rest = post_fn(y_pred)
        mse_list.append(mse(y_np, y_rest))
        mae_list.append(mae(y_np, y_rest))
        smape_list.append(smape(y_np, y_rest))
    
    return {
        "mse": float(np.mean(mse_list)),
        "mae": float(np.mean(mae_list)),
        "smape": float(np.mean(smape_list)),
        "time_sec": round(time.time() - t_start, 2)
    }
def init_parallel_model(model_name, gpu_ids):
    """初始化多GPU并行模型"""
    # 1. 先在CPU上创建模型
    model = get_model(model_name, "cpu", args=None)
    # 2. 用DataParallel包装，指定要用的GPU（比如[0,1]代表用GPU0和GPU1）
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    # 3. 把模型移到GPU上（DP会自动分配）
    model = model.cuda(gpu_ids[0])  # 用第一个GPU作为主GPU
    return model

# ========= 主流程：在验证集上搜索最优，再到测试集对比 =========
def evaluate_with_multi_gpu(model, dataset, cfg, batch_size, gpu_ids):
    mse_list, mae_list, smape_list = [], [], []
    t_start = time.time()
    
    for x_np, y_np in get_eval_batches(dataset, mode="val", batch_size=batch_size):
        # 1. 数据预处理
        pre_fn, post_fn = build_pipeline_fns(cfg, dataset, x_np, x_np)
        x_proc = pre_fn(x_np.copy())
        
        # 2. 多GPU并行推理
        x_tensor = torch.from_numpy(x_proc).float().cuda(gpu_ids[0])
        y_pred_tensor = model.module.forcast(x_tensor, PRED_LEN)  # 确保返回的是张量
        
        # 关键修正：如果模型返回的是NumPy数组，先转为张量
        if isinstance(y_pred_tensor, np.ndarray):
            y_pred_tensor = torch.from_numpy(y_pred_tensor).float().cuda(gpu_ids[0])
        
        # 3. 结果处理（现在可以安全调用.cpu()了）
        y_pred = y_pred_tensor.cpu().numpy()  # 先移到CPU再转NumPy
        y_rest = post_fn(y_pred)
        
        # 4. 计算指标
        mse_list.append(mse(y_np, y_rest))
        mae_list.append(mae(y_np, y_rest))
        smape_list.append(smape(y_np, y_rest))
    
    return {
        "mse": float(np.mean(mse_list)),
        "mae": float(np.mean(mae_list)),
        "smape": float(np.mean(smape_list)),
        "time_sec": round(time.time() - t_start, 2)
    }
# ========= 主函数：触发8GPU并行 =========
def main():
    set_seed(SEED)  # 固定随机种子，确保结果可复现
    
    # 1. 验证8个GPU是否可用
    valid_gpus = []
    for gpu_id in USE_GPUS:
        try:
            torch.cuda.set_device(gpu_id)
            valid_gpus.append(gpu_id)
        except Exception as e:
            print(f"GPU {gpu_id} 不可用，跳过：{str(e)}")
    if len(valid_gpus) != 8:
        print(f"警告：仅检测到 {len(valid_gpus)} 个可用GPU，需8个才能满负载并行！")
        return
    print(f"已确认8个可用GPU：{valid_gpus}")

    # 2. 加载数据集（主进程加载一次，避免多GPU重复加载）
    dataset = get_dataset(DATA_NAME, fast_split=False)
    print(f"数据集 {DATA_NAME} 加载完成，开始生成参数组合")

    # 3. 生成参数搜索组合（比如全空间搜索或抽样）
    used_keys, space, baseline = get_search_space()
    combos = enumerate_combinations(space, max_trials=MAX_TRIALS, ensure_include=baseline)
    total_combos = len(combos)
    print(f"共生成 {total_combos} 个参数组合，搜索维度：{used_keys}")

    # 4. 初始化8GPU并行模型（用DataParallel包装，自动分发数据）
    model = init_parallel_model(MODEL_NAME, valid_gpus)
    print("8GPU并行模型初始化完成，开始评估参数组合")

    # 5. 用8GPU并行评估所有组合（每个数据批自动拆分给8个GPU）
    all_results = []
    for trial_idx, cfg in enumerate(combos, 1):
        # 调用多GPU评估函数，一次处理 EVAL_BATCH_SIZE×8 个数据
        res = evaluate_with_multi_gpu(
            model=model,
            dataset=dataset,
            cfg=cfg,
            batch_size=EVAL_BATCH_SIZE,  # 单GPU批大小：16，8GPU合计：16×8=128
            gpu_ids=valid_gpus
        )
        # 记录结果
        all_results.append({
            "trial": trial_idx,
            "config": json.dumps(cfg, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)),
            "mse": res["mse"],
            "mae": res["mae"],
            "smape": res["smape"],
            "time_sec": res["time_sec"]
        })
        # 打印进度（实时查看并行效果）
        print(f"完成第 {trial_idx}/{total_combos} 个组合 | MSE: {res['mse']:.6f} | 耗时: {res['time_sec']}s")

    # 6. 保存结果到CSV（按MSE排序，方便找最优组合）
    df_results = pd.DataFrame(all_results).sort_values(by=PRIMARY_METRIC, ascending=True)
    df_results.to_csv(RESULT_CSV, index=False, encoding="utf-8")
    print(f"\n所有组合评估完成！结果已保存到 {RESULT_CSV}")

    # 7. 输出最优组合（快速查看效果）
    best_combo = df_results.iloc[0]
    print(f"\n=== 8GPU并行搜索完成：最优组合（按{mse}排序） ===")
    print(f"实验序号：{best_combo['trial']}")
    print(f"参数配置：{json.loads(best_combo['config'])}")
    print(f"最优MSE：{best_combo['mse']:.6f} | MAE：{best_combo['mae']:.6f} | SMAPE：{best_combo['smape']:.2f}%")

# ========= 启动主函数 =========
if __name__ == "__main__":
    main()