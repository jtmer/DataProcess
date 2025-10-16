import json
import time
import itertools
import numpy as np
import pandas as pd
import torch
import argparse  
import logging
import sys
from model import get_model
from dataset import get_dataset
from utils.seed_utils import set_seed
from config import nan_inf_clip_factor
import os
# transforms
from transforms import *

# 搜索空间
from utils.param_utils import get_params_space_and_org
from utils.result_utils import kwargs_to_tag
from feature import PurePythonTSProcessor
import matplotlib.pyplot as plt

# ========= 基本配置 =========
parser = argparse.ArgumentParser(description="指定时序模型运行的显卡编号")
parser.add_argument("--gpu", type=int, default=0, help="显卡编号（默认0，多显卡时可指定1、2等）")
parser.add_argument("--data", type=str, required=True, 
                    choices=["ETTh1","ETTh2","ETTm1","ETTm2","Exchange","Weather","Electricity","Traffic"],
                    help="数据集名称（必填，可选值：ETTh1/ETTh2/ETTm1/ETTm2/Exchange/Weather/Electricity/Traffic）")
args = parser.parse_args()

# 修改：根据命令行参数指定DEVICE，而非固定cuda:0
DEVICE =  f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
SEED = 42
print(f"程序将使用的设备: {DEVICE}") 
MODEL_NAME = "TimerXL"
DATA_NAME  = args.data    # 可选：ETTh1,ETTh2,ETTm1,ETTm2,Exchange,Weather,Electricity,Traffic
EVAL_BATCH_SIZE = 16

TARGET_COLUMN   = "OT"
MAX_SEQ_LEN     = 2880
PRED_LEN        = 192

# 搜索控制
PRIMARY_METRIC = "mse"
MAX_TRIALS = None                # None=全空间；否则随机抽样
RESULT_CSV = f"dp_search_results{args.data}_{args.gpu}.csv"

# ========= 指标 =========
def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def smape(y_true, y_pred, eps=1e-8):
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return float(200.0 * np.mean(np.abs(y_pred - y_true) / denom))

def get_eval_batches(dataset, mode="train", batch_size=32,
                     target=TARGET_COLUMN, max_seq_len=MAX_SEQ_LEN, pred_len=PRED_LEN,
                     stride: int | None = None, show_progress: bool = True):
    """
    生成不重叠（或按 stride 控制重叠）的时序训练样本批次。
    - 每个样本由 (history=Xb[i], future=Yb[i]) 组成；
    - 若 stride=None，则默认 stride=max_seq_len → 相邻窗口历史无重叠；
    - 若 stride<max_seq_len，则历史有重叠；若 stride>max_seq_len，则存在间隔。

    进度：每处理到 10%,20%,...,90% 打印一次，最后补 100%。
    """
    assert mode in ["train", "val", "test"]
    idx_list = dataset.get_available_idx_list(mode, max_seq_len, pred_len)
    if len(idx_list) == 0:
        if show_progress:
            print(f"[{mode}] 无可用样本。")
        return

    # ---- 计算按 stride 采样后的 real_idx 序列 ----
    idx_arr = np.asarray(idx_list, dtype=np.int64)
    idx_arr.sort()
    st = max_seq_len if stride is None else int(stride)

    # 以第一个可用 real_idx 为起点，后续每次寻找 >= (prev + st) 的下一个可用 real_idx
    selected = []
    # 起点：用 idx_arr[0]
    cur = idx_arr[0]
    selected.append(cur)
    # 目标下一个 real_idx 的下限
    target_next = cur + st

    # 用 searchsorted 在 idx_arr 上找 >= target_next 的位置
    while True:
        pos = np.searchsorted(idx_arr, target_next, side='left')
        if pos >= len(idx_arr):
            break
        cur = idx_arr[pos]
        selected.append(cur)
        target_next = cur + st

    Nsel = len(selected)
    if Nsel == 0:
        if show_progress:
            print(f"[{mode}] 无满足 stride 的样本。")
        return

    # ---- 批量遍历与进度输出 ----
    total_batches = (Nsel + batch_size - 1) // batch_size
    next_report = 0.1
    eps = 1e-12

    for b, s in enumerate(range(0, Nsel, batch_size), start=1):
        e = min(Nsel, s + batch_size)
        batch_real_idx = selected[s:e]

        Xb = np.zeros((len(batch_real_idx), max_seq_len, 1), dtype=np.float32)
        Yb = np.zeros((len(batch_real_idx), pred_len,    1), dtype=np.float32)

        for i, real_idx in enumerate(batch_real_idx):
            # 依旧复用你原来的数据接口：其会返回长度 max_seq_len+pred_len 的拼接片段
            hwl = dataset.get_history_with_label(
                target=target, flag=mode, real_idx=int(real_idx),
                max_seq_len=max_seq_len, pred_len=pred_len
            )
            hist = hwl[:max_seq_len].reshape(-1, 1)
            lab  = hwl[max_seq_len:].reshape(-1, 1)
            Xb[i] = hist
            Yb[i] = lab

        if show_progress:
            progress = e / float(Nsel)
            while progress + eps >= next_report and next_report < 1.0:
                print(f"[{mode}] 进度 {int(next_report*100)}% "
                      f"（样本 {e}/{Nsel}，批次 {b}/{total_batches}，batch_size={batch_size}，stride={st}）")
                next_report += 0.1

        yield Xb, Yb

    if show_progress:
        print(f"[{mode}] 进度 100% （样本 {Nsel}/{Nsel}，批次 {total_batches}/{total_batches}，stride={st}）")


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
def evaluate_combo(model, dataset, cfg, split, batch_size=EVAL_BATCH_SIZE):
    mse_list, mae_list, smape_list = [], [], []

    for x_np, y_np in get_eval_batches(dataset, mode=split, batch_size=batch_size,
                                       target=TARGET_COLUMN, max_seq_len=MAX_SEQ_LEN, pred_len=PRED_LEN):
        # 每个 batch 都用它自己的数据构建处理器（关键改动）
        pre_fn, post_fn = build_pipeline_fns(
            cfg, dataset,
            input_data=x_np,    # for normalizer(mode='input')
            history_data=x_np   # for normalizer(mode='history')
        )

        x_proc = pre_fn(x_np.copy())
        y_pred = model_infer(model, x_proc)
        y_rest = post_fn(y_pred)

        mse_list.append(mse(y_np, y_rest))
        mae_list.append(mae(y_np, y_rest))
        smape_list.append(smape(y_np, y_rest))

    return {
        "mse": float(np.mean(mse_list)),
        "mae": float(np.mean(mae_list)),
        "smape": float(np.mean(smape_list)),
        "count_batches": len(mse_list),
    }


# ========= 单一处理项：构造 cfg（其余为 baseline） =========
def make_cfg_for_single_op(op_name: str, op_kwargs: dict):
    """
    基于 baseline（即“什么都不处理”）生成仅启用某一处理项的 cfg。
    op_kwargs 里只填该处理项相关键即可，其它键保持 baseline。
    """
    used_keys, _, baseline = get_search_space()
    cfg = dict(baseline)

    # 映射：每个处理项涉及到的 cfg 键
    op2keys = {
        "sampler": ["sampler_factor"],
        "trimmer": ["trimmer_seq_len"],
        "aligner": ["aligner_mode", "aligner_method"],
        "inputer": ["inputer_detect_method", "inputer_fill_method"],
        "denoiser": ["denoiser_method"],
        "warper": ["warper_method", "clip_factor"],
        "normalizer": ["normalizer_method", "normalizer_mode", "normalizer_ratio", "clip_factor"],
        "differentiator": ["differentiator_n", "clip_factor"],
    }
    if op_name not in op2keys:
        raise ValueError(f"未知处理项: {op_name}")

    # 仅更新该处理项相关键；其它键保持 baseline（=不生效）
    for k in op2keys[op_name]:
        if k in op_kwargs:
            cfg[k] = op_kwargs[k]

    # 只保留 used_keys 范围（防止多余键混入）
    cfg = {k: cfg[k] for k in used_keys}
    return cfg

# ========= 取一个 batch，拿到 pre_fn，并返回处理后的数据 =========
def get_one_processed_batch_for_op(dataset, split: str, op_name: str, op_kwargs: dict):
    """
    1) 基于 baseline + 单一处理项 构造 cfg
    2) 取 split 中的第一个 batch
    3) 用该 batch 作为 input_data/history_data 构建 pre_fn
    4) 返回 (X_raw, X_proc, cfg) 类型：np.ndarray
    """
    cfg = make_cfg_for_single_op(op_name, op_kwargs)

    # 取一个 batch（和你的 evaluate_combo 中一致的取数方式）
    batch_iter = get_eval_batches(dataset, mode=split, batch_size=1,
                                  target=TARGET_COLUMN, max_seq_len=MAX_SEQ_LEN, pred_len=PRED_LEN)
    X_raw, Y_dummy = next(batch_iter)  # 这里只需要 X

    # 构建 pre_fn；post_fn 不用
    pre_fn, post_fn = build_pipeline_fns(cfg, dataset, input_data=X_raw, history_data=X_raw)

    # 应用单一处理（保持 float32，避免某些 numpy 运算升为 float64）
    X_proc = pre_fn(X_raw.copy())
    if X_proc.dtype != np.float32:
        X_proc = X_proc.astype(np.float32, copy=False)

    return X_raw, X_proc, cfg, Y_dummy, post_fn

def iter_processed_batches_for_op_per_batch(dataset, split: str, op_name: str, op_kwargs: dict):
    """
    和你的 evaluate_combo 一致：对每个 batch 都单独构建一次 pre_fn，
    这样每个 batch 的统计（如 normalizer=input/history）都会自适应该 batch。
    """
    cfg = make_cfg_for_single_op(op_name, op_kwargs)

    for Xb, y in get_eval_batches(dataset, mode=split, batch_size=1,
                                  target=TARGET_COLUMN, max_seq_len=MAX_SEQ_LEN, pred_len=PRED_LEN):
        pre_fn, post_fn = build_pipeline_fns(cfg, dataset, input_data=Xb, history_data=Xb)
        Xp = pre_fn(Xb.copy())
        if Xp.dtype != np.float32:
            Xp = Xp.astype(np.float32, copy=False)
        yield Xb, Xp, cfg, y, post_fn
    
# ========= 主流程：在验证集上搜索最优，再到测试集对比 =========
def main():
    log_format = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
    # 配置日志级别（DEBUG: 最详细，INFO: 常规信息，WARNING: 警告，ERROR: 错误）
    logging.basicConfig(
        level=logging.INFO,  # 捕获INFO及以上级别日志
        format=log_format,
        # 同时输出到控制台和文件（但文件输出会被bash脚本重定向覆盖，这里主要确保控制台输出被捕获）
        handlers=[
            logging.StreamHandler(sys.stdout),  # 输出到stdout（会被bash的>重定向到日志文件）
            # 若需要额外保存一份独立日志，可取消下面注释（路径需修改）
            # logging.FileHandler("python_internal.log", encoding="utf-8")
        ]
    )

    set_seed(SEED)
    model = get_model(MODEL_NAME, DEVICE, args=None)
    dataset = get_dataset(DATA_NAME, fast_split=False)

    used_keys, space, baseline = get_search_space()
    combos = enumerate_combinations(space, max_trials=MAX_TRIALS, seed=SEED, ensure_include=baseline)

    print(f"搜索组合数：{len(combos)}（全空间={np.prod([len(v) for v in space.values()])}）")
    print(f"搜索维度：{used_keys}")

    rows = []
    for i, cfg in enumerate(combos, 1):
        print(f'cfg:{cfg}')
        t0 = time.time()
        res = evaluate_combo(model, dataset, cfg, split="val", batch_size=EVAL_BATCH_SIZE)
        elapsed = time.time() - t0
        row = {
            "trial": i,
            "config": json.dumps(cfg, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)),
            "mse": res["mse"], "mae": res["mae"], "smape": res["smape"],
            "time_sec": round(elapsed, 2),
        }
        rows.append(row)
        print(f"[val][{i}/{len(combos)}] mse={row['mse']:.6f} mae={row['mae']:.6f} smape={row['smape']:.3f} ({elapsed:.1f}s) cfg={cfg}")

    df_val = pd.DataFrame(rows).sort_values(PRIMARY_METRIC, ascending=True)
    df_val.to_csv(RESULT_CSV, index=False, encoding="utf-8")

    best_row = df_val.iloc[0]
    best_cfg = json.loads(best_row["config"])
    logging.info(f"验证集上最优组合（按 {PRIMARY_METRIC}）为：{best_cfg}，结果：mse={best_row['mse']:.6f} mae={best_row['mae']:.6f} smape={best_row['smape']:.3f}")
    print("\n=== 验证集最优组合（按 MSE） ===")
    print(best_cfg)
    print(df_val.head(5)[["mse","mae","smape","time_sec","config"]])
    logging.info(f"完整搜索结果已保存: {RESULT_CSV}")
    
    print("\n=== 测试集对比：baseline vs best ===")
    base_res = evaluate_combo(model, dataset, baseline, split="test", batch_size=EVAL_BATCH_SIZE)
    best_res = evaluate_combo(model, dataset, best_cfg, split="test", batch_size=EVAL_BATCH_SIZE)

    def fmt(d): return f"mse={d['mse']:.6f}, mae={d['mae']:.6f}, smape={d['smape']:.3f}"
    print(f"baseline : {fmt(base_res)}")
    print(f"best     : {fmt(best_res)}")
    improve = (base_res["mse"] - best_res["mse"]) / max(1e-12, base_res["mse"]) * 100.0
    print(f"\nMSE 提升：{improve:.2f}%  （正数=更好）")
    logging.info(f"\nMSE 提升：{improve:.2f}%  （正数=更好）")
    print(f"\n完整搜索结果已保存: {RESULT_CSV}")


# def analysis_feature_after_proc(op_name: str, op_kwargs: dict):
#     dataset = get_dataset(DATA_NAME, fast_split=False)
#     processor = PurePythonTSProcessor(output_dir='single_process_analysis')

#     X_raw, X_proc, cfg, ground_truth, post_fn = get_one_processed_batch_for_op(dataset, split="train", op_name=op_name, op_kwargs=op_kwargs)
#     print(f"单一处理项 {op_name} 的 cfg：{cfg}")

#     # 原数据的数据特征
#     stats_raw = processor.process_data(X_raw)
#     save_raw = processor._save_results(stats_raw, f"{DATA_NAME}_raw.csv")

#     # 处理后的数据特征
#     stats_proc = processor.process_data(X_proc)
#     save_proc = processor._save_results(stats_proc, f"{DATA_NAME}_{kwargs_to_tag(op_kwargs)}.csv")
    
#     processor.plot_key_features_bar_compare_single_row(
#         key_raw=save_raw["key_features_df"],
#         key_proc=save_proc["key_features_df"],
#         output_dir=processor.output_dir,
#         title=f"[{DATA_NAME}]Raw vs {op_name}:{kwargs_to_tag(op_kwargs)})"
#     )
    
    

#     # X_raw = np.squeeze(X_raw)
#     # plt.plot(range(len(X_raw)), X_raw, label='*', color='red')
#     # X_proc = np.squeeze(X_proc)
#     # plt.plot(range(0, len(X_proc)), X_proc, label='-', color='green')
#     # plt.savefig('proc.png')


def analysis_feature_after_proc(op_name: str, op_kwargs: dict):
    """
    遍历训练集：
      1) 对每个 batch 的 X_raw / X_proc 提取完整特征 DataFrame 并累积
      2) 对完整特征按列求平均（numeric_only），得到单行“平均完整特征”DataFrame
      3) 用 _save_results 保存（它会同时导出完整特征与关键特征），并用返回的 key_features_df 可视化
      4) 分别评估 Raw / Proc 的预测（Proc 的预测需 post_fn 逆变换）并写入 txt
    """
    # ===== 初始化 =====
    set_seed(SEED)
    model = get_model(MODEL_NAME, DEVICE, args=None)
    dataset = get_dataset(DATA_NAME, fast_split=False)
    processor = PurePythonTSProcessor(output_dir='single_process_analysis')
    os.makedirs(processor.output_dir, exist_ok=True)

    # ===== 累积容器 =====
    # 完整特征的 DataFrame 列表（逐 batch 追加，最后按列求均值）
    full_feats_raw_list: list[pd.DataFrame] = []
    full_feats_proc_list: list[pd.DataFrame] = []

    # 预测误差（逐 batch）
    raw_mse_list, raw_mae_list = [], []
    proc_mse_list, proc_mae_list = [], []

    # ===== 遍历训练集 =====
    for Xb, Xp, cfg, yb, post_fn in iter_processed_batches_for_op_per_batch(
        dataset, split="train", op_name=op_name, op_kwargs=op_kwargs
    ):
        # ---- 特征：Raw ----
        df_raw_full = processor.process_data(Xb)         # 返回完整特征 DataFrame
        if isinstance(df_raw_full, pd.DataFrame) and not df_raw_full.empty:
            full_feats_raw_list.append(df_raw_full)

        # ---- 特征：Proc ----
        df_proc_full = processor.process_data(Xp)
        if isinstance(df_proc_full, pd.DataFrame) and not df_proc_full.empty:
            full_feats_proc_list.append(df_proc_full)

        # ---- 预测表现：Raw 输入 ----
        y_pred_raw = model_infer(model, Xb.astype(np.float32, copy=False))
        raw_mse_list.append(mse(yb, y_pred_raw))
        raw_mae_list.append(mae(yb, y_pred_raw))

        # ---- 预测表现：Proc 输入（预测后需逆变换）----
        y_pred_proc = model_infer(model, Xp.astype(np.float32, copy=False))
        y_pred_proc_restored = post_fn(y_pred_proc)
        proc_mse_list.append(mse(yb, y_pred_proc_restored))
        proc_mae_list.append(mae(yb, y_pred_proc_restored))

    # ===== 计算“平均完整特征” =====
    def _avg_full_features(dfs: list[pd.DataFrame]) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame()
        big = pd.concat(dfs, axis=0, ignore_index=True)
        col_means = big.mean(axis=0, numeric_only=True)
        # 单行 DataFrame（列名为均值计算到的 numeric 列）
        return pd.DataFrame([col_means.values], columns=col_means.index)

    avg_full_raw_df  = _avg_full_features(full_feats_raw_list)
    avg_full_proc_df = _avg_full_features(full_feats_proc_list)

    # ===== 保存平均特征（_save_results 会自动另存关键特征）并可视化 =====
    tag = kwargs_to_tag(op_kwargs) if len(op_kwargs) else "default"

    # 将“平均完整特征”传入 _save_results（它会再提取 key_features 并一并保存）
    save_raw = processor._save_results(avg_full_raw_df,  f"{DATA_NAME}_train_avg_raw.csv")
    save_proc = processor._save_results(avg_full_proc_df, f"{DATA_NAME}_train_avg_{op_name}_{tag}.csv")

    # 可视化对比：使用返回的关键特征 DataFrame
    key_raw_df  = save_raw["key_features_df"] if isinstance(save_raw, dict) and "key_features_df" in save_raw else pd.DataFrame()
    key_proc_df = save_proc["key_features_df"] if isinstance(save_proc, dict) and "key_features_df" in save_proc else pd.DataFrame()

    if not key_raw_df.empty and not key_proc_df.empty:
        processor.plot_key_features_bar_compare_single_row(
            key_raw=key_raw_df,
            key_proc=key_proc_df,
            output_dir=processor.output_dir,
            title=f"[{DATA_NAME}] Train Avg: Raw vs {op_name}({tag})"
        )
    else:
        print("警告：关键特征为空，跳过可视化。")

    # ===== 计算并保存预测指标的平均值 =====
    raw_mse_mean  = float(np.mean(raw_mse_list))  if raw_mse_list  else float("nan")
    raw_mae_mean  = float(np.mean(raw_mae_list))  if raw_mae_list  else float("nan")
    proc_mse_mean = float(np.mean(proc_mse_list)) if proc_mse_list else float("nan")
    proc_mae_mean = float(np.mean(proc_mae_list)) if proc_mae_list else float("nan")

    metrics_txt = os.path.join(processor.output_dir, f"{DATA_NAME}_train_pred_metrics_{op_name}_{tag}.txt")
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(f"=== Train Split Prediction Metrics (Averaged over batches) ===\n")
        f.write(f"Data: {DATA_NAME}\n")
        f.write(f"Op  : {op_name} {op_kwargs}\n\n")
        f.write(f"[Raw Input]\n")
        f.write(f"  MSE: {raw_mse_mean:.6f}\n")
        f.write(f"  MAE: {raw_mae_mean:.6f}\n\n")
        f.write(f"[Processed Input]\n")
        f.write(f"  MSE: {proc_mse_mean:.6f}\n")
        f.write(f"  MAE: {proc_mae_mean:.6f}\n")

    print(f"[Done] 平均特征与关键特征 CSV 已输出到：{processor.output_dir}")
    print(f"[Done] 预测指标写入：{metrics_txt}")



if __name__ == "__main__":
    # main()
    # analysis_feature_after_proc("normalizer", {"normalizer_method":"standard","normalizer_mode":"input","normalizer_ratio":1})
    # analysis_feature_after_proc("sampler", {"sampler_factor":2})
    analysis_feature_after_proc("differentiator", {"differentiator_n":1, "clip_factor":'none'})