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

# ========= 基本配置 =========
parser = argparse.ArgumentParser(description="指定时序模型运行的显卡编号")
parser.add_argument("--gpu", type=int, default=0, help="显卡编号（默认0，多显卡时可指定1、2等）")
parser.add_argument("--data", type=str, required=True, 
                    choices=["ETTh1","ETTh2","ETTm1","ETTm2","Exchange","Weather","Electricity","Traffic"],
                    help="数据集名称（必填，可选值：ETTh1/ETTh2/ETTm1/ETTm2/Exchange/Weather/Electricity/Traffic）")
args = parser.parse_args()

# 修改：根据命令行参数指定DEVICE，而非固定cuda:0 👈
DEVICE =  f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
SEED = 42
print(f"程序将使用的设备: {DEVICE}") 
MODEL_NAME = "TimerXL"
DATA_NAME  = args.data    # 可选：ETTh1,ETTh2,ETTm1,ETTm2,Exchange,Weather,Electricity,Traffic
EVAL_BATCH_SIZE = 16

TARGET_COLUMN = "OT"
MAX_SEQ_LEN   = 96 * 7     # 需满足: MAX_SEQ_LEN <= train_len/2         # ! 输入长度1440
PRED_LEN      = 96

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
def evaluate_combo(model, dataset, cfg, split, batch_size=EVAL_BATCH_SIZE):
    mse_list, mae_list, smape_list = [], [], []

    for x_np, y_np in get_eval_batches(dataset, mode=split, batch_size=batch_size,
                                       target=TARGET_COLUMN, max_seq_len=MAX_SEQ_LEN, pred_len=PRED_LEN):
        # 👇 每个 batch 都用它自己的数据构建处理器（关键改动）
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

if __name__ == "__main__":
    main()
