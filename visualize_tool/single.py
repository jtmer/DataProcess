import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from matplotlib.font_manager import FontManager, findfont
import warnings
import numpy as np
import pandas as pd
import os
import sys


# 配置字体，优先使用系统中已安装的中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 其他全局配置
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300  # 高分辨率保存
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9

def plot_single_operation_analysis(result, baseline_result, save_dir,cfg):
    """单个操作的5维可视化分析"""
    if result is None or baseline_result is None:
        return
    
    op_name = result["operation_name"].replace("/", "_").replace("（", "[").replace("）", "]")
    save_path = os.path.join(save_dir, f"{op_name}.png")
    
    # 创建2×3网格布局
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])  # 数据分布对比
    ax2 = fig.add_subplot(gs[0, 1])  # 时间序列对比
    ax3 = fig.add_subplot(gs[0, 2])  # 预测结果对比
    ax4 = fig.add_subplot(gs[1, 0])  # 误差分布对比
    ax5 = fig.add_subplot(gs[1, 1])  # 指标改进对比
    
    # 计算改进百分比
    baseline_metrics = baseline_result["metrics"]
    current_metrics = result["metrics"]
    improvements = {
        "mse": (baseline_metrics["mse"] - current_metrics["mse"]) / baseline_metrics["mse"] * 100,
        "mae": (baseline_metrics["mae"] - current_metrics["mae"]) / baseline_metrics["mae"] * 100,
        "smape": (baseline_metrics["smape"] - current_metrics["smape"]) / baseline_metrics["smape"] * 100
    }

    # --------------------------
    # 1. 数据分布对比（带核密度估计）
    # --------------------------
    x_original = result["x_original"].flatten()[:10000]
    x_processed = result["x_processed"].flatten()[:10000]
    
    sns.kdeplot(x_original, ax=ax1, label="raw", fill=True, alpha=0.5, color="#1f77b4")
    sns.kdeplot(x_processed, ax=ax1, label="Processed", fill=True, alpha=0.5, color="#ff7f0e")
    
    ax1.set_title(f"dataset: raw={x_original.mean():.2f}, Processed={x_processed.mean():.2f}", fontsize=11)
    ax1.set_xlabel("value")
    ax1.set_ylabel("density")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --------------------------
    # 2. 时间序列趋势对比
    # --------------------------
    sample_idx = 0
    seq_len = min(200, cfg.MAX_SEQ_LEN)
    x_original_seq = result["x_original"][sample_idx, :seq_len, 0]
    x_processed_seq = result["x_processed"][sample_idx, :seq_len, 0]
    
    ax2.plot(range(seq_len), x_original_seq, label="raw", alpha=0.8, color="#1f77b4")
    ax2.plot(range(seq_len), x_processed_seq, label="Processed", alpha=0.8, color="#ff7f0e")
    
    ax2.set_title(f"sample {sample_idx}）", fontsize=11)
    ax2.set_xlabel("time step")
    ax2.set_ylabel("value")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # --------------------------
    # 3. 预测结果对比
    # --------------------------
    pred_len = min(100, cfg.PRED_LEN)
    y_true = result["y_true"][sample_idx, :pred_len, 0]
    y_baseline = baseline_result["y_pred"][sample_idx, :pred_len, 0]
    y_current = result["y_pred"][sample_idx, :pred_len, 0]
    
    ax3.plot(range(pred_len), y_true, label="true", linewidth=2, color="#2ca02c")
    ax3.plot(range(pred_len), y_baseline, label="baseline", linestyle="--", color="#1f77b4")
    ax3.plot(range(pred_len), y_current, label="after_processed", linestyle="-.", color="#ff7f0e")
    
    ax3.set_title(f"outcome comparison {sample_idx}）", fontsize=11)
    ax3.set_xlabel("predicted time step")
    ax3.set_ylabel("value")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # --------------------------
    # 4. 预测误差分布对比
    # --------------------------
    error_baseline = (baseline_result["y_pred"][sample_idx, :, 0] - baseline_result["y_true"][sample_idx, :, 0])[:pred_len]
    error_current = (result["y_pred"][sample_idx, :, 0] - result["y_true"][sample_idx, :, 0])[:pred_len]
    
    sns.histplot(error_baseline, ax=ax4, label="baseline error", alpha=0.5, bins=30, color="#1f77b4", kde=True)
    sns.histplot(error_current, ax=ax4, label="error after processed", alpha=0.5, bins=30, color="#ff7f0e", kde=True)
    
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax4.set_title(f"error distribution\naverage error: {error_baseline.mean():.2f}, 当前: {error_current.mean():.2f}", fontsize=11)
    ax4.set_xlabel("prediction error")
    ax4.set_ylabel("frequency")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # --------------------------
    # 5. 指标改进对比
    # --------------------------
    metrics = ["mse", "mae", "smape"]
    metric_names = ["MSE", "MAE", "SMAPE"]
    baseline_values = [baseline_metrics[m] for m in metrics]
    current_values = [current_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax5.bar(x - width/2, baseline_values, width, label="baseline", color="#1f77b4")
    ax5.bar(x + width/2, current_values, width, label="after", color="#ff7f0e")
    
    # 添加改进百分比标签
    for i, (imp, b_val, c_val) in enumerate(zip(improvements.values(), baseline_values, current_values)):
        ax5.text(i + width/2, c_val, f"{imp:+.1f}%", 
                 ha='center', va='bottom', fontweight='bold',
                 color='green' if imp > 0 else 'red')
    
    ax5.set_title("improvement", fontsize=11)
    ax5.set_xticks(x)
    ax5.set_xticklabels(metric_names)
    ax5.set_ylabel("standard metric value")
    ax5.legend()
    ax5.grid(alpha=0.3, axis='y')

    # 整体标题
    fig.suptitle(f"{result['operation_name']} analysis", fontsize=16)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_global_comparison(all_results, baseline_name, save_dir):
    """全局对比可视化：多个操作之间的对比"""
    # 提取基线结果
    baseline_result = next(r for r in all_results if r["operation_name"] == baseline_name)
    baseline_metrics = baseline_result["metrics"]
    
    # 准备数据
    operations = [r["operation_name"] for r in all_results]
    metrics_data = {
        "mse": [r["metrics"]["mse"] for r in all_results],
        "mae": [r["metrics"]["mae"] for r in all_results],
        "smape": [r["metrics"]["smape"] for r in all_results]
    }
    
    # 计算所有操作的改进百分比
    improvements = {
        "mse": [(baseline_metrics["mse"] - r["metrics"]["mse"]) / baseline_metrics["mse"] * 100 
                for r in all_results],
        "mae": [(baseline_metrics["mae"] - r["metrics"]["mae"]) / baseline_metrics["mae"] * 100 
                for r in all_results],
        "smape": [(baseline_metrics["smape"] - r["metrics"]["smape"]) / baseline_metrics["smape"] * 100 
                 for r in all_results]
    }

    # --------------------------
    # 1. 各操作指标对比条形图
    # --------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle("operation compare(-)", fontsize=16)
    
    for i, (metric, metric_name) in enumerate(zip(["mse", "mae", "smape"], ["MSE", "MAE", "SMAPE(%)"])):
        ax = axes[i]
        y_pos = np.arange(len(operations))
        values = metrics_data[metric]
        
        # 高亮基线
        colors = ['#ff7f0e' if op == baseline_name else '#1f77b4' for op in operations]
        
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(operations)
        ax.set_xlabel(metric_name)
        ax.grid(alpha=0.3, axis='x')
        
        # 在条形末尾添加数值
        for j, v in enumerate(values):
            ax.text(v, j, f"{v:.4f}", va='center', ha='left', fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, "各操作指标对比.png"), bbox_inches='tight')
    plt.close()

    # --------------------------
    # 2. 改进百分比热力图
    # --------------------------
    # 转换为DataFrame
    imp_df = pd.DataFrame(improvements, index=operations)
    imp_df = imp_df.reindex(columns=["mse", "mae", "smape"])
    imp_df.columns = ["MSE improvement(%)", "MAE improvement(%)", "SMAPE improvement(%)"]
    
    plt.figure(figsize=(10, 8))
    # 创建红-白-绿渐变色
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", 
        [(0.8, 0, 0), (1, 1, 1), (0, 0.6, 0)],
        N=100
    )
    
    sns.heatmap(imp_df, annot=True, fmt=".1f", cmap=cmap, center=0,
                cbar_kws={"label": "improvement percentage(%)"}, linewidths=0.5)
    
    plt.title("improvements（+）", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "改进百分比热力图.png"), bbox_inches='tight')
    plt.close()

    # --------------------------
    # 3. 指标雷达图对比
    # --------------------------
    # 归一化指标（便于雷达图对比）
    norm_metrics = {}
    for metric in ["mse", "mae", "smape"]:
        max_val = max(metrics_data[metric])
        norm_metrics[metric] = [v / max_val for v in metrics_data[metric]]
    
    # 准备雷达图数据
    labels = ["MSE", "MAE", "SMAPE"]
    num_vars = len(labels)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # 绘制每个操作的雷达图
    for i, op in enumerate(operations):
        values = [norm_metrics["mse"][i], norm_metrics["mae"][i], norm_metrics["smape"][i]]
        values += values[:1]  # 闭合
        
        # 高亮基线
        if op == baseline_name:
            ax.plot(angles, values, linewidth=2, linestyle='-', label=op, color='#ff7f0e', marker='o')
            ax.fill(angles, values, color='#ff7f0e', alpha=0.25)
        else:
            ax.plot(angles, values, linewidth=2, linestyle='-', label=op, marker='o')
    
    # 设置雷达图属性
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("after 1,the closer the better", fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "指标雷达图对比.png"), bbox_inches='tight')
    plt.close()

    # --------------------------
    # 4. 处理时间对比
    # --------------------------
    processing_times = [r["metrics"]["time_sec"] for r in all_results]
    
    plt.figure(figsize=(12, 6))
    y_pos = np.arange(len(operations))
    
    # 高亮基线
    colors = ['#ff7f0e' if op == baseline_name else '#1f77b4' for op in operations]
    
    plt.barh(y_pos, processing_times, color=colors)
    plt.yticks(y_pos, operations)
    plt.xlabel("lasting time (seconds)")
    plt.title("time comparison", fontsize=14)
    plt.grid(alpha=0.3, axis='x')
    
    # 添加时间标签
    for i, v in enumerate(processing_times):
        plt.text(v, i, f"{v:.1f}s", va='center', ha='left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "处理时间对比.png"), bbox_inches='tight')
    plt.close()

