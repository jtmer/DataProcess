import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import List

max_mse=10000
# --------------------------
# 基础配置：禁用弹窗+统一样式
# --------------------------
plt.switch_backend('Agg')  # 非交互式后端，仅保存图片不弹窗
plt.rcParams.update({
    "axes.unicode_minus": False,  # 修复负号显示
    "font.size": 10,              # 统一基础字体大小
    "figure.facecolor": "white",  # 图表背景为白色（避免保存后透明）
    "savefig.dpi": 300            # 默认高分辨率保存（300dpi）
})


# --------------------------
# 工具函数：路径处理与文件夹创建
# --------------------------
def get_valid_csv_path() -> str:
    """从控制台获取并验证CSV路径有效性，返回绝对路径"""
    while True:
        input_path = input("\n请输入预处理结果CSV文件的完整路径（例：/data/results.csv）：").strip()
        full_path = os.path.abspath(input_path)  # 相对路径转绝对路径
        
        # 多维度验证
        if not os.path.exists(full_path):
            print(f"❌ 错误：路径 '{full_path}' 不存在，请重新输入！")
        elif not os.path.isfile(full_path):
            print(f"❌ 错误：'{full_path}' 不是文件，请输入有效文件路径！")
        elif not full_path.endswith(".csv"):
            print(f"❌ 错误：'{full_path}' 不是CSV文件，请输入 .csv 格式路径！")
        else:
            print(f"✅ 验证通过！CSV路径：{full_path}")
            return full_path


def create_plot_dir(csv_path: str, dir_name: str = "preprocessing_analysis_plots") -> str:
    """在CSV同级目录创建图片文件夹，返回文件夹绝对路径"""
    csv_dir = os.path.dirname(csv_path)
    plot_dir = os.path.join(csv_dir, dir_name)
    new_dir = os.path.join(plot_dir,csv_path.split('/')[-1].replace('.csv',''))
    # 自动创建文件夹（已存在则跳过）
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(new_dir, exist_ok=True)

    print(f"\n📂 图片存储文件夹：{new_dir}")
    return new_dir


# --------------------------
# 核心函数1：数据加载与解析
# --------------------------
def load_parse_data(csv_path: str) -> pd.DataFrame:
    """加载CSV并解析config列的JSON参数，返回完整DataFrame"""
    print("\n🔄 步骤1/7：加载并解析数据...")
    try:
        # 加载原始CSV
        df = pd.read_csv(csv_path)
        
        # 验证必要列（避免后续报错）
        required_cols = ["trial", "config", "mse", "mae", "smape", "time_sec"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV缺少必要列：{', '.join(missing_cols)}")
        
        # 安全解析config列（处理CSV中可能的转义符，如""→"）
        def safe_json_load(x):
            try:
                # 处理CSV中config列的引号转义问题
                clean_x = x.strip('"').replace('""', '"')
                return json.loads(clean_x)
            except json.JSONDecodeError:
                raise ValueError(f"config列JSON格式错误，错误数据片段：{x[:50]}...")
        
        # 解析config为DataFrame并合并
        config_df = pd.DataFrame(list(df["config"].apply(safe_json_load)))
        result_df = pd.concat([df.drop("config", axis=1), config_df], axis=1)
        
        print(f"✅ 数据解析完成！共 {len(result_df)} 组实验，参数列表：{config_df.columns.tolist()}")
        return result_df
    
    except Exception as e:
        print(f"❌ 数据加载失败：{str(e)}")
        exit(1)  # 数据错误无法继续，终止程序


# --------------------------
# 核心函数2：可视化图表生成（共6类图）
# --------------------------
def plot_top_performers(df: pd.DataFrame, plot_dir: str, metric: str = "mse", top_n: int = 10) -> None:
    """生成Top N最优组合柱状图"""
    print(f"\n🎯 步骤2/7：生成Top {top_n} 最优组合图（按{metric.upper()}排序）...")
    try:
        # 排序并提取关键参数
        sorted_df = df.sort_values(metric).head(top_n).copy()
        key_params = ["trimmer_seq_len", "normalizer_method", "denoiser_method"]  # 可自定义关键参数
        sorted_df["label"] = sorted_df.apply(
            lambda x: f"Trial {x['trial']}\n" + "\n".join([f"{p}: {x[p]}" for p in key_params]),
            axis=1
        )
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(sorted_df["label"], sorted_df[metric], color="#4CAF50", alpha=0.8, edgecolor="#2E7D32")
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2, height,
                f"{height:.6f}", ha="center", va="bottom", fontsize=8
            )
        
        # 样式配置
        ax.set_title(f"Top {top_n} Preprocessing Combinations (Sorted by {metric.upper()})", fontsize=12, pad=20)
        ax.set_xlabel("Trial ID & Key Parameters", fontsize=10)
        ax.set_ylabel(f"{metric.upper()} (Lower = Better)", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        
        # 保存图片
        save_path = os.path.join(plot_dir, f"top_{top_n}_by_{metric}.png")
        plt.tight_layout()  # 防止标签截断
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        
        print(f"✅ 已保存：{os.path.basename(save_path)}")
    
    except Exception as e:
        print(f"⚠️ 生成Top N图失败：{str(e)}")


def plot_single_param_impact(df: pd.DataFrame, plot_dir: str, param: str, metric: str = "mse") -> None:
    """生成单个参数对性能的影响折线图（带误差条）"""
    print(f"\n📊 步骤3/7：生成参数 '{param}' 影响图...")
    try:
        # 分组计算统计量
        grouped = df.groupby(param)[metric].agg(["mean", "std", "count"]).reset_index()
        grouped = grouped[grouped["count"] >= 1]  # 过滤无数据分组
        
        if len(grouped) < 2:
            print(f"⚠️ 参数 '{param}' 唯一值过少（仅{len(grouped)}个），跳过绘图！")
            return
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(
            x=grouped[param], y=grouped["mean"], yerr=grouped["std"],
            fmt="o-", color="#FF5722", ecolor="#BDBDBD", capsize=5,
            markersize=6, linewidth=2
        )
        
        # 添加样本量标签
        for _, row in grouped.iterrows():
            ax.text(
                row[param], row["mean"] + row["std"] + 0.0001,
                f"n={int(row['count'])}", ha="center", va="bottom", fontsize=8
            )
        
        # 样式配置
        ax.set_title(f"Impact of {param} on {metric.upper()}", fontsize=12, pad=20)
        ax.set_xlabel(param, fontsize=10)
        ax.set_ylabel(f"Average {metric.upper()}", fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")
        
        # 保存
        save_path = os.path.join(plot_dir, f"param_impact_{param}_vs_{metric}.png")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        
        print(f"✅ 已保存：{os.path.basename(save_path)}")
    
    except Exception as e:
        print(f"⚠️ 生成'{param}'影响图失败：{str(e)}")


def plot_param_interaction(df: pd.DataFrame, plot_dir: str, param1: str, param2: str, metric: str = "mse") -> None:
    """生成两个参数交互影响的热力图"""
    print(f"\n🔗 步骤4/7：生成参数 '{param1}' 与 '{param2}' 交互图...")
    try:
        # 分组并转为热力图格式
        interaction_data = df.groupby([param1, param2])[metric].mean().reset_index()
        pivot_data = interaction_data.pivot(index=param2, columns=param1, values=metric)
        
        if pivot_data.empty:
            print(f"⚠️ 无 '{param1}-{param2}' 交互数据，跳过绘图！")
            return
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            pivot_data, annot=True, fmt=".6f", cmap="YlGnBu_r",  # 反向色：值越低颜色越深
            cbar_kws={"label": f"Average {metric.upper()}", "shrink": 0.8},
            annot_kws={"fontsize": 8}, linewidths=0.5, ax=ax
        )
        
        # 样式配置
        ax.set_title(f"Interaction Between {param1} & {param2}\n(Metric: {metric.upper()})", fontsize=12, pad=20)
        ax.set_xlabel(param1, fontsize=10)
        ax.set_ylabel(param2, fontsize=10)
        
        # 保存
        save_path = os.path.join(plot_dir, f"param_interaction_{param1}_vs_{param2}_{metric}.png")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        
        print(f"✅ 已保存：{os.path.basename(save_path)}")
    
    except Exception as e:
        print(f"⚠️ 生成交互图失败：{str(e)}")


def plot_metric_corr(df: pd.DataFrame, plot_dir: str) -> None:
    """生成性能指标相关性矩阵热力图"""
    print(f"\n📈 步骤5/7：生成指标相关性矩阵图...")
    try:
        # 选择指标列
        metrics = ["mse", "mae", "smape", "time_sec"]
        valid_metrics = [col for col in metrics if col in df.columns]
        
        if len(valid_metrics) < 2:
            print(f"⚠️ 有效指标过少（仅{valid_metrics}），跳过绘图！")
            return
        
        # 计算相关性矩阵
        corr_matrix = df[valid_metrics].corr()
        
        # 绘图
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix, annot=True, fmt=".3f", cmap="coolwarm",
            vmin=-1, vmax=1, cbar_kws={"label": "Pearson Correlation"},
            annot_kws={"fontsize": 10}, linewidths=0.5, ax=ax
        )
        
        # 样式配置
        ax.set_title("Correlation Matrix: Metrics & Inference Time", fontsize=12, pad=20)
        
        # 保存
        save_path = os.path.join(plot_dir, "metric_correlation_matrix.png")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        
        print(f"✅ 已保存：{os.path.basename(save_path)}")
    
    except Exception as e:
        print(f"⚠️ 生成相关性矩阵图失败：{str(e)}")


def plot_time_vs_metric(df: pd.DataFrame, plot_dir: str, metric: str = "mse") -> None:
    """生成推理时间与性能的权衡散点图"""
    print(f"\n⚖️ 步骤6/7：生成推理时间与{metric.upper()}权衡图...")
    try:
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 5))
        scatter = ax.scatter(
            df["time_sec"], df[metric],
            c=df[metric], cmap="viridis_r",  # 颜色映射：性能越好颜色越深
            alpha=0.6, s=50, edgecolors="gray", linewidths=0.5
        )
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f"{metric.upper()} (Lower = Better)", fontsize=10)
        
        # 样式配置
        ax.set_title(f"Inference Time vs. {metric.upper()} (Trade-off Analysis)", fontsize=12, pad=20)
        ax.set_xlabel("Inference Time (sec)", fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")
        
        # 保存
        save_path = os.path.join(plot_dir, f"time_vs_{metric}_tradeoff.png")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        
        print(f"✅ 已保存：{os.path.basename(save_path)}")
    
    except Exception as e:
        print(f"⚠️ 生成权衡图失败：{str(e)}")


# --------------------------
# 核心函数3：最优参数汇总（控制台输出）
# --------------------------
def print_optimal_params(df: pd.DataFrame, metric: str = "mse", top_n: int = 5) -> None:
    """在控制台输出Top N最优预处理组合汇总"""
    print(f"\n🏆 步骤7/7：输出Top {top_n} 最优预处理组合...")
    try:
        # 排序并筛选展示列
        top_df = df.sort_values(metric).head(top_n).copy()
        core_params = [col for col in df.columns if col not in ["trial", "mse", "mae", "smape", "time_sec"]]
        display_df = top_df[["trial", metric] + core_params].reset_index(drop=True)
        display_df.index += 1  # 索引从1开始，更易读
        
        # 打印汇总表
        print(f"\n【Top {top_n} Optimal Combinations (Sorted by {metric.upper()})】")
        print("-" * 150)
        pd.set_option('display.max_columns', None)  # 显示所有列
        pd.set_option('display.width', 150)         # 适配宽屏
        pd.set_option('display.max_colwidth', 20)   # 限制列宽
        print(display_df)
        
        # 统计最优参数出现频率（辅助决策）
        print(f"\n【Parameter Frequency in Top {top_n}】")
        print("-" * 80)
        for param in core_params[:5]:  # 只显示前5个关键参数，避免输出过长
            freq = top_df[param].value_counts().head(2)
            print(f"{param:<20} {freq.to_dict()}")
        
        print(f"\n💡 建议：优先选择Top 3组合的参数交集，平衡性能与稳定性！")
    
    except Exception as e:
        print(f"⚠️ 输出最优参数失败：{str(e)}")

# 修复后的操作组合与基准对比图函数
def plot_operation_vs_baseline(df: pd.DataFrame, plot_dir: str, metric: str = "mse") -> None:
    """
    生成各操作组合与不做操作（基准）的性能对比图
    已修复'trimmer_method'不存在的错误
    """
    print(f"\n📊 生成操作组合与基准（不做操作）的{metric.upper()}对比图...")
    try:
        # 1. 定义"不做操作"的基准条件（使用正确的参数名）
        baseline_mask = (
            (df["denoiser_method"] == "none") & 
            (df["normalizer_method"] == "none") &
            (df["warper_method"] == "none") &
            (df["differentiator_n"] == 0) &
            (df["clip_factor"] == "none")
        )
        baseline_data = df[baseline_mask]
        
        if len(baseline_data) == 0:
            print("⚠️ 未找到完全'不做操作'的基准数据，使用最简化操作作为基准")
            baseline_mask = (df["denoiser_method"] == "none") & (df["normalizer_method"] == "none")
            baseline_data = df[baseline_mask]
            
            if len(baseline_data) == 0:
                raise ValueError("无法找到合适的基准数据（不做操作的记录）")
        
        # 取基准的平均值作为比较标准
        baseline_value = baseline_data[metric].mean()
        print(f"📌 基准{metric.upper()}值: {baseline_value:.8f} (基于{len(baseline_data)}条基准数据)")
        
        # 2. 提取操作组合的数据
        operation_mask = ~baseline_mask
        operation_data = df[operation_mask].copy()
        
        # 创建组合标识（使用正确的trimmer_seq_len参数，移除trimmer_method）
        operation_data["combination"] = operation_data.apply(
            lambda x: f"{x['denoiser_method']}+{x['normalizer_method']}+{x['warper_method']}+{x['trimmer_seq_len']}",
            axis=1
        )
        
        # 按组合分组取平均
        grouped_ops = operation_data.groupby("combination")[metric].mean().reset_index()
        grouped_ops = grouped_ops.sort_values(metric)
        
        # 3. 计算与基准的差异百分比
        grouped_ops["diff_from_baseline"] = (grouped_ops[metric] - baseline_value) / baseline_value * 100
        
        # 4. 绘图
        fig_width = min(14, 2 + len(grouped_ops) * 0.8)
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        
        # 绘制操作组合柱形
        bars = ax.bar(
            grouped_ops["combination"], 
            grouped_ops[metric], 
            color=np.where(grouped_ops["diff_from_baseline"] < 0, "#4CAF50", "#F44336"),
            alpha=0.8, 
            edgecolor="black"
        )
        
        # 绘制基准线
        ax.axhline(y=baseline_value, color="#2196F3", linestyle="--", linewidth=2, label="Baseline (No Operation)")
        
        # 添加差异百分比标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            diff = grouped_ops["diff_from_baseline"].iloc[i]
            label = f"{diff:+.1f}%"
            
            va = "bottom" if height > baseline_value else "top"
            ax.text(
                bar.get_x() + bar.get_width()/2, height,
                label, ha="center", va=va,
                color="black", fontweight="bold", rotation=90
            )
        
        # 样式配置
        ax.set_title(f"Performance Comparison: Operations vs. Baseline ({metric.upper()})", fontsize=14, pad=20)
        ax.set_xlabel("Operation Combinations (denoiser+normalizer+warper+trimmer_len)", fontsize=12)
        ax.set_ylabel(f"{metric.upper()} (Lower = Better)", fontsize=12)
        ax.tick_params(axis="x", rotation=90, labelsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=10)
        
        # 保存图片
        save_path = os.path.join(plot_dir, f"operations_vs_baseline_{metric}.png")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        
        print(f"✅ 已保存：{os.path.basename(save_path)}")
    
    except Exception as e:
        print(f"⚠️ 生成操作对比图失败：{str(e)}")

from math import pi
# 修复后的雷达图对比函数

def get_diverse_combinations(df: pd.DataFrame, max_combinations: int = 16) -> pd.DataFrame:
    """
    从所有组合中选择最具多样性的组合，确保覆盖不同类型的预处理方法
    
    参数:
        df: 包含所有实验数据的DataFrame
        max_combinations: 最大返回组合数量
    返回:
        具有代表性的组合DataFrame
    """
    # 创建组合标识
    df["combination"] = df.apply(
        lambda x: f"{x['denoiser_method']}+{x['normalizer_method']}+{x['warper_method']}+{x['trimmer_seq_len']}",
        axis=1
    )
    
    # 按组合分组计算各指标平均值
    grouped_ops = df.groupby("combination").agg({
        "mse": "mean",
        "mae": "mean",
        "smape": "mean",
        "denoiser_method": "first",
        "normalizer_method": "first",
        "warper_method": "first",
        "trimmer_seq_len": "first"
    }).reset_index()
    
    # 如果总组合数小于等于max_combinations，直接返回所有
    if len(grouped_ops) <= max_combinations:
        return grouped_ops
    
    # 关键参数类别
    key_params = [
        "denoiser_method", 
        "normalizer_method", 
        "warper_method"
    ]
    
    # 为每个关键参数计算其不同取值的覆盖率
    def calculate_coverage(selected, all_values, param):
        """计算特定参数的覆盖率"""
        unique_vals = set(all_values[param].unique())
        selected_vals = set(selected[param].unique())
        return len(selected_vals) / len(unique_vals) if unique_vals else 1.0
    
    # 贪心算法选择最具多样性的组合
    selected = pd.DataFrame(columns=grouped_ops.columns)
    
    # 首先确保包含基准组合（无预处理）
    baseline_mask = (
        (grouped_ops["denoiser_method"] == "none") & 
        (grouped_ops["normalizer_method"] == "none") &
        (grouped_ops["warper_method"] == "none")
    )
    if grouped_ops[baseline_mask].any().any():
        baseline = grouped_ops[baseline_mask].iloc[0:1]
        selected = pd.concat([selected, baseline], ignore_index=True)
    
    # 迭代选择能最大化参数覆盖率的组合
    while len(selected) < max_combinations:
        best_coverage = -1
        best_index = -1
        
        for i, candidate in grouped_ops.iterrows():
            # 跳过已选择的组合
            if candidate["combination"] in selected["combination"].values:
                continue
                
            # 临时添加候选组合
            temp_selected = pd.concat([selected, grouped_ops.iloc[i:i+1]], ignore_index=True)
            
            # 计算当前覆盖率
            coverage_scores = [
                calculate_coverage(temp_selected, grouped_ops, param) 
                for param in key_params
            ]
            avg_coverage = np.mean(coverage_scores)
            
            # 记录最佳候选
            if avg_coverage > best_coverage:
                best_coverage = avg_coverage
                best_index = i
        
        # 添加最佳候选
        if best_index != -1:
            selected = pd.concat([selected, grouped_ops.iloc[best_index:best_index+1]], ignore_index=True)
        else:
            break  # 没有更多组合可供选择
    
    return selected

def plot_radar_diverse_combinations(df: pd.DataFrame, plot_dir: str, max_combinations: int = 16) -> None:
    """
    生成雷达图，展示最多16个具有代表性的操作组合，每个组合包含MSE、MAE、SMAPE三条指标线
    
    参数:
        df: 包含所有实验数据的DataFrame
        plot_dir: 图片保存目录
        max_combinations: 最大展示的组合数量，默认16
    """
    print("\n📊 生成16个代表性组合的MSE、MAE、SMAPE雷达对比图...")
    try:
        # 1. 选择最具多样性的组合
        combinations = get_diverse_combinations(df, max_combinations)
        num_combinations = len(combinations)
        print(f"📌 展示{num_combinations}个具有代表性的操作组合")
        
        # 2. 准备雷达图数据
        metrics = ["mse", "mae", "smape"]
        metric_labels = ["MSE", "MAE", "SMAPE (%)"]
        num_metrics = len(metrics)
        
        # 数据标准化（使用所有组合的最大值进行标准化）
        all_groups = df.groupby("combination").agg({
            "mse": "mean",
            "mae": "mean",
            "smape": "mean"
        }).reset_index()
        
        max_values = {
            "mse": all_groups["mse"].max(),
            "mae": all_groups["mae"].max(),
            "smape": all_groups["smape"].max()
        }
        
        def normalize(value, metric):
            """将值标准化到0-1范围（值越小越好）"""
            return value / max_values[metric] if max_values[metric] != 0 else 0
        
        # 3. 绘制雷达图
        angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
        angles += [angles[0]]  # 闭合图形
        metric_labels += [metric_labels[0]]  # 闭合标签
        
        # 创建画布
        fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))
        
        # 为不同指标定义颜色和样式
        metric_styles = {
            "mse": {"color": "#2196F3", "marker": "o", "linewidth": 1.5},    # 蓝色
            "mae": {"color": "#4CAF50", "marker": "s", "linewidth": 1.5},    # 绿色
            "smape": {"color": "#FF9800", "marker": "^", "linewidth": 1.5}   # 橙色
        }
        
        # 为每个组合绘制三条指标线
        for _, row in combinations.iterrows():
            combo_name = row["combination"]
            
            for metric in metrics:
                # 准备该指标的数据
                values = [0 if metric != m else normalize(row[metric], metric) for m in metrics]
                values += [values[0]]  # 闭合图形
                
                # 只在第一个组合添加图例
                label = metric_labels[metrics.index(metric)] if combo_name == combinations.iloc[0]["combination"] else ""
                ax.plot(angles, values, 
                        color=metric_styles[metric]["color"],
                        marker=metric_styles[metric]["marker"],
                        linewidth=metric_styles[metric]["linewidth"],
                        alpha=0.7, label=label)
        
        # 添加标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels[:-1], size=12)
        
        # 添加图例
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11,
                 title="Metrics", title_fontsize=13)
        
        # 优化组合名称显示（分两列放置以避免重叠）
        radius = ax.get_ylim()[1] * 1.05
        for i, row in combinations.iterrows():
            # 偶数行放右侧，奇数行放左侧
            angle = angles[1] if i % 2 == 0 else angles[1] + pi
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # 文本旋转角度与放置位置匹配
            rotation = 0 if i % 2 == 0 else 180
            ha = 'left' if i % 2 == 0 else 'right'
            
            ax.text(x, y + (i//2)*0.1, row["combination"], 
                    ha=ha, va='center', rotation=rotation, 
                    fontsize=9, alpha=0.8)
        
        # 添加标题
        ax.set_title(f"Diverse Operation Combinations Performance Comparison ({num_combinations} combinations)", 
                    size=16, pad=20)
        
        # 设置径向轴范围
        ax.set_ylim(0, 1.1)
        
        # 添加网格线使阅读更方便
        ax.grid(True, alpha=0.3)
        
        # 保存图片
        save_path = os.path.join(plot_dir, f"radar_diverse_{num_combinations}_combinations.png")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        
        print(f"✅ 已保存：{os.path.basename(save_path)}")
    
    except Exception as e:
        print(f"⚠️ 生成雷达图失败：{str(e)}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import pi

def get_16_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """获取16个具有代表性的组合（确保覆盖不同类型）"""
    # 创建组合标识
    df["combination"] = df.apply(
        lambda x: f"{x['denoiser_method']}+{x['normalizer_method']}+{x['warper_method']}+{x['trimmer_seq_len']}",
        axis=1
    )
    
    # 按组合分组计算各指标平均值
    grouped_ops = df.groupby("combination").agg({
        "mse": "mean",
        "mae": "mean",
        "smape": "mean"
    }).reset_index()
    
    # 确保我们有正好16个组合（如果不足则全部使用，如果过多则采样）
    if len(grouped_ops) <= 16:
        return grouped_ops
    else:
        # 按MSE排序后均匀采样16个，确保覆盖不同性能区间
        grouped_ops = grouped_ops.sort_values("mse")
        indices = np.linspace(0, len(grouped_ops)-1, 16, dtype=int)
        return grouped_ops.iloc[indices]

def plot_three_lines_radar(df: pd.DataFrame, plot_dir: str) -> None:
    """
    生成雷达图，包含三条线（MSE、MAE、SMAPE），每条线上有16个点代表16个组合
    
    参数:
        df: 包含所有实验数据的DataFrame
        plot_dir: 图片保存目录
    """
    print("\n📊 生成三条线（MSE、MAE、SMAPE）各16个点的雷达图...")
    try:
        # 1. 获取16个组合
        combinations = get_16_combinations(df)
        num_combinations = len(combinations)
        print(f"📌 展示{num_combinations}个组合，每条指标线有{num_combinations}个点")
        
        # 2. 准备雷达图数据
        metrics = ["mse", "mae", "smape"]
        metric_labels = ["MSE", "MAE", "SMAPE (%)"]
        
        # 为每个指标单独标准化（同一指标内的组合比较）
        metric_scalers = {}
        for metric in metrics:
            # 使用该指标的最大值进行标准化
            max_val = combinations[metric].max()
            min_val = combinations[metric].min()
            metric_scalers[metric] = {"max": max_val, "min": min_val}
        
        def normalize(value, metric):
            """将值标准化到0-1范围（值越小越好）"""
            scaler = metric_scalers[metric]
            range_val = scaler["max"] - scaler["min"]
            return (value - scaler["min"]) / range_val if range_val != 0 else 0
        
        # 3. 绘制雷达图
        # 计算角度（16个点 + 闭合点）
        num_points = num_combinations
        angles = [n / float(num_points) * 2 * pi for n in range(num_points)]
        angles += [angles[0]]  # 闭合图形
        
        # 创建画布
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
        
        # 为不同指标定义样式
        metric_styles = {
            "mse": {"color": "#2196F3", "marker": "o", "linewidth": 2, "label": "MSE"},
            "mae": {"color": "#4CAF50", "marker": "s", "linewidth": 2, "label": "MAE"},
            "smape": {"color": "#FF9800", "marker": "^", "linewidth": 2, "label": "SMAPE"}
        }
        
        # 为每个指标绘制一条线，线上有16个点（每个组合一个点）
        for metric in metrics:
            # 准备该指标的16个点数据
            values = [normalize(combinations.iloc[i][metric], metric) for i in range(num_points)]
            values += [values[0]]  # 闭合图形
            
            # 绘制线
            style = metric_styles[metric]
            ax.plot(angles, values, 
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=style["linewidth"],
                    label=style["label"],
                    alpha=0.8)
        
        # 添加组合标签（16个点的标签）
        ax.set_xticks(angles[:-1])  # 排除最后一个闭合点
        ax.set_xticklabels(combinations["combination"], size=8, rotation=0)
        
        # 添加图例
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        
        # 添加标题
        ax.set_title(f"Performance Metrics Across 16 Operation Combinations", 
                    size=14, pad=20)
        
        # 设置径向轴范围（0表示该指标最佳，1表示该指标最差）
        ax.set_ylim(0, 1.1)
        
        # 添加网格线
        ax.grid(True, alpha=0.3)
        
        # 调整标签角度以避免重叠
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 保存图片
        save_path = os.path.join(plot_dir, "radar_three_lines_16_points.png")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        
        print(f"✅ 已保存：{os.path.basename(save_path)}")
    
    except Exception as e:
        print(f"⚠️ 生成雷达图失败：{str(e)}")


# --------------------------
# 主函数：流程控制（串联所有模块）
# --------------------------
def main():
    # 1. 欢迎信息
    print("=" * 80)
    print("                Preprocessing Parameter Analysis Tool")
    print("=" * 80)
    print("功能说明：")
    print("1. 从控制台输入CSV路径（自动验证有效性）")
    print("2. 在CSV同级目录创建图片文件夹（无弹窗，仅保存文件）")
    print("3. 生成6类分析图：Top N组合、参数影响、交互热力图等")
    print("4. 输出最优预处理组合汇总，辅助参数选择")
    print("=" * 80)
    
    # 2. 获取有效CSV路径
    csv_path = get_valid_csv_path()
    
    # 3. 创建图片文件夹
    plot_dir = create_plot_dir(csv_path)
    
    # 4. 加载解析数据
    df = load_parse_data(csv_path)
    df = df[df["mse"] <= max_mse].copy()
    # 5. 生成Top N最优组合图
    plot_top_performers(df, plot_dir, metric="mse", top_n=10)
    
    # 6. 生成单个参数影响图（可根据实际参数调整列表）
    key_params = ["trimmer_seq_len", "normalizer_method", "denoiser_method", "sampler_factor"]
    for param in key_params:
        plot_single_param_impact(df, plot_dir, param, metric="mse")
    
    # 7. 生成参数交互图（选择关键参数对）
    plot_param_interaction(df, plot_dir, "trimmer_seq_len", "normalizer_method", metric="mse")
    plot_param_interaction(df, plot_dir, "normalizer_method", "denoiser_method", metric="mse")
    
    # 8. 生成指标相关性矩阵图
    plot_metric_corr(df, plot_dir)

    plot_operation_vs_baseline(df, plot_dir, metric="mse")
    plot_operation_vs_baseline(df, plot_dir, metric="smape")
    plot_radar_diverse_combinations(df, plot_dir, max_combinations=16)
    plot_three_lines_radar(df, plot_dir)


    # # 9. 生成推理时间与性能权衡图
    # plot_time_vs_metric(df, plot_dir, metric="mse")
    # plot_time_vs_metric(df, plot_dir, metric="smape")  # 额外生成SMAPE的权衡图
    
    # 10. 输出最优参数汇总
    print_optimal_params(df, metric="mse", top_n=5)
    
    # 11. 完成提示
    print("\n" + "=" * 80)
    print("分析完成！所有图表已保存至：")
    print(f"📁 {plot_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
