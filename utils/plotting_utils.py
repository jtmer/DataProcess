import matplotlib.pyplot as plt
import os
import numpy as np
import logging
import plotly.graph_objs as go
from .param_utils import get_params_space_and_org, get_valid_params_mode_data

def many_plot(pd_data, mode, res_dir):
    # 只需要space而不需要params_list是因为 画图需要遍历所有的组合，而不仅仅是HPO选中的！
    logging.info('\nBegin to plot...')

    params_space, origin_param_dict = get_params_space_and_org()

    # 先把split_idx用mse的mean聚合掉！！！
    grouped_cols = list(params_space.keys()) + ['mae', 'mse', 'rmse', 'mape', 'mspe']
    data_grouped_by_params = pd_data[pd_data['mode'] == mode][grouped_cols] \
        .groupby(list(params_space.keys())).mean().reset_index()
    param_names = list(params_space.keys())


    # 平行坐标图！
    # n*x=n*param_name+mse
    # 目标：展示不同超参数组合下的 平均mse （使用聚合后的数据）
    # 创建维度对象
    logging.info('Creating parallel coordinates plot...')
    dimensions = []
    logging.info('Adding dimensions for hyperparameters...')
    for param_name in param_names:
        param_values = params_space[param_name]['values']
        param_type = params_space[param_name]['type']
        if param_type in ['float', 'int']:
            range_min = min(param_values)
            range_max = max(param_values)
            values = data_grouped_by_params[param_name].tolist()
            dimension = dict(
                range=[range_min, range_max],
                label=param_name,
                values=values,
            )
        elif param_type == 'str':
            range_min = 0
            range_max = len(param_values) - 1  # Use index as range for string values
            values = [param_values.index(value) for value in data_grouped_by_params[param_name].tolist()]
            texts = param_values
            dimension = dict(
                range=[range_min, range_max],
                label=param_name,
                values=values,
                tickvals=list(range(len(texts))),  # Use index as tickvals
                ticktext=texts,
            )
        else:
            raise ValueError(f"Unknown type: {param_type}")
        dimensions.append(dimension)
    # 加上mse等metrics的dimension
    logging.info('Adding dimensions for metrics...')
    for metric in ['mae', 'mse', 'rmse', 'mape', 'mspe']:
        dimensions.append(dict(
            range=[min(data_grouped_by_params[metric]), max(data_grouped_by_params[metric])],
            label=metric,
            values=data_grouped_by_params[metric].tolist()
        ))
    # 创建数据对象
    logging.info('Creating data object...')
    data = [
        go.Parcoords(
            line=dict(color='blue'),
            dimensions=dimensions,
            labelfont=dict(size=8, color='black'),
            tickfont=dict(size=8, color='black'),
        )
    ]
    # 创建布局对象
    logging.info('Creating layout object...')
    layout = go.Layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        title=f"Parallel Coordinates Plot of MSE by Hyperparameters",
    )
    # 创建图对象
    logging.info('Creating figure object...')
    fig = go.Figure(data=data, layout=layout)
    # 保存图对象
    # py.offline.plot(fig, filename=os.path.join(res_dir, mode, '_parallel_coordinates_plot.html'))
    logging.info('Writing HTML...')
    fig.write_html(os.path.join(res_dir, mode, '_parallel_coordinates_plot.html'), auto_open=False)

    plot_hpo_progress(pd_data, res_dir, mode, params_space)

    logging.info('Plotting finished!')


def plot_hpo_progress(pd_data, res_dir, mode, params_space):  # 仅画mean_mse
    # FIXME:提前被prune掉的也被画上了。。。

    logging.info('Creating HPO progress plot...')
    # 获取未被剪枝的params的个数：
    total_num_combinations = len(pd_data[pd_data['mode'] == mode].groupby(list(params_space.keys()))
                                 .agg({'mse': 'mean'})['mse'].reset_index())
    logging.info(f'total_num_combinations={total_num_combinations}')

    # 去除了被剪枝的params数据
    valid_params_mode_data = get_valid_params_mode_data(pd_data, mode, params_space)
    data_grouped = valid_params_mode_data.groupby(list(params_space.keys())).agg({'mse': 'mean'}).reset_index()
    valid_num_combinations = len(data_grouped)
    valid_mse_values = data_grouped['mse'].values
    logging.info(f'valid_num_combinations={valid_num_combinations}')
    logging.info(f'valid mse_values={valid_mse_values}')

    # 去除掉异常值以免影响画图：-> 画图时限制即可
    lower_bound = 0  # 无限制
    upper_bound = min(valid_mse_values) * 30  # 一个数量级多一点
    # mse_values = np.clip(mse_values, lower_bound, upper_bound)
    # logging.info(f'mse_values after clipping={mse_values}')

    # 初始化最佳 MSE 的跟踪列表
    best_mse_values = []
    best_mse = float('inf')
    for mse in valid_mse_values:
        if mse < best_mse:
            best_mse = mse
        best_mse_values.append(best_mse)
    logging.info(f'best_mse_values={best_mse_values}')

    plt.figure(figsize=(10, 6))

    # 绘制散点图
    plt.scatter(range(1, valid_num_combinations + 1), valid_mse_values, label='MSE', color='blue', s=10)

    # 绘制最佳 MSE 折线图
    plt.plot(range(1, valid_num_combinations + 1), best_mse_values, label='Best MSE', color='red')

    plt.xlabel('Number of parameter combinations')
    plt.ylabel('MSE (log scale)')
    plt.yscale('log')  # 设置 y 轴为对数刻度
    plt.ylim(lower_bound, upper_bound)  # 限制 y 轴范围
    plt.title(f'HPO Progress: MSE over parameter combinations {mode} Mode (Total: {total_num_combinations})')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plot_path = os.path.join(res_dir, mode, '_hpo_progress_plot.png')
    plt.savefig(plot_path, bbox_inches='tight')
    logging.info(f'HPO progress plot saved to {plot_path}')



def plot_metric_hist_comparison(pd_data, mode, metric_names, res_dir, val_top1_param_dict):
    logging.info(f"Begin to plot histograms for {metric_names} in mode={mode}...")

    params_space, origin_param_dict = get_params_space_and_org()

    mode_data = pd_data[pd_data['mode'] == mode]
    mask_origin = np.logical_and.reduce([mode_data[key] == value for key, value in origin_param_dict.items()])
    mask_val_top1 = np.logical_and.reduce([mode_data[key] == value for key, value in val_top1_param_dict.items()])

    for metric_name in metric_names:
        logging.info(f"Plotting histogram for {metric_name}...")
        origin_metric_values = mode_data[mask_origin][metric_name].values
        val_top1_metric_values = mode_data[mask_val_top1][metric_name].values

        assert len(origin_metric_values) == len(val_top1_metric_values), \
            f"Lengths do not match: {len(origin_metric_values)} != {len(val_top1_metric_values)}"

        plt.figure(figsize=(12, 6))

        # # Plot histograms for origin and val_top1 metric values
        # plt.hist(origin_metric_values, bins=50, alpha=0.5, label='Origin', color='blue', log=True)
        # plt.hist(val_top1_metric_values, bins=50, alpha=0.5, label='Val Top1', color='orange', log=True)

        # Compute histograms
        bins = np.linspace(min(origin_metric_values.min(), val_top1_metric_values.min()),
                           max(origin_metric_values.max(), val_top1_metric_values.max()), 50)
        origin_hist, bins = np.histogram(origin_metric_values, bins=bins)
        val_top1_hist, _ = np.histogram(val_top1_metric_values, bins=bins)
        # Plot histograms with same bar width
        bin_centers = (bins[:-1] + bins[1:]) / 2
        width = (bins[1] - bins[0]) * 1  # Set bar width to 40% of bin width # 中间白色太亮眼 0.4 0.5 1
        plt.bar(bin_centers - width / 2 * 0, origin_hist, width=width, label='Origin', color='orange', alpha=0.5)
        plt.bar(bin_centers + width / 2 * 0, val_top1_hist, width=width, label='Val Top1', color='blue', alpha=0.5)
        # FIXME: log or not
        # plt.xscale('log') # bin宽度形状会变得不一致
        # plt.yscale('log') # 面积比例会不一致？
        # 看不出来明显优势？？？ -》 xlog才能显示出大小值的差异？
        # plt.bar(bin_centers - width / 2, origin_hist, width=width, alpha=0.5, label='Origin', color='blue')
        # plt.bar(bin_centers + width / 2, val_top1_hist, width=width, alpha=0.5, label='Val Top1', color='orange')

        plt.xlabel(metric_name)
        plt.ylabel('Frequency')
        plt.title(f'{metric_name} Distribution in {mode} Mode')
        plt.legend(loc='upper right')

        plot_path = os.path.join(res_dir, f'_{mode}_{metric_name}_histogram.png')
        plt.savefig(plot_path)
        logging.info(f"Histogram saved to {plot_path}")
        plt.close()

