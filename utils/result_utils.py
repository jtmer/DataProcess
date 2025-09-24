# result_utils.py
import logging
import os
import itertools
import numpy as np
import pandas as pd
import re
from .param_utils import get_params_space_and_org, get_valid_params_mode_data


def calc_statistics(valid_params_mode_data, metric_names, our_param_dict):
    logging.info(f"Begin to calculate statistics ...")

    # Construct mask for the val_top1 param dict
    param_mask = np.logical_and.reduce([valid_params_mode_data[key] == value for key, value in our_param_dict.items()])

    res = {}
    # Get the metric values for the filtered data
    for metric_name in metric_names:
        logging.info(f"Calculating statistics for {metric_name}...")

        val_top1_metric_values = valid_params_mode_data[param_mask][metric_name].values

        # Calculate the mean, median, std, and IQR for val_top1 metric values
        val_top1_mean = np.mean(val_top1_metric_values)
        val_top1_median = np.median(val_top1_metric_values)
        val_top1_std = np.std(val_top1_metric_values)
        val_top1_q1 = np.percentile(val_top1_metric_values, 25)
        val_top1_q3 = np.percentile(val_top1_metric_values, 75)
        val_top1_iqr = val_top1_q3 - val_top1_q1
        val_top1_max = np.max(val_top1_metric_values)
        val_top1_min = np.min(val_top1_metric_values)

        res[f"{metric_name}_mean"] = val_top1_mean
        res[f"{metric_name}_median"] = val_top1_median
        res[f"{metric_name}_std"] = val_top1_std
        res[f"{metric_name}_iqr"] = val_top1_iqr
        res[f"{metric_name}_max"] = val_top1_max
        res[f"{metric_name}_min"] = val_top1_min
    return res




def find_top_param_dict_list(pd_data, mode, params_space, top_k, metric_weight_dict, res_dir):
    logging.info(f"Begin to find top-{top_k} param dict list in mode={mode}...")
    metric_names = list(metric_weight_dict.keys())
    rank_metric_names = [f'{metric}_rank' for metric in metric_names]

    # Filter the dataframe to include only mode data and un-pruned params
    valid_params_mode_data = get_valid_params_mode_data(pd_data, mode, params_space)

    data_by_valid_params = valid_params_mode_data.groupby(list(params_space.keys()))
    mean_metrics_by_valid_params = data_by_valid_params[metric_names].mean().reset_index()

    # 计算加权平均值
    # Calculate ranks for each metric and add to DataFrame
    for metric in metric_weight_dict.keys():
        mean_metrics_by_valid_params[f'{metric}_rank'] = mean_metrics_by_valid_params[metric].rank()
        mean_metrics_by_valid_params[f'{metric}_rank'] *= metric_weight_dict.get(metric, 1)
    # Calculate mean rank across all metrics
    total_weight = sum(metric_weight_dict.values())
    mean_metrics_by_valid_params['final_rank'] = \
        mean_metrics_by_valid_params[rank_metric_names].sum(axis=1) / total_weight

    # save the mean_metrics_by_valid_params
    mean_metrics_by_valid_params.to_csv(os.path.join(res_dir, f'_{mode}_mean_metrics_by_params.csv'), index=False)

    # Sort parameter combinations based on mean rank
    sorted_pd_data = mean_metrics_by_valid_params.sort_values(by='final_rank', ascending=True) \
        .head(top_k if top_k != 'all' else len(mean_metrics_by_valid_params))

    # Print the sorted data for inspection
    logging.debug(f"sorted_pd_data=\n{sorted_pd_data.to_string()}")

    # Convert the sorted dataframe to a list of parameter dictionaries
    best_param_dict_list = sorted_pd_data.drop(columns=metric_names + ['final_rank'] + rank_metric_names) \
        .to_dict(orient='records')
    return best_param_dict_list


def find_top_param_dict_list_pareto(pd_data, mode, params_space, top_k, metric_weight_dict, metrics_for_pareto_must,
                                    multi_pareto_mode, res_dir):
    logging.info(f"Begin to find top-{top_k} param dict list in mode={mode} for multiple metric combinations...")
    metric_names = list(metric_weight_dict.keys())

    # Filter the dataframe to include only mode data and un-pruned params
    valid_params_mode_data = get_valid_params_mode_data(pd_data, mode, params_space)
    # Compute mean MSE and other metrics for each parameter combination
    data_by_valid_params = valid_params_mode_data.groupby(list(params_space.keys()))
    mean_metrics_by_params = data_by_valid_params[metric_names].mean().reset_index()

    # mean_metrics_by_params = mean_metrics_by_params.copy()
    for metric in metric_names:
        mean_metrics_by_params[f'{metric}_rank'] = mean_metrics_by_params[metric].rank()
        mean_metrics_by_params[f'{metric}_rank'] *= metric_weight_dict.get(metric, 1)

    if multi_pareto_mode is True:  # 多个指标的排列组合 形成多个帕里托前沿
        # Generate all possible metric combinations
        metric_names_list = [list(combo) for length in range(1, len(metric_names) + 1) for combo in
                             itertools.combinations(metric_names, length)]
        rm_idxes = []
        for idx, cur_metric_names in enumerate(metric_names_list):
            # must要求必须包含的所有must指标 或者 本身是must指标的子集合
            if set(metrics_for_pareto_must).issubset(cur_metric_names):
                continue
            elif set(cur_metric_names).issubset(metrics_for_pareto_must):
                continue
            else:
                rm_idxes.append(idx)
        for idx in rm_idxes[::-1]:
            metric_names_list.pop(idx)
    else:
        metric_names_list = [metrics_for_pareto_must]  # !!!

    # Store all results
    all_best_param_dict_list = []
    for cur_metric_names in metric_names_list:
        logging.info(f"cur_metric_names={cur_metric_names}")
        # Calculate mean rank across all metrics
        cur_metric_rank_names = [f'{metric}_rank' for metric in cur_metric_names]
        total_weight = sum(metric_weight_dict[metric] for metric in cur_metric_names)
        mean_metrics_by_params['final_rank'] = mean_metrics_by_params[cur_metric_rank_names] \
                                                   .sum(axis=1) / total_weight

        # if last then save
        if cur_metric_names == metric_names_list[-1]:  # 期望比较全面的保存
            mean_metrics_by_params.to_csv(os.path.join(res_dir, f'_{mode}_mean_metrics_by_params.csv'), index=False)

        # Sort parameter combinations based on mean rank
        sorted_pd_data = mean_metrics_by_params.sort_values(by='final_rank', ascending=True) \
            .head(top_k if top_k != 'all' else len(mean_metrics_by_params))
        # Convert the sorted dataframe to a list of parameter dictionaries
        best_param_dict_list = sorted_pd_data[list(params_space.keys())].to_dict(orient='records')
        all_best_param_dict_list.extend(best_param_dict_list)

    logging.info(f"len(all_best_param_dict_list)={len(all_best_param_dict_list)}")
    return all_best_param_dict_list



def find_top_param_dict_list_by_statistics(pd_data, mode, params_space, top_k, metric_weight_dict, stats_weight_dict,
                                           res_dir):
    logging.info(f"Begin to find top-{top_k} param dict list by statistics in mode={mode}...")

    metric_names = list(metric_weight_dict.keys())
    statistics_names = list(stats_weight_dict.keys())
    statistics_metric_names = [f'{metric}_{stat}' for metric in metric_names for stat in statistics_names]
    rank_metric_names = [f'{stat_metric}_rank' for stat_metric in statistics_metric_names]

    # Filter the dataframe to include only mode data and un-pruned params
    valid_params_mode_data = get_valid_params_mode_data(pd_data, mode, params_space)
    # FIXME：计算出不重复的valid_params_dict_list(先用group)
    valid_params_dict_list = valid_params_mode_data[list(params_space.keys())].drop_duplicates() \
        .to_dict(orient='records')
    logging.info(f"len(valid_params_dict_list)={len(valid_params_dict_list)}")
    logging.info(f"valid_params_mode_data=\n{valid_params_mode_data}")

    statistics_metrics_df = pd.DataFrame(columns=list(params_space.keys()) + statistics_metric_names)
    logging.info(f"Begin to calculate statistics for each parameter combination...")
    for param_dict in valid_params_dict_list:
        logging.info(f"Calculating statistics for param_dict={param_dict}...")
        statistics_dict = calc_statistics(valid_params_mode_data, metric_names, param_dict)
        logging.info(f"statistics_dict={statistics_dict}")
        # statistics_metrics_df = statistics_metrics_df.append({**param_dict, **statistics_dict}, ignore_index=True)
        statistics_metrics_df.loc[len(statistics_metrics_df)] = {**param_dict, **statistics_dict}

    logging.info(f"Begin to calculate ranks for each statistics metric...")
    for stat_metric in statistics_metric_names:
        stat = stat_metric.split('_')[-1]
        metric = '_'.join(stat_metric.split('_')[:-1])
        weight = metric_weight_dict[metric] * stats_weight_dict[stat]
        # 所有的都是越小越好
        statistics_metrics_df[f'{stat_metric}_rank'] = statistics_metrics_df[stat_metric].rank(ascending=True)
        statistics_metrics_df[f'{stat_metric}_rank'] *= weight

    total_weight = sum(metric_weight_dict[metric] * stats_weight_dict[stat]
                       for metric in metric_names for stat in statistics_names)
    statistics_metrics_df['final_rank'] = statistics_metrics_df[rank_metric_names].sum(axis=1) / total_weight

    statistics_metrics_df.to_csv(os.path.join(res_dir, f'_{mode}_statistics_metrics_by_params.csv'), index=False)

    sorted_pd_data = statistics_metrics_df.sort_values(by='final_rank', ascending=True).head(
        top_k if top_k != 'all' else len(statistics_metrics_df))

    best_param_dict_list = sorted_pd_data[list(params_space.keys())].to_dict(orient='records')
    return best_param_dict_list


def find_top_param_dict_list_pareto_by_statistics(pd_data, mode, params_space, top_k, metric_weight_dict,
                                                  metrics_for_pareto_must, stats_weight_dict,
                                                  res_dir):
    logging.info(f"Begin to find top-{top_k} param dict list by statistics in mode={mode} with Pareto optimization...")

    metric_names = list(metric_weight_dict.keys())
    statistics_names = list(stats_weight_dict.keys())
    statistics_metric_names = [f'{metric}_{stat}' for metric in metric_names for stat in statistics_names]
    rank_stat_metric_names = [f'{stat_metric}_rank' for stat_metric in statistics_metric_names]

    # Filter the dataframe to include only mode data and un-pruned params
    valid_params_mode_data = get_valid_params_mode_data(pd_data, mode, params_space)
    # FIXME：计算出不重复的valid_params_dict_list(先用group)
    valid_params_dict_list = valid_params_mode_data[list(params_space.keys())].drop_duplicates() \
        .to_dict(orient='records')
    logging.info(f"len(valid_params_dict_list)={len(valid_params_dict_list)}")

    logging.info(f"Begin to calculate statistics for each parameter combination...")
    statistics_metrics_df = pd.DataFrame(columns=list(params_space.keys()) + statistics_metric_names)
    for param_dict in valid_params_dict_list:
        statistics_dict = calc_statistics(valid_params_mode_data, metric_names, param_dict)
        statistics_metrics_df.loc[len(statistics_metrics_df)] = {**param_dict, **statistics_dict}

    logging.info(f"Begin to calculate ranks for each statistics metric...")
    for stat_metric in statistics_metric_names:
        stat = stat_metric.split('_')[-1]
        metric = '_'.join(stat_metric.split('_')[:-1])
        weight = metric_weight_dict[metric] * stats_weight_dict[stat]
        statistics_metrics_df[f'{stat_metric}_rank'] = statistics_metrics_df[stat_metric].rank(
            ascending=True)  # 所有的都是越小越好
        statistics_metrics_df[f'{stat_metric}_rank'] *= weight

    # 单独计算最全的final_rank并保存
    logging.info(f"Begin to calculate final rank for all statistics metrics...")
    total_weight = sum(metric_weight_dict[metric] * stats_weight_dict[stat]
                       for metric in metric_names for stat in statistics_names)
    statistics_metrics_df['final_rank'] = statistics_metrics_df[rank_stat_metric_names].sum(axis=1) / total_weight
    statistics_metrics_df.to_csv(os.path.join(res_dir, f'_{mode}_statistics_metrics_by_params.csv'), index=False)

    # # Generate all possible metric combinations
    # statistics_metric_names_list = [list(combo) for length in range(1, len(statistics_metric_names) + 1) for combo in
    #                                 itertools.combinations(statistics_metric_names, length)]
    # # 排除不包含must指标的组合
    # rm_index_list = []
    # for idx, stat_metric_names in enumerate(statistics_metric_names_list):
    #     # 要求必须包含的所有must指标 或者 本身是must指标的子集合
    #     if set(stat_metric_names_for_pareto_must).issubset(stat_metric_names):
    #         continue
    #     elif set(stat_metric_names).issubset(stat_metric_names_for_pareto_must):
    #         continue
    #     else:
    #         rm_index_list.append(idx)
    # for idx in rm_index_list[::-1]:
    #     statistics_metric_names_list.pop(idx)

    # Generate all possible metric combinations of metric_names
    metric_names_list = [list(combo) for length in range(1, len(metric_names) + 1) for combo in
                         itertools.combinations(metric_names, length)]
    rm_idxes = []
    for idx, cur_metric_names in enumerate(metric_names_list):
        # 要求必须包含的所有must指标 或者 本身是must指标的子集合
        if set(metrics_for_pareto_must).issubset(cur_metric_names):
            continue
        elif set(cur_metric_names).issubset(metrics_for_pareto_must):
            continue
        else:
            rm_idxes.append(idx)
    for idx in rm_idxes[::-1]:
        metric_names_list.pop(idx)
    # Add statistics for each metric combination
    statistics_metric_names_list = []
    for cur_metric_names in metric_names_list:
        cur_stat_metric_names = []
        for metric in cur_metric_names:
            cur_stat_metric_names.extend([f'{metric}_{stat}' for stat in statistics_names])
        statistics_metric_names_list.append(cur_stat_metric_names)

    logging.info(f"Begin to find Pareto optimal param dict list...")
    pareto_optimal_dict_list = []
    for cur_stat_metric_names in statistics_metric_names_list:
        logging.info(f"cur_stat_metric_names={cur_stat_metric_names}")
        cur_rank_metric_names = [f'{metric}_rank' for metric in cur_stat_metric_names]
        total_weight = sum(metric_weight_dict[metric.split('_')[0]] * stats_weight_dict[metric.split('_')[1]]
                           for metric in cur_stat_metric_names)
        statistics_metrics_df['pareto_rank'] = statistics_metrics_df[cur_rank_metric_names].sum(axis=1) / total_weight
        sorted_pareto_pd_data = statistics_metrics_df.sort_values(by='pareto_rank', ascending=True).head(
            top_k if top_k != 'all' else len(statistics_metrics_df))
        pareto_dict_list = sorted_pareto_pd_data[list(params_space.keys())].to_dict(orient='records')
        pareto_optimal_dict_list.extend(pareto_dict_list)

    logging.info(f"len(pareto_optimal_dict_list)={len(pareto_optimal_dict_list)}")
    return pareto_optimal_dict_list


def find_result_by_param_dict(pd_data, mode, params_space, param_dict):
    metric_name_list = ['mae', 'mse', 'rmse', 'mape', 'mspe']
    grouped_cols = list(params_space.keys()) + metric_name_list
    data_grouped = pd_data[pd_data['mode'] == mode][grouped_cols] \
        .groupby(list(params_space.keys())).mean().reset_index()
    # print('data_grouped', data_grouped)
    # 构造一个列表，包含每个键值对的字典
    filters = [data_grouped[key] == value for key, value in param_dict.items()]
    # 使用all()方法确保所有过滤器都匹配
    matched_rows = data_grouped[np.logical_and.reduce(filters)]
    assert len(matched_rows) == 1, f"len(matched_rows)={len(matched_rows)}"  # 保证只有一个匹配
    row = matched_rows.iloc[0]
    # print('row', row)
    res = {metric_name: row[metric_name] for metric_name in metric_name_list}
    # 加入选出的具体的超参数取值
    res.update({key: row[key] for key in params_space.keys()})
    return res

def calc_improve_percent1(pd_data, mode, params_space, metric_names, val_top1_param_dict):  # FIXME: 顺序！
    # FIXME：之前粒度最小是task，现在是samples
    logging.info(f"Begin to calculate improvement percent1 in mode={mode}...")
    origin_param_dict = get_params_space_and_org()[1]

    result_dict_org = find_result_by_param_dict(pd_data, mode, params_space, origin_param_dict)
    result_dict_our = find_result_by_param_dict(pd_data, mode, params_space, val_top1_param_dict)
    result_dict = {}
    for key in metric_names:
        org, our = result_dict_org[key], result_dict_our[key]
        result_dict[f'org_{key}'] = org
        result_dict[f'our_{key}'] = our
        # 计算improvement_percent （正数越大越好（又因为our越小越好所以是org-our
        # FIXME：
        result_dict[f'improve_percent1_{key}'] = (org - our) / org * 100
        # result_dict[f'improve_percent_{key}'] = (org - our) / ((org + our) / 2) * 100  # SMAPE 平均百分比误差
        # result_dict[f'improve_percent_{key}'] = (org - our) / (org + our) * 100
    return result_dict


# per sample ...
def calc_improve_percent2(pd_data, mode, metric_names, val_top1_param_dict):  # FIXME: 顺序！
    # FIXME：之前粒度最小是task，现在是samples -> 逐sample百分比improve异常值更猛！！！（而且普遍差...）
    logging.info(f"Begin to calculate improvement percent2 in mode={mode}...")
    params_space, origin_param_dict = get_params_space_and_org()

    # Filter data for the specified mode
    data_filtered = pd_data[pd_data['mode'] == mode]
    # Construct masks for the origin and val_top1 param dicts
    mask_origin = np.logical_and.reduce([data_filtered[key] == value for key, value in origin_param_dict.items()])
    mask_val_top1 = np.logical_and.reduce([data_filtered[key] == value for key, value in val_top1_param_dict.items()])
    res = {}
    # Get the metric values for the filtered data
    for metric_name in metric_names:
        logging.info(f"Calculating improvement percent for {metric_name}...")
        origin_metric_values = data_filtered[mask_origin][metric_name].values
        val_top1_metric_values = data_filtered[mask_val_top1][metric_name].values
        assert len(origin_metric_values) == len(val_top1_metric_values), \
            f"Lengths do not match: {len(origin_metric_values)} != {len(val_top1_metric_values)}"
        improve_percent = ((origin_metric_values - val_top1_metric_values) / origin_metric_values * 100).mean()
        res[f"improve_percent2_{metric_name}"] = improve_percent
    return res


def calc_improve_percent_statistics(pd_data, mode, metric_names, val_top1_param_dict):
    logging.info(f"Begin to calculate improvement percent statistics in mode={mode}...")
    params_space, origin_param_dict = get_params_space_and_org()

    # Filter data for the specified mode
    data_filtered = pd_data[pd_data['mode'] == mode]

    # Calculate statistics for origin and val_top1 parameter dictionaries
    origin_statistics = calc_statistics(data_filtered, metric_names, origin_param_dict)
    val_top1_statistics = calc_statistics(data_filtered, metric_names, val_top1_param_dict)

    res = {}
    # Calculate the improvement percent for each statistic
    for metric_name in metric_names:
        logging.info(f"Calculating improvement percent for {metric_name}...")

        origin_mean = origin_statistics[f"{metric_name}_mean"]
        origin_median = origin_statistics[f"{metric_name}_median"]
        origin_std = origin_statistics[f"{metric_name}_std"]
        origin_iqr = origin_statistics[f"{metric_name}_iqr"]
        origin_max = origin_statistics[f"{metric_name}_max"]
        origin_min = origin_statistics[f"{metric_name}_min"]

        val_top1_mean = val_top1_statistics[f"{metric_name}_mean"]
        val_top1_median = val_top1_statistics[f"{metric_name}_median"]
        val_top1_std = val_top1_statistics[f"{metric_name}_std"]
        val_top1_iqr = val_top1_statistics[f"{metric_name}_iqr"]
        val_top1_max = val_top1_statistics[f"{metric_name}_max"]
        val_top1_min = val_top1_statistics[f"{metric_name}_min"]

        improve_percent_mean = (origin_mean - val_top1_mean) / origin_mean * 100
        improve_percent_median = (origin_median - val_top1_median) / origin_median * 100
        improve_percent_std = (origin_std - val_top1_std) / origin_std * 100
        improve_percent_iqr = (origin_iqr - val_top1_iqr) / origin_iqr * 100
        improve_percent_max = (origin_max - val_top1_max) / origin_max * 100
        improve_percent_min = (origin_min - val_top1_min) / origin_min * 100

        res[f"improve_percent_mean_{metric_name}"] = improve_percent_mean
        res[f"improve_percent_median_{metric_name}"] = improve_percent_median
        res[f"improve_percent_std_{metric_name}"] = improve_percent_std
        res[f"improve_percent_iqr_{metric_name}"] = improve_percent_iqr
        res[f"improve_percent_max_{metric_name}"] = improve_percent_max
        res[f"improve_percent_min_{metric_name}"] = improve_percent_min

        # 原始值应该也要做记录到summary。。。以后补上
        # res[f"origin_mean_{metric_name}"] = origin_mean
        # res[f"origin_median_{metric_name}"] = origin_median
        # res[f"origin_std_{metric_name}"] = origin_std
        # res[f"origin_iqr_{metric_name}"] = origin_iqr
        # res[f"origin_max_{metric_name}"] = origin_max
        # res[f"origin_min_{metric_name}"] = origin_min
    return res


def calc_better_draw_percent(pd_data, mode, metric_names, val_top1_param_dict):
    logging.info(f"Begin to calculate better percent in mode={mode}...")

    for metric_name in metric_names:
        assert metric_name in ['mae', 'mse', 'rmse', 'mape', 'mspe'], f"Invalid metric name: {metric_name}"
    params_space, origin_param_dict = get_params_space_and_org()

    # Filter data for the specified mode
    data_filtered = pd_data[pd_data['mode'] == mode]

    # Construct masks for the origin and val_top1 param dicts
    mask_origin = np.logical_and.reduce([data_filtered[key] == value for key, value in origin_param_dict.items()])
    mask_val_top1 = np.logical_and.reduce([data_filtered[key] == value for key, value in val_top1_param_dict.items()])

    res = {}
    # Get the metric values for the filtered data
    for metric_name in metric_names:
        logging.info(f"Calculating better percent for {metric_name}...")
        origin_metric_values = data_filtered[mask_origin][metric_name].values
        val_top1_metric_values = data_filtered[mask_val_top1][metric_name].values

        assert len(origin_metric_values) == len(val_top1_metric_values), \
            f"Lengths do not match: {len(origin_metric_values)} != {len(val_top1_metric_values)}"
        # Calculate the better percent
        better_count = np.sum(val_top1_metric_values < origin_metric_values)
        draw_count = np.sum(val_top1_metric_values == origin_metric_values)  # 好像不可能相等？少数情况会有的
        total_count = len(origin_metric_values)
        better_percent = (better_count / total_count) * 100
        draw_percent = (draw_count / total_count) * 100
        res[f"better_percent_{metric_name}"] = better_percent
        res[f"draw_percent_{metric_name}"] = draw_percent
    return res


#     # 计算val_top1_param_dict和org在test上Bett的sample和Wrse的sample分别提升和降低的比率
#     improve_percent_in_better_and_worse_dict = calc_improve_percent_in_better_and_bad(pd_data, 'test', metric_names,
def calc_improve_percent_in_better_and_worse(pd_data, mode, metric_names, val_top1_param_dict):
    logging.info(f"Begin to calculate improvement percent in better and worse samples in mode={mode}...")
    params_space, origin_param_dict = get_params_space_and_org()

    # Filter data for the specified mode
    data_filtered = pd_data[pd_data['mode'] == mode]
    # Construct masks for the origin and val_top1 param dicts
    mask_origin = np.logical_and.reduce([data_filtered[key] == value for key, value in origin_param_dict.items()])
    mask_val_top1 = np.logical_and.reduce([data_filtered[key] == value for key, value in val_top1_param_dict.items()])
    res = {}
    # Get the metric values for the filtered data
    for metric_name in metric_names:
        logging.info(f"Calculating improvement percent for {metric_name}...")

        origin_metric_values = data_filtered[mask_origin][metric_name].values
        val_top1_metric_values = data_filtered[mask_val_top1][metric_name].values

        assert len(origin_metric_values) == len(val_top1_metric_values), \
            f"Lengths do not match: {len(origin_metric_values)} != {len(val_top1_metric_values)}"

        # Calculate the better and worse samples # mse等越小越好
        better_mask = val_top1_metric_values < origin_metric_values
        worse_mask = val_top1_metric_values > origin_metric_values  # 取等会导致abs(Imp%_MSE(Wrse))偏低

        better_origin_values = origin_metric_values[better_mask]
        better_val_top1_values = val_top1_metric_values[better_mask]

        worse_origin_values = origin_metric_values[worse_mask]
        worse_val_top1_values = val_top1_metric_values[worse_mask]

        # # Calculate the improvement percent
        # if len(better_origin_values) > 0:
        #     improve_percent_better = (
        #             (better_origin_values - better_val_top1_values) / better_origin_values * 100).mean()
        # else:
        #     # improve_percent_better = np.nan  # No better samples
        #     improve_percent_better = 0  # No better samples
        #
        # if len(worse_origin_values) > 0:
        #     improve_percent_worse = ((worse_origin_values - worse_val_top1_values) / worse_origin_values * 100).mean()
        # else:
        #     # improve_percent_bad = np.nan  # No bad samples
        #     improve_percent_worse = 0

        # Calculate the improvement percent
        if len(better_origin_values) > 0:
            mean_better_origin = np.mean(better_origin_values)
            mean_better_val_top1 = np.mean(better_val_top1_values)
            improve_percent_better = (mean_better_origin - mean_better_val_top1) / mean_better_origin * 100
        else:
            improve_percent_better = 0  # No better samples

        if len(worse_origin_values) > 0:
            mean_worse_origin = np.mean(worse_origin_values)
            mean_worse_val_top1 = np.mean(worse_val_top1_values)
            improve_percent_worse = (mean_worse_origin - mean_worse_val_top1) / mean_worse_origin * 100
        else:
            improve_percent_worse = 0  # No worse samples

        res[f"improve_percent_in_better_{metric_name}"] = improve_percent_better
        res[f"improve_percent_in_worse_{metric_name}"] = improve_percent_worse

    return res


def calc_improve_percent_in_hard_medium_easy(pd_data, mode, metric_names, val_top1_param_dict):
    logging.info(f"Begin to calculate improvement percent in hard, medium and easy samples in mode={mode}...")

    params_space, origin_param_dict = get_params_space_and_org()

    # Filter data for the specified mode
    data_filtered = pd_data[pd_data['mode'] == mode]

    # Construct masks for the origin and val_top1 param dicts
    mask_origin = np.logical_and.reduce([data_filtered[key] == value for key, value in origin_param_dict.items()])
    mask_val_top1 = np.logical_and.reduce([data_filtered[key] == value for key, value in val_top1_param_dict.items()])

    res = {}
    # Get the metric values for the filtered data
    for metric_name in metric_names:
        logging.info(f"Calculating improvement percent for {metric_name}...")

        origin_metric_values = data_filtered[mask_origin][metric_name].values
        val_top1_metric_values = data_filtered[mask_val_top1][metric_name].values

        assert len(origin_metric_values) == len(val_top1_metric_values), \
            f"Lengths do not match: {len(origin_metric_values)} != {len(val_top1_metric_values)}"

        # Sort the origin_metric_values to find hard, medium, and easy samples
        sorted_indices = np.argsort(origin_metric_values)
        total_count = len(origin_metric_values)

        # hard_indices = sorted_indices[-(total_count // 3):]
        # medium_indices = sorted_indices[total_count // 3: 2 * total_count // 3]
        # easy_indices = sorted_indices[:total_count // 3]

        hard_indices = sorted_indices[-(total_count // 2):]
        medium_indices = sorted_indices[:]  # 不考虑了
        easy_indices = sorted_indices[:total_count // 2]

        def calculate_improve_percent(indices):
            if len(indices) > 0:
                mean_origin = np.mean(origin_metric_values[indices])
                mean_val_top1 = np.mean(val_top1_metric_values[indices])
                return (mean_origin - mean_val_top1) / mean_origin * 100
            else:
                return 0  # No samples

        improve_percent_easy = calculate_improve_percent(easy_indices)
        improve_percent_medium = calculate_improve_percent(medium_indices)
        improve_percent_hard = calculate_improve_percent(hard_indices)

        res[f"improve_percent_in_easy_{metric_name}"] = improve_percent_easy
        res[f"improve_percent_in_medium_{metric_name}"] = improve_percent_medium
        res[f"improve_percent_in_hard_{metric_name}"] = improve_percent_hard

    return res


def evaluate_and_save_results(
    pd_data, exp_config, metrics_config, metric_names, 
    val_top1_param_dict, actual_num_params, 
    data_name, model_name, pred_len
):
    res_dict = {}
    # mean-based
    res_dict.update(calc_improve_percent1(pd_data, 'test', exp_config.params_space, metric_names, val_top1_param_dict))
    # sample-based
    res_dict.update(calc_improve_percent2(pd_data, 'test', metric_names, val_top1_param_dict))
    # mean-based statistics
    res_dict.update(calc_improve_percent_statistics(pd_data, 'test', metric_names, val_top1_param_dict))
    # basic (可选)
    res_dict.update(calc_better_draw_percent(pd_data, 'test', metric_names, val_top1_param_dict))
    res_dict.update(calc_improve_percent_in_better_and_worse(pd_data, 'test', metric_names, val_top1_param_dict))
    res_dict.update(calc_improve_percent_in_hard_medium_easy(pd_data, 'test', metric_names, val_top1_param_dict))
    # params info
    res_dict[f'num_params'] = actual_num_params
    for key, value in val_top1_param_dict.items():
        res_dict[key] = value

    logging.info(f"res_dict={res_dict}")
    result_dict = res_dict
    record = {**result_dict, 'data_name': data_name, 'model_name': model_name,
              'target_column': 'OT', 'pred_len': pred_len}
    
    detailed_results = pd.DataFrame(columns=metrics_config.all_columns(exp_config.params_columns))
    detailed_results.loc[len(detailed_results)] = record
    detailed_results.to_csv(os.path.join(exp_config.res_dir, 'detailed_results.csv'))
    logging.info(f"detailed_results=\n{detailed_results.to_string()}")
    return detailed_results


def kwargs_to_tag(op_kwargs: dict) -> str:
    """
    把 kwargs 转成简短的 tag，用于文件名。
    - 先拼接 key=val
    - 把不适合文件名的字符替换成下划线
    """
    if not op_kwargs:
        return "default"
    parts = [f"{k}={v}" for k, v in op_kwargs.items()]
    tag = "_".join(parts)
    # 替换掉可能非法的字符（如空格、冒号、大括号）
    tag = re.sub(r"[^0-9a-zA-Z._=-]+", "_", tag)
    return tag

