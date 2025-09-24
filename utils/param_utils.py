# param_utils.py
import numpy as np
import logging

def get_params_space_and_org(fast_mode=None):
    # if fast_mode is None:
    #     from parser import fast_mode, ablation
    #     patch_len=96
    # else:
    #     patch_len = 96
    #     ablation = 'none'
    patch_len=96
    
    params_space = {
        'sampler_factor': {
            'type': 'int',
            'values': np.arange(1, 3 + 1, 1)  # 减小space
        },
        'trimmer_seq_len': {
            'type': 'int',  # (40,96,1) Chronos+Cuda稳定报错
            'values': np.arange(6 * patch_len, (10 + 1) * patch_len, int(patch_len * 1))  # 补align危险？？？
        },
        'aligner_mode': {
            'type': 'str',
            'values': ['none', 'data_patch']  # 减小Space
 
        },
        'aligner_method': {
            'type': 'str',  # zero_pad会被明显排除 不过会影响整体可视化和debug
            'values': ['edge_pad']  # 减小Space

        },
        'normalizer_method': {
            'type': 'str',  # 'minmax', 'maxabs' 减小搜索空间 ????????????'minmax', 'maxabs'???????????
            'values': ['none', 'standard', 'robust']  # robust略慢...！！

        },
        'normalizer_mode': {
            'type': 'str',  # train overfit??? ???????????????????????leak！ 'history'导致Weather坏?????
            'values': ['input']

        },
        'normalizer_ratio': {  # new!!!
            'type': 'str',
            'values': [1]  # 减小space

        },
        'inputer_detect_method': {
            'type': 'str',  # iqr计算时间长了一点(1/2 model)！！
            'values': ['none', '3_sigma', '1.5_iqr']#加回来了
        },
        'inputer_fill_method': {  # forward_fill感觉很多时候也不如不impute... # forward_fill在ETT上差！
            'type': 'str',  # 'forward_fill', 'backward_fill' 减小搜索空间 # rolling_mean有时候遇大倾斜很坏！！！
            'values': ['linear_interpolate']
        },
        'warper_method': {  # 'boxcox'+Uni2ts -> 有时候nan 而且秒级 # yeojohnson 有时候会-1847倍...overfit..??
            'type': 'str',  # 'log' 'sqrt' 坏
            'values': ['none', 'log']  # 减小space
        },
        'decomposer_period': {  # 有点太慢了...并行抢cpu # 现在还行 # 整体貌似会变差？rand？
            'type': 'str',
            'values': ['none']
        },
        'decomposer_components': {
            'type': 'str',  # 'trend+season', 'season+residual', 'trend+residual'
            'values': ['none']
        },
        # Differentiator
        'differentiator_n': {#
            'type': 'int',
            'values': [0]
        },
        'pipeline_name': {
            'type': 'str',
            'values': ['infer1', 'infer3'] 
        },
        'denoiser_method': {
            'type': 'str', 
            'values': ['none', 'ewma'] 
        },
        'clip_factor': {
            'type': 'str',
            'values': ['none', '0', '0.25']
        }
    }

    origin_param_dict = {  # FIXME：
        'sampler_factor': 1,
        'trimmer_seq_len': patch_len * 7,
        'aligner_mode': 'none',
        'aligner_method': 'edge_pad',  # model_patch之后org一定需要是none！！！
        'normalizer_method': 'none',  # FIXME： 目前已经使用了Timer内置的std的scaler
        'normalizer_mode': 'input',
        'normalizer_ratio': 1,
        'inputer_detect_method': 'none',
        'inputer_fill_method': 'linear_interpolate',
        'warper_method': 'none',
        'decomposer_period': 'none',
        'decomposer_components': 'none',
        'differentiator_n': 0,
        'pipeline_name': 'infer3',
        'denoiser_method': 'none',
        'clip_factor': 'none'  # FIXME: 原来是0 但实际上不影响，只是算子内部的clip
    }
    # logging.info(f"origin_param_dict={origin_param_dict}")
    return params_space, origin_param_dict



def make_param_dict_unique(param_dict_list):
    unique_param_dict_list = []
    for param_dict in param_dict_list:
        if param_dict not in unique_param_dict_list:
            unique_param_dict_list.append(param_dict)
    logging.info(f"Unique param_dict_list: {unique_param_dict_list}")
    return unique_param_dict_list

def get_valid_params_mode_data(pd_data, mode, params_space):
    logging.info(f"Begin to get valid params mode data in mode={mode}...")
    mode_data = pd_data[pd_data['mode'] == mode]
    data_by_params = mode_data.groupby(list(params_space.keys())).size().reset_index(name='counts')
    max_step = data_by_params['counts'].max()
    logging.debug(f"max_step={max_step}")
    valid_params = data_by_params[data_by_params['counts'] == max_step].drop(columns='counts')
    valid_params_mode_data = mode_data.merge(valid_params, on=list(params_space.keys()))
    return valid_params_mode_data

