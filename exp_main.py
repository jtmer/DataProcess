import argparse
import atexit
import cProfile
import logging
import os
import pstats
import random
import sys

from configs.experiment_runtime_config import ExperimentRuntimeConfig
from exp.train import train
from exp.valid import valid
from exp.test import test
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.experiment_stages_config import ExperimentStagesConfig
from configs.metrics_config import MetricsConfig
from utils.ablation_substitude import ablation_substitute
from utils.result_utils import evaluate_and_save_results

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from AnyTransform.parser import *
from AnyTransform.config import *
from AnyTransform.terminator import *
from dataset import *
from model import get_model
import numpy as np
from AnyTransform.transforms import *
from AnyTransform.utils import *
import os
import sys





if __name__ == "__main__":
    set_seed(seed)

    if fast_mode:
        profiler = cProfile.Profile()
        profiler.enable()

    device = f"cuda:{args.gpu}" if use_gpu else 'cpu'

    model = get_model(model_name, device, args)
    dataset = get_dataset(data_name)
    exp_config = ExperimentRuntimeConfig.from_yaml()
    metrics_config = MetricsConfig.from_yaml(ablation=exp_config.ablation)
    exp_config.initialize(
        model_name=model_name,
        data_name=data_name,
        metrics_config=metrics_config,
        get_params_space_and_org_fn=get_params_space_and_org,
        atexit_handler=atexit_handler
    )
    pd_data = pd.DataFrame(columns=exp_config.get_columns_for_record())

    # ########################################################### train
    # FIXME 不考虑至今有多少需要比原本好的指标 理论上至少为1
    terminator_manager = TerminatorManager([  MaxIterationsTerminator(exp_config.num_params),], 0, 0) 
    
    exp_stages = ExperimentStagesConfig.from_yaml()
    train_stage_config = exp_stages.get_runtime_config( mode='train',exp_config=exp_config)

    pd_data = train(exp_config,model, dataset,train_stage_config,exp_config.num_params,  terminator_manager,pd_data)
    print(f"pd_data after train:\n{pd_data}")
    
    # 计算实际参数数量
    actual_num_params = num_params
    for terminator in terminator_manager.terminators:
        if isinstance(terminator, MaxIterationsTerminator):
            actual_num_params = terminator.iteration_count

    # ########################################################### valid
    
    pareto_top_k = max(1, floor(actual_num_params / 30))
    pareto_param_dict_list = find_top_param_dict_list_pareto(
        pd_data, 'train', exp_config.params_space, pareto_top_k,
        metrics_config.weights, metrics_config.pareto_metrics(exp_config.ablation),
        False, exp_config.res_dir
    )
    

    val_unique_param_dict_list = make_param_dict_unique(
        [exp_config.origin_param_dict] + pareto_param_dict_list
    )

    val_stage_config = exp_stages.get_runtime_config(mode='val',exp_config=exp_config,param_dict_list=val_unique_param_dict_list)

    pd_data = valid(
        exp_config=exp_config,
        model=model,
        dataset=dataset,
        stage_config=val_stage_config,
        param_count=len(val_unique_param_dict_list),
        pd_data=pd_data
    )
    # ########################################################### test
    # 选出 val 的 top1
    val_top1_param_dict = find_top_param_dict_list_by_statistics(
        pd_data, 'val', exp_config.params_space, 1,
        metrics_config.test_weights(),
        metrics_config.test_stats_weights(),
        exp_config.res_dir
    )[0]


    test_unique_param_dict_list = make_param_dict_unique([
        exp_config.origin_param_dict, val_top1_param_dict
    ])

    # 获取 test 阶段执行配置
    test_stage_config = exp_stages.get_runtime_config(mode='test',exp_config=exp_config,param_dict_list=test_unique_param_dict_list)

    # 执行测试
    pd_data = test(
        exp_config=exp_config,
        model=model,
        dataset=dataset,
        stage_config=test_stage_config,
        param_count=len(test_unique_param_dict_list),
        pd_data=pd_data
    )

    metric_names = metrics_config.metric_names


    detailed_results = evaluate_and_save_results(
        pd_data, exp_config, metrics_config, metric_names, 
        val_top1_param_dict, actual_num_params, 
        data_name, model_name, pred_len
    )

    logging.info(f"detailed_results=\n{detailed_results.to_string()}")

    if fast_mode:  
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)  # Print the top 10 time-consuming functions
        