from pipeline.executor import PipelineExecutor
from pipeline.pipeline_order import PIPELINE_ORDERS
from math import ceil
import logging
import numpy as np

def adaptive_infer(**kwargs):
    pipeline_name = kwargs.get("pipeline_name")
    if pipeline_name not in PIPELINE_ORDERS:
        raise ValueError(f"pipeline_name={pipeline_name} not supported!")
    logging.info(f"pipeline_name={pipeline_name}")
    return infer_template(**kwargs)

def infer_template(pipeline_name, history_seqs, model, dataset, patch_len, pred_len, mode, target_column, **kwargs):
    config = {
        **kwargs,
        "original_data": history_seqs,
        "patch_len": patch_len,
        "pred_len": pred_len,
        "target_column": target_column
    }
    stages = PIPELINE_ORDERS[pipeline_name]
    executor = PipelineExecutor(stages, config, model=model, dataset=dataset)

    x = executor.preprocess(history_seqs)
    pred_len_needed = ceil(pred_len / config["sampler_factor"] / patch_len) * patch_len

    preds = model.forcast(x, pred_len_needed)
    if np.isnan(preds).any() or np.isinf(preds).any():
        logging.warning("NaN/Inf in model prediction")

    preds = executor.postprocess(preds)
    return preds


