def get_max_batch_size_for_cuda(model_name):
    # batch_size -> CUDA OOM 或者一些model相关的奇怪的报错
    # 模型越大下内存占用越多，但是可能patch_size变大，继而自回归占用显存变小
    # 推理也可能存在数值不稳定的问题->Uni2ts的logit报错 （amp
    if 'Chronos-tiny' in model_name:
        res = 600
    elif 'Timer' in model_name:
        res = 600
    elif 'MOIRAI-small' in model_name:
        res = 5000  # 6000
    elif 'MOIRAI-base' in model_name:
        res = 5000
    elif 'MOIRAI-large' in model_name:
        res = 4000
    elif 'Arima' in model_name:
        res = 1
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    # return min(res, maximum)  # for train speed
    return res
