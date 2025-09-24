import logging
import numpy as np
import torch

def smart_clip(seq, min_allowed, max_allowed, seq_in_last_values):
    assert seq.ndim == 3, "Input sequence must be 3D: (batch, time, feature)"
    assert min_allowed.shape == max_allowed.shape == (seq.shape[0], 1, seq.shape[2]), \
        "Min and max must have shape (batch, 1, feature)"

    batch, time, feature = seq.shape
    first_elements = seq[:, 0:1, :]  # Preserve the first elements
    # assert np.all(first_elements < max_values) and np.all(first_elements > min_values), \
    #     f"The first elements must be within min and max values: \n" \
    #     f"first_elements:{first_elements}, \nmin_values:{min_values}, \nmax_values:{max_values}"
    # FIXME:
    if np.any(first_elements > max_allowed) or np.any(first_elements < min_allowed):
        logging.info(f"The first elements must be within min and max allowed!!!\n")
        logging.debug(f"The first elements must be within min and max allowed: \n"
                      f"first_elements:{first_elements}, \nmin_allowed:{min_allowed}, \nmax_allowed:{max_allowed}")
        # return np.clip(seq, min_allowed, max_allowed)
        # FIXME：平移first使得跟last一样,即在(first,last)之间进行scale
        seq = seq - first_elements + seq_in_last_values

    # Calculate scaling factors for each batch
    seq_max_values = np.max(seq, axis=1, keepdims=True)  # Include the first element
    seq_min_values = np.min(seq, axis=1, keepdims=True)  # Include the first element

    # Apply scaling to the sequences that exceed the max values
    for i in range(batch):
        for j in range(feature):
            # 如果存在大于max的值，则把值介于(first,max)的数值进行scale
            tmp_seq = seq[i, :, j]
            first_value = first_elements[i, 0, j]
            max_value = max_allowed[i, 0, j]
            min_value = min_allowed[i, 0, j]
            seq_max_value = seq_max_values[i, 0, j]
            seq_min_value = seq_min_values[i, 0, j]
            if seq_max_value > max_value:
                scale = (max_value - first_value) / (seq_max_value - first_value)
                upper_mask = tmp_seq > first_value
                seq[i, upper_mask, j] = first_value + (seq[i, upper_mask, j] - first_value) * scale
            if seq_min_value < min_value:
                scale = (min_value - first_value) / (seq_min_value - first_value)
                lower_mask = tmp_seq < first_value
                seq[i, lower_mask, j] = first_value + (seq[i, lower_mask, j] - first_value) * scale
    return seq

def my_clip(seq_in, seq_out, nan_inf_clip_factor=None, min_max_clip_factor=None):
    # nan_inf_clip_factor=3, min_max_clip_factor=2 ...
    # mean+1.5IQR-> max+0.25range
    if isinstance(seq_in, torch.Tensor):
        seq_in = seq_in.detach().cpu().numpy()
    if isinstance(seq_out, torch.Tensor):
        seq_out = seq_out.detach().cpu().numpy()
    
    max_values = np.max(seq_in, axis=1, keepdims=True)
    min_values = np.min(seq_in, axis=1, keepdims=True)
    range_values = max_values - min_values

    assert nan_inf_clip_factor is not None or min_max_clip_factor is not None, \
        "nan_inf_clip_factor and min_max_clip_factor cannot be both None!"

    if nan_inf_clip_factor is not None and (np.isnan(seq_out).any() or np.isinf(seq_out).any()):
        max_allowed = max_values + nan_inf_clip_factor * range_values
        min_allowed = min_values - nan_inf_clip_factor * range_values
        logging.info(f"seq_out contains NaN values!!! \n")
        logging.debug(f"seq_out contains NaN values!!!: {seq_out}")
        # seq_out = np.nan_to_num(seq_out, nan=(max_values + min_values) / 2, posinf=max_allowed, neginf=min_allowed)
        seq_out = np.nan_to_num(seq_out, nan=max_allowed, posinf=max_allowed, neginf=min_allowed)  # nan hard punish
        # logging.warning(f"seq_out after filling NaN values: {seq_out}")
    if min_max_clip_factor is not None:
        max_allowed = max_values + min_max_clip_factor * range_values
        min_allowed = min_values - min_max_clip_factor * range_values
        seq_in_last_values = seq_in[:, -1:, :]
        if (seq_out > max_allowed).any() or (seq_out < min_allowed).any():
            logging.info(f"seq_out out of range!!!: \n")
            logging.debug(f"seq_out out of range!!!: {seq_out}")
            # seq_out = np.clip(seq_out, min_allowed, max_allowed)
            # logging.warning(f"seq_out after clipping: {seq_out}")
            # FIXME：助长scale的气焰....危险 # 使用allowed！Ok ->对scale的处理还是有点问题？weizhi
            seq_out = smart_clip(seq_out, min_allowed, max_allowed, seq_in_last_values)
            # logging.warning(f"seq_out after smart scaling: {seq_out}")
    return seq_out