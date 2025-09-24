import numpy as np

import torch
import torch.nn.functional as F

def assert_timeseries_3d_np(data):
    assert type(data) is np.ndarray and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'

def assert_timeseries_3d_tensor(data):
    assert type(data) is torch.Tensor and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'
    
# 序列的上下文长度选择
class Trimmer:
    def __init__(self, seq_l, pred_l):
        self.seq_l = seq_l
        self.pred_l = pred_l

    def pre_process(self, data):
        if data.shape[1] <= self.seq_l:
            return data
        if type(data) is np.ndarray:
            assert_timeseries_3d_np(data)
        elif type(data) is torch.Tensor:
            assert_timeseries_3d_tensor(data)
        assert data.shape[1] >= self.seq_l, f'Invalid data shape: {data.shape} for seq_l={self.seq_l}'

        res = data[:, -self.seq_l:, :]
        return res

    def post_process(self, data):
        if data.shape[1] == self.pred_l:
            return data
        if type(data) is np.ndarray:
            assert_timeseries_3d_np(data)
        elif type(data) is torch.Tensor:
            assert_timeseries_3d_tensor(data)
        assert data.shape[1] >= self.pred_l
        res = data[:, :self.pred_l, :]
        return res
