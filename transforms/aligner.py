import argparse
import logging
from math import ceil

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy import signal
from scipy.fft import fft, ifft
from scipy.stats import boxcox, boxcox_normmax, yeojohnson
from scipy.special import inv_boxcox
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from statsmodels.tsa.stl._stl import STL

from utils import my_clip

def assert_timeseries_3d_np(data):
    assert type(data) is np.ndarray and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'

def assert_timeseries_3d_tensor(data):
    assert type(data) is torch.Tensor and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'

# 序列输入到模型前需要对序列进行对齐到Patch整数倍
class Aligner:
    def __init__(self, mode, method, data_patch_len, model_patch_len):
        assert mode in ['none', 'data_patch', 'model_patch']
        assert method in ['none', 'trim', 'zero_pad', 'mean_pad', 'edge_pad']
        self.mode = mode
        self.method = method
        self.patch_len = data_patch_len if mode == 'data_patch' else model_patch_len

    def pre_process(self, data):  # padding mostly
        if self.mode == 'none' or self.method == 'none':
            return data
        if type(data) is np.ndarray:
            assert_timeseries_3d_np(data)
        elif type(data) is torch.Tensor:
            assert_timeseries_3d_tensor(data)
        batch, time, feature = data.shape
        if time % self.patch_len == 0:
            return data
        pad_l = self.patch_len - time % self.patch_len if time % self.patch_len != 0 else 0
        if isinstance(data, np.ndarray):
            res = np.zeros((batch, pad_l + time, feature))
        elif isinstance(data, torch.Tensor):
            res = torch.zeros((batch, pad_l + time, feature)).to(data.device)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        if self.method == 'trim':  # trim其实应该相信model自己完成。。。-》可能是0-pad
            if time < self.patch_len:  # 理论上不会出现
                self.method = 'edge_pad'
            else:
                valid_len = time // self.patch_len * self.patch_len
                return data[:, -valid_len:, :]

        for b in range(batch):
            for f in range(feature):
                if isinstance(data, np.ndarray):
                    # FIXME：应在头部而不是尾部填充数据！！！(到这时单array了！
                    if self.method == 'zero_pad':
                        res[b, :, f] = np.pad(data[b, :, f], (pad_l, 0), 'constant', constant_values=0)
                    elif self.method == 'mean_pad':  # FIXME:mean?axis?
                        res[b, :, f] = np.pad(data[b, :, f], (pad_l, 0), 'constant', constant_values=np.mean(data[b, :, f]))
                    elif self.method == 'edge_pad':
                        res[b, :, f] = np.pad(data[b, :, f], (pad_l, 0), 'edge')
                    else:
                        raise Exception('Invalid aligner: {}'.format(self.method))
                elif isinstance(data, torch.Tensor):
                    if self.method == 'zero_pad':
                        # 零填充
                        res[b, :, f] = F.pad(data[b, :, f].unsqueeze(0).unsqueeze(0), (pad_l, 0), mode='constant', value=0).squeeze()
                    elif self.method == 'mean_pad':
                        # 均值填充
                        mean_val = torch.mean(data[b, :, f])
                        res[b, :, f] = F.pad(data[b, :, f].unsqueeze(0).unsqueeze(0), (pad_l, 0), mode='constant', value=mean_val).squeeze()
                    elif self.method == 'edge_pad':
                        # 边缘填充
                        res[b, :, f] = F.pad(data[b, :, f].unsqueeze(0).unsqueeze(0), (pad_l, 0), mode='replicate').squeeze()
                    else:
                        raise Exception('Invalid aligner: {}'.format(self.method))
                else:
                    raise ValueError(f"Unsupported data type: {type(data)}")
        return res

    def post_process(self, data):
        return data
