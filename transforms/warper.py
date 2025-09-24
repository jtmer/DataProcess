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

from config import nan_inf_clip_factor
from utils import my_clip

def assert_timeseries_3d_np(data):
    assert type(data) is np.ndarray and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'

def assert_timeseries_3d_tensor(data):
    assert type(data) is torch.Tensor and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'
    
class Warper:
    def __init__(self, method, clip_factor):
        self.method = method
        self.shift_values = None
        self.box_cox_lambda = None
        self.fail = False
        self.data_in = None
        self.clip_factor = float(clip_factor) if clip_factor != 'none' else nan_inf_clip_factor

    def pre_process(self, data):
        assert len(data.shape)==3, f'Invalid data shape: {data.shape}'
        if self.method == 'none':
            return data

        batch_size, time_len, feature_dim = data.shape
        self.data_in = data

        if self.method == 'log':
            if isinstance(data, np.ndarray):
                min_values = np.min(data, axis=1, keepdims=True)
                self.shift_values = np.where(min_values <= 1, 1 - min_values, 0)
                data_shifted = data + self.shift_values
                res = np.log(data_shifted)
            elif isinstance(data, torch.Tensor):
                min_values = torch.min(data, dim=1, keepdim=True).values
                self.shift_values = torch.where(min_values <= 1, 1 - min_values, torch.tensor(0., dtype=data.dtype, device=data.device))
                data_shifted = data + self.shift_values
                res = torch.log(data_shifted)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        
        elif self.method == 'sqrt':
            if isinstance(data, np.ndarray):
                min_values = np.min(data, axis=1, keepdims=True)
                self.shift_values = np.where(min_values < 0, 1 - min_values, 0)
                data_shifted = data + self.shift_values
                res = np.sqrt(data_shifted)
            elif isinstance(data, torch.Tensor):
                min_values = torch.min(data, dim=1, keepdim=True).values
                zero_tensor = torch.tensor(0., dtype=data.dtype, device=data.device)
                self.shift_values = torch.where(min_values < 0, 1 - min_values, zero_tensor)
                data_shifted = data + self.shift_values
                res = torch.sqrt(data_shifted)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        else:
            raise Exception('Invalid warper method: {}'.format(self.method))
        
        assert len(res.shape) == len(data.shape), f'Invalid data shape: {res.shape}'
        if isinstance(res, np.ndarray):
            assert np.isnan(res).sum() == 0 and np.isinf(res).sum() == 0, \
                f'Invalid data: {res}, method: {self.method}'

            if np.isnan(res).any() or np.isinf(res).any():
                logging.error(f"NaN or Inf values in transformed data: {res}")
                self.fail = True
                return data
        elif isinstance(res, torch.Tensor):
            assert torch.isnan(res).sum() == 0 and torch.isinf(res).sum() == 0, \
                f'Invalid data: {res}, method: {self.method}'
            
            if torch.isnan(res).any() or torch.isinf(res).any():
                logging.error(f"NaN or Inf values in transformed data: {res}")
                self.fail = True
                return data
        else:
            raise ValueError(f"Unsupported data type: {type(res)}")
        return res

    def post_process(self, data):
        if self.method == 'none':
            return data

        if self.fail:
            return data

        if self.method == 'log':
            if isinstance(data, np.ndarray):
                _data = np.exp(data)
            elif isinstance(data, torch.Tensor):
                _data = torch.exp(data)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            data_restored = _data - self.shift_values

        elif self.method == 'sqrt':
            if isinstance(data, np.ndarray):
                data_restored = np.square(data)
            elif isinstance(data, torch.Tensor):
                data_restored = torch.square(data)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            data_restored = data_restored - self.shift_values

        else:
            raise Exception('Invalid warper method: {}'.format(self.method))

        res = data_restored
        if isinstance(data, np.ndarray):
            if np.isnan(res).any() or np.isinf(res).any():
                logging.error(f"NaN or Inf values in restored data: {res}")
                res = my_clip(self.data_in, res, nan_inf_clip_factor=5)  # 表示出结果很差
            elif self.clip_factor is not None:
                res = my_clip(self.data_in, res, min_max_clip_factor=self.clip_factor)  # 要求严格
        elif isinstance(data, torch.Tensor):
            if torch.isnan(res).any() or torch.isinf(res).any():
                logging.error(f"NaN or Inf values in restored data: {res}")
                res = my_clip(self.data_in, res, nan_inf_clip_factor=5)  # 表示出结果很差
            elif self.clip_factor is not None:
                res = my_clip(self.data_in, res, min_max_clip_factor=self.clip_factor)  # 要求严格
        return res
