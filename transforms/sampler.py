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

def assert_timeseries_3d_np(data):
    assert type(data) is np.ndarray and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'

def assert_timeseries_3d_tensor(data):
    assert type(data) is torch.Tensor and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'
    
    
# 序列的采样方法
class Sampler:
    def __init__(self, factor):
        self.factor = factor
    
    def torch_resample(self, x: torch.Tensor, num: int, dim: int = -1) -> torch.Tensor:
        """
        对张量进行重采样（类似 scipy.signal.resample）

        参数：
            x: 输入张量（支持任意维度）
            num: 目标采样点数
            dim: 沿着哪个维度进行重采样（默认最后一个维度）

        返回：
            重采样后的张量
        """
        # 获取原始长度
        N = x.size(dim)
        
        # 计算傅里叶变换
        X = torch.fft.rfft(x, dim=dim)
        
        # 调整频域分量长度
        if num > N:
            # 上采样：在中间填充零
            X_resampled = torch.zeros((num,), dtype=X.dtype)
            slices = [slice(None)] * X.ndim
            slices[dim] = slice(0, X.size(dim))
            X_resampled[slices] = X
        else:
            # 下采样：截断高频分量
            X_resampled = X.narrow(dim, 0, num // 2 + 1)
        
        # 逆傅里叶变换并取实部
        x_resampled = torch.fft.irfft(X_resampled, n=num, dim=dim)
        
        # 调整能量缩放（与 SciPy 一致）
        x_resampled *= (num / N) ** 0.5
        
        return x_resampled

    def pre_process(self, data):
        if self.factor == 1:
            return data
 
        if type(data) is np.ndarray:
            assert_timeseries_3d_np(data)
            batch, time, feature = data.shape
            res = np.zeros((batch, ceil(time / self.factor), feature))
            for b in range(batch):
                for f in range(feature):
                    res[b, :, f] = signal.resample(data[b, :, f], ceil(time / self.factor))
        elif type(data) is torch.Tensor:
            assert_timeseries_3d_tensor(data)
            batch, time, feature = data.shape
            res = torch.zeros((batch, ceil(time * self.factor), feature)).to(data.device)
            for b in range(batch):
                for f in range(feature):
                    res[b, :, f] = self.torch_resample(data[b, :, f], ceil(time * self.factor)).to(data.device)
        else:
            res = data
 
        return res

    def post_process(self, data):
        if self.factor == 1:
            return data
        if type(data) is np.ndarray:
            assert_timeseries_3d_np(data)
            batch, time, feature = data.shape
            res = np.zeros((batch, ceil(time * self.factor), feature))
            for b in range(batch):
                for f in range(feature):
                    res[b, :, f] = signal.resample(data[b, :, f], ceil(time * self.factor))
        elif type(data) is torch.Tensor:
            assert_timeseries_3d_tensor(data)
            batch, time, feature = data.shape
            res = torch.zeros((batch, ceil(time * self.factor), feature)).to(data.device)
            for b in range(batch):
                for f in range(feature):
                    res[b, :, f] = self.torch_resample(data[b, :, f], ceil(time * self.factor))
        else:
            res = data
        return res
