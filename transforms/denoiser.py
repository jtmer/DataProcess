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

#降噪
class Denoiser:
    def __init__(self, method):
        assert method in ['none', 'moving_average', 'ewma', 'fft']  # median比较慢
        self.method = method
        self.window_size = 3
        self.alpha = 0.1

    def pre_process(self, data):
        if self.method == 'none':
            return data

        if self.method == 'moving_average':
            return self.moving_average(data)
        elif self.method == 'ewma':
            return self.ewma(data)
        # elif self.method == 'median':
        #     return self.median_filter(data)
        elif self.method == 'fft':
            return self.fft_filter(data)
        else:
            raise ValueError(f"Unsupported denoise method: {self.method}")

    def moving_average(self, data):
        window_size = self.window_size
        if isinstance(data, np.ndarray):
            kernel = np.ones(window_size) / window_size
            smoothed_data = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=data)
        elif isinstance(data, torch.Tensor):
            kernel = torch.ones(window_size, dtype=data.dtype, device=data.device) / window_size
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            data = data.unsqueeze(1)
            smoothed_data = torch.nn.functional.conv1d(data, kernel, padding=window_size // 2)
            smoothed_data = smoothed_data.squeeze(1)
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
        return smoothed_data

    def ewma(self, data):
        alpha = self.alpha
        batch, time, feature = data.shape
        if isinstance(data, np.ndarray):
            smoothed_data = np.zeros_like(data)
        elif isinstance(data, torch.Tensor):
            smoothed_data = torch.zeros_like(data)
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
        smoothed_data[:, 0, :] = data[:, 0, :]  # Initialize with the first value

        for t in range(1, time):
            smoothed_data[:, t, :] = alpha * data[:, t, :] + (1 - alpha) * smoothed_data[:, t - 1, :]
        return smoothed_data

    def _apply_median_filter(self, data, window_size):
        pad_size = window_size // 2
        if isinstance(data, np.ndarray):
            padded_data = np.pad(data, pad_size, mode='edge')
            smoothed_data = np.zeros_like(data)
            for i in range(len(data)):
                smoothed_data[i] = np.median(padded_data[i:i + window_size])
        elif isinstance(data, torch.Tensor):
            padded_data = torch.nn.functional.pad(data, (pad_size, pad_size), mode='replicate')
            smoothed_data = torch.zeros_like(data)
            for i in range(data.size(0)):
                # 使用 torch.median 计算中位数
                smoothed_data[i], _ = torch.median(padded_data[i:i + window_size], dim=0)
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
        return smoothed_data


    def fft_filter(self, data):
        percentile = 80
        batch, time, feature = data.shape
        if isinstance(data, np.ndarray):
            denoised_data = np.zeros_like(data)
    
            for b in range(batch):
                for f in range(feature):
                    fft_coeffs = np.fft.fft(data[b, :, f])
                    magnitudes = np.abs(fft_coeffs)
                    upper_magnitude = np.percentile(magnitudes, percentile)
                    fft_coeffs[magnitudes < upper_magnitude] = 0 + 0j
                    denoised_data[b, :, f] = np.fft.ifft(fft_coeffs).real
        elif isinstance(data, torch.Tensor):
            denoised_data = torch.zeros_like(data)
            
            for b in range(batch):
                for f in range(feature):
                    # 进行快速傅里叶变换
                    fft_coeffs = torch.fft.fft(data[b, :, f])
                    # 计算傅里叶系数的幅值
                    magnitudes = torch.abs(fft_coeffs)
                    # 计算指定百分位数对应的幅值
                    upper_magnitude = torch.quantile(magnitudes, percentile / 100)
                    # 将低于该幅值的傅里叶系数置为零
                    fft_coeffs[magnitudes < upper_magnitude] = torch.tensor(0 + 0j, dtype=fft_coeffs.dtype, device=fft_coeffs.device)
                    # 进行逆快速傅里叶变换并取实部
                    denoised_data[b, :, f] = torch.fft.ifft(fft_coeffs).real
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
        return denoised_data

    def post_process(self, data):
        return data
