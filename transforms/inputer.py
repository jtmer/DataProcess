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

# 异常检测和填充
class Inputer:
    def __init__(self, detect_method, fill_method, history_seq):
        # history_seq: (batch, time, feature)
        self.detect_method = detect_method
        self.fill_method = fill_method
        self.statistics_dict = self.get_statistics_dict(history_seq)

    def get_statistics_dict(self, history_seq):
        if self.detect_method == 'none' or self.fill_method == 'none':
            return None
        if isinstance(history_seq, np.ndarray):
            if 'sigma' in self.detect_method:
                mean = np.mean(history_seq, axis=1, keepdims=True)
                std = np.std(history_seq, axis=1, keepdims=True)
                statistics_dict = {'mean': mean, 'std': std}
            elif 'iqr' in self.detect_method:
                q1 = np.percentile(history_seq, 25, axis=1, keepdims=True)
                q3 = np.percentile(history_seq, 75, axis=1, keepdims=True)
                statistics_dict = {'q1': q1, 'q3': q3}
            else:
                raise ValueError(f"Unsupported detect method: {self.detect_method}")
        elif isinstance(history_seq, torch.Tensor):
            if 'sigma' in self.detect_method:
                mean = torch.mean(history_seq, dim=1, keepdim=True)
                std = torch.std(history_seq, dim=1, keepdim=True)
                statistics_dict = {'mean': mean, 'std': std}
            elif 'iqr' in self.detect_method:
                q1 = torch.quantile(history_seq, 0.25, dim=1, keepdim=True)
                q3 = torch.quantile(history_seq, 0.75, dim=1, keepdim=True)
                statistics_dict = {'q1': q1, 'q3': q3}
            else:
                raise ValueError(f"Unsupported detect method: {self.detect_method}")
        else:
            raise ValueError(f"Unsupported data type: {type(history_seq)}")
        return statistics_dict

    def pre_process(self, data):
        if self.detect_method == 'none' or self.fill_method == 'none':
            return data

        if 'sigma' in self.detect_method:
            k_sigma = float(self.detect_method.split('_')[0])
            fill_indices = self.detect_outliers_k_sigma(data, k_sigma)
        elif 'iqr' in self.detect_method:
            ratio = float(self.detect_method.split('_')[0])
            fill_indices = self.detect_outliers_iqr(data, ratio)
        else:
            raise ValueError(f"Unsupported detect method: {self.detect_method}")

        # # FIXME: 取出为不大量连续的fill_indices
        # tail_ratio = 1 / 4
        tail_ratio = 1
        batch_size, seq_len, feature_dim = data.shape
        rm_indices = set()


        consecutive_count = 0  # 用于记录连续异常点的数量
        threshold = 1  # FIXME:需要移除的连续异常点数量阈值 （认为n个趋势！大了10和3不太好
        # threshold = seq_len / 4
        if len(fill_indices) > 0:  # ! Is 'fill_indices' possible to be None?
            for idx in range(1, len(fill_indices[0])):
                # 若在同一个batch和同一个feature
                batch_idx_last, batch_idx_cur = fill_indices[0][idx - 1], fill_indices[0][idx]
                feature_idx_last, feature_idx_cur = fill_indices[2][idx - 1], fill_indices[2][idx]
                time_idx_last, time_idx_cur = fill_indices[1][idx - 1], fill_indices[1][idx]
                if batch_idx_cur == batch_idx_last and feature_idx_last == feature_idx_cur \
                        and time_idx_cur - time_idx_last == 1 and time_idx_cur > seq_len * (1 - tail_ratio):
                    consecutive_count += 1
                    if consecutive_count >= threshold:
                        # rm_indices.extend(range(idx - threshold + 1, idx + 1))  # 移除连续的异常点
                        rm_indices.update(range(idx - threshold, idx))  # 移除连续的异常点
                else:
                    consecutive_count = 1  # 重新计数
            # TODO
            new_fill_indices = [[], [], []]
            for idx in range(len(fill_indices[0])):
                if idx not in rm_indices:
                    new_fill_indices[0].append(fill_indices[0][idx])
                    new_fill_indices[1].append(fill_indices[1][idx])
                    new_fill_indices[2].append(fill_indices[2][idx])
            if isinstance(data, np.ndarray):
                new_fill_indices[0] = np.array(new_fill_indices[0])
                new_fill_indices[1] = np.array(new_fill_indices[1])
                new_fill_indices[2] = np.array(new_fill_indices[2])
            elif isinstance(data, torch.Tensor):
                new_fill_indices[0] = torch.tensor(new_fill_indices[0]).to(data.device)
                new_fill_indices[1] = torch.tensor(new_fill_indices[1]).to(data.device)
                new_fill_indices[2] = torch.tensor(new_fill_indices[2]).to(data.device)
            new_fill_indices = tuple(new_fill_indices)
            logging.debug(f"fill_indices: {fill_indices}")
            logging.debug(f"new_fill_indices: {new_fill_indices}")
            fill_indices = new_fill_indices
    
            filled_data = self.fill_outliers(data, fill_indices)
            if isinstance(data, np.ndarray):
                if np.isnan(filled_data).any() or np.isinf(filled_data).any():
                    logging.error(f"NaN or Inf values in filled data: {filled_data}")
                    return data
            elif isinstance(data, torch.Tensor):
                if torch.isnan(filled_data).any() or torch.isinf(filled_data).any():
                    logging.error(f"NaN or Inf values in filled data: {filled_data}")
                    return data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        else:
            filled_data = data
        return filled_data

    def post_process(self, data):
        return data


    def detect_outliers_k_sigma(self, data, k_sigma):
        seq_len = data.shape[1]
        # cutoff_index = seq_len - seq_len // 4  # Exclude the last 1/10 of the sequence for separate handling
        cutoff_index = seq_len  # 相信2-sigma
        mean = self.statistics_dict['mean']
        std = self.statistics_dict['std']
        lower_bound = mean - k_sigma * std
        upper_bound = mean + k_sigma * std
        mask = (data[:, :cutoff_index] < lower_bound) | (data[:, :cutoff_index] > upper_bound)
        if type(data) is np.ndarray:
            fill_indices = np.where(mask)
        elif type(data) is torch.Tensor:
            # fill_indices = torch.nonzero(mask)
            fill_indices = torch.where(mask)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        return fill_indices


    def detect_outliers_iqr(self, data, ratio):
        seq_len = data.shape[1]
        # cutoff_index = seq_len - seq_len // 4  # Exclude the last 1/10 of the sequence for separate handling
        cutoff_index = seq_len  # 相信2-sigma
        q1 = self.statistics_dict['q1']
        q3 = self.statistics_dict['q3']
        iqr = q3 - q1
        lower_bound = q1 - ratio * iqr
        upper_bound = q3 + ratio * iqr
        if type(data) is torch.Tensor:
            lower_bound = lower_bound.to(data.device)
            upper_bound = upper_bound.to(data.device)
        
        mask = (data[:, :cutoff_index] < lower_bound) | (data[:, :cutoff_index] > upper_bound)
        if type(data) is np.ndarray:
            fill_indices = np.where(mask)
        elif type(data) is torch.Tensor:
            # fill_indices = torch.nonzero(mask)
            fill_indices = torch.where(mask)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        return fill_indices
    
    def fill_outliers(self, data, fill_indices):
        if self.detect_method == 'none' or self.fill_method == 'none':
            return data

        if self.fill_method == 'linear_interpolate':
            filled_data = self.linear_interpolate(data, fill_indices)
        elif self.fill_method == 'rolling_mean':
            filled_data = self.rolling_mean(data, fill_indices)
        elif self.fill_method == 'forward_fill':
            filled_data = self.forward_fill(data, fill_indices)
        elif self.fill_method == 'backward_fill':
            filled_data = self.backward_fill(data, fill_indices)
        else:
            raise ValueError(f"Unsupported fill method: {self.fill_method}")

        return filled_data
    
    def linear_interpolate_torch(self, data, indices, normal_indices, values):
        """
        使用 PyTorch 实现一维线性插值。
        :param data: 输入数据张量 (torch.Tensor)
        :param indices: 需要插值的时间索引 (torch.Tensor)
        :param normal_indices: 正常的时间索引 (torch.Tensor)
        :param values: 正常时间索引对应的数据值 (torch.Tensor)
        :return: 插值后的值
        """
        # 计算差值和斜率
        delta = (normal_indices[1:] - normal_indices[:-1]).type_as(values)
        delta_values = (values[1:] - values[:-1]) / delta
        
        # 计算累积差值和累积斜率
        cumsum_delta = torch.cumsum(delta, dim=0)
        cumsum_delta = torch.hstack([torch.zeros(1, device=data.device), cumsum_delta])
        cumsum_values = torch.cumsum(delta_values, dim=0)
        cumsum_values = torch.hstack([torch.zeros(1, device=data.device), cumsum_values])
        
        # 在正常索引中找到插值点的左侧索引
        left_idx = torch.searchsorted(normal_indices, indices) - 1
        left_idx[left_idx < 0] = 0
        
        # 获取左侧已知点的坐标和值
        left_normal = normal_indices[left_idx]
        left_values = values[left_idx]
        
        # 计算插值斜率
        slope = delta_values[left_idx]
        slope = torch.nan_to_num(slope)  # 避免除以零产生的 NaN
        
        # 计算插值结果
        interpolated_values = left_values + slope * (indices - left_normal)
        return interpolated_values
    
    def get_normal_indices(self, seq_len, indices):
        all_indices = torch.arange(seq_len, device=indices.device)
        mask = torch.ones_like(all_indices, dtype=torch.bool)
        # 检查 indices 是否越界
        valid_indices = indices[(indices >= 0) & (indices < seq_len)]
        if valid_indices.numel() > 0:
            mask[valid_indices] = False
        normal_indices = all_indices[mask]
        return normal_indices

    def linear_interpolate(self, data, fill_indices):
        batch_size, seq_len, feature_dim = data.shape
        if type(data) is np.ndarray:
            filled_data = data.copy()
        elif type(data) is torch.Tensor:
            filled_data = data.clone()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        if len(fill_indices) > 0:  # ! Is 'fill_indices' possible to be None?
            for b in range(batch_size):
                for f in range(feature_dim):
                    # 使用布尔索引从 fill_indices[1] 中筛选出属于当前批次 b 的时间索引
                    if isinstance(data, np.ndarray):
                        indices = fill_indices[1][fill_indices[0] == b]
                    elif isinstance(data, torch.Tensor):
                        # import pdb; pdb.set_trace()
                        indices = fill_indices[1][fill_indices[0] == b]
                    else:
                        raise ValueError(f"Unsupported data type: {type(data)}")
                    if len(indices) > 0:
                        
                        if type(data) is np.ndarray:
                            normal_indices = np.setdiff1d(np.arange(seq_len), indices)
                            if len(normal_indices) == 0:
                                logging.warning(f"No normal indices for batch {b} feature {f} data: {data[b, :, f]}")
                                continue
                            filled_data[b, indices, f] = np.interp(indices, normal_indices, data[b, normal_indices, f])
                        elif type(data) is torch.Tensor:
 
                            normal_indices = self.get_normal_indices(seq_len, indices)
                            if normal_indices.numel() == 0:
                                logging.warning(f"No normal indices for batch {b} feature {f} data: {data[b, :, f]}")
                                return data[b, :, f]
                            
                            # 线性插值的输入
                            x = normal_indices.to(torch.float32)
                            y = data[b, normal_indices, f].to(torch.float32)
                            x_new = indices.to(torch.float32)
                            
                            # 线性插值实现
                            def interp_torch(x_new, x, y):
                                """
								自定义的 PyTorch 线性插值函数
								"""
                                ind = torch.searchsorted(x, x_new)
                                ind = torch.clamp(ind, 1, x.numel() - 1)
                                lo = ind - 1
                                hi = ind
                                dx = x[hi] - x[lo]
                                dy = y[hi] - y[lo]
                                slope = dy / dx
                                return y[lo] + slope * (x_new - x[lo])
                            
                            interpolated_values = interp_torch(x_new, x, y)
                            filled_data = data.clone()
                            filled_data[b, indices, f] = interpolated_values
                            return filled_data
                        
                        else:
                            raise ValueError(f"Unsupported data type: {type(data)}")
        else:
            filled_data = data
                        
        return filled_data
    
    def rolling_mean(self, data, fill_indices):
        window_size = 1000  # FIXME: magic number
        if type(data) is np.ndarray:
            filled_data = data.copy()
        elif type(data) is torch.Tensor:
            filled_data = data.clone()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        batch_size, seq_len, feature_dim = data.shape
        for b in range(batch_size):
            for f in range(feature_dim):
                indices = fill_indices[1][fill_indices[0] == b]
                for idx in indices:
                    start = max(0, idx - window_size)
                    end = min(seq_len, idx + window_size + 1)
                    neighbors = data[b, start:end, f]
                    valid_neighbors = neighbors[neighbors != 0]
                    if len(valid_neighbors) > 0:
                        if type(data) is np.ndarray:
                            filled_data[b, idx, f] = np.mean(valid_neighbors)
                        elif type(data) is torch.Tensor:
                            filled_data[b, idx, f] = torch.mean(valid_neighbors)
                        else:
                            raise ValueError(f"Unsupported data type: {type(data)}")
        return filled_data

    def forward_fill(self, data, fill_indices):
        if type(data) is np.ndarray:
            filled_data = data.copy()
        elif type(data) is torch.Tensor:
            filled_data = data.clone()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        batch_size, seq_len, feature_dim = data.shape
        for b in range(batch_size):
            for f in range(feature_dim):
                indices = fill_indices[1][fill_indices[0] == b]
                for idx in indices:
                    if idx > 0:
                        filled_data[b, idx, f] = filled_data[b, idx - 1, f]
                    else:
                        filled_data[b, idx, f] = filled_data[b, idx + 1, f]
        return filled_data

    def backward_fill(self, data, fill_indices):
        if type(data) is np.ndarray:
            filled_data = data.copy()
        elif type(data) is torch.Tensor:
            filled_data = data.clone()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        batch_size, seq_len, feature_dim = data.shape
        for b in range(batch_size):
            for f in range(feature_dim):
                indices = fill_indices[1][fill_indices[0] == b]
                for idx in indices[::-1]:
                    if idx < seq_len - 1:
                        filled_data[b, idx, f] = filled_data[b, idx + 1, f]
                    else:
                        filled_data[b, idx, f] = filled_data[b, idx - 1, f]
        return filled_data
