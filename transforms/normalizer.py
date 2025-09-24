import logging
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

from config import nan_inf_clip_factor
from utils import my_clip

class Normalizer:
    def __init__(self, method, mode, input_data, history_data, dataset_scaler, ratio, clip_factor):  # FIXME:ratio
        assert method in ['none', 'standard', 'minmax', 'maxabs', 'robust']
        assert mode in ['none', 'dataset', 'input', 'history']
        self.method = method
        self.mode = mode
        self.data_in = None
        self.clip_factor = float(clip_factor) if clip_factor != 'none' else nan_inf_clip_factor

        if mode == 'none' or method == 'none':
            return
        if mode == 'dataset':
            assert isinstance(dataset_scaler, (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler)), \
                'Invalid dataset scaler type: {}'.format(type(dataset_scaler))
            self.scaler = dataset_scaler
        elif mode in ['input', 'history']:
            data = input_data if mode == 'input' else history_data
            assert 0 < ratio <= 1, 'Invalid ratio: {}'.format(ratio)
            self.scaler_params = self._compute_scaler_params(data, ratio)
        else:
            raise Exception('Invalid normalizer mode: {}'.format(self.mode))

    def _compute_scaler_params(self, data, look_back_ratio):
        assert data.ndim == 3  # (batch, time, feature)
        batch, time, feature = data.shape

        look_back_len = int(time * look_back_ratio)

        self.data_in = data
        params = {}
        for i in range(feature):
            feature_data = data[:, :, i].reshape(batch, -1)[:, -look_back_len:]
            if self.method == 'standard':
                if isinstance(data, np.ndarray):
                    mean = np.mean(feature_data, axis=1, keepdims=True)
                    std = np.std(feature_data, axis=1, keepdims=True)
                elif isinstance(data, torch.Tensor):
                    mean = torch.mean(feature_data, dim=1, keepdims=True)
                    std = torch.std(feature_data, dim=1, keepdims=True)
                else:
                    raise ValueError('Invalid data type: {}'.format(type(data)))
                params[i] = (mean, std)
            elif self.method == 'minmax':
                if isinstance(data, np.ndarray):
                    min_val = np.min(feature_data, axis=1, keepdims=True)
                    max_val = np.max(feature_data, axis=1, keepdims=True)
                elif isinstance(data, torch.Tensor):
                    min_val = torch.min(feature_data, dim=1, keepdims=True)
                    max_val = torch.max(feature_data, dim=1, keepdims=True)
                else:
                    raise ValueError('Invalid data type: {}'.format(type(data)))
                params[i] = (min_val, max_val)
            elif self.method == 'maxabs':
                if isinstance(data, np.ndarray):
                    max_abs_val = np.max(np.abs(feature_data), axis=1, keepdims=True)
                elif isinstance(data, torch.Tensor):
                    max_abs_val = torch.max(torch.abs(feature_data), dim=1, keepdims=True)
                else:
                    raise ValueError('Invalid data type: {}'.format(type(data)))
                params[i] = max_abs_val
            elif self.method == 'robust':
                if isinstance(data, np.ndarray):
                    median = np.median(feature_data, axis=1, keepdims=True)
                    q1 = np.percentile(feature_data, 25, axis=1, keepdims=True)
                    q3 = np.percentile(feature_data, 75, axis=1, keepdims=True)
                elif isinstance(data, torch.Tensor):
                    median = torch.median(feature_data, dim=1, keepdims=True).values
                    q1 = torch.quantile(feature_data, 0.25, dim=1, keepdim=True)
                    q3 = torch.quantile(feature_data, 0.75, dim=1, keepdim=True)
                else:
                    raise ValueError('Invalid data type: {}'.format(type(data)))
                params[i] = (median, q1, q3)
        return params

    def pre_process(self, data):
        if self.mode == 'none' or self.method == 'none':
            return data
        assert len(data.shape) == 3  # (batch, time, feature)
        batch, time, feature = data.shape
        if isinstance(data, np.ndarray):
            res = np.zeros_like(data)
        elif isinstance(data, torch.Tensor):
            res = torch.zeros_like(data)
        else:
            raise ValueError('Invalid data type: {}'.format(type(data)))
        if self.mode == 'dataset':
            # 使用 sklearn 的 scaler 进行变换
            for i in range(feature):
                feature_data = data[:, :, i].reshape(-1, 1)
                res[:, :, i] = self.scaler.transform(feature_data).reshape(batch, time)
        else:
            # 使用自定义的 scaler 参数进行变换
            for i in range(feature):
                feature_data = data[:, :, i].reshape(batch, -1)
                if self.method == 'standard':
                    mean, std = self.scaler_params[i]
                    res[:, :, i] = ((feature_data - mean) / (std + 1e-8)).reshape(batch, time)
                elif self.method == 'minmax':
                    min_val, max_val = self.scaler_params[i]
                    res[:, :, i] = ((feature_data - min_val) / (max_val - min_val + 1e-8)).reshape(batch, time)
                elif self.method == 'maxabs':
                    max_abs_val = self.scaler_params[i]
                    res[:, :, i] = (feature_data / (max_abs_val + 1e-8)).reshape(batch, time)
                elif self.method == 'robust':
                    median, q1, q3 = self.scaler_params[i]
                    res[:, :, i] = ((feature_data - median) / (q3 - q1 + 1e-8)).reshape(batch, time)

        return res

    def post_process(self, data):
        if self.mode == 'none' or self.method == 'none':
            return data
        assert len(data.shape) == 3  # (batch, time, feature)
        batch, time, feature = data.shape
        if isinstance(data, np.ndarray):
            res = np.zeros_like(data)
        elif isinstance(data, torch.Tensor):
            res = torch.zeros_like(data)
        else:
            raise ValueError('Invalid data type: {}'.format(type(data)))
        if self.mode == 'dataset':
            # 使用 sklearn 的 scaler 进行反变换
            for i in range(feature):
                feature_data = data[:, :, i].reshape(-1, 1)
                res[:, :, i] = self.scaler.inverse_transform(feature_data).reshape(batch, time)
        else:
            # 使用自定义的 scaler 参数进行反变换
            for i in range(feature):
                feature_data = data[:, :, i].reshape(batch, -1)
                if self.method == 'standard':
                    mean, std = self.scaler_params[i]
                    res[:, :, i] = (feature_data * std + mean).reshape(batch, time)
                elif self.method == 'minmax':
                    min_val, max_val = self.scaler_params[i]
                    res[:, :, i] = (feature_data * (max_val - min_val) + min_val).reshape(batch, time)
                elif self.method == 'maxabs':
                    max_abs_val = self.scaler_params[i]
                    res[:, :, i] = (feature_data * max_abs_val).reshape(batch, time)
                elif self.method == 'robust':
                    median, q1, q3 = self.scaler_params[i]
                    res[:, :, i] = (feature_data * (q3 - q1) + median).reshape(batch, time)
        if isinstance(self.data_in, np.ndarray):
            if np.isnan(res).any() or np.isinf(res).any():
                logging.error(f"NaN or Inf values in restored data: {res}")
                res = my_clip(self.data_in, res, nan_inf_clip_factor=nan_inf_clip_factor)  # 表示出结果很差
            elif self.clip_factor is not None:
                res = my_clip(self.data_in, res, min_max_clip_factor=self.clip_factor)  # 要求严格
        elif isinstance(self.data_in, torch.Tensor):
            if torch.isnan(res).any() or torch.isinf(res).any():
                res = my_clip(self.data_in, res, nan_inf_clip_factor=nan_inf_clip_factor)  # 表示出结果很差
            elif self.clip_factor is not None:
                res = my_clip(self.data_in, res, min_max_clip_factor=self.clip_factor)  # 要求严格
        else:
            raise ValueError('Invalid data type: {}'.format(type(data)))
        

        return res