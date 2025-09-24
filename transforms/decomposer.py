
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy import signal
from scipy.fft import fft, ifft

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from statsmodels.tsa.stl._stl import STL


def assert_timeseries_3d_np(data):
    assert type(data) is np.ndarray and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'

def assert_timeseries_3d_tensor(data):
    assert type(data) is torch.Tensor and data.ndim == 3, \
        f'type(data)={type(data)}, data.ndim={data.ndim}, data.shape={data.shape}'
    

# 序列的分解方法(废弃
class Decomposer:
    def __init__(self, period, component_for_model, parallel=True):
        self.period = int(period) if period != 'none' else None
        self.component_for_model = component_for_model.split('+')
        self.none_flag = component_for_model == 'none' or self.period is None
        self.trend = None
        self.season = None
        self.residual = None
        self.parallel = parallel

    def pre_process(self, data):
        if self.none_flag:
            return data

        def _decompose_series1(series):

            stl = STL(series, period=self.period, seasonal=13)
            result = stl.fit()
            return result.trend, result.seasonal, result.resid

        def _decompose_series5(series):

            freqs = np.fft.fftfreq(len(series))
            fft_values = fft(series)
            trend_fft = fft_values.copy()
            trend_fft[np.abs(freqs) > 1 / self.period] = 0
            trend = ifft(trend_fft).real

            # Compute seasonal component using FFT high frequency components
            season_fft = fft_values.copy()
            season_fft[np.abs(freqs) <= 1 / self.period] = 0
            season = ifft(season_fft).real

            # Compute residual
            residual = series - trend - season
            return trend, season, residual

        def _decompose_series6(series):
            # Compute trend using linear regression
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series, 1)
            trend = np.polyval(coeffs, x)

            # Compute seasonal component by subtracting the trend and detrending the result
            detrended = series - trend
            # season = self._compute_seasonal(detrended, self.period)
            season = np.zeros_like(data)
            period = self.period
            for i in range(period):
                seasonal_mean = np.mean(data[i::period])
                season[i::period] = seasonal_mean

            # Compute residual
            residual = series - trend - season
            return trend, season, residual

        if self.parallel:
            # 使用 joblib 并行处理每个批次的数据
            results = Parallel(n_jobs=-1)(delayed(_decompose_series1)(data[i, :, 0]) for i in range(data.shape[0]))
            # 将结果分解成趋势、季节性和残差
            trends, seasons, residuals = zip(*results)
        else:
            trends, seasons, residuals = [], [], []
            for i in range(data.shape[0]):
                series = data[i, :, 0]  # Extract the series for each batch
                trend, seasonal, residual = _decompose_series5(series)
                # stl = STL(series, period=self.period, seasonal=13)
                # result = stl.fit()
                trends.append(trend)
                seasons.append(seasonal)
                residuals.append(residual)
        self.trend = np.array(trends).reshape(data.shape)
        self.season = np.array(seasons).reshape(data.shape)
        self.residual = np.array(residuals).reshape(data.shape)

        # Determine which components to return based on the target
        combined = np.zeros_like(data)
        if 'trend' in self.component_for_model:
            combined += self.trend
        if 'season' in self.component_for_model:
            combined += self.season
        if 'residual' in self.component_for_model:
            combined += self.residual

        return combined

    def post_process(self, pred):
        if self.none_flag:
            return pred
        pred_trend, pred_season, pred_residual = 0, 0, 0

        if 'trend' not in self.component_for_model:
            pred_trend = self._predict_trend(pred.shape[1])

        if 'season' not in self.component_for_model:
            pred_season = self._predict_season(pred.shape[1])

        if 'residual' not in self.component_for_model:
            pred_residual = self._predict_residual(pred.shape[1])

        return pred + pred_trend + pred_season + pred_residual

    def _predict_trend(self, pred_len):  # 用线性回归预测
        pred_trend = np.zeros((self.trend.shape[0], pred_len, 1))
        for i in range(self.trend.shape[0]):
            y = self.trend[i, :, 0]
            x = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            future_x = np.arange(len(y), len(y) + pred_len).reshape(-1, 1)
            pred_trend[i, :, 0] = model.predict(future_x)
        return pred_trend

    
    def _predict_season(self, pred_len):
        pred_season = np.zeros((self.season.shape[0], pred_len, 1))
        for i in range(self.season.shape[0]):
            y = self.season[i, :, 0]
            n = len(y)
            f = fft(y)
            frequencies = np.fft.fftfreq(n)
            # 只保留重要的频率
            threshold = 0.1
            f[np.abs(frequencies) > threshold] = 0
            future_frequencies = np.fft.fftfreq(n + pred_len)
            future_f = np.zeros_like(future_frequencies, dtype=complex)
            future_f[:n] = f
            future_f = ifft(future_f).real
            pred_season[i, :, 0] = future_f[-pred_len:]
        return pred_season

    def _predict_residual(self, pred_len):
        pred_residual = np.zeros((self.residual.shape[0], pred_len, 1))
        for i in range(self.residual.shape[0]):
            y = self.residual[i, :, 0]
            # 使用最后一个窗口的均值进行预测
            mean_value = np.mean(y[-pred_len:])
            pred_residual[i, :, 0] = mean_value
        return pred_residual
