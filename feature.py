import os
import warnings
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, welch
from scipy.stats import skew, kurtosis, norm, entropy, pearsonr
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt

# 常量定义
DEFAULT_PERIODS = [4, 7, 12, 24, 48, 52, 96, 144, 168, 336, 672, 1008, 1440]
CATCH22_FEATURES = [
    "DN_HistogramMode_5", "DN_HistogramMode_10", "SB_BinaryStats_mean_longstretch1",
    "SB_BinaryStats_mean_longstretch0", "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Basic_tau_incr1_embDim2", "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "MD_hrv_classic_pnn40", "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_10", "CO_Embed2_Basic_tau_incr1_embDim10",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1", "SP_Summaries_welch_rect_area_5_1",
    "SP_Summaries_welch_rect_centroid", "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh", "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r2",
    "FC_LocalSimple_mean1_tauresrat", "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd", "SP_Summaries_welch_rect_area_1_2"
]

# 抑制警告
warnings.filterwarnings("ignore")
plt.switch_backend('Agg')  # 非交互式后端，避免显示图形窗口


class PurePythonTSProcessor:
    def __init__(self, output_dir: str = "pure_python_characteristics"):
        """初始化纯Python时间序列处理器"""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def read_data(self, path: str, nrows=None) -> pd.Series:
        """读取单变量时间序列数据"""
        data = pd.read_csv(path)
        
        # 处理可能的日期列
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
        
        # 只取第一列数据作为单变量序列
        ts_series = data.iloc[:, 0].dropna()
        
        # 限制行数
        if nrows is not None and isinstance(nrows, int) and len(ts_series) >= nrows:
            ts_series = ts_series.iloc[:nrows]
            
        return ts_series.astype(float)

    def adjust_period(self, period_value: int) -> int:
        """调整周期值到最近的标准周期"""
        if abs(period_value - 4) <= 1:
            return 4
        if abs(period_value - 7) <= 1:
            return 7
        if abs(period_value - 12) <= 2:
            return 12
        if abs(period_value - 24) <= 3:
            return 24
        if abs(period_value - 48) <= 4:
            return 48
        if abs(period_value - 52) <= 2:
            return 52
        if abs(period_value - 96) <= 10:
            return 96
        if abs(period_value - 144) <= 10:
            return 144
        if abs(period_value - 168) <= 10:
            return 168
        if abs(period_value - 336) <= 50:
            return 336
        if abs(period_value - 672) <= 20:
            return 672
        if abs(period_value - 1008) <= 100:
            return 1008
        if abs(period_value - 1440) <= 200:
            return 1440
        return period_value

    def fft_transfer(self, timeseries: np.ndarray, fmin: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
        """使用FFT提取周期特征"""
        yf = abs(np.fft.fft(timeseries))
        yfnormlize = yf / len(timeseries)
        yfhalf = yfnormlize[: len(timeseries) // 2] * 2

        fwbest = yfhalf[argrelextrema(yfhalf, np.greater)]
        xwbest = argrelextrema(yfhalf, np.greater)

        fwbest = fwbest[fwbest >= fmin].copy()

        return len(timeseries) / xwbest[0][: len(fwbest)], fwbest

    def count_inversions(self, series: np.ndarray) -> int:
        """计算序列中的逆序数量"""
        def merge_sort(arr):
            if len(arr) <= 1:
                return arr, 0

            mid = len(arr) // 2
            left, inversions_left = merge_sort(arr[:mid])
            right, inversions_right = merge_sort(arr[mid:])

            merged = []
            inversions = inversions_left + inversions_right

            i, j = 0, 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    merged.append(left[i])
                    i += 1
                else:
                    merged.append(right[j])
                    j += 1
                    inversions += len(left) - i

            merged.extend(left[i:])
            merged.extend(right[j:])
            return merged, inversions

        _, inversions_count = merge_sort(series.tolist())
        return inversions_count

    def count_peaks_and_valleys(self, sequence: np.ndarray) -> int:
        """计算序列中的峰和谷数量"""
        peaks = 0
        valleys = 0

        for i in range(1, len(sequence) - 1):
            if sequence[i] > sequence[i - 1] and sequence[i] > sequence[i + 1]:
                peaks += 1
            elif sequence[i] < sequence[i - 1] and sequence[i] < sequence[i + 1]:
                valleys += 1

        return peaks + valleys

    def count_series(self, sequence: np.ndarray, threshold: float) -> int:
        """计算超过/低于阈值的连续序列数量"""
        if len(sequence) == 0:
            return 0

        positive_series = 0
        negative_series = 0
        current_class = None

        for value in sequence:
            if value > threshold:
                if current_class == "negative":
                    negative_series += 1
                current_class = "positive"
            else:
                if current_class == "positive":
                    positive_series += 1
                current_class = "negative"

        if current_class == "positive":
            positive_series += 1
        elif current_class == "negative":
            negative_series += 1

        return positive_series + negative_series

    def extract_statistical_features(self, series: pd.Series) -> dict:
        """提取统计特征"""
        series_values = series.values
        length = len(series_values)
        
        # 基础统计量
        mean = np.mean(series_values)
        std = np.std(series_values)
        
        return {
            "length": length,
            "mean": mean,
            "std": std,
            "skewness": skew(series_values),
            "kurtosis": kurtosis(series_values),
            "rsd": abs((std / mean) * 100) if mean != 0 else 0,
            "std_of_first_derivative": np.std(np.diff(series_values)) if length > 1 else 0,
            "inversions_ratio": self.count_inversions(series_values) / length,
            "turning_points_ratio": self.count_peaks_and_valleys(series_values) / length,
            "threshold_crossing_ratio": self.count_series(series_values, np.median(series_values)) / length
        }

    def extract_seasonal_trend_features(self, series: pd.Series) -> dict:
        """提取季节性和趋势特征"""
        series_values = series.values
        length = len(series_values)
        
        # 使用FFT识别周期
        periods, amplitude = self.fft_transfer(series_values, fmin=0)
        
        # 处理周期
        periods_list = []
        for index_j in range(min(3, len(amplitude))):
            periods_list.append(
                round(periods[amplitude.tolist().index(sorted(amplitude, reverse=True)[index_j])])
            )
        
        # 调整周期到标准值
        final_periods = []
        for period in periods_list:
            adjusted = self.adjust_period(period)
            if adjusted not in final_periods and adjusted >= 4:
                final_periods.append(adjusted)
        
        # 补充默认周期
        final_periods += [p for p in DEFAULT_PERIODS if p not in final_periods]
        
        # 计算季节性和趋势强度
        season_dict = {}
        yuzhi = max(int(length / 3), 12)  # 周期阈值
        
        for period_value in final_periods[:3]:  # 只取前3个周期
            if period_value < yuzhi and period_value > 0:
                try:
                    stl = STL(series, period=period_value).fit()
                    trend = stl.trend
                    seasonal = stl.seasonal
                    resid = stl.resid
                    
                    detrend = series_values - trend
                    deseasonal = series_values - seasonal
                    
                    # 计算趋势强度和季节性强度
                    trend_strength = max(0, 1 - (resid.var() / deseasonal.var())) if deseasonal.var() != 0 else 0
                    seasonal_strength = max(0, 1 - (resid.var() / detrend.var())) if detrend.var() != 0 else 0
                    
                    season_dict[seasonal_strength] = [period_value, seasonal_strength, trend_strength]
                except:
                    continue
        
        # 确保至少有3个周期特征（不足的用0填充）
        while len(season_dict) < 3:
            season_dict[0.0] = [0, 0, 0]
            
        # 按季节性强度排序
        sorted_seasons = sorted(season_dict.items(), key=lambda x: x[0], reverse=True)
        
        # 提取最强的三个周期特征
        features = {}
        for i, (_, vals) in enumerate(sorted_seasons[:3]):
            features[f"period_{i+1}"] = vals[0]
            features[f"seasonal_strength_{i+1}"] = vals[1]
            features[f"trend_strength_{i+1}"] = vals[2]
        
        # 总体季节性和趋势判断
        max_seasonal_strength = sorted_seasons[0][1][1] if sorted_seasons else 0
        max_trend_strength = sorted_seasons[0][1][2] if sorted_seasons else 0
        
        features["has_seasonality"] = max_seasonal_strength >= 0.9
        features["has_trend"] = max_trend_strength >= 0.85
        
        return features

    def extract_stationarity_features(self, series: pd.Series) -> dict:
        """提取平稳性特征"""
        series_values = series.values
        
        # ADF检验
        try:
            adf_result = adfuller(series_values, autolag="AIC")
            adf_pvalue = adf_result[1]
        except:
            adf_pvalue = None
            
        # KPSS检验
        try:
            kpss_result = kpss(series_values, regression="c")
            kpss_pvalue = kpss_result[1]
        except:
            kpss_pvalue = None
            
        # 稳定性判断
        if adf_pvalue is not None and kpss_pvalue is not None:
            stability = adf_pvalue <= 0.05 or kpss_pvalue >= 0.05
        else:
            stability = None
            
        return {
            "adf_pvalue": adf_pvalue,
            "kpss_pvalue": kpss_pvalue,
            "stability": stability
        }

    def calculate_jsd(self, series: pd.Series, window_size: int) -> float:
        """计算Jensen-Shannon散度"""
        data = series.values
        num_windows = len(data) // window_size
        
        if num_windows < 2:
            return np.nan
            
        jsd_list = []
        for i in range(num_windows):
            window_data = data[i * window_size : (i + 1) * window_size]
            hist, bin_edges = np.histogram(window_data, bins="stone", density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            mu = np.mean(window_data)
            sigma = np.std(window_data)
            
            if sigma == 0:
                jsd_list.append(0)
                continue
                
            pdf = norm.pdf(bin_centers, mu, sigma)
            
            # 计算JSD
            m = 0.5 * (hist + pdf)
            kl_p_m = entropy(hist, m) if not np.any(m == 0) else 0
            kl_q_m = entropy(pdf, m) if not np.any(m == 0) else 0
            jsd = 0.5 * (kl_p_m + kl_q_m)
            jsd_list.append(jsd)
            
        return np.mean(jsd_list)

    def extract_jsd_features(self, series: pd.Series) -> dict:
        """提取JSD特征"""
        return {
            "short_term_jsd": self.calculate_jsd(series, window_size=30),
            "long_term_jsd": self.calculate_jsd(series, window_size=336)
        }

    def extract_acf_pacf_features(self, series: pd.Series) -> dict:
        """提取ACF和PACF特征"""
        series_values = series.values
        nlags = min(20, len(series_values) // 2)
        
        if nlags < 2:
            return {
                "acf1": 0, "acf5": 0, "pacf1": 0, "pacf5": 0,
                "firstmin_ac": 0, "firstzero_ac": 0
            }
            
        # 计算ACF和PACF
        acf_vals = acf(series_values, nlags=nlags, fft=False)
        pacf_vals = pacf(series_values, nlags=nlags)
        
        # 提取ACF特征
        firstmin_ac = None
        for i in range(1, len(acf_vals)):
            if i > 1 and acf_vals[i-1] > 0 and acf_vals[i] < 0:
                firstmin_ac = i
                break
                
        firstzero_ac = None
        for i in range(1, len(acf_vals)):
            if abs(acf_vals[i]) < 0.01:
                firstzero_ac = i
                break
                
        return {
            "acf1": acf_vals[1],  # 滞后1的ACF值
            "acf5": acf_vals[5] if 5 < len(acf_vals) else 0,  # 滞后5的ACF值
            "pacf1": pacf_vals[1],  # 滞后1的PACF值
            "pacf5": pacf_vals[5] if 5 < len(pacf_vals) else 0,  # 滞后5的PACF值
            "firstmin_ac": firstmin_ac or nlags,
            "firstzero_ac": firstzero_ac or nlags
        }

    def extract_holt_parameters(self, series: pd.Series) -> dict:
        """提取Holt指数平滑参数"""
        try:
            # 尝试简单指数平滑
            model = ExponentialSmoothing(series)
            fit = model.fit()
            return {
                "holt_alpha": fit.params['smoothing_level'],
                "holt_beta": fit.params.get('smoothing_trend', 0),
                "holt_gamma": fit.params.get('smoothing_seasonal', 0)
            }
        except:
            return {"holt_alpha": 0, "holt_beta": 0, "holt_gamma": 0}

    def hurst_exponent(self, series: pd.Series) -> float:
        """计算Hurst指数"""
        series_values = series.values
        n = len(series_values)
        if n < 10:
            return 0.5  # 随机序列的Hurst指数
        
        # 计算均值
        mean = np.mean(series_values)
        
        # 计算累积偏差
        cum_dev = np.cumsum(series_values - mean)
        
        # 计算范围和标准差
        range_vals = []
        std_vals = []
        sizes = np.floor(n / 2 **np.arange(0, int(np.log2(n)))).astype(int)
        sizes = sizes[sizes >= 10]  # 只考虑足够大的窗口
        
        for s in sizes:
            num_windows = n // s
            windows = np.array_split(cum_dev[:num_windows*s], num_windows)
            
            # 每个窗口的范围
            window_ranges = [np.max(w) - np.min(w) for w in windows]
            range_vals.append(np.mean(window_ranges))
            
            # 每个窗口的标准差
            window_std = [np.std(series_values[i*s:(i+1)*s]) for i in range(num_windows)]
            std_vals.append(np.mean(window_std))
        
        # 避免除以零
        std_vals = np.array(std_vals)
        range_vals = np.array(range_vals)
        valid = (std_vals > 0) & (range_vals > 0)
        
        if np.sum(valid) < 2:
            return 0.5
            
        # 计算对数和Hurst指数
        log_sizes = np.log(sizes[valid])
        log_rs = np.log(range_vals[valid] / std_vals[valid])
        
        # 线性回归求斜率
        hurst = np.polyfit(log_sizes, log_rs, 1)[0]
        return max(0, min(1, hurst))  # 限制在0-1之间

    def arch_statistic(self, series: pd.Series) -> float:
        """计算ARCH统计量"""
        try:
            # 拟合AR模型
            model = AutoReg(series, lags=1)
            result = model.fit()
            residuals = result.resid
            
            # 平方残差的自相关
            acf_resid = acf(residuals**2, nlags=1, fft=False)
            return acf_resid[1] if len(acf_resid) > 1 else 0
        except:
            return 0

    def extract_catch22_features(self, series: pd.Series) -> dict:
        """提取类似Catch22的特征（Python实现）"""
        series_values = series.values
        n = len(series_values)
        
        # 基本统计
        mean_val = np.mean(series_values)
        std_val = np.std(series_values)
        
        # 标准化序列
        if std_val > 0:
            normalized = (series_values - mean_val) / std_val
        else:
            normalized = np.zeros_like(series_values)
        
        features = {}
        
        # 1. 直方图模式特征
        hist5, _ = np.histogram(series_values, bins=5)
        features["DN_HistogramMode_5"] = np.argmax(hist5)
        
        hist10, _ = np.histogram(series_values, bins=10)
        features["DN_HistogramMode_10"] = np.argmax(hist10)
        
        # 2. 二进制统计特征（基于均值的阈值）
        binary = (series_values > mean_val).astype(int)
        streaks1 = []
        streaks0 = []
        current_streak = 1
        current_val = binary[0]
        
        for val in binary[1:]:
            if val == current_val:
                current_streak += 1
            else:
                if current_val == 1:
                    streaks1.append(current_streak)
                else:
                    streaks0.append(current_streak)
                current_streak = 1
                current_val = val
        
        features["SB_BinaryStats_mean_longstretch1"] = np.mean(streaks1) if streaks1 else 0
        features["SB_BinaryStats_mean_longstretch0"] = np.mean(streaks0) if streaks0 else 0
        features["SB_BinaryStats_diff_longstretch0"] = np.std(streaks0) if streaks0 else 0
        
        # 3. 功率谱特征
        if n >= 10:
            freqs, psd = welch(series_values)
            features["SP_Summaries_welch_rect_centroid"] = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
            
            # 特定频段的面积
            mask1 = (freqs >= 1/2) & (freqs <= 1)
            features["SP_Summaries_welch_rect_area_1_2"] = np.sum(psd[mask1]) if np.any(mask1) else 0
            
            mask2 = (freqs >= 1/5) & (freqs <= 1)
            features["SP_Summaries_welch_rect_area_5_1"] = np.sum(psd[mask2]) if np.any(mask2) else 0
        
        # 4. 离群值特征
        mad = np.median(np.abs(series_values - np.median(series_values)))
        outlier_mask = np.abs(series_values - np.median(series_values)) > 3 * mad
        features["DN_OutlierInclude_n_001_mdrmd"] = np.sum(outlier_mask)
        features["DN_OutlierInclude_p_001_mdrmd"] = np.mean(outlier_mask) if n > 0 else 0
        
        # 5. 其他特征使用默认值或简化实现
        features["PD_PeriodicityWang_th0_01"] = 0
        features["PD_PeriodicityWang_th0_10"] = 0
        features["CO_Embed2_Basic_tau_incr1_embDim2"] = 0
        features["CO_Embed2_Basic_tau_incr1_embDim10"] = 0
        features["IN_AutoMutualInfoStats_40_gaussian_fmmi"] = 0
        features["MD_hrv_classic_pnn40"] = 0
        features["SB_TransitionMatrix_3ac_sumdiagcov"] = 0
        features["SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1"] = 0
        features["SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1"] = 0
        features["SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r2"] = 0
        features["SB_MotifThree_quantile_hh"] = 0
        features["FC_LocalSimple_mean1_tauresrat"] = 0
        
        return features

    def extract_additional_features(self, series: pd.Series) -> dict:
        """提取其他补充特征"""
        return {
            "hurst": self.hurst_exponent(series),
            "arch_stat": self.arch_statistic(series),
            **self.extract_holt_parameters(series),** self.extract_acf_pacf_features(series)
        }

    def process_file(self, file_path: str) -> pd.DataFrame:
        """处理单个文件并提取所有特征"""
        print(f"处理文件: {file_path}")
        ts_series = self.read_data(file_path)
        return self.process_data(ts_series.values)
    
    def process_data(self, data: np.ndarray) -> pd.DataFrame:
        """处理直接传入的时间序列数据"""
        ts_series = pd.Series(data).dropna().astype(float)
        if len(ts_series) < 10:
            raise ValueError("时间序列长度过短，无法处理")
        
        # 提取各类特征
        stat_features = self.extract_statistical_features(ts_series)
        seasonal_trend_features = self.extract_seasonal_trend_features(ts_series)
        stationarity_features = self.extract_stationarity_features(ts_series)
        jsd_features = self.extract_jsd_features(ts_series)
        additional_features = self.extract_additional_features(ts_series)
        catch22_features = self.extract_catch22_features(ts_series)
        
        # 合并所有特征
        all_features = {
            **stat_features,** seasonal_trend_features,
            **stationarity_features,** jsd_features,
            **additional_features,** catch22_features
        }
        
        # 转换为DataFrame
        result_df = pd.DataFrame([all_features])
        
        return result_df

    def _save_results(self, result_df: pd.DataFrame, file_path: str) -> None:
        """保存特征结果"""
        file_basename = os.path.splitext(os.path.basename(file_path))[0]
        
        # 保存完整特征
        full_features_path = os.path.join(
            self.output_dir, f"full_features_{file_basename}.csv"
        )
        result_df.to_csv(full_features_path, index=False)
        
        # 提取关键特征并保存
        key_features = self._extract_key_features(result_df)
        key_features_path = os.path.join(
            self.output_dir, f"key_features_{file_basename}.csv"
        )
        key_features.to_csv(key_features_path, index=False)
        
        print(f"特征已保存至: {full_features_path} 和 {key_features_path}")

    def _extract_key_features(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """提取关键特征子集"""
        key_columns = [
            "length", "mean", "std", "skewness", "kurtosis",
            "seasonal_strength_1", "trend_strength_1",
            "adf_pvalue", "kpss_pvalue", "stability",
            "hurst", "arch_stat", "short_term_jsd", "long_term_jsd"
        ]
        
        # 保留存在的列
        available_columns = [col for col in key_columns if col in result_df.columns]
        return result_df[available_columns].rename(columns={
            "seasonal_strength_1": "seasonality_strength",
            "trend_strength_1": "trend_strength",
            "adf_pvalue": "adf_test_pvalue",
            "kpss_pvalue": "kpss_test_pvalue"
        })


if __name__ == "__main__":
    # 示例用法
    processor = PurePythonTSProcessor()
    # 替换为你的单变量时间序列CSV文件路径
    filelist=['./data/ETT-small/ETTh1.csv','./data/ETT-small/ETTh2.csv','./data/ETT-small/ETTm1.csv','./data/ETT-small/ETTm2.csv','./data/exchange_rate/exchange_rate.csv','./data/traffic/traffic.csv','./data/weather/weather.csv']
    for i in filelist:
        processor.process_file(i)
    print("处理完成")
    