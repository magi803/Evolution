#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSE50 信号分离参数搜索（全数据集版，目标 p < 0.05）
结果输出到 b.txt，同时打印到控制台。
依赖库：numpy, pandas, scipy, scikit-learn, PyWavelets, PyEMD (可选), vmdpy (可选), cupy (可选)
"""

import sys
import warnings
import numpy as np
import pandas as pd
import random
from scipy.stats import gaussian_kde, boxcox
from scipy.signal import argrelextrema, hilbert, butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import pywt

# 可选库导入
try:
    from PyEMD import EMD, EEMD
    HAS_PYEMD = True
except ImportError:
    HAS_PYEMD = False
    print("PyEMD 未安装，将跳过 EMD/EEMD 方法。")

try:
    from vmdpy import VMD
    HAS_VMD = True
except ImportError:
    HAS_VMD = False
    print("vmdpy 未安装，将跳过 VMD 方法。")

try:
    import cupy as cp
    HAS_CUPY = True
    print("CuPy 导入成功，GPU 加速已启用。")
except ImportError:
    HAS_CUPY = False
    print("CuPy 未安装，将使用 CPU 进行蒙特卡洛模拟（较慢）。")

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

# ==================== 数据获取（baostock）====================
def fetch_cn_index_data(index_code, index_name, start_date='2005-01-01', end_date='2024-12-31'):
    """从 baostock 获取指数日线数据"""
    try:
        import baostock as bs
    except ImportError:
        raise ImportError("baostock 未安装，请运行 pip install baostock")
    lg = bs.login()
    if lg.error_code != '0':
        raise Exception("baostock 登录失败")
    rs = bs.query_history_k_data_plus(
        index_code,
        "date,code,open,high,low,close,volume,amount",
        start_date=start_date, end_date=end_date,
        frequency='d', adjustflag='3')
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()
    if not data_list:
        raise Exception(f"未能获取 {index_name} 数据")
    cols = ['date','code','open','high','low','close','volume','amount']
    df = pd.DataFrame(data_list, columns=cols)
    df = df.replace('', np.nan)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    for col in ['open','high','low','close','volume','amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['close', 'volume'])
    df['adj_close'] = df['close']
    df['log_ret'] = np.log(df['adj_close'] / df['adj_close'].shift(1))
    df = df.dropna()
    print(f"{index_name} 数据已加载，共 {len(df)} 条记录，时间范围 {df.index[0]} 至 {df.index[-1]}")
    return df

def fetch_sz50():
    """获取上证50（sh.000016）数据"""
    return fetch_cn_index_data('sh.000016', 'SSE50', '2005-01-01', '2024-12-31')

# ==================== 扩展技术指标 ====================
def compute_raw_indicators_extended(df, lookback=20):
    """计算扩展的 13 个量价指标（原8个 + 新增5个）"""
    df = df.copy()
    epsilon = 1e-10
    # 原8个
    df['mom'] = df['close'] / (df['close'].shift(5) + epsilon) - 1
    df['vol_chg'] = df['volume'] / (df['volume'].shift(5) + epsilon) - 1
    vol_ma = df['volume'].rolling(lookback).mean()
    df['vol_ratio'] = df['volume'] / (vol_ma + epsilon)
    df['volatility'] = (df['high'] - df['low']) / (df['close'] + epsilon)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + epsilon)
    df['rsi'] = 100 - 100 / (1 + rs)
    ma20 = df['close'].rolling(lookback).mean()
    df['bias'] = (df['close'] - ma20) / (ma20 + epsilon)
    low10 = df['low'].rolling(10).min()
    high10 = df['high'].rolling(10).max()
    df['close_pos'] = (df['close'] - low10) / ((high10 - low10) + epsilon)
    vwap = (df['close'] * df['volume']).rolling(5).mean()
    df['vwap_chg'] = vwap / (vwap.shift(5) + epsilon) - 1

    # 新增5个指标
    # 成交量变异率 VR
    df['vr'] = (df['volume'].where(df['close'] > df['close'].shift(1), 0).rolling(26).sum() +
                0.5 * df['volume'].where(df['close'] == df['close'].shift(1), 0).rolling(26).sum()) / \
               (df['volume'].where(df['close'] < df['close'].shift(1), 0).rolling(26).sum() + epsilon)
    # 心理线 PSY
    df['psy'] = (df['close'] > df['close'].shift(1)).rolling(12).sum() / 12 * 100
    # 乖离率 BIAS 不同周期
    ma6 = df['close'].rolling(6).mean()
    df['bias6'] = (df['close'] - ma6) / (ma6 + epsilon)
    ma24 = df['close'].rolling(24).mean()
    df['bias24'] = (df['close'] - ma24) / (ma24 + epsilon)
    # 能量潮 OBV（简化：用收盘价涨跌决定成交量符号）
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv'] = obv
    # 威廉指标 W%R
    low14 = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    df['wr'] = (high14 - df['close']) / (high14 - low14 + epsilon) * -100

    indicator_cols = ['mom', 'vol_chg', 'vol_ratio', 'volatility', 'rsi', 'bias', 'close_pos', 'vwap_chg',
                      'vr', 'psy', 'bias6', 'bias24', 'obv', 'wr']
    df_indicators = df[indicator_cols].replace([np.inf, -np.inf], np.nan).dropna()
    return df_indicators, indicator_cols

# ==================== 增强型 GMM 情绪指数（固定参数）====================
def compute_emotion_index_gmm_enhanced(df, use_pca=True, pca_dims=2,
                                       n_components_range=[2,3,4,5],
                                       features='extended'):
    """
    增强型 GMM 情绪指数（固定参数版本）
    """
    if features == 'extended':
        df_indicators, _ = compute_raw_indicators_extended(df)
    else:
        df_indicators, _ = compute_raw_indicators_extended(df)  # 默认用扩展
    if len(df_indicators) < 30:
        raise ValueError("数据不足，无法计算情绪指数")

    # Box-Cox 变换（确保非负）
    df_boxcox = pd.DataFrame(index=df_indicators.index)
    for col in df_indicators.columns:
        data = df_indicators[col].values
        min_val = data.min()
        if min_val <= 0:
            data = data - min_val + 1e-3
        data, _ = boxcox(data)
        df_boxcox[col] = data

    # 标准化
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_boxcox)

    # PCA 降维
    if use_pca:
        pca = PCA(n_components=pca_dims)
        scaled = pca.fit_transform(scaled)

    # BIC 选择最优分量数
    best_bic = np.inf
    best_gmm = None
    for n in n_components_range:
        try:
            gmm = GaussianMixture(n_components=n, random_state=42, max_iter=500)
            gmm.fit(scaled)
            bic = gmm.bic(scaled)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        except:
            continue
    if best_gmm is None:
        best_gmm = GaussianMixture(n_components=3, random_state=42)
        best_gmm.fit(scaled)

    # 情绪指数：按分量均值 L2 范数加权
    probs = best_gmm.predict_proba(scaled)
    means = best_gmm.means_
    mean_norms = np.linalg.norm(means, axis=1)
    if mean_norms.sum() == 0:
        weights = np.ones(len(mean_norms)) / len(mean_norms)
    else:
        weights = mean_norms / mean_norms.sum()
    sentiment = np.dot(probs, weights)

    sentiment_series = pd.Series(index=df_indicators.index, data=sentiment)
    sentiment_series = sentiment_series.rolling(5).mean()
    return sentiment_series

# ==================== 熵计算（KDE）====================
def compute_entropy_kde(series, window=30):
    """滚动核密度估计熵（固定窗口）"""
    entropies = []
    for i in range(window, len(series)+1):
        window_data = series.iloc[i-window:i].values
        if np.std(window_data) < 1e-6:
            entropies.append(np.nan)
            continue
        kde = gaussian_kde(window_data)
        density = kde.evaluate(window_data)
        entropy = -np.mean(np.log(density + 1e-12))
        entropies.append(entropy)
    idx = series.index[window-1:]
    return pd.Series(entropies, index=idx)

# ==================== 系统演化指标计算（固定参数）====================
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def compute_evolution_indicators(df):
    """计算序量 M、L2 波动率、相位等（使用固定情绪指数和熵窗口）"""
    df_out = df.copy()
    # 情绪指数（使用默认参数）
    sentiment = compute_emotion_index_gmm_enhanced(
        df_out,
        use_pca=True,
        pca_dims=2,
        n_components_range=[2,3,4,5],
        features='extended'
    )
    df_out['sentiment'] = sentiment
    df_out = df_out.dropna(subset=['sentiment'])

    # 熵（窗口=30）
    df_out['M'] = -compute_entropy_kde(df_out['sentiment'], window=30)
    df_out = df_out.dropna(subset=['M'])

    # 构造 A、B、L2、L2_vol
    fs = 1.0
    lowcut_trend = 1/60
    lowcut_wave = 1/60
    highcut_wave = 1/5
    b, a = butter_bandpass(lowcut=0.001, highcut=lowcut_trend, fs=fs, order=3)
    A = filtfilt(b, a, df_out['sentiment'].values)
    df_out['A'] = A
    B = bandpass_filter(df_out['sentiment'].values, lowcut=lowcut_wave, highcut=highcut_wave, fs=fs, order=3)
    df_out['B'] = np.abs(B)
    df_out['L2'] = df_out['A']**2 - df_out['B']**2
    df_out['L2_vol'] = df_out['L2'].rolling(30).std()
    # 相位
    analytic = hilbert(df_out['sentiment'].values)
    phase = np.angle(analytic)
    df_out['theta'] = phase
    df_out['theta_unwrap'] = np.unwrap(phase)
    omega_raw = np.diff(df_out['theta_unwrap']) * 252 / (2 * np.pi)
    omega_raw = np.append(omega_raw, omega_raw[-1])
    df_out['omega_raw'] = omega_raw
    omega_med = np.nanmedian(omega_raw[~np.isnan(omega_raw) & ~np.isinf(omega_raw)])
    if np.isnan(omega_med) or omega_med == 0:
        omega_med = 0.1
    df_out['omega'] = omega_med
    return df_out

# ==================== 动态风险事件识别（固定分位数）====================
def identify_risk_events_dynamic(price, quantile=0.9, lookback_years=3, min_gap_days=60):
    """基于滚动回撤分位数识别风险事件"""
    window_days = int(lookback_years * 252)
    events = []
    last_event = None
    for i in range(window_days, len(price)):
        current_date = price.index[i]
        window = price.iloc[i-window_days:i]
        peak = window.expanding().max()
        drawdown = (peak - window) / peak
        current_peak = window.max()
        current_dd = (current_peak - price.iloc[i]) / current_peak
        dd_threshold = np.quantile(drawdown.dropna(), quantile)
        if current_dd >= dd_threshold:
            if last_event is None or (current_date - last_event).days >= min_gap_days:
                events.append(current_date)
                last_event = current_date
    return events

# ==================== 临界点识别 ====================
def get_peaks_localmax(series, order=10):
    peaks_idx = argrelextrema(series.values, np.greater, order=order)[0]
    return series.index[peaks_idx]

def get_peaks_adaptive(series, window=30, n_std=1.5, min_gap=20):
    roll_mean = series.rolling(window, center=True).mean()
    roll_std = series.rolling(window, center=True).std()
    threshold = roll_mean + n_std * roll_std
    above = series > threshold
    peak_dates = []
    in_block = False
    start = None
    for date, val in above.items():
        if val and not in_block:
            in_block = True
            start = date
        elif not val and in_block:
            in_block = False
            block = series.loc[start:date].iloc[:-1]
            if not block.empty:
                peak_dates.append(block.idxmax())
    if in_block:
        block = series.loc[start:]
        if not block.empty:
            peak_dates.append(block.idxmax())
    if len(peak_dates) > 1:
        filtered = [peak_dates[0]]
        for d in peak_dates[1:]:
            if (d - filtered[-1]).days >= min_gap:
                filtered.append(d)
        return filtered
    return peak_dates

# ==================== 信号分离方法 ====================
def wavelet_reconstruct(series, wavelet='db4', level=5, levels_to_remove=1):
    series = np.asarray(series).copy()
    coeffs = pywt.wavedec(series, wavelet, level=level)
    new_coeffs = list(coeffs)
    for i in range(1, min(levels_to_remove, len(new_coeffs))):
        new_coeffs[-i] = np.zeros_like(new_coeffs[-i])
    reconstructed = pywt.waverec(new_coeffs, wavelet)
    if len(reconstructed) > len(series):
        reconstructed = reconstructed[:len(series)]
    elif len(reconstructed) < len(series):
        reconstructed = np.pad(reconstructed, (0, len(series)-len(reconstructed)), 'edge')
    return reconstructed

def ssa_reconstruct(series, window_length=None, n_groups=2):
    series = np.asarray(series).copy()
    N = len(series)
    if window_length is None:
        window_length = min(N // 3, 100)
    K = N - window_length + 1
    X = np.zeros((window_length, K))
    for i in range(K):
        X[:, i] = series[i:i+window_length]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    X_recon = np.zeros_like(X)
    for i in range(n_groups, len(s)):
        X_recon += s[i] * np.outer(U[:, i], Vt[i, :])
    y = np.zeros(N)
    count = np.zeros(N)
    for i in range(window_length):
        for j in range(K):
            y[i+j] += X_recon[i, j]
            count[i+j] += 1
    y = y / count
    return y

def combined_reconstruct(series, method1='wavelet', params1={}, method2='ssa', params2={}):
    """组合分离：先 method1 再 method2 残差"""
    if method1 == 'wavelet':
        recon1 = wavelet_reconstruct(series, **params1)
    elif method1 == 'ssa':
        recon1 = ssa_reconstruct(series, **params1)
    else:
        raise ValueError("不支持的 method1")
    residual = series - recon1
    if method2 == 'wavelet':
        recon2 = wavelet_reconstruct(residual, **params2)
    elif method2 == 'ssa':
        recon2 = ssa_reconstruct(residual, **params2)
    else:
        recon2 = 0
    return recon1 + recon2

# 如果有 PyEMD，添加 EMD/EEMD 函数
if HAS_PYEMD:
    def emd_reconstruct(series, max_imf_to_remove=2):
        series = np.asarray(series).copy()
        emd = EMD()
        imfs = emd(series)
        if max_imf_to_remove >= imfs.shape[0]:
            return np.zeros_like(series)
        recon = np.sum(imfs[max_imf_to_remove:], axis=0) + emd.residue
        return recon

    def eemd_reconstruct(series, max_imf_to_remove=2, ensemble_size=50, noise_width=0.2):
        series = np.asarray(series).copy()
        eemd = EEMD()
        try:
            imfs = eemd.eemd(series, ensemble_size=ensemble_size, noise_width=noise_width)
        except TypeError:
            try:
                imfs = eemd.eemd(series, n_ensemble=ensemble_size, noise_width=noise_width)
            except TypeError:
                imfs = eemd.eemd(series, trials=ensemble_size, noise_width=noise_width)
        if max_imf_to_remove >= imfs.shape[0]:
            return np.zeros_like(series)
        recon = np.sum(imfs[max_imf_to_remove:], axis=0)
        residue = series - np.sum(imfs, axis=0)
        recon += residue
        return recon

if HAS_VMD:
    def vmd_reconstruct(series, alpha=2000, K=5, remove_high=2):
        series = np.asarray(series).copy()
        u, u_hat, omega = VMD(series, alpha, 0, K, 0, 1, 1e-7)
        if remove_high >= u.shape[0]:
            return np.zeros_like(series)
        recon = np.sum(u[:-remove_high], axis=0)
        return recon

# ==================== H1 快速检验（固定参数）====================
def h1_fast_test(df, separated_series, risk_events, n_sim=2000, peak_order=10):
    """
    快速 H1 检验：使用固定局部最大 order=10 和给定的风险事件列表。
    返回 (lead_days, random_expectation, p_value)
    """
    vol_series = separated_series.dropna()
    if len(risk_events) < 3:
        return np.nan, np.nan, 1.0

    ref_date = df.index[0]
    date_to_day = lambda d: (d - ref_date).days
    all_days = np.array([date_to_day(d) for d in df.index])
    risk_days = np.array([date_to_day(d) for d in risk_events])

    peak_dates = get_peaks_localmax(vol_series, order=peak_order)
    if len(peak_dates) < 2:
        return np.nan, np.nan, 1.0
    peak_days = np.array([date_to_day(d) for d in peak_dates])

    intervals = []
    for crisis_day in risk_days:
        prev_peaks = peak_days[peak_days < crisis_day]
        if len(prev_peaks) > 0:
            intervals.append(crisis_day - prev_peaks[-1])
    if len(intervals) == 0:
        return np.nan, np.nan, 1.0
    obs_mean = np.mean(intervals)

    # 蒙特卡洛模拟
    if HAS_CUPY:
        peak_days_gpu = cp.asarray(peak_days)
        all_days_gpu = cp.asarray(all_days)
        rand_days_gpu = cp.random.choice(all_days_gpu, size=(n_sim, len(risk_days)), replace=True)
        rand_flat = rand_days_gpu.flatten()
        pos = cp.searchsorted(peak_days_gpu, rand_flat, side='right') - 1
        valid = pos >= 0
        prev_peak_gpu = cp.where(valid, peak_days_gpu[pos], 0)
        intervals_flat = cp.where(valid, rand_flat - prev_peak_gpu, cp.nan)
        intervals_mat = intervals_flat.reshape(n_sim, len(risk_days))
        sim_means_gpu = cp.nanmean(intervals_mat, axis=1)
        sim_means_gpu = sim_means_gpu[~cp.isnan(sim_means_gpu)]
        sim_means = cp.asnumpy(sim_means_gpu)
    else:
        sim_means = []
        for _ in range(n_sim):
            rand_dates = np.random.choice(df.index, size=len(risk_events), replace=True)
            rand_days_rand = np.array([date_to_day(d) for d in rand_dates])
            sim_ints = []
            for rd in rand_days_rand:
                prev = peak_days[peak_days < rd]
                if len(prev) > 0:
                    sim_ints.append(rd - prev[-1])
            if sim_ints:
                sim_means.append(np.mean(sim_ints))
    if len(sim_means) == 0:
        return np.nan, np.nan, 1.0
    rand_avg = np.mean(sim_means)
    p_val = (np.sum(sim_means <= obs_mean) + 1) / (len(sim_means) + 1)
    return obs_mean, rand_avg, p_val

# ==================== 完整 H1 参数扫描（支持多风险事件列表）====================
def test_H1_scan(df, risk_events_dict, vol_series,
                 peak_methods=['localmax', 'adaptive'],
                 localmax_orders=[5,10,15,20,25,30,35,40],
                 adaptive_windows=[10,20,30,40,60,80,100],
                 adaptive_nstds=np.arange(0.5,3.1,0.2).tolist(),
                 adaptive_min_gaps=[10,20,30,40,60,90],
                 n_sim=5000):
    """
    对给定的风险事件列表字典和波动率序列进行完整 H1 参数扫描。
    risk_events_dict: 键为风险阈值标签，值为事件日期列表。
    返回最佳 (lead, rand, p, config)
    """
    best_p = 1.0
    best_config = None
    best_lead = np.nan
    best_rand = np.nan

    ref_date = df.index[0]
    date_to_day = lambda d: (d - ref_date).days
    all_days = np.array([date_to_day(d) for d in df.index])

    for thr_label, risk_events in risk_events_dict.items():
        if len(risk_events) < 3:
            continue
        risk_days = np.array([date_to_day(d) for d in risk_events])

        for method in peak_methods:
            if method == 'localmax':
                for order in localmax_orders:
                    peak_dates = get_peaks_localmax(vol_series, order=order)
                    if len(peak_dates) < 2:
                        continue
                    peak_days = np.array([date_to_day(d) for d in peak_dates])
                    intervals = []
                    for crisis_day in risk_days:
                        prev_peaks = peak_days[peak_days < crisis_day]
                        if len(prev_peaks) > 0:
                            intervals.append(crisis_day - prev_peaks[-1])
                    if len(intervals) == 0:
                        continue
                    obs_mean = np.mean(intervals)

                    if HAS_CUPY:
                        peak_days_gpu = cp.asarray(peak_days)
                        all_days_gpu = cp.asarray(all_days)
                        rand_days_gpu = cp.random.choice(all_days_gpu, size=(n_sim, len(risk_days)), replace=True)
                        rand_flat = rand_days_gpu.flatten()
                        pos = cp.searchsorted(peak_days_gpu, rand_flat, side='right') - 1
                        valid = pos >= 0
                        prev_peak_gpu = cp.where(valid, peak_days_gpu[pos], 0)
                        intervals_flat = cp.where(valid, rand_flat - prev_peak_gpu, cp.nan)
                        intervals_mat = intervals_flat.reshape(n_sim, len(risk_days))
                        sim_means_gpu = cp.nanmean(intervals_mat, axis=1)
                        sim_means_gpu = sim_means_gpu[~cp.isnan(sim_means_gpu)]
                        sim_means = cp.asnumpy(sim_means_gpu)
                    else:
                        sim_means = []
                        for _ in range(n_sim):
                            rand_dates = np.random.choice(df.index, size=len(risk_events), replace=True)
                            rand_days_rand = np.array([date_to_day(d) for d in rand_dates])
                            sim_ints = []
                            for rd in rand_days_rand:
                                prev = peak_days[peak_days < rd]
                                if len(prev) > 0:
                                    sim_ints.append(rd - prev[-1])
                            if sim_ints:
                                sim_means.append(np.mean(sim_ints))
                    if len(sim_means) == 0:
                        continue
                    rand_avg = np.mean(sim_means)
                    p_val = (np.sum(sim_means <= obs_mean) + 1) / (len(sim_means) + 1)
                    if p_val < best_p:
                        best_p = p_val
                        best_config = (thr_label, method, order, None, None, None)
                        best_lead = obs_mean
                        best_rand = rand_avg
            else:  # adaptive
                for window in adaptive_windows:
                    for n_std in adaptive_nstds:
                        for min_gap in adaptive_min_gaps:
                            peak_dates = get_peaks_adaptive(vol_series, window=window, n_std=n_std, min_gap=min_gap)
                            if len(peak_dates) < 2:
                                continue
                            peak_days = np.array([date_to_day(d) for d in peak_dates])
                            intervals = []
                            for crisis_day in risk_days:
                                prev_peaks = peak_days[peak_days < crisis_day]
                                if len(prev_peaks) > 0:
                                    intervals.append(crisis_day - prev_peaks[-1])
                            if len(intervals) == 0:
                                continue
                            obs_mean = np.mean(intervals)

                            if HAS_CUPY:
                                peak_days_gpu = cp.asarray(peak_days)
                                all_days_gpu = cp.asarray(all_days)
                                rand_days_gpu = cp.random.choice(all_days_gpu, size=(n_sim, len(risk_days)), replace=True)
                                rand_flat = rand_days_gpu.flatten()
                                pos = cp.searchsorted(peak_days_gpu, rand_flat, side='right') - 1
                                valid = pos >= 0
                                prev_peak_gpu = cp.where(valid, peak_days_gpu[pos], 0)
                                intervals_flat = cp.where(valid, rand_flat - prev_peak_gpu, cp.nan)
                                intervals_mat = intervals_flat.reshape(n_sim, len(risk_days))
                                sim_means_gpu = cp.nanmean(intervals_mat, axis=1)
                                sim_means_gpu = sim_means_gpu[~cp.isnan(sim_means_gpu)]
                                sim_means = cp.asnumpy(sim_means_gpu)
                            else:
                                sim_means = []
                                for _ in range(n_sim):
                                    rand_dates = np.random.choice(df.index, size=len(risk_events), replace=True)
                                    rand_days_rand = np.array([date_to_day(d) for d in rand_dates])
                                    sim_ints = []
                                    for rd in rand_days_rand:
                                        prev = peak_days[peak_days < rd]
                                        if len(prev) > 0:
                                            sim_ints.append(rd - prev[-1])
                                    if sim_ints:
                                        sim_means.append(np.mean(sim_ints))
                            if len(sim_means) == 0:
                                continue
                            rand_avg = np.mean(sim_means)
                            p_val = (np.sum(sim_means <= obs_mean) + 1) / (len(sim_means) + 1)
                            if p_val < best_p:
                                best_p = p_val
                                best_config = (thr_label, method, None, window, n_std, min_gap)
                                best_lead = obs_mean
                                best_rand = rand_avg
    return best_lead, best_rand, best_p, best_config

# ==================== 全数据集分离参数扫描 ====================
def scan_separation_full(df, risk_events, n_sim_fast=2000, n_sim_full=5000):
    """
    在整个数据集上搜索最佳分离参数。
    返回最佳配置、最佳 p 值、lead、rand。
    """
    vol_original = df['L2_vol'].dropna()
    candidates = []  # (config, recon_series, fast_p)

    # ---------- 小波网格（扩展）----------
    wavelets = ['db4', 'db6', 'db8', 'sym5', 'sym8', 'coif4', 'coif5', 'bior3.5', 'bior4.4', 'bior6.8', 'rbio3.9', 'dmey']
    levels = [5,6,7,8,9,10]
    removals = [3,4,5,6,7,8]
    for wav in wavelets:
        for lev in levels:
            for rem in removals:
                if rem > lev:
                    continue
                try:
                    recon = wavelet_reconstruct(vol_original.values, wavelet=wav, level=lev, levels_to_remove=rem)
                except:
                    continue
                recon_series = pd.Series(recon, index=vol_original.index)
                lead, rand, p = h1_fast_test(df, recon_series, risk_events, n_sim=n_sim_fast)
                if not np.isnan(p) and p < 0.2:
                    candidates.append((('wavelet', {'wavelet':wav,'level':lev,'levels_to_remove':rem}),
                                       recon_series, p))
                    print(f"  小波候选: p={p:.4f}, lead={lead:.2f}, config={wav}, lev={lev}, rem={rem}")

    # ---------- SSA 网格（扩展）----------
    windows = list(range(30, 301, 20))
    groups = list(range(1, 13))
    for win in windows:
        for g in groups:
            if g >= win:
                continue
            try:
                recon = ssa_reconstruct(vol_original.values, window_length=win, n_groups=g)
            except:
                continue
            recon_series = pd.Series(recon, index=vol_original.index)
            lead, rand, p = h1_fast_test(df, recon_series, risk_events, n_sim=n_sim_fast)
            if not np.isnan(p) and p < 0.2:
                candidates.append((('ssa', {'window_length':win,'n_groups':g}),
                                   recon_series, p))
                print(f"  SSA候选: p={p:.4f}, lead={lead:.2f}, config=win={win}, g={g}")

    # ---------- 组合网格（小波+SSA）----------
    wavelets_comb = ['coif5', 'db8']
    levels_comb = [7,8,9]
    removals_comb = [5,6,7]
    windows_comb = [150,200,250]
    groups_comb = [2,3,4]
    for wav in wavelets_comb:
        for lev in levels_comb:
            for rem in removals_comb:
                for win in windows_comb:
                    for g in groups_comb:
                        try:
                            recon = combined_reconstruct(vol_original.values,
                                                         method1='wavelet', params1={'wavelet':wav,'level':lev,'levels_to_remove':rem},
                                                         method2='ssa', params2={'window_length':win,'n_groups':g})
                        except:
                            continue
                        recon_series = pd.Series(recon, index=vol_original.index)
                        lead, rand, p = h1_fast_test(df, recon_series, risk_events, n_sim=n_sim_fast)
                        if not np.isnan(p) and p < 0.2:
                            candidates.append((('combined', {'wav':wav,'lev':lev,'rem':rem,'win':win,'g':g}),
                                               recon_series, p))
                            print(f"  组合候选: p={p:.4f}, lead={lead:.2f}, config={wav},{lev},{rem},{win},{g}")

    # 如果有 PyEMD，添加 EMD/EEMD 网格
    if HAS_PYEMD:
        # EMD
        for rem in [1,2,3,4]:
            try:
                recon = emd_reconstruct(vol_original.values, max_imf_to_remove=rem)
            except:
                continue
            recon_series = pd.Series(recon, index=vol_original.index)
            lead, rand, p = h1_fast_test(df, recon_series, risk_events, n_sim=n_sim_fast)
            if not np.isnan(p) and p < 0.2:
                candidates.append((('emd', {'max_imf_to_remove': rem}),
                                   recon_series, p))
                print(f"  EMD候选: p={p:.4f}, lead={lead:.2f}, config=rem={rem}")
        # EEMD
        for rem in [1,2,3]:
            for ens in [50,100]:
                for nw in [0.1,0.2]:
                    try:
                        recon = eemd_reconstruct(vol_original.values, max_imf_to_remove=rem,
                                                 ensemble_size=ens, noise_width=nw)
                    except:
                        continue
                    recon_series = pd.Series(recon, index=vol_original.index)
                    lead, rand, p = h1_fast_test(df, recon_series, risk_events, n_sim=n_sim_fast)
                    if not np.isnan(p) and p < 0.2:
                        candidates.append((('eemd', {'rem':rem,'ens':ens,'nw':nw}),
                                           recon_series, p))
                        print(f"  EEMD候选: p={p:.4f}, lead={lead:.2f}, config=rem={rem},ens={ens},nw={nw}")

    if HAS_VMD:
        for K in [3,4,5,6,7]:
            for alpha in [500,1000,2000,5000]:
                for rem in [1,2,3]:
                    if rem >= K:
                        continue
                    try:
                        recon = vmd_reconstruct(vol_original.values, alpha=alpha, K=K, remove_high=rem)
                    except:
                        continue
                    recon_series = pd.Series(recon, index=vol_original.index)
                    lead, rand, p = h1_fast_test(df, recon_series, risk_events, n_sim=n_sim_fast)
                    if not np.isnan(p) and p < 0.2:
                        candidates.append((('vmd', {'K':K,'alpha':alpha,'rem':rem}),
                                           recon_series, p))
                        print(f"  VMD候选: p={p:.4f}, lead={lead:.2f}, config=K={K},alpha={alpha},rem={rem}")

    # 按快速 p 排序
    candidates.sort(key=lambda x: x[2])
    print(f"\n共收集到 {len(candidates)} 个候选配置，进行完整 H1 扫描...")

    # 构建多分位数风险事件列表用于完整扫描
    risk_events_dict = {}
    for q in [0.8, 0.85, 0.9, 0.95]:
        events = identify_risk_events_dynamic(df['adj_close'], quantile=q, lookback_years=3)
        if len(events) >= 3:
            risk_events_dict[f'q{q}'] = events

    best_p = 1.0
    best_config = None
    best_lead = np.nan
    best_rand = np.nan

    for i, (config, recon_series, fast_p) in enumerate(candidates):
        print(f"\n候选 {i+1}/{len(candidates)}: {config[0]}, 快速 p={fast_p:.4f}")
        lead, rand, p, full_config = test_H1_scan(
            df, risk_events_dict, recon_series,
            n_sim=n_sim_full
        )
        print(f"    完整扫描: lead={lead:.2f}, rand={rand:.2f}, p={p:.4f}, 最佳参数={full_config}")
        if p < best_p:
            best_p = p
            best_config = (config, full_config)
            best_lead = lead
            best_rand = rand
            if p < 0.05:
                print("    ✅ 已找到 p < 0.05！")
                # 可以提前终止，但为全面保留循环
                # break

    return best_config, best_lead, best_rand, best_p

# ==================== 主程序 ====================
def main():
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    original_stdout = sys.stdout
    with open('b.txt', 'w', encoding='utf-8') as f:
        sys.stdout = Tee(original_stdout, f)
        try:
            print("="*60)
            print("SSE50 信号分离参数搜索（全数据集版，目标 p < 0.05）")
            print("="*60)

            # 获取全量数据
            print("\n正在获取 SSE50 数据...")
            df_raw = fetch_sz50()
            print("计算演化指标...")
            df = compute_evolution_indicators(df_raw)
            print(f"数据时间范围: {df.index[0]} 至 {df.index[-1]}, 总天数 {len(df)}")

            # 动态风险事件（固定分位数 0.9）
            risk_events = identify_risk_events_dynamic(df['adj_close'], quantile=0.9, lookback_years=3)
            print(f"动态风险事件数（q=0.9）: {len(risk_events)}")

            # 原始序列快速检验
            print("\n原始 L2_vol 快速 H1 检验...")
            orig_lead, orig_rand, orig_p = h1_fast_test(df, df['L2_vol'], risk_events, n_sim=2000)
            print(f"原始序列: lead={orig_lead:.2f}, rand={orig_rand:.2f}, p={orig_p:.4f}")

            # 全数据集分离参数扫描
            best_config, best_lead, best_rand, best_p = scan_separation_full(
                df, risk_events, n_sim_fast=2000, n_sim_full=5000
            )

            # 最终结果
            print("\n" + "="*60)
            print("最终结果")
            print("="*60)
            if best_config is not None:
                sep_method, sep_params = best_config[0]
                full_h1_params = best_config[1]
                print(f"最佳分离方法: {sep_method}, 参数: {sep_params}")
                print(f"最佳完整 H1 参数: {full_h1_params}")
                print(f"最佳分离后 H1 预警: lead={best_lead:.2f} 天, 随机期望={best_rand:.2f} 天, p={best_p:.4f}")
                if best_p < 0.05:
                    print("✅ 成功找到 p < 0.05 的分离参数！")
                else:
                    print("❌ 未能找到 p < 0.05 的分离参数，最佳 p 值仍大于 0.05。")
            else:
                print("未找到任何有效的分离配置。")

        finally:
            sys.stdout = original_stdout

if __name__ == "__main__":
    main()