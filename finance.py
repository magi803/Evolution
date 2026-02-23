"""
演化论金融市场实证完整代码（最终版）
包含 H1、H2、H3 检验，以及高级信号分离参数搜索（针对 S&P500）。
数据源：S&P500、SSE50、HSI、DAX 从 stooq 或 baostock 获取，无模拟数据。
GPU加速：H1的蒙特卡洛模拟和H3的置换检验使用 CuPy 加速。
所有中间输出及最终结果同时打印到控制台和写入 RR.txt 文件。
"""

import sys
import numpy as np
import pandas as pd
import random
import time
import warnings
from scipy.stats import gaussian_kde, t
from scipy.signal import argrelextrema, hilbert, butter, filtfilt
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pywt

# 可选库
try:
    from PyEMD import EMD, EEMD
    HAS_PYEMD = True
except ImportError:
    HAS_PYEMD = False
    print("PyEMD not installed, EMD/EEMD separation will be skipped.")

try:
    from vmdpy import VMD
    HAS_VMD = True
except ImportError:
    HAS_VMD = False
    print("vmdpy not installed, VMD separation will be skipped.")

try:
    import cupy as cp
    HAS_CUPY = True
    print("CuPy imported successfully, GPU acceleration enabled.")
except ImportError:
    HAS_CUPY = False
    print("CuPy not found, using CPU (slower). Install CuPy for GPU acceleration.")

warnings.filterwarnings('ignore')

# 固定随机种子
np.random.seed(42)
random.seed(42)

# ==================== 数据获取函数 ====================
def fetch_stooq_data(symbol, start_date, end_date, max_retries=3):
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    for attempt in range(max_retries):
        try:
            df_raw = pd.read_csv(url)
            date_col_candidates = ['Date', 'Data', 'Datum', '日期', 'date', 'data']
            date_col = None
            for col in date_col_candidates:
                if col in df_raw.columns:
                    date_col = col
                    break
            if date_col is None:
                raise ValueError("No date column found")
            df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors='coerce')
            df_raw = df_raw.dropna(subset=[date_col])
            df_raw = df_raw.set_index(date_col).sort_index()
            rename_map = {}
            for old in df_raw.columns:
                lower = old.lower()
                if lower in ['open', 'high', 'low', 'close', 'volume']:
                    rename_map[old] = lower
            df_raw = df_raw.rename(columns=rename_map)
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required if col not in df_raw.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            df_raw = df_raw.loc[start_date:end_date].copy()
            if df_raw.empty:
                raise ValueError("No data in selected date range")
            df_raw['adj_close'] = df_raw['close']
            df_raw['amount'] = df_raw['volume'] * df_raw['adj_close']
            df_raw['log_ret'] = np.log(df_raw['adj_close'] / df_raw['adj_close'].shift(1))
            df_raw = df_raw.dropna()
            print(f"Data loaded for {symbol}, {len(df_raw)} records from {df_raw.index[0]} to {df_raw.index[-1]}")
            return df_raw
        except Exception as e:
            print(f"Attempt {attempt+1} for {symbol} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = random.uniform(2, 5)
                time.sleep(wait_time)
    raise Exception(f"Failed to fetch {symbol} from stooq after {max_retries} attempts.")

def fetch_sp500_stooq():
    return fetch_stooq_data('^spx', '1990-01-01', '2024-12-31')

def fetch_hsi_stooq():
    return fetch_stooq_data('^hsi', '2000-01-01', '2024-12-31')

def fetch_dax_stooq():
    return fetch_stooq_data('^dax', '1990-01-01', '2024-12-31')

def fetch_cn_index_data(index_code, index_name, start_date='2005-01-01', end_date='2024-12-31'):
    try:
        import baostock as bs
    except ImportError:
        raise ImportError("baostock not installed. Please install it via: pip install baostock")
    lg = bs.login()
    if lg.error_code != '0':
        raise Exception("baostock login failed")
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
        raise Exception(f"No data for {index_name}")
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
    print(f"{index_name} data loaded, {len(df)} records from {df.index[0]} to {df.index[-1]}")
    return df

def fetch_sz50():
    return fetch_cn_index_data('sh.000016', 'SSE50', '2005-01-01', '2024-12-31')

def load_market(market_name):
    if market_name == 'sp500':
        return fetch_sp500_stooq()
    elif market_name == 'sz50':
        return fetch_sz50()
    elif market_name == 'hsi':
        return fetch_hsi_stooq()
    elif market_name == 'dax':
        return fetch_dax_stooq()
    else:
        raise ValueError(f"Unknown market: {market_name}")

# ==================== 情绪指数与序量 ====================
def compute_emotion_index(df, lookback=20):
    df = df.copy()
    epsilon = 1e-10
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

    indicator_cols = ['mom', 'vol_chg', 'vol_ratio', 'volatility', 'rsi', 'bias', 'close_pos', 'vwap_chg']
    df_indicators = df[indicator_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df_indicators) < 30:
        raise ValueError("Insufficient data to compute emotion index")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_indicators)
    pca = PCA(n_components=1)
    sentiment = pca.fit_transform(scaled).flatten()
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_[0]:.4f}")
    sentiment_series = pd.Series(index=df_indicators.index, data=sentiment)
    sentiment_series = sentiment_series.rolling(5).mean()
    return sentiment_series

def compute_entropy(series, window=30):
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

# ==================== 滤波器 ====================
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

# ==================== 核心指标计算 ====================
def compute_evolution_indicators(df):
    df_out = df.copy()
    sentiment = compute_emotion_index(df_out)
    df_out['sentiment'] = sentiment
    df_out = df_out.dropna(subset=['sentiment'])
    H = compute_entropy(df_out['sentiment'], window=30)
    df_out['M'] = -H
    df_out = df_out.dropna(subset=['M'])
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
    # 默认峰值（用于H2）
    peaks_idx = argrelextrema(df_out['L2_vol'].values, np.greater, order=10)[0]
    df_out['peak'] = False
    if len(peaks_idx) > 0:
        df_out.iloc[peaks_idx, df_out.columns.get_loc('peak')] = True
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

# ==================== 风险事件识别 ====================
def identify_risk_events(price, threshold=0.15, min_gap_days=60):
    peak = price.expanding().max()
    drawdown = (peak - price) / peak
    events = []
    last_event = None
    for date, dd in drawdown.items():
        if dd >= threshold:
            if last_event is None or (date - last_event).days >= min_gap_days:
                events.append(date)
                last_event = date
    return events

# ==================== 临界点识别方法 ====================
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

def emd_reconstruct(series, max_imf_to_remove=2):
    if not HAS_PYEMD:
        raise ImportError("PyEMD not available")
    series = np.asarray(series).copy()
    emd = EMD()
    imfs = emd(series)
    if max_imf_to_remove >= imfs.shape[0]:
        return np.zeros_like(series)
    recon = np.sum(imfs[max_imf_to_remove:], axis=0) + emd.residue
    return recon

# 修复 EEMD 接口，根据 PyEMD 版本调整参数
def eemd_reconstruct(series, max_imf_to_remove=2, ensemble_size=50, noise_width=0.2):
    if not HAS_PYEMD:
        raise ImportError("PyEMD not available")
    series = np.asarray(series).copy()
    eemd = EEMD()
    # 注意：不同版本参数名可能不同，尝试常见组合
    try:
        # 新版可能用 ensemble_size
        imfs = eemd.eemd(series, ensemble_size=ensemble_size, noise_width=noise_width)
    except TypeError:
        try:
            # 旧版可能用 n_ensemble
            imfs = eemd.eemd(series, n_ensemble=ensemble_size, noise_width=noise_width)
        except TypeError:
            # 最旧版可能用 trials
            imfs = eemd.eemd(series, trials=ensemble_size, noise_width=noise_width)
    if max_imf_to_remove >= imfs.shape[0]:
        return np.zeros_like(series)
    recon = np.sum(imfs[max_imf_to_remove:], axis=0)
    # 残差
    residue = series - np.sum(imfs, axis=0)
    recon += residue
    return recon

def vmd_reconstruct(series, alpha=2000, tau=0, K=5, DC=0, init=1, tol=1e-7, remove_high=2):
    if not HAS_VMD:
        raise ImportError("vmdpy not available")
    series = np.asarray(series).copy()
    u, u_hat, omega = VMD(series, alpha, tau, K, DC, init, tol)
    # omega 升序排列，后几个是高频
    if remove_high >= u.shape[0]:
        return np.zeros_like(series)
    recon = np.sum(u[:-remove_high], axis=0)
    return recon

# ==================== H1 参数扫描函数 ====================
def test_H1_scan(df, market_name, 
                 risk_thresholds=np.arange(0.05, 0.26, 0.01).tolist(),
                 peak_methods=['localmax', 'adaptive'],
                 localmax_orders=[5, 10, 15, 20, 25, 30, 35, 40],
                 adaptive_windows=[20, 30, 40, 60, 80, 100],
                 adaptive_nstds=np.arange(0.5, 4.1, 0.3).tolist(),
                 adaptive_min_gaps=[10, 20, 30, 40, 60],
                 min_risk_events=5,
                 n_sim=5000,
                 separated_series=None):
    best_p = 1.0
    best_config = None
    best_lead = np.nan
    best_rand = np.nan

    risk_events_dict = {}
    for thr in risk_thresholds:
        events = identify_risk_events(df['adj_close'], threshold=thr)
        risk_events_dict[thr] = events
        print(f"  Threshold {thr:.2f}: {len(events)} risk events")

    if separated_series is not None:
        vol_series = separated_series.dropna()
    else:
        vol_series = df['L2_vol'].dropna()

    ref_date = df.index[0]
    date_to_day = lambda d: (d - ref_date).days
    all_days = np.array([date_to_day(d) for d in df.index])

    for thr in risk_thresholds:
        risk_events = risk_events_dict[thr]
        if len(risk_events) < min_risk_events:
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
                        best_config = (thr, method, order, None, None, None)
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
                                best_config = (thr, method, None, window, n_std, min_gap)
                                best_lead = obs_mean
                                best_rand = rand_avg

    if best_config is not None:
        print(f"{market_name} best H1: lead={best_lead:.2f}, rand={best_rand:.2f}, p={best_p:.4f}, config={best_config}")
    else:
        print(f"{market_name} H1: no valid configuration found")
    return best_lead, best_rand, best_p, best_config

# ==================== H2 检验 ====================
def test_H2(df, market_name):
    print(f"\n--- H2 Equal Interval Test ({market_name}) ---")
    if 'peak' not in df.columns:
        print(f"Warning: 'peak' column not found in {market_name} data. Skipping H2.")
        return np.nan, np.nan, np.nan, np.nan
    peak_dates = df[df['peak']].index
    if len(peak_dates) < 2:
        print("Too few peaks")
        return np.nan, np.nan, np.nan, np.nan
    intervals = (peak_dates[1:] - peak_dates[:-1]).days
    mean_int = np.mean(intervals)
    std_int = np.std(intervals)
    cv = std_int / mean_int if mean_int != 0 else np.nan
    omega = df['omega'].iloc[0]
    if np.isnan(omega) or omega <= 0:
        theo_int = None
    else:
        omega_daily = omega / 252
        theo_int = np.pi / (2 * omega_daily)
        print(f"omega annualized={omega:.4f}, theoretical interval={theo_int:.2f} days")
        if theo_int > 0:
            t_stat = abs(mean_int - theo_int) / (std_int / np.sqrt(len(intervals)))
            p_val = 2 * (1 - t.cdf(t_stat, df=len(intervals)-1))
            print(f"t-test p={p_val:.4f}")
        else:
            p_val = np.nan
    print(f"Actual mean interval={mean_int:.2f} days, CV={cv:.3f}")
    return mean_int, theo_int, p_val, cv

# ==================== H3 检验（GPU加速） ====================
def test_H3(df1, df2, name1, name2):
    print(f"\n--- H3 Phase Synchronization Test (GPU accelerated) ({name1} vs {name2}) ---")
    common_idx = df1.index.intersection(df2.index)
    if len(common_idx) == 0:
        return np.nan, np.nan, np.nan
    df1 = df1.loc[common_idx]
    df2 = df2.loc[common_idx]

    ref_date = common_idx[0]
    date_to_day = lambda d: (d - ref_date).days
    N = len(common_idx)

    delta_theta = df1['theta_unwrap'].values - df2['theta_unwrap'].values
    delta_theta_norm = (delta_theta + np.pi) % (2 * np.pi) - np.pi

    window = 60
    delta_mean = pd.Series(delta_theta_norm).rolling(window, center=True).mean().values
    delta_std = pd.Series(delta_theta_norm).rolling(window, center=True).std().values

    std_threshold = np.nanmedian(delta_std)
    sync_cond = ( (np.abs(delta_mean) < 0.5) | (np.abs(np.abs(delta_mean) - np.pi) < 0.5) ) & (delta_std < std_threshold)
    sync_periods = sync_cond & ~np.isnan(delta_std)
    sync_dates_mask = sync_periods

    peaks1 = df1[df1['peak']].index
    peaks2 = df2[df2['peak']].index
    peaks1_mask = np.zeros(N, dtype=bool)
    peaks2_mask = np.zeros(N, dtype=bool)
    for i, d in enumerate(common_idx):
        if d in peaks1:
            peaks1_mask[i] = True
        if d in peaks2:
            peaks2_mask[i] = True

    near1_mask = np.zeros(N, dtype=bool)
    near2_mask = np.zeros(N, dtype=bool)
    for i in range(N):
        left = max(0, i-3)
        right = min(N-1, i+3)
        if np.any(peaks1_mask[left:right+1]):
            near1_mask[i] = True
        if np.any(peaks2_mask[left:right+1]):
            near2_mask[i] = True
    both_mask = near1_mask & near2_mask

    def sync_rate(mask, dates_mask):
        if not np.any(dates_mask):
            return 0.0
        both_in_dates = both_mask & dates_mask
        return np.sum(both_in_dates) / np.sum(dates_mask)

    sync_sync = sync_rate(both_mask, sync_dates_mask)
    sync_unsync = sync_rate(both_mask, ~sync_dates_mask)
    print(f"Sync rate in sync periods={sync_sync:.3f}, in non-sync={sync_unsync:.3f}")

    if HAS_CUPY:
        n_perm = 1000
        sync_mask_gpu = cp.asarray(sync_dates_mask)
        both_gpu = cp.asarray(both_mask)
        rand = cp.random.rand(n_perm, N)
        perm_indices = cp.argsort(rand, axis=1)
        perm_masks = sync_mask_gpu[perm_indices]

        sync_both = cp.sum(perm_masks & both_gpu, axis=1)
        sync_total = cp.sum(perm_masks, axis=1)
        unsync_both = cp.sum(~perm_masks & both_gpu, axis=1)
        unsync_total = cp.sum(~perm_masks, axis=1)

        sync_rates = cp.where(sync_total > 0, sync_both / sync_total, 0.0)
        unsync_rates = cp.where(unsync_total > 0, unsync_both / unsync_total, 0.0)
        diff_perm_gpu = sync_rates - unsync_rates
        diff_perm = cp.asnumpy(diff_perm_gpu)

        diff_obs = sync_sync - sync_unsync
        p_val = (np.sum(diff_perm >= diff_obs) + 1) / (n_perm + 1)
    else:
        n_perm = 1000
        diff_obs = sync_sync - sync_unsync
        diff_perm = []
        for _ in range(n_perm):
            perm_labels = np.random.permutation(sync_dates_mask)
            rate_s = sync_rate(both_mask, perm_labels)
            rate_u = sync_rate(both_mask, ~perm_labels)
            diff_perm.append(rate_s - rate_u)
        p_val = (sum(np.array(diff_perm) >= diff_obs) + 1) / (n_perm + 1)

    print(f"Permutation p={p_val:.4f}")
    return sync_sync, sync_unsync, p_val

# ==================== 信号分离高级参数扫描 ====================
def scan_separation_advanced_for_sp500(df, n_sim_fast=2000):
    print("\n" + "="*60)
    print("ADVANCED SIGNAL SEPARATION PARAMETER SCAN FOR S&P500")
    print("="*60)

    vol_series = df['L2_vol'].dropna()
    risk_events = identify_risk_events(df['adj_close'], threshold=0.15)
    if len(risk_events) < 5:
        print("Too few risk events, skipping separation scan.")
        return None, None, np.nan, np.nan, np.nan

    ref_date = df.index[0]
    date_to_day = lambda d: (d - ref_date).days
    all_days = np.array([date_to_day(d) for d in df.index])
    risk_days = np.array([date_to_day(d) for d in risk_events])

    best_p = 1.0
    best_config = None
    best_lead = np.nan
    best_rand = np.nan
    best_recon = None

    # ---------- 小波网格 ----------
    wavelets = ['db4', 'db6', 'sym5', 'coif5', 'bior3.5']
    levels = [4, 5, 6, 7, 8]
    removals = [1, 2, 3, 4, 5]
    for wav in wavelets:
        for lev in levels:
            for rem in removals:
                if rem > lev:
                    continue
                print(f"  Testing wavelet: {wav}, level={lev}, remove={rem}")
                recon = wavelet_reconstruct(vol_series.values, wavelet=wav, level=lev, levels_to_remove=rem)
                recon_series = pd.Series(recon, index=vol_series.index)
                peak_dates = get_peaks_localmax(recon_series, order=10)
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
                # 快速蒙特卡洛
                if HAS_CUPY:
                    peak_days_gpu = cp.asarray(peak_days)
                    all_days_gpu = cp.asarray(all_days)
                    rand_days_gpu = cp.random.choice(all_days_gpu, size=(n_sim_fast, len(risk_days)), replace=True)
                    rand_flat = rand_days_gpu.flatten()
                    pos = cp.searchsorted(peak_days_gpu, rand_flat, side='right') - 1
                    valid = pos >= 0
                    prev_peak_gpu = cp.where(valid, peak_days_gpu[pos], 0)
                    intervals_flat = cp.where(valid, rand_flat - prev_peak_gpu, cp.nan)
                    intervals_mat = intervals_flat.reshape(n_sim_fast, len(risk_days))
                    sim_means_gpu = cp.nanmean(intervals_mat, axis=1)
                    sim_means_gpu = sim_means_gpu[~cp.isnan(sim_means_gpu)]
                    sim_means = cp.asnumpy(sim_means_gpu)
                else:
                    sim_means = []
                    for _ in range(n_sim_fast):
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
                print(f"    p={p_val:.4f}, lead={obs_mean:.2f}, rand={rand_avg:.2f}")
                if p_val < best_p:
                    best_p = p_val
                    best_config = ('wavelet', {'wavelet': wav, 'level': lev, 'levels_to_remove': rem})
                    best_lead = obs_mean
                    best_rand = rand_avg
                    best_recon = recon_series

    # ---------- SSA 网格 ----------
    windows = [30, 60, 90, 120, 150, 200]
    groups = [1, 2, 3, 4, 5, 6]
    for win in windows:
        for g in groups:
            if g >= win:
                continue
            print(f"  Testing SSA: window={win}, n_groups={g}")
            recon = ssa_reconstruct(vol_series.values, window_length=win, n_groups=g)
            recon_series = pd.Series(recon, index=vol_series.index)
            peak_dates = get_peaks_localmax(recon_series, order=10)
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
                rand_days_gpu = cp.random.choice(all_days_gpu, size=(n_sim_fast, len(risk_days)), replace=True)
                rand_flat = rand_days_gpu.flatten()
                pos = cp.searchsorted(peak_days_gpu, rand_flat, side='right') - 1
                valid = pos >= 0
                prev_peak_gpu = cp.where(valid, peak_days_gpu[pos], 0)
                intervals_flat = cp.where(valid, rand_flat - prev_peak_gpu, cp.nan)
                intervals_mat = intervals_flat.reshape(n_sim_fast, len(risk_days))
                sim_means_gpu = cp.nanmean(intervals_mat, axis=1)
                sim_means_gpu = sim_means_gpu[~cp.isnan(sim_means_gpu)]
                sim_means = cp.asnumpy(sim_means_gpu)
            else:
                sim_means = []
                for _ in range(n_sim_fast):
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
            print(f"    p={p_val:.4f}, lead={obs_mean:.2f}, rand={rand_avg:.2f}")
            if p_val < best_p:
                best_p = p_val
                best_config = ('ssa', {'window_length': win, 'n_groups': g})
                best_lead = obs_mean
                best_rand = rand_avg
                best_recon = recon_series

    # ---------- EMD 网格 ----------
    if HAS_PYEMD:
        removals_emd = [1, 2, 3, 4, 5]
        for rem in removals_emd:
            print(f"  Testing EMD: remove {rem} IMFs")
            try:
                recon = emd_reconstruct(vol_series.values, max_imf_to_remove=rem)
                recon_series = pd.Series(recon, index=vol_series.index)
                peak_dates = get_peaks_localmax(recon_series, order=10)
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
                    rand_days_gpu = cp.random.choice(all_days_gpu, size=(n_sim_fast, len(risk_days)), replace=True)
                    rand_flat = rand_days_gpu.flatten()
                    pos = cp.searchsorted(peak_days_gpu, rand_flat, side='right') - 1
                    valid = pos >= 0
                    prev_peak_gpu = cp.where(valid, peak_days_gpu[pos], 0)
                    intervals_flat = cp.where(valid, rand_flat - prev_peak_gpu, cp.nan)
                    intervals_mat = intervals_flat.reshape(n_sim_fast, len(risk_days))
                    sim_means_gpu = cp.nanmean(intervals_mat, axis=1)
                    sim_means_gpu = sim_means_gpu[~cp.isnan(sim_means_gpu)]
                    sim_means = cp.asnumpy(sim_means_gpu)
                else:
                    sim_means = []
                    for _ in range(n_sim_fast):
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
                print(f"    p={p_val:.4f}, lead={obs_mean:.2f}, rand={rand_avg:.2f}")
                if p_val < best_p:
                    best_p = p_val
                    best_config = ('emd', {'max_imf_to_remove': rem})
                    best_lead = obs_mean
                    best_rand = rand_avg
                    best_recon = recon_series
            except Exception as e:
                print(f"    EMD failed: {e}")

    # ---------- EEMD 网格（修复参数）----------
    if HAS_PYEMD:
        removals_eemd = [1, 2, 3, 4]
        ensembles = [50, 100]
        noise_widths = [0.1, 0.2, 0.3]
        for rem in removals_eemd:
            for ens in ensembles:
                for nw in noise_widths:
                    print(f"  Testing EEMD: remove {rem} IMFs, ensemble={ens}, noise_width={nw}")
                    try:
                        recon = eemd_reconstruct(vol_series.values, max_imf_to_remove=rem, ensemble_size=ens, noise_width=nw)
                        recon_series = pd.Series(recon, index=vol_series.index)
                        peak_dates = get_peaks_localmax(recon_series, order=10)
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
                            rand_days_gpu = cp.random.choice(all_days_gpu, size=(n_sim_fast, len(risk_days)), replace=True)
                            rand_flat = rand_days_gpu.flatten()
                            pos = cp.searchsorted(peak_days_gpu, rand_flat, side='right') - 1
                            valid = pos >= 0
                            prev_peak_gpu = cp.where(valid, peak_days_gpu[pos], 0)
                            intervals_flat = cp.where(valid, rand_flat - prev_peak_gpu, cp.nan)
                            intervals_mat = intervals_flat.reshape(n_sim_fast, len(risk_days))
                            sim_means_gpu = cp.nanmean(intervals_mat, axis=1)
                            sim_means_gpu = sim_means_gpu[~cp.isnan(sim_means_gpu)]
                            sim_means = cp.asnumpy(sim_means_gpu)
                        else:
                            sim_means = []
                            for _ in range(n_sim_fast):
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
                        print(f"    p={p_val:.4f}, lead={obs_mean:.2f}, rand={rand_avg:.2f}")
                        if p_val < best_p:
                            best_p = p_val
                            best_config = ('eemd', {'max_imf_to_remove': rem, 'ensemble_size': ens, 'noise_width': nw})
                            best_lead = obs_mean
                            best_rand = rand_avg
                            best_recon = recon_series
                    except Exception as e:
                        print(f"    EEMD failed: {e}")

    # ---------- VMD 网格（如果有）----------
    if HAS_VMD:
        Ks = [3, 5, 7, 9]
        alphas = [1000, 2000, 5000]
        remove_high_opts = [1, 2, 3]
        for K in Ks:
            for alpha in alphas:
                for rem in remove_high_opts:
                    if rem >= K:
                        continue
                    print(f"  Testing VMD: K={K}, alpha={alpha}, remove_high={rem}")
                    try:
                        recon = vmd_reconstruct(vol_series.values, alpha=alpha, K=K, remove_high=rem)
                        recon_series = pd.Series(recon, index=vol_series.index)
                        peak_dates = get_peaks_localmax(recon_series, order=10)
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
                            rand_days_gpu = cp.random.choice(all_days_gpu, size=(n_sim_fast, len(risk_days)), replace=True)
                            rand_flat = rand_days_gpu.flatten()
                            pos = cp.searchsorted(peak_days_gpu, rand_flat, side='right') - 1
                            valid = pos >= 0
                            prev_peak_gpu = cp.where(valid, peak_days_gpu[pos], 0)
                            intervals_flat = cp.where(valid, rand_flat - prev_peak_gpu, cp.nan)
                            intervals_mat = intervals_flat.reshape(n_sim_fast, len(risk_days))
                            sim_means_gpu = cp.nanmean(intervals_mat, axis=1)
                            sim_means_gpu = sim_means_gpu[~cp.isnan(sim_means_gpu)]
                            sim_means = cp.asnumpy(sim_means_gpu)
                        else:
                            sim_means = []
                            for _ in range(n_sim_fast):
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
                        print(f"    p={p_val:.4f}, lead={obs_mean:.2f}, rand={rand_avg:.2f}")
                        if p_val < best_p:
                            best_p = p_val
                            best_config = ('vmd', {'K': K, 'alpha': alpha, 'remove_high': rem})
                            best_lead = obs_mean
                            best_rand = rand_avg
                            best_recon = recon_series
                    except Exception as e:
                        print(f"    VMD failed: {e}")

    if best_config is not None:
        print(f"\nBest separation config: {best_config}, p={best_p:.4f}, lead={best_lead:.2f}, rand={best_rand:.2f}")
    else:
        print("No valid separation config found.")
    return best_config, best_recon, best_lead, best_rand, best_p

# ==================== 主程序 ====================
def main():
    print("="*60)
    print("EVOLUTION THEORY FINANCIAL MARKET EMPIRICAL STUDY (FINAL VERSION)")
    print("="*60)

    market_keys = ['sp500', 'sz50', 'hsi', 'dax']
    market_names = {'sp500':'S&P500', 'sz50':'SSE50', 'hsi':'HSI', 'dax':'DAX'}
    data = {}
    for key in market_keys:
        print(f"\nLoading {market_names[key]}...")
        df = load_market(key)
        print(f"Computing evolution indicators...")
        data[key] = compute_evolution_indicators(df)

    # --- H1 标准扫描（所有市场）---
    print("\n" + "="*60)
    print("H1 EARLY WARNING EFFECTIVENESS (EXTENSIVE PARAMETER SCAN)")
    print("="*60)
    h1_best = []
    for key, name in market_names.items():
        lead, rand, p, config = test_H1_scan(data[key], name)
        h1_best.append((name, lead, rand, p, config))

    print("\nH1 BEST RESULTS SUMMARY (original L2_vol):")
    print("{:<20} {:>12} {:>12} {:>10} {:>40}".format("Market","Lead(days)","Random(days)","p-value","Best Config"))
    for name, lead, rand, p, config in h1_best:
        print("{:<20} {:>12.2f} {:>12.2f} {:>10.4f} {:>40}".format(name, lead, rand, p, str(config)))

    # --- 高级信号分离扫描（S&P500）---
    if 'sp500' in data:
        best_sep_config, best_recon, sep_lead, sep_rand, sep_p = scan_separation_advanced_for_sp500(data['sp500'], n_sim_fast=2000)
        if best_sep_config is not None:
            print("\n" + "="*60)
            print("S&P500 H1 WITH BEST SEPARATION (full parameter scan)")
            print("="*60)
            sep_lead_full, sep_rand_full, sep_p_full, sep_config_full = test_H1_scan(
                data['sp500'], 'S&P500 (separated)',
                separated_series=best_recon
            )
            print(f"\nBest separated result: lead={sep_lead_full:.2f}, rand={sep_rand_full:.2f}, p={sep_p_full:.4f}, config={sep_config_full}")
            h1_best.append(('S&P500 (separated best)', sep_lead_full, sep_rand_full, sep_p_full, sep_config_full))

    # --- H2 检验 ---
    print("\n" + "="*60)
    print("H2 EQUAL INTERVAL TEST")
    print("="*60)
    h2_results = []
    for key, name in market_names.items():
        avg, theo, p, cv = test_H2(data[key], name)
        h2_results.append((name, avg, theo, p, cv))
    print("\nH2 RESULTS SUMMARY:")
    print("{:<20} {:>12} {:>12} {:>10} {:>10}".format("Market","Actual(d)","Theoretical(d)","t-test p","CV"))
    for name, avg, theo, p, cv in h2_results:
        theo_str = f"{theo:.2f}" if theo is not None else "nan"
        print("{:<20} {:>12.2f} {:>12} {:>10.4f} {:>10.3f}".format(name, avg, theo_str, p, cv))

    # --- H3 检验 ---
    print("\n" + "="*60)
    print("H3 PHASE SYNCHRONIZATION (SSE50 vs S&P500)")
    print("="*60)
    if 'sz50' in data and 'sp500' in data:
        sync_s, sync_u, p_h3 = test_H3(data['sz50'], data['sp500'], 'SSE50', 'S&P500')
        print(f"\nSync rate in sync periods={sync_s:.3f}, in non-sync={sync_u:.3f}, p={p_h3:.4f}")
    else:
        print("Data missing for H3.")

    print("\nAll done.")

# ==================== 双重输出（控制台+文件）====================
if __name__ == "__main__":
    # 同时输出到控制台和文件 RR.txt
    import sys
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
    with open('RR.txt', 'w', encoding='utf-8') as f:
        sys.stdout = Tee(original_stdout, f)
        try:
            main()
        finally:
            sys.stdout = original_stdout