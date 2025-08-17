import numpy as np
from scipy.fft import fft
from scipy.signal import detrend, find_peaks, correlate
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from math import pi, atan2, degrees

def adjust_phase_diff(diff):
    if diff > pi:
        diff -= 2 * pi
    elif diff < -pi:
        diff += 2 * pi
    return diff

def analyze_fft_phase_diff(left_signal, right_signal, fs=30):
    left = detrend(left_signal)
    right = detrend(right_signal)
    N = len(left)
    fft_left = fft(left)
    fft_right = fft(right)
    freq = np.arange(N) * fs / N

    idx_main = np.argmax(np.abs(fft_left[1:N // 2])) + 1
    phase_left = np.angle(fft_left[idx_main])
    phase_right = np.angle(fft_right[idx_main])
    phase_diff_main = adjust_phase_diff(phase_right - phase_left)

    min_idx = np.argmax(freq > 0.5)
    max_idx = N // 2
    weights = np.abs(fft_left[min_idx:max_idx]) * np.abs(fft_right[min_idx:max_idx])
    total_weight = np.sum(weights)

    phase_diffs = []
    if total_weight > 0:
        for i in range(min_idx, max_idx):
            pl = np.angle(fft_left[i])
            pr = np.angle(fft_right[i])
            pd = adjust_phase_diff(pr - pl)
            phase_diffs.append(pd)
        weighted_phase_diff = np.sum(np.array(phase_diffs) * weights) / total_weight
    else:
        weighted_phase_diff = np.nan

    return {
        "main_freq": freq[idx_main],
        "main_phase_diff_rad": phase_diff_main,
        "main_phase_diff_deg": degrees(phase_diff_main),
        "weighted_phase_diff_rad": weighted_phase_diff,
        "weighted_phase_diff_deg": degrees(weighted_phase_diff) if not np.isnan(weighted_phase_diff) else np.nan
    }

def analyze_fft_zero_padding(left, right, fs=30):
    # === 1. 去趨勢 detrend（對應 MATLAB 的 detrend(x, 'linear')）
    left_detrended = detrend(left, type='linear')
    right_detrended = detrend(right, type='linear')

    # === 2. Zero padding 長度
    N = len(left_detrended)
    window_size = int(2 ** np.ceil(np.log2(N)))

    # === 3. Zero-padding
    left_padded = np.concatenate([left_detrended, np.zeros(window_size - N)])
    right_padded = np.concatenate([right_detrended, np.zeros(window_size - N)])

    # === 4. Hamming window（對應 MATLAB 的 hamming(window_size)）
    window = np.hamming(window_size)
    left_windowed = left_padded * window
    right_windowed = right_padded * window

    # === 5. FFT
    fft1 = np.fft.fft(left_windowed)
    fft2 = np.fft.fft(right_windowed)

    # === 6. 相位差（全部頻點）
    phase1 = np.angle(fft1)
    phase2 = np.angle(fft2)
    phase_diff = np.angle(np.exp(1j * (phase2 - phase1)))  # wrap to [-pi, pi]

    # === 7. 頻率軸（對應 freq2）
    freq = np.arange(window_size) * fs / window_size

    # === 8. 找最大頻率點 index（和 MATLAB 一模一樣）
    max_idx = np.argmax(np.abs(fft1))

    return {
        'zero_freq': freq[max_idx],
        'zero_phase_diff_rad': phase_diff[max_idx],
        'zero_phase_diff_deg': np.rad2deg(phase_diff[max_idx])
    }


def calculate_dynamic_parameters(data, fs=30):
    y = detrend(data)
    L = len(y)
    Y = fft(y)
    P2 = np.abs(Y / L)
    P1 = P2[:(L // 2 + 1)]
    freqs = np.linspace(0, fs / 2, L // 2 + 1)

    # 忽略 DC 分量，從 index 1 開始找最大值
    max_idx = np.argmax(P1[1:]) + 1
    main_freq = freqs[max_idx]

    if main_freq > 0:
        min_peak_distance = round(fs / (main_freq * 2))
        window_size = round(fs / (main_freq * 5))
    else:
        min_peak_distance = round(fs * 0.8)
        window_size = round(fs * 0.1)

    min_peak_distance = max(min_peak_distance, round(fs * 0.5))
    window_size = max(3, min(window_size, round(fs * 0.2)))
    return min_peak_distance, window_size

def gaussian_smooth(x, win_size):
    std = win_size / 6.0  # MATLAB的'gaussian'濾波近似 std ≈ window / 6
    return gaussian_filter1d(x, sigma=std, mode='nearest')

def detect_gait_events(knee, ankle, heel, min_peak_distance, window_size, fs=30):
    # 平滑 + 去趨勢
    knee_s = gaussian_smooth(detrend(knee), window_size)
    ankle_s = gaussian_smooth(detrend(ankle), window_size)
    heel_s = gaussian_smooth(detrend(heel), window_size)

    # 複合信號
    comp = heel_s + 0.5 * ankle_s
    adaptive_prominence = np.std(comp) * 0.2

    # 初步偵測波峰
    peaks, _ = find_peaks(comp, distance=min_peak_distance, prominence=adaptive_prominence)

    valid_events = []
    for i in peaks:
        start = max(0, i - round(fs * 0.15))
        end = min(len(comp), i + round(fs * 0.15))

        heel_val = heel_s[i]
        heel_win = heel_s[start:end]
        ankle_win = ankle_s[start:end]
        knee_win = knee_s[start:end]

        v_disp = heel_val - np.mean(heel_win) if len(heel_win) > 0 else 0
        heel_vel = np.diff(np.append(heel_win, heel_win[-1])) if len(heel_win) > 1 else np.array([0])

        if len(knee_win) > 1 and len(ankle_win) > 1:
            corr = np.corrcoef(knee_win, ankle_win)[0, 1]
        else:
            corr = 0

        is_valid = ((v_disp > adaptive_prominence * 0.2) or
                    (np.max(np.abs(heel_vel)) > np.std(heel_vel) * 1.2)) and (corr > -0.5)

        if is_valid:
            valid_events.append(i)

    # 過近移除
    valid_events = np.array(valid_events)
    if len(valid_events) > 1:
        diff_e = np.diff(valid_events)
        keep = np.insert(diff_e > round(fs * 0.5), 0, True)
        valid_events = valid_events[keep]

    # 計算週期
    if len(valid_events) >= 2:
        cycle_times = np.diff(valid_events) / fs
        valid_cycles = cycle_times[(cycle_times > 0.7) & (cycle_times < 1.8)]
    else:
        valid_cycles = np.array([])

    return valid_events.tolist(), valid_cycles.tolist()

def calculate_ratios(smoothed):
    Ldiff1 = np.abs(smoothed['Lknee'] - smoothed['Lankle'])
    Rdiff1 = np.abs(smoothed['Rknee'] - smoothed['Rankle'])
    Ldiff2 = np.abs(smoothed['Ls'] - smoothed['Lankle'])
    Rdiff2 = np.abs(smoothed['Rs'] - smoothed['Rankle'])

    # 避免除以 0
    Lper = np.divide(Ldiff1, Ldiff2, out=np.zeros_like(Ldiff1), where=Ldiff2!=0)
    Rper = np.divide(Rdiff1, Rdiff2, out=np.zeros_like(Rdiff1), where=Rdiff2!=0)
    return Lper, Rper

def analyze_lift_ratios(Lper, Rper, serious_threshold=0.6, mild_threshold=0.2, outlier_std=3):
    Lper = np.array(Lper)
    Rper = np.array(Rper)

    frame_count = len(Lper)
    left_higher = Lper > Rper
    right_higher = Rper > Lper

    left_high_count = np.sum(left_higher)
    right_high_count = np.sum(right_higher)

    above = left_high_count / frame_count
    below = right_high_count / frame_count
    difference_value = abs(above - below)

    if above >= below:
        higher_side = "left"
    else:
        higher_side = "right"

    if difference_value >= serious_threshold:
        severity = "serious"
    elif difference_value >= mild_threshold:
        severity = "mild"
    else:
        severity = "normal"

    return {
        "frame_count": frame_count,
        "above": above,
        "below": below,
        "difference_value": difference_value,
        "higher_side": higher_side,
        "severity": severity
    }

def calculate_ratios2(smoothed, left_events, right_events, hip_y):
    L_knee_to_heel = np.abs(smoothed['Lknee'] - smoothed['Lheel'])
    R_knee_to_heel = np.abs(smoothed['Rknee'] - smoothed['Rheel'])

    L_distances = np.array([abs(hip_y['Lhip'][i] - smoothed['Lheel'][i]) for i in left_events])
    R_distances = np.array([abs(hip_y['Rhip'][i] - smoothed['Rheel'][i]) for i in right_events])
    ref_height = np.mean(np.concatenate([L_distances, R_distances]))

    L_ratios, R_ratios = [], []
    for i in range(len(left_events) - 1):
        rng = slice(left_events[i], left_events[i+1])
        lift = np.max(L_knee_to_heel[rng]) - np.min(L_knee_to_heel[rng])
        L_ratios.append(lift / ref_height)
    for i in range(len(right_events) - 1):
        rng = slice(right_events[i], right_events[i+1])
        lift = np.max(R_knee_to_heel[rng]) - np.min(R_knee_to_heel[rng])
        R_ratios.append(lift / ref_height)

    return {
        'Lper': np.array(L_ratios),
        'Rper': np.array(R_ratios)
    }

def analyze_pca_features(smoothed):
    n = len(smoothed['Lknee'])

    # 左腳特徵
    left_features = np.stack([
        smoothed['Lknee'] - smoothed['Lankle'],
        smoothed['Lknee'] - smoothed['Lhip'],
        smoothed['Lankle'] - smoothed['Lheel'],
        smoothed['Lhip'] - smoothed['Lheel']
    ], axis=1)

    # 右腳特徵
    right_features = np.stack([
        smoothed['Rknee'] - smoothed['Rankle'],
        smoothed['Rknee'] - smoothed['Rhip'],
        smoothed['Rankle'] - smoothed['Rheel'],
        smoothed['Rhip'] - smoothed['Rheel']
    ], axis=1)

    # 標準化
    scaler = StandardScaler()
    left_std = scaler.fit_transform(left_features)
    right_std = scaler.fit_transform(right_features)

    # PCA
    pca_left = PCA(n_components=4).fit(left_std)
    pca_right = PCA(n_components=4).fit(right_std)

    # PC1 loading correlation
    coeff_corr = np.corrcoef(pca_left.components_[0], pca_right.components_[0])[0, 1]

    return {
        'left': {
            'explained': pca_left.explained_variance_ratio_ * 100,
            'coeff': pca_left.components_,
        },
        'right': {
            'explained': pca_right.explained_variance_ratio_ * 100,
            'coeff': pca_right.components_,
        },
        'pc1_corr': coeff_corr
    }

def analyze_pelvic_angles(hip_x, hip_y, left_events, right_events, fs=30):
    Lhip_x = np.array(hip_x['Lhip'])
    Rhip_x = np.array(hip_x['Rhip'])
    Lhip_y = np.array(hip_y['Lhip'])
    Rhip_y = np.array(hip_y['Rhip'])

    num_frames = len(Lhip_x)
    angles = np.zeros(num_frames)

    for i in range(num_frames):
        dx = Rhip_x[i] - Lhip_x[i]
        dy = Rhip_y[i] - Lhip_y[i]
        angles[i] = degrees(atan2(dy, dx))

    mean_angle = np.mean(angles)
    zero_centered = angles - mean_angle
    stats = {
        'min': np.min(zero_centered),
        'max': np.max(zero_centered),
        'std': np.std(zero_centered),
        'mean': np.mean(zero_centered)
    }

    # 高度差分析
    height_diff = Lhip_y - Rhip_y
    mean_diff = np.mean(height_diff)
    centered_diff = height_diff - mean_diff

    # 週期切片
    results = []
    for i in range(len(left_events)-1):
        start = left_events[i]
        end = left_events[i+1]
        segment = centered_diff[start:end]
        if len(segment) > 0:
            results.append({
                'mean': np.mean(segment),
                'max': np.max(segment),
                'min': np.min(segment),
                'std': np.std(segment),
                'duration': (end - start) / fs
            })

    # 統計嚴重程度與方向
    overall_mean = np.mean([r['mean'] for r in results])
    if abs(overall_mean) <= 0.7:
        severity = '正常'
        direction = '正常'
    else:
        direction = '右高左低' if overall_mean > 0 else '右低左高'
        if abs(overall_mean) > 2.5 * 3:
            severity = '嚴重'
        elif abs(overall_mean) > 2.5 * 1.5:
            severity = '中度'
        else:
            severity = '輕度'

    return {
        'angle_stats': stats,
        'height_diff_mean': mean_diff,
        'cycle_stats': results,
        'overall_direction': direction,
        'severity': severity
    }

def remove_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data

def analyze_pelvis(data, points_x, points_y, left_events, right_events, fs):

    def check_ball_data(df):
        keys = ['left_ball_x', 'right_ball_x', 'left_ball_y', 'right_ball_y']
        return all(k in df.columns and df[k].notnull().all() for k in keys)

    def get_angle(x1, y1, x2, y2):
        dx = np.array(x2) - np.array(x1)
        dy = np.array(y2) - np.array(y1)
        return np.degrees(np.arctan2(dy, dx))

    def hist_weighted_percentages(values, bins):
        hist, _ = np.histogram(values, bins=bins)
        percent = hist / len(values) * 100
        return percent

    if check_ball_data(data):
        Lhip_x, Lhip_y = data['left_ball_x'].values, data['left_ball_y'].values
        Rhip_x, Rhip_y = data['right_ball_x'].values, data['right_ball_y'].values
        # print("[INFO] 使用：球的座標")
        # print(f"Lhip_y[0] = {Lhip_y[0]:.2f} (from left_ball_y)")
    else:
        Lhip_x, Lhip_y = data['x_23'].values, data['y_23'].values
        Rhip_x, Rhip_y = data['x_24'].values, data['y_24'].values
        # print("[INFO] 使用：關節的座標")
        # print(f"Lhip_y[0] = {Lhip_y[0]:.2f} (from y_23)")

    if not left_events or not right_events or left_events[0] >= right_events[-1]:
        print("無法進行骨盆分析：事件數不足或排序異常")
        return {}

    start, end = left_events[0], right_events[-1]
    frames_of_interest = list(range(start, end + 1))

    # print("\n=== Python: 擷取資料範圍 ===")
    # print(f"start frame: {start}, end frame: {end}")
    # print(f"擷取幀數: {len(frames_of_interest)}")

    # Trimmed events
    trimmed_left = [i - start for i in left_events if start <= i <= end]

    # Trimmed hip data
    Lhip_x, Lhip_y = Lhip_x[start:end+1], Lhip_y[start:end+1]
    Rhip_x, Rhip_y = Rhip_x[start:end+1], Rhip_y[start:end+1]

    # 角度分析
    angles = get_angle(Lhip_x, Lhip_y, Rhip_x, Rhip_y)
    angle_mean = np.mean(angles)
    centered_angles = angles - angle_mean
    angle_std = np.std(angles)

    # 高度差分析
    height_diff = Lhip_y - Rhip_y
    mean_height_diff = np.mean(height_diff)
    centered_height_diff = height_diff - mean_height_diff

    bins = [-np.inf, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, np.inf]
    percents = hist_weighted_percentages(centered_angles, bins)
    right_high_large = percents[0]*3 + percents[1]*2 + percents[2]*1
    right_low_large  = percents[11]*3 + percents[10]*2 + percents[9]*1
    right_high_small = percents[3]*0.5 + percents[4]*0.3
    right_low_small  = percents[7]*0.3 + percents[8]*0.5

    # 判斷方向與嚴重程度
    large_diff = abs(right_high_large - right_low_large)
    small_diff = abs(right_high_small - right_low_small)
    total_large = np.sum(percents[[0,1,2,9,10,11]])

    if total_large > 5:
        if large_diff > 10:
            severity = '嚴重'
        elif large_diff > 6:
            severity = '中度'
        elif large_diff > 2:
            severity = '輕度'
        else:
            severity = '無'
        direction = '右高左低' if right_high_large > right_low_large else '右低左高'
    else:
        if small_diff > 3:
            severity = '輕度'
        elif small_diff > 1:
            severity = '傾向'
        else:
            severity = '無'
        direction = '右高左低' if right_high_small > right_low_small else '右低左高'

    if severity == '無':
        direction = '正常'

    # 週期高度差分析
    mean_sums = []
    for i in range(len(trimmed_left)-1):
        seg = centered_height_diff[trimmed_left[i]:trimmed_left[i+1]]
        mean_sums.append(max(seg) + min(seg))
    mean_sums = np.array(mean_sums)
    clean = mean_sums[np.abs(mean_sums - np.mean(mean_sums)) <= 2*np.std(mean_sums)]
    filtered_mean = np.mean(clean) if len(clean) else np.mean(mean_sums)

    final_direction = '正常'
    final_severity = '正常'
    if abs(filtered_mean) > 0.7:
        final_direction = '右高左低' if filtered_mean > 0 else '右低左高'
        if abs(filtered_mean) > 7.5:
            final_severity = '嚴重'
        elif abs(filtered_mean) > 3.75:
            final_severity = '中度'
        else:
            final_severity = '輕度'

    return {
        'angle_stats': {
            'min': np.min(centered_angles),
            'max': np.max(centered_angles),
            'std': angle_std,
        },
        'height_diff_mean': mean_height_diff,
        'weighted': {
            'right_high_large': right_high_large,
            'right_low_large': right_low_large,
            'right_high_small': right_high_small,
            'right_low_small': right_low_small
        },
        'direction': direction,
        'severity': severity,
        'overall_direction': final_direction,
        'overall_severity': final_severity,
        'filtered_mean': filtered_mean
    }

def _linear_detrend(x: np.ndarray) -> np.ndarray:
    """線性 detrend：擬合 y=ax+b 後相減（等價 MATLAB detrend(...,'linear')）。"""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2 or np.allclose(x, x[0]):
        return x - np.mean(x)
    t = np.arange(n, dtype=float)
    A = np.vstack([t, np.ones_like(t)]).T
    a, b = np.linalg.lstsq(A, x, rcond=None)[0]
    trend = a * t + b
    return x - trend

def _zscore(x: np.ndarray) -> np.ndarray:
    """使用樣本標準差（ddof=1）做標準化，貼齊 MATLAB 慣例。"""
    x = np.asarray(x, dtype=float).ravel()
    mu = x.mean()        
    sd = x.std(ddof=1)      # 關鍵：n-1
    if not np.isfinite(sd) or sd == 0:
        return x * 0.0
    return (x - mu) / sd

def _phi_cross(x: np.ndarray, y: np.ndarray, m: int, r: float) -> float:
    """計算 Cross-ApEn 的 phi(m)，距離用 Chebyshev（max|x-y|），包含自配對。"""
    N = len(x)
    if N < m + 1:
        return np.nan
    X = np.lib.stride_tricks.sliding_window_view(x, window_shape=m)  # (N-m+1, m)
    Y = np.lib.stride_tricks.sliding_window_view(y, window_shape=m)  # (N-m+1, m)
    diff = np.abs(X[:, None, :] - Y[None, :, :])  # (i,j,m)
    dmax = diff.max(axis=2)                       # (i,j)
    C = (dmax <= r).mean(axis=1)                  # 每個 i 的相對頻率
    C = np.where(C <= 0, np.finfo(float).eps, C)  # 避免 log(0)
    return float(np.mean(np.log(C)))

def compute_cross_apen_heel(
    left_heel: np.ndarray,
    right_heel: np.ndarray,
    m: int = 2,
    r_coeff: float = 0.3,
    use_existing_detrend: bool = False
) -> float:
    """
    交叉近似熵（腳跟）：
      1) 線性 detrend
      2) 各自 z-score（ddof=1）
      3) r = r_coeff * std([L_norm, R_norm], ddof=1)
      4) Cross-ApEn = phi(m) - phi(m+1)
    備註：含自配對；資料含 NaN 會傳染為 NaN（與 MATLAB 一致）。
    """
    L = np.asarray(left_heel, dtype=float).ravel()
    R = np.asarray(right_heel, dtype=float).ravel()
    N = min(L.size, R.size)
    L, R = L[:N], R[:N]

    if use_existing_detrend:
        Ld = _linear_detrend(L)  
        Rd = _linear_detrend(R)  
    else:
        Ld = _linear_detrend(L)
        Rd = _linear_detrend(R)

    Ln = _zscore(Ld)
    Rn = _zscore(Rd)

    pooled_sd = np.std(np.concatenate([Ln, Rn]), ddof=1)
    r = r_coeff * (pooled_sd if pooled_sd > 0 else 1.0)

    phi_m  = _phi_cross(Ln, Rn, m, r)
    phi_m1 = _phi_cross(Ln, Rn, m + 1, r)
    return float(phi_m - phi_m1)