import argparse
import pandas as pd
import numpy as np
from back_analysis_modules import (
    analyze_fft_phase_diff,
    analyze_fft_zero_padding,
    calculate_dynamic_parameters,
    detect_gait_events,
    calculate_ratios,
    analyze_lift_ratios,
    calculate_ratios2,
    analyze_pca_features,
    analyze_pelvis,
    compute_cross_apen_heel
)

# === 讀取資料 ===
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='輸入的xlsx路徑')
parser.add_argument('--image_folder', type=str, default='result_images', help='儲存圖像的資料夾')
args = parser.parse_args()

file_path = args.input
image_save_folder = args.image_folder

df = pd.read_excel(file_path)

# print("=== Python: 資料總幀數 ===")
# print(f"總幀數: {len(df)}")
# print(f"欄位名稱：{list(df.columns)}")
# print(f"第一筆數據 Lhip_y: {df['y_23'].iloc[0]}")

fs = 30
Lankle = df['y_27'].values
Rankle = df['y_28'].values
Lknee = df['y_25'].values
Rknee = df['y_26'].values
Lheel = df['y_29'].values
Rheel = df['y_30'].values
Ls = df['y_11'].values
Rs = df['y_12'].values
hip_x = {'Lhip': df['x_23'].values, 'Rhip': df['x_24'].values}
hip_y = {'Lhip': df['y_23'].values, 'Rhip': df['y_24'].values}

# === FFT 分析 ===
print("===== FFT 相位分析 =====")
fft_result = analyze_fft_phase_diff(Lankle, Rankle, fs)
print(f"主要頻率成分: {fft_result['main_freq']:.3f} Hz")
print(f"主頻相位差: {fft_result['main_phase_diff_deg']:.2f}°")

fft_zero = analyze_fft_zero_padding(Lankle, Rankle, fs)
print(f"主頻相位差(補零): {fft_zero['zero_phase_diff_deg']:.2f}°")

print(f"加權相位差: {fft_result['weighted_phase_diff_deg']:.2f}°")

# === 週期偵測 ===
print("\n===== 步態週期分析 =====")
min_peak_distance, window_size = calculate_dynamic_parameters(Lankle, fs)
left_events, left_cycles = detect_gait_events(Lknee, Lankle, Lheel, min_peak_distance, window_size, fs)
right_events, right_cycles = detect_gait_events(Rknee, Rankle, Rheel, min_peak_distance, window_size, fs)
if left_cycles:
    l_mean = np.mean(left_cycles)
    l_std = np.std(left_cycles)
    l_cv = (l_std / l_mean) * 100 if l_mean != 0 else 0
    print(f"左平均週期時間: {l_mean:.2f}秒 標準差: {l_std:.2f}秒 (變異數: {l_cv:.2f}%)")
else:
    print("左腳週期無有效事件")

if right_cycles:
    r_mean = np.mean(right_cycles)
    r_std = np.std(right_cycles)
    r_cv = (r_std / r_mean) * 100 if r_mean != 0 else 0
    print(f"右平均週期時間: {r_mean:.2f}秒 標準差: {r_std:.2f}秒 (變異數: {r_cv:.2f}%)")
else:
    print("右腳週期無有效事件")

#print("\n=== Python: 偵測事件數 ===")
#print(f"左腳事件數: {len(left_events)}")
#print(f"右腳事件數: {len(right_events)}")
#print(f"左腳事件位置: {left_events[:10]} ...")
#print(f"右腳事件位置: {right_events[:10]} ...")

# === 較高幀數比例分析 ===
smoothed = {
    'Lknee': pd.Series(Lknee).rolling(window_size, center=True, min_periods=1).mean().values,
    'Rknee': pd.Series(Rknee).rolling(window_size, center=True, min_periods=1).mean().values,
    'Lankle': pd.Series(Lankle).rolling(window_size, center=True, min_periods=1).mean().values,
    'Rankle': pd.Series(Rankle).rolling(window_size, center=True, min_periods=1).mean().values,
    'Lheel': pd.Series(Lheel).rolling(window_size, center=True, min_periods=1).mean().values,
    'Rheel': pd.Series(Rheel).rolling(window_size, center=True, min_periods=1).mean().values,
    'Ls': pd.Series(Ls).rolling(window_size, center=True, min_periods=1).mean().values,
    'Rs': pd.Series(Rs).rolling(window_size, center=True, min_periods=1).mean().values
}
Lper, Rper = calculate_ratios(smoothed)
stats = analyze_lift_ratios(Lper, Rper)

print("\n===== 較高比例幀數分析 =====")
# print(f"總幀數: {stats['frame_count']}")
print(f"左側較高幀數比例: {stats['above']:.2f}")
print(f"右側較高幀數比例: {stats['below']:.2f}")
print(f"差異程度: {stats['difference_value']:.4f}")

print("推測: ", end="")
if stats["severity"] == "serious":
    print(f"{'左' if stats['higher_side'] == 'left' else '右'}側嚴重偏高!")
elif stats["severity"] == "mild":
    print(f"{'左' if stats['higher_side'] == 'left' else '右'}側輕微偏高")
else:
    print("左右側差異在正常範圍內")


# === 週期抬腿幅度分析 ===
print("\n===== 週期抬腿幅度分析 =====")
ratios = calculate_ratios2(smoothed, left_events, right_events, hip_y)
print(f"左腳抬腿比例平均 : {np.mean(ratios['Lper']) * 100:.1f}% (SD: {np.std(ratios['Lper']) * 100:.1f}%)")
print(f"右腳抬腿比例平均 : {np.mean(ratios['Rper']) * 100:.1f}% (SD: {np.std(ratios['Rper']) * 100:.1f}%)")

# === PCA 分析 ===
print("\n===== PCA 分析 =====")
pca_result = analyze_pca_features({
    'Lknee': smoothed['Lknee'],
    'Lankle': smoothed['Lankle'],
    'Lhip': hip_y['Lhip'],
    'Lheel': smoothed['Lheel'],
    'Rknee': smoothed['Rknee'],
    'Rankle': smoothed['Rankle'],
    'Rhip': hip_y['Rhip'],
    'Rheel': smoothed['Rheel']
})

left_exp = pca_result['left']['explained']
right_exp = pca_result['right']['explained']
print(f"左腳: PC1: {left_exp[0]:.1f}%, PC2: {left_exp[1]:.1f}%")
print(f"右腳: PC1: {right_exp[0]:.1f}%, PC2: {right_exp[1]:.1f}%")

# ===== 交叉近似熵（腳跟） =====
cross_apen_heel = compute_cross_apen_heel(
    df['y_29'].to_numpy(),   # 左腳跟
    df['y_30'].to_numpy(),   # 右腳跟
    m=2,
    r_coeff=0.3,
    use_existing_detrend=False  # 若要改用你現成 detrend()，設 True 並在模組中接上
)
print("\n===== 交叉近似熵 =====")
print(f"交叉近似熵: {cross_apen_heel:.4f}")

# === 骨盆分析 ===
print("\n===== 骨盆分析 =====")
points_x = {
    'Lknee': df['x_25'],
    'Rknee': df['x_26'],
    'Lankle': df['x_27'],
    'Rankle': df['x_28'],
    'Lheel': df['x_29'],
    'Rheel': df['x_30']
}
points_y = {
    'Lknee': df['y_25'],
    'Rknee': df['y_26'],
    'Lankle': df['y_27'],
    'Rankle': df['y_28'],
    'Lheel': df['y_29'],
    'Rheel': df['y_30']
}
pelvis = analyze_pelvis(df, points_x, points_y, left_events, right_events, fs)
print("--- (1) 骨盆角度分布 ---")
print(f"角度變化範圍: {pelvis['angle_stats']['min']:.2f}° ~ {pelvis['angle_stats']['max']:.2f}°")
print(f"標準差: {pelvis['angle_stats']['std']:.2f}°")

large_diff = abs(pelvis['weighted']['right_high_large'] - pelvis['weighted']['right_low_large'])
if pelvis['weighted']['right_high_large'] > pelvis['weighted']['right_low_large']:
    print(f"大角度加權: 右高左低多({large_diff:.1f})")
else:
    print(f"大角度加權: 右低左高多({large_diff:.1f})")

small_diff = abs(pelvis['weighted']['right_high_small'] - pelvis['weighted']['right_low_small'])
if pelvis['weighted']['right_high_small'] > pelvis['weighted']['right_low_small']:
    print(f"小角度加權: 右高左低多({small_diff:.1f})")
else:
    print(f"小角度加權: 右低左高多({small_diff:.1f})")

print(f"骨盆狀態: {pelvis['direction']}")
print(f"嚴重程度: {pelvis['severity']}")

print("--- (2) 骨盆週期高度差 ---")
print(f"排除異常後平均值: {pelvis['filtered_mean']:.2f}")
print(f"骨盆狀態: {pelvis['overall_direction']}")
print(f"嚴重程度: {pelvis['overall_severity']}")

# print("Python Left Events:", left_events)
# print("Python Right Events:", right_events)