import cv2
import mediapipe as mp
import os
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

def swap_left_right_coordinates(df):
    """
    檢查並修正左右關節點的x座標，確保右側關節點的x座標大於左側關節點
    
    參數:
    df: DataFrame 包含關節點座標
    """
    df_corrected = df.copy()
    
    # 定義左右對應的關節點
    pairs = [
        (11, 12),  # 肩膀
        (23, 24),  # 臀部
        (25, 26),  # 膝蓋
        (27, 28),  # 腳踝
        (29, 30),  # 腳跟
        (31, 32)   # 腳趾
    ]
    
    print("\n開始檢查並修正左右關節點座標...")
    
    for left, right in pairs:
        print(f"檢查並修正關節點對 {left}-{right}...")
        # 取得對應的列名
        left_x = f'x_{left}'
        left_y = f'y_{left}'
        right_x = f'x_{right}'
        right_y = f'y_{right}'
        
        # 遍歷每一行數據
        for idx in range(len(df_corrected)):
            # 如果左側x座標大於右側x座標，交換左右側的x和y座標
            if df_corrected.loc[idx, left_x] > df_corrected.loc[idx, right_x]:
                # 暫存左側座標
                temp_x = df_corrected.loc[idx, left_x]
                temp_y = df_corrected.loc[idx, left_y]
                
                # 將右側座標移到左側
                df_corrected.loc[idx, left_x] = df_corrected.loc[idx, right_x]
                df_corrected.loc[idx, left_y] = df_corrected.loc[idx, right_y]
                
                # 將暫存的左側座標移到右側
                df_corrected.loc[idx, right_x] = temp_x
                df_corrected.loc[idx, right_y] = temp_y
    
    return df_corrected

def fix_gait_data(df, window_size=7):
    """
    修正步態數據，使用 IQR 方法檢測異常值
    
    參數:
    df: DataFrame 包含關節點座標
    window_size: int, 平滑化窗口大小
    """
    df_fixed = df.copy()
    
    # 取得所有y座標的欄位（排除乒乓球欄位）
    y_columns = [col for col in df.columns if col.startswith('y_')]
    
    print("\n開始修正步態數據...")
    for col in y_columns:
        print(f"處理 {col}...")
        
        # 1. 使用 IQR 方法計算異常值範圍
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 找出typical的步態峰值（使用正常範圍內的數據）
        valid_data = df[col][(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        peaks = []
        for i in range(1, len(valid_data)-1):
            if valid_data.iloc[i] > valid_data.iloc[i-1] and valid_data.iloc[i] > valid_data.iloc[i+1]:
                peaks.append(valid_data.iloc[i])
        
        typical_peak_height = np.mean(peaks) if peaks else valid_data.mean()
        
        # 2. 標記和修正異常值
        for idx in range(len(df_fixed)):
            current_val = df_fixed.loc[idx, col]
            
            # 檢查是否為異常值
            if current_val < lower_bound or current_val > upper_bound or np.isnan(current_val):
                # 取得前後有效值
                prev_valid = df_fixed.loc[:idx-1, col][
                    (df_fixed.loc[:idx-1, col] >= lower_bound) & 
                    (df_fixed.loc[:idx-1, col] <= upper_bound)
                ]
                next_valid = df_fixed.loc[idx+1:, col][
                    (df_fixed.loc[idx+1:, col] >= lower_bound) & 
                    (df_fixed.loc[idx+1:, col] <= upper_bound)
                ]
                
                if len(prev_valid) > 0 and len(next_valid) > 0:
                    prev_val = prev_valid.iloc[-1]
                    next_val = next_valid.iloc[0]
                    prev_idx = prev_valid.index[-1]
                    next_idx = next_valid.index[0]
                    
                    # 使用typical peak height來修正
                    if abs(prev_val - next_val) > IQR:  # 可能是步態週期的峰值
                        df_fixed.loc[idx, col] = typical_peak_height
                    else:
                        # 線性插值
                        ratio = (idx - prev_idx) / (next_idx - prev_idx)
                        interpolated_value = prev_val + ratio * (next_val - prev_val)
                        df_fixed.loc[idx, col] = interpolated_value
                elif len(prev_valid) > 0:
                    df_fixed.loc[idx, col] = prev_valid.iloc[-1]
                elif len(next_valid) > 0:
                    df_fixed.loc[idx, col] = next_valid.iloc[0]
                else:
                    df_fixed.loc[idx, col] = valid_data.mean()
        
        # 3. 輕度平滑化，保持峰值特徵
        df_fixed[col] = df_fixed[col].rolling(window=window_size, center=True, min_periods=1).mean()
    
    return df_fixed

# 設定路徑
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True, help="影片路徑")
parser.add_argument("--output", type=str, required=True, help="XLSX儲存路徑")
args = parser.parse_args()

video_file_path = args.video
output_excel_path = args.output

# 再根據 video_file_path 設定 PATH 路徑：
PATH = os.path.dirname(video_file_path)
video_file = os.path.basename(video_file_path)

# 嘗試開啟影片
vidcap = cv2.VideoCapture(video_file_path)
if not vidcap.isOpened():
    print(f"Error: Could not open video file '{video_file_path}'.")
    exit()

# 獲取影片的原始 FPS 和尺寸
original_fps = vidcap.get(cv2.CAP_PROP_FPS)
original_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Original FPS: {original_fps}, Width: {original_width}, Height: {original_height}")

# 設置固定的 FPS
fps = original_fps if original_fps < 30 else 30
print(f"Using FPS: {fps}")

success, image = vidcap.read()
count = 0

os.makedirs(os.path.join(PATH, 'FRAMES'), exist_ok=True)
os.makedirs(os.path.join(PATH, 'FRAMES_MODIFY'), exist_ok=True)
os.makedirs(os.path.join(PATH, 'FRAMES_TRACKING'), exist_ok=True)  # 修改資料夾名稱

# 將影片分割成幀，並存入指定文件夾
while success:
    if count % (original_fps // fps) == 0:  # 根據原影片的 FPS 和使用的 FPS 比較
        cv2.imwrite(os.path.join(PATH, "FRAMES", f"{count}.jpg"), image)  # 儲存幀影像
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

vidcap.release()

# 設定 MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# 設置固定的肩膀距離 (從原始版本)
fixed_shoulder_distance = 300  

# 定義橘色範圍 (HSV)
lower_orange = np.array([5, 100, 150])
upper_orange = np.array([25, 255, 255])

# 定義著色的關鍵點索引
colored_landmarks = [11, 23, 25, 27, 29, 31, 12, 24, 26, 28, 30, 32]  # 包含左右側關鍵點

# 初始化DataFrame，添加乒乓球追蹤的欄位
columns = [f"x_{lm}" for lm in colored_landmarks] + [f"y_{lm}" for lm in colored_landmarks] + \
          ["left_ball_x", "left_ball_y", "right_ball_x", "right_ball_y"]
df = pd.DataFrame(columns=columns)

# 用於追蹤是否已計算縮放比例
scale_factor_calculated = False

# 開始用 MediaPipe 處理畫面，顯示進度條
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    frame_list = sorted(glob.glob(os.path.join(PATH, 'FRAMES', '*.jpg')), key=os.path.getmtime)

    for idx, path in tqdm(enumerate(frame_list), total=len(frame_list), desc="Processing frames"):
        frame = cv2.imread(path)
        
        # 創建全白背景 (從原始版本)
        fixed_width = 2500
        fixed_height = 2500
        blank_image_scaled = 255 * np.ones(shape=[fixed_height, fixed_width, 3], dtype=np.uint8)
        
        # 取得目標ROI區域 - 使用MediaPipe先取得髖部位置
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        # 儲存原始幀作為比較
        original_frame = frame.copy()
        
        # 檢測到骨架關鍵點時
        if results.pose_landmarks:
            # 繪製原始骨架
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=4),
            )
            
            # 獲取肩膀的關鍵點 (從原始版本)
            shoulder_left = results.pose_landmarks.landmark[11]
            shoulder_right = results.pose_landmarks.landmark[12]
            
            # 計算實際肩膀距離 (從原始版本)
            actual_shoulder_distance = np.sqrt((shoulder_right.x - shoulder_left.x) ** 2 + (shoulder_right.y - shoulder_left.y) ** 2) * original_width

            scale_factor_calculated = False
            if not scale_factor_calculated:
                # 計算縮放比例 (從原始版本)
                scale_factor = fixed_shoulder_distance / actual_shoulder_distance if actual_shoulder_distance != 0 else 1
                scale_factor_calculated = True  # 標記已經計算過縮放比例
            
            # 使用右肩作為基準點 (從原始版本)
            sh_right = results.pose_landmarks.landmark[12]
            x_offset = int(fixed_width / 2 - sh_right.x * original_width * scale_factor)
            y_offset = int(fixed_height / 2 - sh_right.y * original_height * scale_factor) - 500
            
            # 獲取左右髖部位置 (球追蹤用)
            left_hip = results.pose_landmarks.landmark[23]
            right_hip = results.pose_landmarks.landmark[24]
            
            # 計算髖部中心 (球追蹤用)
            hip_center_x = (left_hip.x + right_hip.x) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            
            # 計算左右髖部距離 (球追蹤用)
            hip_distance = np.sqrt((right_hip.x - left_hip.x) ** 2 + (right_hip.y - left_hip.y) ** 2) * frame.shape[1]
            
            # 設定感興趣區域(ROI)為髖部附近的寬範圍 (球追蹤用)
            roi_width = int(hip_distance * 4)
            roi_height = int(hip_distance * 2)
            
            roi_x_start = max(0, int(hip_center_x * frame.shape[1] - roi_width / 2))
            roi_y_start = max(0, int(hip_center_y * frame.shape[0] - roi_height / 2))
            roi_x_end = min(frame.shape[1], roi_x_start + roi_width)
            roi_y_end = min(frame.shape[0], roi_y_start + roi_height)
            
            # 截取ROI區域 (球追蹤用)
            roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            
            # 初始化左右乒乓球位置 (球追蹤用)
            left_ball_pos = None
            right_ball_pos = None
            left_ball_pos_float = None
            right_ball_pos_float = None
            
            # 只有當ROI區域有效時才進行處理 (球追蹤用)
            if roi.size > 0:
                # 轉換ROI到HSV色彩空間以進行顏色過濾
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # 創建橘色遮罩
                mask = cv2.inRange(hsv_roi, lower_orange, upper_orange)
                
                # 形態學操作改善遮罩
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
                mask = cv2.dilate(mask, kernel, iterations=2)
                
                # 找到輪廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 找到最多兩個符合條件的橘色球
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    # 根據預期的乒乓球大小過濾
                    if 50 < area < 1000:
                        # 計算圓度來確保是球形
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                        if circularity > 0.7:  # 圓形閾值
                            # 計算輪廓中心
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                # 計算精確的浮點數中心
                                cx_float = M["m10"] / M["m00"]
                                cy_float = M["m01"] / M["m00"]
                                # 轉換為整數用於繪圖
                                cx_int = int(cx_float)
                                cy_int = int(cy_float)
                                # 轉換回原始幀座標（保留整數和浮點數版本）
                                global_cx_int = roi_x_start + cx_int
                                global_cy_int = roi_y_start + cy_int
                                global_cx_float = roi_x_start + cx_float
                                global_cy_float = roi_y_start + cy_float
                                # 儲存整數版本用於繪圖，但在元組最後附加浮點數版本用於資料儲存
                                valid_contours.append((global_cx_int, global_cy_int, area, global_cx_float, global_cy_float))
                
                # 根據位置將球分為左右兩側 (根據髖部中心位置)
                hip_center_x_pixel = int(hip_center_x * frame.shape[1])
                
                # 排序輪廓按照面積從大到小
                valid_contours.sort(key=lambda x: x[2], reverse=True)
                
                # 僅保留前兩個最大的輪廓
                valid_contours = valid_contours[:2]
                
                # 將輪廓根據x座標分為左右
                if len(valid_contours) == 2:
                    if valid_contours[0][0] < valid_contours[1][0]:
                        # 整數位置用於繪圖
                        left_ball_pos = (valid_contours[0][0], valid_contours[0][1])
                        right_ball_pos = (valid_contours[1][0], valid_contours[1][1])
                        # 浮點數位置用於資料儲存
                        left_ball_pos_float = (valid_contours[0][3], valid_contours[0][4]) 
                        right_ball_pos_float = (valid_contours[1][3], valid_contours[1][4])
                    else:
                        # 整數位置用於繪圖
                        left_ball_pos = (valid_contours[1][0], valid_contours[1][1])
                        right_ball_pos = (valid_contours[0][0], valid_contours[0][1])
                        # 浮點數位置用於資料儲存
                        left_ball_pos_float = (valid_contours[1][3], valid_contours[1][4])
                        right_ball_pos_float = (valid_contours[0][3], valid_contours[0][4])
                elif len(valid_contours) == 1:
                    # 只找到一個球，根據位置判斷是左側還是右側
                    if valid_contours[0][0] < hip_center_x_pixel:
                        left_ball_pos = (valid_contours[0][0], valid_contours[0][1])  # 整數
                        left_ball_pos_float = (valid_contours[0][3], valid_contours[0][4])  # 浮點數
                    else:
                        right_ball_pos = (valid_contours[0][0], valid_contours[0][1])  # 整數
                        right_ball_pos_float = (valid_contours[0][3], valid_contours[0][4])  # 浮點數
                
                # 在原始幀上繪製乒乓球位置
                if left_ball_pos:
                    cv2.circle(frame, left_ball_pos, 8, (0, 0, 255), -1)  # 左側乒乓球紅色
                    cv2.putText(frame, "Left Ball", (left_ball_pos[0]+15, left_ball_pos[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                if right_ball_pos:
                    cv2.circle(frame, right_ball_pos, 8, (0, 0, 255), -1)  # 右側乒乓球紅色
                    cv2.putText(frame, "Right Ball", (right_ball_pos[0]+15, right_ball_pos[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 140, 105), 2)
                
                # 顯示髖部ROI範圍
                cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2)
            
            # 繪製骨架並儲存數據 (整合原始版本和乒乓球版本)
            frame_data = {}
            
            # 處理每個骨架關鍵點
            for i in colored_landmarks:
                landmark = results.pose_landmarks.landmark[i]
                
                # 無縮放版的原始座標 (用於繪製)
                x_actual = landmark.x * original_width
                y_actual = landmark.y * original_height
                
                # 在原始影像上繪製關鍵點
                if i in [11, 23, 25, 27, 29, 31]:  # 左側關鍵點
                    color = (0, 165, 255)  # 橘色 (BGR)
                elif i in [12, 24, 26, 28, 30, 32]:  # 右側關鍵點
                    color = (255, 140, 105)  # 淺藍色 (BGR)
                
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), radius=5, color=color, thickness=-1)
                
                # 標記髖部關鍵點
                if i in [23, 24]:
                    label = "L-Hip" if i == 23 else "R-Hip"
                    cv2.putText(frame, label, (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 縮放後的平移座標 (從原始版本)
                x_scaled = x_actual * scale_factor + x_offset
                y_scaled = y_actual * scale_factor + y_offset

                # 限制座標範圍 (從原始版本)
                x_scaled = min(max(x_scaled, 0), fixed_width - 1)
                y_scaled = min(max(y_scaled, 0), fixed_height - 1)

                # 存入DataFrame (縮放後的座標)
                frame_data[f"x_{i}"] = x_scaled
                frame_data[f"y_{i}"] = y_scaled

                # 繪製縮放版本的關鍵點 (在白色背景上)
                cv2.circle(blank_image_scaled, (int(x_scaled), int(y_scaled)), radius=5, color=color, thickness=-1)
            
            # 存入乒乓球位置 (如果沒有找到則為 None)
            frame_data["left_ball_x"] = left_ball_pos_float[0] if left_ball_pos_float else None
            frame_data["left_ball_y"] = left_ball_pos_float[1] if left_ball_pos_float else None
            frame_data["right_ball_x"] = right_ball_pos_float[0] if right_ball_pos_float else None
            frame_data["right_ball_y"] = right_ball_pos_float[1] if right_ball_pos_float else None
            
            # 添加到DataFrame
            df = df._append(frame_data, ignore_index=True)
            
            # 儲存包含骨架的圖片到 FRAMES_MODIFY 資料夾 (從原始版本)
            cv2.imwrite(os.path.join(PATH, "FRAMES_MODIFY", f"joint_point_{idx}.jpg"), blank_image_scaled)
        
        # 儲存處理後的幀
        cv2.imwrite(os.path.join(PATH, "FRAMES_TRACKING", f"tracking_{idx}.jpg"), frame)

# 檢測並修正步態數據 (從原始版本)
print("\n正在進行步態數據修正...")
df_fixed = fix_gait_data(df, window_size=7)

# 檢查並修正左右關節點座標 (從原始版本)
df_fixed = swap_left_right_coordinates(df_fixed)

# 對乒乓球座標進行插值處理 (從乒乓球版本)
# 記錄原始的缺失值位置（遮罩）
left_ball_x_mask = df_fixed['left_ball_x'].isna()
left_ball_y_mask = df_fixed['left_ball_y'].isna()
right_ball_x_mask = df_fixed['right_ball_x'].isna()
right_ball_y_mask = df_fixed['right_ball_y'].isna()

# 只對連續缺失不超過特定幀數的數據進行插值
max_gap = int(fps * 0.5)  # 假設最多容許缺失半秒

# 使用三次樣條插值填補缺失值，但不修改已存在的值
df_fixed['left_ball_x'] = df_fixed['left_ball_x'].interpolate(method='cubic', limit=max_gap)
df_fixed['left_ball_y'] = df_fixed['left_ball_y'].interpolate(method='cubic', limit=max_gap)
df_fixed['right_ball_x'] = df_fixed['right_ball_x'].interpolate(method='cubic', limit=max_gap)
df_fixed['right_ball_y'] = df_fixed['right_ball_y'].interpolate(method='cubic', limit=max_gap)

# 如果還有剩餘的空值，使用線性插值嘗試填補（更簡單但效果較差）
df_fixed['left_ball_x'] = df_fixed['left_ball_x'].interpolate(method='linear', limit=max_gap)
df_fixed['left_ball_y'] = df_fixed['left_ball_y'].interpolate(method='linear', limit=max_gap)
df_fixed['right_ball_x'] = df_fixed['right_ball_x'].interpolate(method='linear', limit=max_gap)
df_fixed['right_ball_y'] = df_fixed['right_ball_y'].interpolate(method='linear', limit=max_gap)

# 只儲存最終帶有乒乓球座標的 Excel
excel_file_name = f"{os.path.splitext(video_file)[0]}_with_pingpong.xlsx"
excel_path = os.path.join(PATH, excel_file_name)
df_fixed.to_excel(output_excel_path, index=False)
print(f"\n修正後的數據已儲存至: {excel_path}")

# 生成一個影片 (整合骨架和乒乓球追蹤)
output_file_path = os.path.join(PATH, f"{os.path.splitext(video_file)[0]}_tracking.mp4")
frame_files = sorted(glob.glob(os.path.join(PATH, 'FRAMES_TRACKING', '*.jpg')), key=os.path.getmtime)

if frame_files:
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

    # 將每一幀寫入影片
    for frame_path in tqdm(frame_files, desc="Generating video"):
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()
    print(f"\nTracking video saved as {output_file_path}")

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

print("\nPose Tracking, Skeleton Images, and Ping-pong Ball Tracking Complete!")