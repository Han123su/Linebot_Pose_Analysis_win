# 雙側運動分析工具 - 自動判斷方向並整合左右側數據
import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import pandas as pd
from tqdm import tqdm
import shutil
from typing import Tuple, Optional, List, Dict, Any

# 設定 MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

STANDARD_LENGTH = 280  # 統一縮放基準

def calc_angle(a, b, c):
    """
    考慮 x,y,z 座標計算角度
    a, b, c 分別為座標點
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # 計算三角形的邊長
    ab = np.linalg.norm(b - a)
    bc = np.linalg.norm(c - b)
    ac = np.linalg.norm(c - a)
    
    # 使用餘弦定理計算角度
    if ab > 0 and bc > 0:
        cosine_angle = (ab**2 + bc**2 - ac**2) / (2 * ab * bc)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # 防止數值超出範圍
        return np.degrees(angle)  # 轉換為度數
    return None

def calc_angle_2d(a, b, c):
    """
    只考慮 x,y 座標計算角度
    a, b, c 分別為座標點
    """
    # 確保輸入數據有效
    if any(p is None or len(p) < 2 for p in [a, b, c]):
        return None
            
    # 只取 x,y 座標
    a_2d = np.array(a[:2], dtype=np.float64)
    b_2d = np.array(b[:2], dtype=np.float64)
    c_2d = np.array(c[:2], dtype=np.float64)
        
    # 計算 2D 平面上的邊長
    ab = np.linalg.norm(b_2d - a_2d)
    bc = np.linalg.norm(c_2d - b_2d)
    ac = np.linalg.norm(c_2d - a_2d)
        
    # 使用餘弦定理計算角度
    if ab > 0 and bc > 0:
        cosine_angle = (ab**2 + bc**2 - ac**2) / (2 * ab * bc)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
            
    return None

def compute_scale_distance(landmarks, direction, image_width, image_height):
    def get_coords(idx):
        lm = landmarks[idx]
        return np.array([lm.x * image_width, lm.y * image_height])
    
    if direction == '左':
        shoulder = get_coords(11)
        hip = get_coords(23)
    else:
        shoulder = get_coords(12)
        hip = get_coords(24)
    
    return np.linalg.norm(shoulder - hip)

def determine_video_direction(pose_landmarks_list):
    """
    基於多幀的關節點深度資訊判斷影片拍攝方向
    使用膝和踝的平均深度進行雙重確認
    返回 '左' 表示左側拍攝，'右' 表示右側拍攝
    """
    # 初始化儲存深度值的列表
    left_knee_depths = []
    right_knee_depths = []
    left_ankle_depths = []
    right_ankle_depths = []
    
    # 遍歷每一幀的關節點資訊
    for landmarks in pose_landmarks_list:
        if landmarks and landmarks.landmark:
            # 獲取左右膝蓋和踝關節的深度值
            left_knee_z = landmarks.landmark[25].z
            right_knee_z = landmarks.landmark[26].z
            left_ankle_z = landmarks.landmark[27].z
            right_ankle_z = landmarks.landmark[28].z
            
            # 收集深度值
            left_knee_depths.append(left_knee_z)
            right_knee_depths.append(right_knee_z)
            left_ankle_depths.append(left_ankle_z)
            right_ankle_depths.append(right_ankle_z)
    
    # 計算平均深度值
    left_knee_avg = np.mean(left_knee_depths)
    right_knee_avg = np.mean(right_knee_depths)
    left_ankle_avg = np.mean(left_ankle_depths)
    right_ankle_avg = np.mean(right_ankle_depths)
    
    # 判斷方向：右側深度大於左側代表是左側拍攝
    knee_is_left = right_knee_avg > left_knee_avg
    ankle_is_left = right_ankle_avg > left_ankle_avg
    
    if knee_is_left and ankle_is_left:
        return '左'
    elif not knee_is_left and not ankle_is_left:
        return '右'
    else:
        return '左' if knee_is_left else '右'

def get_column_names(direction):
    """
    根據影片方向返回需要記錄的欄位名稱
    """
    if direction == '左':
        return [
            'L_Shoulder_11_x', 'L_Shoulder_11_y', 'L_Shoulder_11_z',
            'L_Hip_23_x', 'L_Hip_23_y', 'L_Hip_23_z', 
            'L_Knee_25_x', 'L_Knee_25_y', 'L_Knee_25_z', 
            'L_Ankle_27_x', 'L_Ankle_27_y', 'L_Ankle_27_z', 
            'L_Heel_29_x', 'L_Heel_29_y', 'L_Heel_29_z',
            'L_Foot_31_x', 'L_Foot_31_y', 'L_Foot_31_z',
            'Angle_23_25_27', 'Angle_11_23_25',
            'Angle_25_27_31', 'Angle_27_29_31', 'Angle_25_29_31',
            'Angle_23_29_31', 'Angle_23_27_31'
        ]
    elif direction == '右':
        return [
            'R_Shoulder_12_x', 'R_Shoulder_12_y', 'R_Shoulder_12_z',
            'R_Hip_24_x', 'R_Hip_24_y', 'R_Hip_24_z', 
            'R_Knee_26_x', 'R_Knee_26_y', 'R_Knee_26_z', 
            'R_Ankle_28_x', 'R_Ankle_28_y', 'R_Ankle_28_z', 
            'R_Heel_30_x', 'R_Heel_30_y', 'R_Heel_30_z',
            'R_Foot_32_x', 'R_Foot_32_y', 'R_Foot_32_z',
            'Angle_24_26_28', 'Angle_12_24_26',
            'Angle_26_28_32', 'Angle_28_30_32', 'Angle_26_30_32',
            'Angle_24_30_32', 'Angle_24_28_32'
        ]
    else:
        raise ValueError("未知方向")

def process_frame_data_scaled(landmarks, direction, image_width, image_height, scale_factor):
    def get_coords(idx):
        lm = landmarks[idx]
        return [lm.x * image_width / scale_factor, lm.y * image_height / scale_factor, lm.z]

    if direction == '左':
        shoulder = get_coords(11)
        hip = get_coords(23)
        knee = get_coords(25)
        ankle = get_coords(27)
        heel = get_coords(29)
        foot = get_coords(31)
    else:
        shoulder = get_coords(12)
        hip = get_coords(24)
        knee = get_coords(26)
        ankle = get_coords(28)
        heel = get_coords(30)
        foot = get_coords(32)

    angles = [
        calc_angle_2d(hip, knee, ankle),
        calc_angle_2d(shoulder, hip, knee),
        calc_angle_2d(knee, ankle, foot),
        calc_angle_2d(ankle, heel, foot),
        calc_angle_2d(knee, heel, foot),
        calc_angle_2d(hip, heel, foot),
        calc_angle_2d(hip, ankle, foot)
    ]

    return shoulder + hip + knee + ankle + heel + foot + angles

def draw_skeleton(frame, results, direction):
    """
    根據方向繪製骨架，突出顯示重要側邊
    """
    if results.pose_landmarks:
        # 繪製基本骨架
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(128, 128, 128), thickness=1),
            mp_drawing.DrawingSpec(color=(128, 128, 128), thickness=1)
        )
        
        # 根據方向突出顯示重要關節點
        landmarks = results.pose_landmarks.landmark
        if direction == 'left':
            key_indices = [11, 23, 25, 27]  # 左側關節點（加入肩膀）
            highlight_color = (0, 165, 255)  # 橘色 (BGR)
        else:
            key_indices = [12, 24, 26, 28]  # 右側關節點（加入肩膀）
            highlight_color = (255, 140, 105)  # 淺藍色 (BGR)
        
        # 突出顯示重要關節點
        for idx in key_indices:
            landmark = landmarks[idx]
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), radius=5, color=highlight_color, thickness=-1)

def check_existing_data(xlsx_path: str) -> Tuple[bool, Optional[str]]:
    """
    檢查是否存在 xlsx 檔案，並確認其中已有的側向數據
    返回: (檔案是否存在, 已存在的側向)
    """
    if not os.path.exists(xlsx_path):
        return False, None
        
    try:
        df = pd.read_excel(xlsx_path)
        columns = df.columns.tolist()
        
        # 通過欄位名稱判斷已有的側向
        if any('L_' in col for col in columns):
            return True, '左'
        elif any('R_' in col for col in columns):
            return True, '右'
        
        return True, None
    except Exception as e:
        print(f"Error reading existing file: {e}")
        return False, None
    
def merge_bilateral_data(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    合併左右側數據
    """
    # 確保兩個 DataFrame 的行數相同
    min_rows = min(len(existing_df), len(new_df))
    existing_df = existing_df.iloc[:min_rows].reset_index(drop=True)
    new_df = new_df.iloc[:min_rows].reset_index(drop=True)
    
    # 合併數據
    merged_df = pd.concat([existing_df, new_df], axis=1)
    
    return merged_df

def quick_direction_check(video_path: str, base_path: str) -> str:
    """
    快速檢查影片方向，只處理前25幀
    """
    # 建立暫存資料夾
    os.makedirs(os.path.join(base_path, 'FRAMES_TEMP'), exist_ok=True)
    
    # 讀取影片
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise ValueError(f"Could not open video file '{video_path}'")

    # 只讀取前25幀
    landmarks_collection = []
    frame_count = 0
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for _ in tqdm(range(25)):
            success, frame = vidcap.read()
            if not success:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks_collection.append(results.pose_landmarks)
            
            frame_count += 1
    
    vidcap.release()
    
    # 清理暫存資料夾
    try:
        shutil.rmtree(os.path.join(base_path, 'FRAMES_TEMP'))
    except Exception as e:
        print(f"Error cleaning up directories: {e}")
    
    # 判斷方向
    if landmarks_collection:
        return determine_video_direction(landmarks_collection)
    return None

def process_video(video_path: str, base_path: str, video_direction: str) -> List[List[float]]:
    """
    使用預先判斷的方向處理影片並返回處理後的數據
    """
    # 建立暫存資料夾
    os.makedirs(os.path.join(base_path, 'FRAMES'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'FRAMES_MP'), exist_ok=True)

    # 讀取影片
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise ValueError(f"Could not open video file '{video_path}'")

    # 獲取影片參數
    original_fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(f"Original FPS: {original_fps}")

    # 設置 FPS 條件：如果原始 FPS > 30，則使用 30，否則使用原始 FPS
    fps = min(original_fps, 30)
    print(f"Using FPS: {fps}")

    # 獲取總幀數以顯示進度
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n總幀數: {total_frames}")
    
    # 讀取並保存幀
    success, image = vidcap.read()
    count = 0
    print("\n正在讀取並保存幀...")
    with tqdm(total=total_frames) as pbar:
        while success:
            if count % max(1, int(original_fps // fps)) == 0:
                cv2.imwrite(os.path.join(base_path, "FRAMES", f"{count}.jpg"), image)
            success, image = vidcap.read()
            count += 1
            pbar.update(1)
    vidcap.release()

    # 收集幀列表
    frame_list = sorted(glob.glob(os.path.join(base_path, 'FRAMES', '*.jpg')))
    print("frame count: ", len(frame_list))
    
    # 獲取影片尺寸
    first_frame = cv2.imread(frame_list[0])
    image_height, image_width = first_frame.shape[:2]
    print(f"frame size: ({image_width}, {image_height})")

    # 計算中位數肩髖距離與縮放因子
    distances = []
    with mp_holistic.Holistic() as holistic:
        for path in tqdm(frame_list[:70]):
            frame = cv2.imread(path)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            if results.pose_landmarks:
                d = compute_scale_distance(results.pose_landmarks.landmark, video_direction, image_width, image_height)
                distances.append(d)

    median_dist = np.median(distances)
    scale_factor = median_dist / STANDARD_LENGTH

    # 處理所有幀
    processed_data = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        print("\nProcessing frames...")
        for idx, path in tqdm(enumerate(frame_list), total=len(frame_list)):
            frame = cv2.imread(path)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            if results.pose_landmarks:
                try:
                    frame_data = process_frame_data_scaled(
                        results.pose_landmarks.landmark,
                        video_direction,
                        image_width,
                        image_height,
                        scale_factor
                    )
                    processed_data.append(frame_data)
                except Exception as e:
                    print(f"Error processing frame {idx}: {e}")
                    continue

            # 繪製骨架
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            draw_skeleton(frame, results, video_direction)
            cv2.imwrite(os.path.join(base_path, "FRAMES_MP", f"{idx}.jpg"), frame)

    # 清理暫存資料夾
    try:
        shutil.rmtree(os.path.join(base_path, 'FRAMES'))
        shutil.rmtree(os.path.join(base_path, 'FRAMES_MP'))
    except Exception as e:
        print(f"Error cleaning up directories: {e}")

    return processed_data

def main():

    # 設定路徑
    PATH = "/data/2TSSD/Han212/skeleton/data/"
    video_file = "video/750g_right1b.mp4"  
    video_file_path = os.path.join(PATH, video_file)
    
    # 使用固定的 xlsx 檔案名稱
    xlsx_path = os.path.join(PATH, "Data.xlsx")
    
    try:
        # 檢查現有數據
        file_exists, existing_side = check_existing_data(xlsx_path)
        
        # 檢查影片方向
        video_direction = quick_direction_check(video_file_path, PATH)
        print(f"影片判斷方向為: {video_direction}側")
        
        # 檢查是否已存在相同側向的數據
        if file_exists and existing_side and existing_side == video_direction:
            print(f"\n警告：檔案中已存在{video_direction}側數據！！！")
            print(f"請提供另一側（{'右' if video_direction == '左' else '左'}）的影片。")
            return
        
        # 使用預先判斷的方向處理影片和收集數據
        print("\n開始處理影片和收集數據...")
        processed_data = process_video(video_file_path, PATH, video_direction)
        
        # 建立新數據的 DataFrame
        new_df = pd.DataFrame(processed_data, columns=get_column_names(video_direction))
        
        if file_exists and existing_side:
            print("\n發現現有數據檔案，正在合併兩側數據...")
            # 讀取現有數據
            existing_df = pd.read_excel(xlsx_path)
            
            # 合併數據
            merged_df = merge_bilateral_data(existing_df, new_df)
            
            # 保存完整數據
            merged_df.to_excel(xlsx_path, index=False)
            print("\n已成功合併左右側完整數據！！")
            
        else:
            # 保存新的數據檔案
            new_df.to_excel(xlsx_path, index=False)
            print(f"\n已保存{video_direction}側數據")
            print(f"請提供 {'右' if video_direction == '左' else '左'}側 影片以完成完整分析。")
            
    except Exception as e:
        print(f"\n處理過程中發生錯誤: {e}")
        raise

if __name__ == "__main__":
    main()
