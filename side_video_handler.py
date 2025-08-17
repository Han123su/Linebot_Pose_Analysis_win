import os
import shutil
import pandas as pd
from Decide_direction import quick_direction_check, process_video, get_column_names, check_existing_data, merge_bilateral_data
from side_analysis_modules import analyze_side_gait

def handle_side_video(video_path: str, base_dir: str) -> str:
    """
    處理側面影片邏輯，包含判斷方向、儲存數據、分析資料。

    Args:
        video_path: 使用者上傳的影片路徑
        base_dir: 儲存分析數據與 Data_side.xlsx 的資料夾路徑

    Returns:
        給使用者的文字訊息
    """
    xlsx_path = os.path.join(base_dir, "Data_side.xlsx")

    try:
        # 檢查已存在哪側資料
        file_exists, existing_side = check_existing_data(xlsx_path)

        # 判斷影片方向
        direction = quick_direction_check(video_path, base_dir)
        if direction is None:
            return "無法從影片判斷方向，請重新上傳較清晰的影片"

        # 檢查是否上傳了重複方向的影片
        if file_exists and existing_side == direction:
            other_side = "右" if direction == "左" else "左"
            return f"已有{direction}側數據，請上傳{other_side}側的影片"

        # 處理影片並取得關節資料
        data = process_video(video_path, base_dir, direction)
        df_new = pd.DataFrame(data, columns=get_column_names(direction))

        if file_exists and existing_side:
            # 合併左右資料
            df_existing = pd.read_excel(xlsx_path)
            df_merged = merge_bilateral_data(df_existing, df_new)
            df_merged.to_excel(xlsx_path, index=False)

            # 兩側齊全後執行分析
            result_text = analyze_side_gait(xlsx_path, fs=30)

            # 分析完成後清理 Data_side.xlsx 和中繼資料夾
            try:
                if os.path.exists(xlsx_path):
                    os.remove(xlsx_path)

                for folder_name in ["FRAMES", "FRAMES_MP", "FRAMES_TRACKING"]:
                    folder_path = os.path.join(base_dir, folder_name)
                    if os.path.exists(folder_path):
                        shutil.rmtree(folder_path)

            except Exception as cleanup_error:
                print(f"清理暫存資料時出錯：{cleanup_error}")

            return f"分析結果：\n\n{result_text}"

        else:
            # 儲存單側資料
            df_new.to_excel(xlsx_path, index=False)
            other_side = "右" if direction == "左" else "左"
            return f"成功接收{direction}側影片\n請接著上傳{other_side}側影片以進行完整分析"

    except Exception as e:
        return f"錯誤：{str(e)}"
