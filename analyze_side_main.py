import os
import sys
from side_analysis_modules import analyze_side_gait

def main():
    # 設定輸入檔案路徑
    if len(sys.argv) > 1:
        xlsx_path = sys.argv[1]
    else:
        xlsx_path = "Data_side.xlsx"  
    
    # 取樣率（影片通常是 30 FPS）
    fs = 30
    
    if not os.path.exists(xlsx_path):
        print(f"[錯誤] 找不到檔案：{xlsx_path}")
        return
    
    # 分析步態數據
    result_text = analyze_side_gait(xlsx_path, fs=fs)
    print(result_text)
    

if __name__ == "__main__":
    main()