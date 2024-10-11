import cv2
import time
import matplotlib
from concurrent.futures import ThreadPoolExecutor
import scipy
import matplotlib.font_manager
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from tqdm import tqdm

# 閾值比例 與最高最低的差/100
pro_ratio = 30

# 設定影片的幀率
fps = 60  

# 設置支持中文的字體
matplotlib.rcParams['font.family'] = 'MingLiU'
print("start")
# 讀取影片
cap = cv2.VideoCapture('video.mp4')

# 檢查影片是否成功打開
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start_time = time.time()

def process_frame(frame):
    try:
        # 將BGR影像轉換為HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 提取色相通道
        hue, _, _ = cv2.split(hsv)

        # 計算色相的平均值
        mean_hue = np.mean(hue)
        return mean_hue
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

def main():
    hue_values = []

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                futures.append(executor.submit(process_frame, frame))
                pbar.update(1)

            for future in futures:
                result = future.result()
                if result is not None:
                    hue_values.append(result)

    cap.release()

    dh = (max(hue_values) - min(hue_values)) / 100

    time_values = np.arange(len(hue_values)) / fps

    # 找到色相值的顯著低點
    peaks, properties = scipy.signal.find_peaks(-np.array(hue_values), prominence=pro_ratio * dh)  # 設置顯著性閾值
    low_points = [(time_values[p], hue_values[p]) for p in peaks]

    # 繪製色相變化圖表並標註低點
    plt.plot(time_values, hue_values, label='平均色相值')
    plt.title('平均色相值對時間的關係圖')
    plt.xlabel('時間 (秒)')
    plt.ylabel('平均色相值')

    # 標註低點
    for t, h in low_points:
        plt.scatter([t], [h], color='red')
        plt.text(t, h, f'{t:.2f}', fontsize=12, position=(t, h - 5 * dh))

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
