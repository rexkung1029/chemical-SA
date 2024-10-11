import numpy as np
import matplotlib.pyplot as plt

t = input("輸入時間點，以空格為間隔:")

# 時間點數據（秒）
time_points = list(map(float, t.split()))

# 計算間隔
intervals = np.diff(time_points)

# 計算平均間隔
average_interval = np.mean(intervals)
print(f"平均間隔: {average_interval:.4f} 秒")

# 繪製間隔圖表
plt.plot(intervals, marker='o')
for i, interval in enumerate(intervals):
    plt.text(i, interval, f'{interval:.2f}', ha='center', va='bottom')
plt.title(f'Intervals Between Time Points\nAverge={average_interval:.4f}sec')
plt.xlabel('Interval Index')
plt.ylabel('Interval Duration (seconds)')
plt.grid(True)
plt.show()
