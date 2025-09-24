# 读取weather.csv，并可视化数据

import pandas as pd
import matplotlib.pyplot as plt

# 读取数据并设置索引列
data = pd.read_csv("weather/weather.csv", index_col="date")
data = data[5000:6000]
print(data.head())

# 可视化温度数据
plt.plot(data["T (degC)"])
plt.xlabel("Date")
plt.ylabel("Temperature (degC)")
plt.title("Temperature over Time")
plt.savefig("image.png")  # 保存图像到文件
plt.close()  # 关闭图像

# # 可视化所有列的数据
# for column in data.columns:
#     plt.figure()
#     plt.plot(data[column])
#     plt.xlabel('Date')
#     plt.ylabel(column)
#     plt.title(f'{column} over Time')
#     plt.show()
