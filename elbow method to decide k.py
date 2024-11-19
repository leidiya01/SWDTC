import os
os.environ["OMP_NUM_THREADS"] = '4'
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_excel(r'C:\Users\leidy\Desktop\normalized_data.xlsx')

# 提取列X1至X7作为聚类对象，并将每一行的7列数据当作一个七维向量
X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']].values

# 定义肘部法则，计算SSE并绘制肘部曲线
sse = []
k_values = range(1, 16)  # 选择聚类数的范围，可以根据需要调整

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    # 计算SSE: 每个数据点（七维向量）到所属簇中心的距离平方和
    sse.append(kmeans.inertia_)

# 绘制肘部法则曲线
plt.figure(figsize=(8, 6))
plt.plot(k_values, sse, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k')
plt.show()
