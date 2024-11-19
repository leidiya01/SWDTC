import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# 读取数据
data = pd.read_excel(r'C:\Users\leidy\Desktop\normalized_data.xlsx')

# 提取X1至X7列的数据，作为初始七维向量
columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']
X = data[columns].values

# 定义聚类数
k = 4

# 原始七维聚类
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# 聚类中心按照距离原点的欧氏距离排序并编号
centers = kmeans.cluster_centers_
distances_from_origin = np.linalg.norm(centers, axis=1)
sorted_indices = np.argsort(distances_from_origin)
cluster_labels = np.array([np.where(sorted_indices == i)[0][0] for i in kmeans.labels_])

# 将七维聚类结果添加到数据框中
data['Cluster_Original_7D'] = cluster_labels

# 逐一剔除一列，进行六维聚类并保存结果
for i in range(len(columns)):
    # 剔除第i列，保留其他六列作为新的六维向量
    subset_columns = columns[:i] + columns[i+1:]
    X_subset = data[subset_columns].values

    # 执行六维聚类
    kmeans_subset = KMeans(n_clusters=k, random_state=42)
    kmeans_subset.fit(X_subset)

    # 聚类中心按距离原点排序并编号
    centers_subset = kmeans_subset.cluster_centers_
    distances_from_origin_subset = np.linalg.norm(centers_subset, axis=1)
    sorted_indices_subset = np.argsort(distances_from_origin_subset)
    cluster_labels_subset = np.array([np.where(sorted_indices_subset == j)[0][0] for j in kmeans_subset.labels_])

    # 将六维聚类结果添加到数据框中
    col_name = f'Cluster_Drop_{columns[i]}'
    data[col_name] = cluster_labels_subset

# 保存结果到新的Excel文件
output_path = 'clustered_data_results.xlsx'
data.to_excel(output_path, index=False)


