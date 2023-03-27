from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

filename = 'features.npy'
# filename = 'features_SEWA.npy'

feature_matrix = np.load(filename)
print(feature_matrix.shape)

reduced_data = PCA(n_components=2).fit_transform(feature_matrix)
print(reduced_data.shape)
kmeans = KMeans(n_clusters=2, random_state=0).fit(reduced_data)

# X is the data matrix, kmeans.labels_ contains the cluster assignments
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker='*', c='black')
plt.show()