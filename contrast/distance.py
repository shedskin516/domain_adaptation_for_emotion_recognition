import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA

def get_pairs(source_feature, target_feature, n_clusters):
    source_feature = source_feature.detach().numpy()
    target_feature = target_feature.detach().numpy()

    source_kmeans = KMeans(n_clusters=n_clusters).fit(source_feature)
    source_cluster_centers = source_kmeans.cluster_centers_
    source_labels = source_kmeans.labels_

    target_kmeans = KMeans(n_clusters=n_clusters).fit(target_feature)
    target_cluster_centers = target_kmeans.cluster_centers_
    target_labels = target_kmeans.labels_

    wasserstein_distances = np.zeros((n_clusters, n_clusters))

    for i in range(n_clusters):
        for j in range(n_clusters):
            wasserstein_distances[i,j] = wasserstein_distance(source_cluster_centers[i], target_cluster_centers[j])
            # print(i,j,wasserstein_distances[i,j])

    # print(wasserstein_distances)

    # Hungarian algorithm to pair the clusters
    row_ind, col_ind = linear_sum_assignment(wasserstein_distances)

    for i in range(n_clusters):
        print("aff cluster", i, "is paired with sewa cluster", col_ind[i])

    # Allocate identical numbers to the related clusters in both datasets."
    for i in range(len(source_labels)):
        source_labels[i] = col_ind[source_labels[i]]

    return source_labels, target_labels
