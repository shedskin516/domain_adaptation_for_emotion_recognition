import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA

def get_pairs(path_aff, path_sewa):
    aff_features = np.load(path_aff)
    sewa_features = np.load(path_sewa)

    n_clusters = 10

    aff_kmeans = KMeans(n_clusters=n_clusters).fit(aff_features)
    aff_cluster_centers = aff_kmeans.cluster_centers_
    aff_labels = aff_kmeans.labels_

    sewa_kmeans = KMeans(n_clusters=n_clusters).fit(sewa_features)
    sewa_cluster_centers = sewa_kmeans.cluster_centers_
    sewa_labels = sewa_kmeans.labels_

    wasserstein_distances = np.zeros((n_clusters, n_clusters))

    for i in range(n_clusters):
        for j in range(n_clusters):
            wasserstein_distances[i,j] = wasserstein_distance(aff_cluster_centers[i], sewa_cluster_centers[j])
            # print(i,j,wasserstein_distances[i,j])

    # print(wasserstein_distances)

    # Hungarian algorithm to pair the clusters
    row_ind, col_ind = linear_sum_assignment(wasserstein_distances)

    for i in range(n_clusters):
        print("aff cluster", i, "is paired with sewa cluster", col_ind[i])

    # Allocate identical numbers to the related clusters in both datasets."
    for i in range(len(aff_labels)):
        aff_labels[i] = col_ind[aff_labels[i]]


    # plot
    # plt.scatter(aff_cluster_centers[:,0], aff_cluster_centers[:,1], c='red', marker='o', label='aff')
    # plt.scatter(sewa_cluster_centers[:,0], sewa_cluster_centers[:,1], c='blue', marker='s', label='sewa')
    # for i in range(n_clusters):
    #     plt.annotate(str(i), xy=(aff_cluster_centers[i,0], aff_cluster_centers[i,1]), color='red')
    #     plt.annotate(str(col_ind[i]), xy=(sewa_cluster_centers[col_ind[i],0], sewa_cluster_centers[col_ind[i],1]), color='blue')
    # plt.legend()
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.savefig("images/figure1.png")

    # plot clusters
    # reduced_data = PCA(n_components=2).fit_transform(sewa_features)
    # # print(reduced_data.shape)
    # kmeans = KMeans(n_clusters=10, random_state=0).fit(reduced_data)
    # plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, s=50, cmap='viridis')
    # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker='*', c='black')
    # plt.show()

    return aff_labels, sewa_labels
