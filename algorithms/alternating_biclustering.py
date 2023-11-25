import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List

from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_biclusters
from sklearn.metrics import consensus_score
from tqdm import trange
from generate import Dataset, make_checkerboard_with_custom_distribution, Distribution
from utils import get_biclusters_from_labels, get_submatrix_from_labels, get_reordered_row_labels
from algorithms.kmeans import k_means


# def norm(x: NDArray, indices: NDArray) -> float:
#     """dimensionality-reducing norm"""
#     return np.sqrt(np.square(x[indices]) / indices.size)

def get_loss(X: NDArray, C: List[NDArray], I: NDArray, J: NDArray, k: int) -> float:
    loss = 0
    for i in range(X.shape[0]):
        row = X[i]
        norms = np.array([np.linalg.norm(row[np.where(I == j)] - C[j]) ** 2 / C[j].size
                          for j in range(k)])
        idx = np.argmin(norms)
        loss += norms[idx]
    return loss / X.shape[0]


def update_step(X: NDArray, I: NDArray, J: NDArray, k: int) -> List[NDArray]:
    C = []
    for j in range(k):
        submatrix = get_submatrix_from_labels(X, J, I, j, j)
        centroid_j = np.mean(submatrix, axis=0)
        C.append(centroid_j)
    return C


def assignment_step(
        X: NDArray,
        I: NDArray,
        J: NDArray,
        C: List[NDArray],
        k: int
) -> NDArray:
    updated_labels = np.zeros(X.shape[0], dtype=np.int32)
    I_ = [I[np.where(I == j)] for j in range(k)]
    for i in range(X.shape[0]):
        norms = np.array([np.linalg.norm(X[i][I_[j]] - C[j]) ** 2 / C[j].size for j in range(k)])
        updated_labels[i] = np.argmin(norms)
    return updated_labels


def alternate_iteration(
        X: NDArray,
        I: NDArray,
        J: NDArray,
        k,
        eps: float
) -> Tuple[float, NDArray]:
    alt_loss = np.inf
    continue_flag = True
    while continue_flag:
        try:
            alt_centroids = update_step(X, I, J, k)
            J = assignment_step(X, I, np.copy(J), alt_centroids, k)
            new_loss = get_loss(X, alt_centroids, I, J, k)
            if alt_loss - new_loss < eps:
                continue_flag = False
            alt_loss = new_loss
        except IndexError:
            continue_flag = False

    return alt_loss, J


def alternating_k_means_biclustering(
        data_matrix: NDArray,
        n_clusters: int,
        *,
        eps=1.e-6
) -> Tuple[NDArray, NDArray, float, NDArray, NDArray]:
    total_loss = np.inf

    row_labels, _ = k_means(data_matrix, n_clusters)
    col_labels, _ = k_means(data_matrix.T, n_clusters)

    try: row_labels = get_reordered_row_labels(data_matrix, row_labels, col_labels, n_clusters)
    except IndexError: return row_labels, col_labels, total_loss, row_labels, col_labels

    init_row_labels, init_col_labels = np.copy(row_labels), np.copy(col_labels)

    while True:
        row_loss, row_labels = alternate_iteration(data_matrix, col_labels, row_labels, n_clusters, eps / 2)
        col_loss, col_labels = alternate_iteration(data_matrix.T, row_labels, col_labels, n_clusters, eps / 2)

        if col_loss + row_loss == np.inf: break
        if total_loss - (row_loss + col_loss) < eps: break

        total_loss = row_loss + col_loss
    #     print(f"Loss: {total_loss}")
    # print("End")
    try: row_labels = get_reordered_row_labels(data_matrix, row_labels, col_labels, n_clusters)
    finally: return row_labels, col_labels, total_loss, init_row_labels, init_col_labels


# fig, (ax1, ax2, ax3) = plt.subplots(
#     1, 3, figsize=(11, 4), layout='constrained', dpi=200)
#
# shape = (20, 20)
# n_clusters = 3
# noise = 5
#
# # generate data matrix with cluster assignments
# data, rows, cols = make_biclusters(
#     shape=shape, n_clusters=n_clusters, noise=noise, shuffle=False)
#
# # show original dataset with visible clusters
# ax1.matshow(data, cmap=plt.cm.Blues)
# ax1.set_title("Original dataset")
#
# # shuffle data
# rng = np.random.RandomState(0)
# row_idx = rng.permutation(data.shape[0])
# col_idx = rng.permutation(data.shape[1])
# data = data[row_idx][:, col_idx]
#
# # show shuffled data
# ax2.matshow(data, cmap=plt.cm.Blues)
# ax2.set_title("Shuffled dataset")
#
# # calculate cluster assignments using an iterative algorithm
#
# n_runs = 10
# labels: List[Tuple[NDArray, NDArray]] = []
# losses: List[float] = []
# for _ in range(n_runs):
#     row_labels, col_labels, loss = alternating_k_means_biclustering(data, n_clusters)
#     labels.append((row_labels, col_labels))
#     losses.append(loss)
# row_labels, col_labels = labels[np.argmin(losses)]
#
# # reorder rows and cols of a data matrix to show clusters
# fit_data = data[np.argsort(row_labels)]
# fit_data = fit_data[:, np.argsort(col_labels)]
#
# # show data matrix with reordered rows and cols according to calculated cluster assignments
# # ax3.matshow(fit_data, cmap=plt.cm.Blues)
# # ax3.set_title(f"Alternating KMeans")
#
# # calculate consensus score between expected and actual biclusters
# score = consensus_score(get_biclusters_from_labels(shape, n_clusters, row_labels, col_labels),
#                         (rows[:, row_idx], cols[:, col_idx]))

# print(f"Consensus Score: {score:.3f}")

# plt.show()

def run(n: int):
    shape = (100, 100)
    n_clusters = 5
    noise = 20

    data, rows, cols = make_biclusters(
        shape=shape, n_clusters=n_clusters, noise=noise, shuffle=False)

    # shuffle data
    rng = np.random.RandomState(0)
    row_idx = rng.permutation(data.shape[0])
    col_idx = rng.permutation(data.shape[1])
    data = data[row_idx][:, col_idx]

    k_means_scores, alt_k_means_scores = [0] * n, [0] * n

    for i in range(n):
        row_labels, col_labels, loss, init_row_labels, init_col_labels = alternating_k_means_biclustering(data,
                                                                                                          n_clusters)

        # reorder rows and cols of a data matrix to show clusters
        fit_data = data[np.argsort(row_labels)]
        fit_data = fit_data[:, np.argsort(col_labels)]

        score = consensus_score(get_biclusters_from_labels(shape, n_clusters, row_labels, col_labels),
                                (rows[:, row_idx], cols[:, col_idx]))

        alt_k_means_scores[i] = score

        # reorder rows and cols of a data matrix to show clusters
        fit_data = data[np.argsort(init_row_labels)]
        fit_data = fit_data[:, np.argsort(init_col_labels)]

        score = consensus_score(get_biclusters_from_labels(shape, n_clusters, init_row_labels, init_col_labels),
                                (rows[:, row_idx], cols[:, col_idx]))

        k_means_scores[i] = score

    return alt_k_means_scores, k_means_scores


alt, reg = run(10)
print(alt)
print(reg)
