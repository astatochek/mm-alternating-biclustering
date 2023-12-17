import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from utils import get_reordered_row_labels


def get_clusters_from_labels(labels: NDArray, n_clusters: int) -> list[NDArray]:
    return [np.array([j for j in range(labels.size) if labels[j] == i]) for i in range(n_clusters)]


def update_means(model: NDArray, n_clusters: int, row_labels: NDArray, col_labels: NDArray) -> NDArray:
    row_clusters = get_clusters_from_labels(row_labels, n_clusters)
    col_clusters = get_clusters_from_labels(col_labels, n_clusters)
    mu = np.zeros((n_clusters, n_clusters))
    for g in range(n_clusters):
        for h in range(n_clusters):
            if row_clusters[g].size != 0 and col_clusters[h].size != 0:
                block = model[row_clusters[g][:, np.newaxis], col_clusters[h]]
                mu[g][h] = np.mean(block.reshape(block.size))
    return mu


def calc_loss(model: NDArray, mu: NDArray, row_labels: NDArray, col_labels: NDArray) -> float:
    n_rows = model.shape[0]
    n_cols = model.shape[0]
    return np.mean([(model[i][j] - mu[row_labels[i]][col_labels[j]]) ** 2 for j in range(n_cols) for i in range(n_rows)])


def bregman_block_average_coclustering(
        data_matrix: NDArray,
        n_clusters: int,
        *,
        eps=1.e-6
) -> Tuple[NDArray, NDArray, float]:
    """Bregman Block Average Co-clustering"""

    n_rows, n_cols = data_matrix.shape

    row_labels = np.random.randint(0, n_clusters, size=n_rows)
    col_labels = np.random.randint(0, n_clusters, size=n_rows)
    mu = np.zeros((n_clusters, n_clusters))
    prev_loss = calc_loss(data_matrix, mu, row_labels, col_labels) + 1.
    loss = prev_loss - 1.

    while prev_loss - loss > eps:
        mu = update_means(data_matrix, n_clusters, row_labels, col_labels)
        col_clusters = get_clusters_from_labels(col_labels, n_clusters)
        for u in range(n_rows):
            row_labels[u] = np.argmin([np.sum([np.sum([(data_matrix[u][v] - mu[g][h]) ** 2 for v in col_clusters[h]])
                                               for h in range(n_clusters)]) for g in range(n_clusters)])
        row_clusters = get_clusters_from_labels(row_labels, n_clusters)
        for v in range(n_cols):
            col_labels[v] = np.argmin([np.sum([np.sum([(data_matrix[u][v] - mu[g][h]) ** 2 for u in row_clusters[g]])
                                               for g in range(n_clusters)]) for h in range(n_clusters)])

        loss, prev_loss = calc_loss(data_matrix, mu, row_labels, col_labels), loss

    row_labels = get_reordered_row_labels(data_matrix, row_labels, col_labels, n_clusters)

    return row_labels, col_labels, loss
