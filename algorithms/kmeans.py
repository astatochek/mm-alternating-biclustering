from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from utils import get_reordered_row_labels


def k_means(data_matrix: NDArray, n_clusters: int, *, eps=1.e-6) -> Tuple[NDArray, float]:
    def expectation_step(current_matrix: NDArray, new_centroids: NDArray) -> NDArray:
        n_rows = current_matrix.shape[0]
        updated_labels = np.zeros(n_rows, dtype=np.int32)
        for i in range(n_rows):
            updated_labels[i] = np.argmin([np.linalg.norm(current_matrix[i] - centroid) for centroid in new_centroids])
        return updated_labels

    def maximization_step(current_matrix: NDArray, new_labels: NDArray, n_clusters: int) -> NDArray:
        n_rows, m_cols = current_matrix.shape
        updated_centroids = np.zeros((n_clusters, m_cols))
        labels_count = np.zeros(n_clusters)
        for i in range(n_rows):
            updated_centroids[new_labels[i]] += current_matrix[i]
            labels_count[new_labels[i]] += 1
        for i in range(n_clusters):
            if labels_count[i] > 0:
                updated_centroids[i] /= labels_count[i]
            else:
                updated_centroids[i] = current_matrix[np.random.randint(n_rows)]
        return updated_centroids

    def get_loss(current_matrix: NDArray, updated_labels: NDArray, updated_centroids: NDArray) -> float:
        return np.mean([np.linalg.norm(current_matrix[i] - updated_centroids[updated_labels[i]]) for i in
                        range(current_matrix.shape[0])])

    size, _ = data_matrix.shape
    labels = np.random.randint(0, n_clusters, size)
    loss = np.inf
    centroids = maximization_step(data_matrix, labels, n_clusters)

    while True:
        labels = expectation_step(data_matrix, centroids)
        centroids = maximization_step(data_matrix, labels, n_clusters)
        updated_loss = get_loss(data_matrix, labels, centroids)
        if loss - updated_loss < eps:
            break
        loss = updated_loss

    return labels, loss


def k_means_biclustering(
    data_matrix: NDArray,
    n_clusters: int,
    *,
    eps=1.e-6
) -> Tuple[NDArray, NDArray, float]:
    row_labels, row_loss = k_means(data_matrix, n_clusters, eps=eps)
    col_labels, col_loss = k_means(data_matrix.T, n_clusters, eps=eps)
    try: row_labels = get_reordered_row_labels(data_matrix, row_labels, col_labels, n_clusters)
    finally: return row_labels, col_labels, row_loss + col_loss
