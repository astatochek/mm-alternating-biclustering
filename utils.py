from typing import Tuple, Callable, List, Any
from numpy.typing import NDArray
import numpy as np


def get_biclusters_from_labels(
    shape: Tuple[int, int],
    n_clusters: int,
    row_labels: NDArray,
    col_labels: NDArray
) -> Tuple[NDArray, NDArray]:

    n_rows = shape[0]
    n_cols = shape[1]

    row_clusters = np.zeros((n_clusters, n_rows), dtype=bool)
    col_clusters = np.zeros((n_clusters, n_cols), dtype=bool)

    for row_idx in range(row_labels.size):
        row_clusters[row_labels[row_idx]][row_idx] = True

    for col_idx in range(col_labels.size):
        col_clusters[col_labels[col_idx]][col_idx] = True

    return row_clusters, col_clusters


def get_submatrix_from_labels(
        data_matrix: NDArray,
        row_labels: NDArray, col_labels: NDArray,
        row_cluster: int, col_cluster: int
) -> NDArray:

    if row_cluster not in row_labels:
        raise IndexError('row cluster not in labels')
    row_indices = np.where(row_labels == row_cluster)

    if col_cluster not in col_labels:
        raise IndexError('col cluster not in labels')
    col_indices = np.where(col_labels == col_cluster)

    return data_matrix[np.ix_(*row_indices, *col_indices)]


def get_reordered_row_labels(
        data_matrix: NDArray,
        row_labels: NDArray,
        col_labels: NDArray,
        n_clusters
) -> NDArray:
    blocks = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            blocks[i][j] = np.mean(get_submatrix_from_labels(data_matrix, row_labels, col_labels, i, j))
    reordered_row_labels = np.array([np.argmax(row) for row in blocks])
    reorder = np.vectorize(lambda i: reordered_row_labels[i])
    return reorder(row_labels)


def run_n_times(
        algorithm: Callable,
        args: Any,
        n_runs: int
) -> Tuple[NDArray, NDArray]:
    labels: List[Tuple[NDArray, NDArray]] = []
    losses: List[float] = []

    for _ in range(n_runs):
        row_labels, col_labels, loss = algorithm(**args)
        labels.append((row_labels, col_labels))
        losses.append(loss)

    return labels[np.argmin(losses)]
