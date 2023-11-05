from typing import Tuple
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
