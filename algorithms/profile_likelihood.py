import numpy as np
from numpy.typing import NDArray
from utils import get_submatrix_from_labels
from typing import Tuple


def entropy(size: int, sum_: float) -> float:
    """Gaussian"""
    if size <= 0:
        return 0
    mean = sum_ / size
    # return 0.5 * sum * mean
    return mean ** 2


def proportion(labels: NDArray, cluster_idx: int) -> float:
    size = labels.size
    count = np.count_nonzero(labels == cluster_idx)
    return count / size


def criterion(
    data_matrix: NDArray, row_labels: NDArray, col_labels: NDArray, n_clusters: int
) -> float:
    res = 0.0
    for row_cluster_idx in range(n_clusters):
        for col_cluster_idx in range(n_clusters):
            submatrix = get_submatrix_from_labels(
                data_matrix, row_labels, col_labels, row_cluster_idx, col_cluster_idx
            )
            size = submatrix.size
            sum = np.sum(submatrix)
            res += (
                proportion(row_labels, row_cluster_idx)
                * proportion(col_labels, col_cluster_idx)
                * entropy(size, sum)
            )
    return res


def profile_likelihood_biclustering(
    data_matrix: NDArray, n_clusters: int
) -> Tuple[NDArray, NDArray, float]:
    n_rows, n_cols = data_matrix.shape

    row_labels = np.random.randint(0, n_clusters, n_rows)
    col_labels = np.random.randint(0, n_clusters, n_cols)

    assignment_dtype = [
        ("type", "S3"),
        ("from_idx", int),
        ("to_cluster_idx", int),
        ("criterion_eval", float),
    ]

    cur_criterion = -1

    while True:

        new_cur_criterion = criterion(data_matrix, row_labels, col_labels, n_clusters)
        if new_cur_criterion - cur_criterion <= 1e-4:
            break

        cur_criterion = new_cur_criterion

        assignment_list = []

        for i in range(n_rows):
            row_label = row_labels[i]
            criterion_list = np.zeros(n_clusters)

            for k in range(n_clusters):
                if k != row_label:
                    temp_row_labels = np.copy(row_labels)
                    temp_row_labels[i] = k
                    try:
                        criterion_list[k] = criterion(
                            data_matrix, temp_row_labels, col_labels, n_clusters
                        )
                    except:
                        criterion_list[k] = -1
                else:
                    criterion_list[k] = cur_criterion

            optimal_assignment_idx = np.argmax(criterion_list)
            assignment_list.append(
                (
                    "row",
                    i,
                    optimal_assignment_idx,
                    criterion_list[optimal_assignment_idx],
                )
            )

        for j in range(n_cols):
            col_label = col_labels[j]
            criterion_list = np.zeros(n_clusters)

            for k in range(n_clusters):
                if k != col_label:
                    temp_col_labels = np.copy(col_labels)
                    temp_col_labels[j] = k
                    try:
                        criterion_list[k] = criterion(
                            data_matrix, row_labels, temp_col_labels, n_clusters
                        )
                    except:
                        criterion_list[k] = -1
                else:
                    criterion_list[k] = cur_criterion

            optimal_assignment_idx = np.argmax(criterion_list)
            assignment_list.append(
                (
                    "col",
                    j,
                    optimal_assignment_idx,
                    criterion_list[optimal_assignment_idx],
                )
            )

        sequence = np.array(assignment_list, dtype=assignment_dtype)
        sequence = np.flip(
            np.sort(sequence, order=["criterion_eval"], kind="quicksort")
        )

        s_row_labels = np.copy(row_labels)
        s_col_labels = np.copy(col_labels)

        s_criterion_list = []

        for assignment in sequence:
            assignment_type, idx, assignment_label, _ = assignment

            match assignment_type:
                case b"row":
                    s_row_labels[idx] = assignment_label
                case b"col":
                    s_col_labels[idx] = assignment_label

            try:
                s_criterion = criterion(
                    data_matrix, s_row_labels, s_col_labels, n_clusters
                )
            except:
                break

            s_criterion_list.append(s_criterion)

        best_idx = np.argmax(s_criterion_list)
        assignment_type, idx, assignment_label, _ = sequence[best_idx]
        match assignment_type:
            case b"row":
                row_labels[idx] = assignment_label
            case b"col":
                col_labels[idx] = assignment_label

    final_criterion = criterion(data_matrix, row_labels, col_labels, n_clusters)
    # print(final_criterion, row_labels, col_labels)

    return (
        row_labels,
        col_labels,
        -final_criterion,
    )
