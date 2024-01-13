import numpy as np
from numpy.typing import NDArray
from utils import get_submatrix_from_labels
from typing import Tuple, Literal


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


def calc_proportion_vector(labels: NDArray, n_clusters: int) -> NDArray:
    proportion_vector = np.zeros(n_clusters)
    for k in range(n_clusters):
        proportion_vector[k] = proportion(labels, k)
    return proportion_vector


def calc_cluster_sum_and_size(
        data_matrix: NDArray,
        row_labels: NDArray, col_labels: NDArray,
        row_cluster: int, col_cluster: int
) -> Tuple[float, int]:
    submatrix = get_submatrix_from_labels(
        data_matrix, row_labels, col_labels, row_cluster, col_cluster
    )
    return np.sum(submatrix), submatrix.size


def calc_confusion_matrix(
        row_proportion_vector: NDArray,
        col_proportion_vector: NDArray,
        mu: NDArray,
        n_clusters: int
) -> NDArray:
    confusion_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            confusion_matrix[i][j] = row_proportion_vector[i] * col_proportion_vector[j] * mu[i][j]
    return confusion_matrix


def updated_confusion_matrix(
        row_labels: NDArray,
        col_labels: NDArray,
        row_proportion_vector: NDArray,
        col_proportion_vector: NDArray,
        sum_matrix: NDArray,
        count_matrix: NDArray,
        n_clusters: int,
        assignment_vector: NDArray,
        assignment_type: Literal['row', 'col'],
        assignment_from_idx: int,
        assignment_to_cluster_idx: int
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:

    new_sum_matrix = np.copy(sum_matrix)
    new_count_matrix = np.copy(count_matrix)

    n_rows = row_labels.size
    n_cols = col_labels.size

    match assignment_type:
        case 'row':
            sums = np.array([np.sum(assignment_vector[np.where(row_labels == i)]) for i in range(n_clusters)])
            counts = np.array([np.count_nonzero(row_labels == i) for i in range(n_clusters)])
            from_cluster_idx = row_labels[assignment_from_idx]

            for k in range(n_clusters):
                new_count_matrix[from_cluster_idx][k] -= counts[k]
                new_sum_matrix[from_cluster_idx][k] -= sums[k]
                new_count_matrix[assignment_to_cluster_idx][k] += counts[k]
                new_sum_matrix[assignment_to_cluster_idx][k] += sums[k]

                if new_count_matrix[from_cluster_idx][k] == 0.:
                    raise ZeroDivisionError

            mu = new_sum_matrix / new_count_matrix

            new_row_proportion_vector = np.copy(row_proportion_vector)
            new_row_proportion_vector[from_cluster_idx] -= 1 / n_rows
            new_row_proportion_vector[assignment_to_cluster_idx] += 1 / n_rows

            new_confusion_matrix = calc_confusion_matrix(
                new_row_proportion_vector, col_proportion_vector, mu, n_clusters
            )

            return (
                new_confusion_matrix,
                new_row_proportion_vector, col_proportion_vector,
                new_sum_matrix, new_count_matrix
            )

        case 'col':
            sums = np.array([np.sum(assignment_vector[np.where(col_labels == i)]) for i in range(n_clusters)])
            counts = np.array([np.count_nonzero(col_labels == i) for i in range(n_clusters)])
            from_cluster_idx = col_labels[assignment_from_idx]

            for k in range(n_clusters):
                new_count_matrix[k][from_cluster_idx] -= counts[k]
                new_sum_matrix[k][from_cluster_idx] -= sums[k]
                new_count_matrix[k][assignment_to_cluster_idx] += counts[k]
                new_sum_matrix[k][assignment_to_cluster_idx] += sums[k]

                if new_count_matrix[k][from_cluster_idx] == 0.:
                    raise ZeroDivisionError

            mu = new_sum_matrix / new_count_matrix

            new_col_proportion_vector = np.copy(col_proportion_vector)
            new_col_proportion_vector[from_cluster_idx] -= 1 / n_cols
            new_col_proportion_vector[assignment_to_cluster_idx] += 1 / n_cols

            new_confusion_matrix = calc_confusion_matrix(
                row_proportion_vector, new_col_proportion_vector, mu, n_clusters
            )

            return (
                new_confusion_matrix,
                row_proportion_vector, new_col_proportion_vector,
                new_sum_matrix, new_count_matrix
            )


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

    row_proportion_vector = calc_proportion_vector(row_labels, n_clusters)
    col_proportion_vector = calc_proportion_vector(col_labels, n_clusters)

    sum_matrix = np.zeros((n_clusters, n_clusters))
    count_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int32)

    for i in range(n_clusters):
        for j in range(n_clusters):
            _sum, size = calc_cluster_sum_and_size(data_matrix, row_labels, col_labels, i, j)
            sum_matrix[i][j] = _sum
            count_matrix[i][j] = size

    cur_criterion = -1
    new_cur_criterion = criterion(data_matrix, row_labels, col_labels, n_clusters)

    while True:
        print(new_cur_criterion)
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
                        new_confusion_matrix, *_ = updated_confusion_matrix(
                            row_labels=row_labels,
                            col_labels=col_labels,
                            row_proportion_vector=row_proportion_vector,
                            col_proportion_vector=col_proportion_vector,
                            sum_matrix=sum_matrix,
                            count_matrix=count_matrix,
                            n_clusters=n_clusters,
                            assignment_vector=data_matrix[i],
                            assignment_type='row',
                            assignment_from_idx=i,
                            assignment_to_cluster_idx=k
                        )
                        criterion_list[k] = np.sum(new_confusion_matrix)
                    except ZeroDivisionError:
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
                        new_confusion_matrix, *_ = updated_confusion_matrix(
                            row_labels=row_labels,
                            col_labels=col_labels,
                            row_proportion_vector=row_proportion_vector,
                            col_proportion_vector=col_proportion_vector,
                            sum_matrix=sum_matrix,
                            count_matrix=count_matrix,
                            n_clusters=n_clusters,
                            assignment_vector=data_matrix.T[j],
                            assignment_type='col',
                            assignment_from_idx=j,
                            assignment_to_cluster_idx=k
                        )
                        criterion_list[k] = np.sum(new_confusion_matrix)
                    except ZeroDivisionError:
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

        if assignment_list == []:
            break

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
                (
                    confusion_matrix,
                    row_proportion_vector, col_proportion_vector,
                    sum_matrix, count_matrix
                ) = updated_confusion_matrix(
                    row_labels=row_labels,
                    col_labels=col_labels,
                    row_proportion_vector=row_proportion_vector,
                    col_proportion_vector=col_proportion_vector,
                    sum_matrix=sum_matrix,
                    count_matrix=count_matrix,
                    n_clusters=n_clusters,
                    assignment_vector=data_matrix[idx],
                    assignment_type='row',
                    assignment_from_idx=idx,
                    assignment_to_cluster_idx=assignment_label
                )
                row_labels[idx] = assignment_label
                new_cur_criterion = np.sum(confusion_matrix)
            case b"col":
                (
                    confusion_matrix,
                    row_proportion_vector, col_proportion_vector,
                    sum_matrix, count_matrix
                ) = updated_confusion_matrix(
                    row_labels=row_labels,
                    col_labels=col_labels,
                    row_proportion_vector=row_proportion_vector,
                    col_proportion_vector=col_proportion_vector,
                    sum_matrix=sum_matrix,
                    count_matrix=count_matrix,
                    n_clusters=n_clusters,
                    assignment_vector=data_matrix.T[idx],
                    assignment_type='col',
                    assignment_from_idx=idx,
                    assignment_to_cluster_idx=assignment_label
                )
                col_labels[idx] = assignment_label
                new_cur_criterion = np.sum(confusion_matrix)


    final_criterion = criterion(data_matrix, row_labels, col_labels, n_clusters)
    # print(final_criterion, row_labels, col_labels)

    return (
        row_labels,
        col_labels,
        -final_criterion,
    )
