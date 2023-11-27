import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List
import pandas as pd

from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_biclusters
from sklearn.metrics import consensus_score
from tqdm import trange
from generate import Dataset, make_checkerboard_with_custom_distribution, Distribution, make_biclusters_simulation
from utils import get_biclusters_from_labels, get_submatrix_from_labels, get_reordered_row_labels, run_n_times
from algorithms.kmeans import k_means


# def norm(x: NDArray, indices: NDArray) -> float:
#     """dimensionality-reducing norm"""
#     return np.sqrt(np.square(x[indices]) / indices.size)

def get_loss(X: NDArray, C: List[NDArray], I: NDArray, J: NDArray, k: int) -> float:
    n_rows, _ = X.shape
    loss = 0.
    I_ = [np.where(I == j) for j in range(k)]
    for i in range(n_rows):
        j = J[i]
        loss += np.linalg.norm(X[i][I_[j]] - C[j]) ** 2 / C[j].size
    return loss / n_rows


def get_penalized_loss(X: NDArray, C: List[NDArray], I: NDArray, J: NDArray, k: int, lamda: float) -> float:
    loss = get_loss(X, C, I, J, k)
    X_F = np.linalg.norm(X) ** 2
    for j in range(1, k):
        try: loss += lamda * X_F / (np.linalg.norm(get_submatrix_from_labels(X, J, I, j, j)) ** 2 + 1)
        except IndexError: return np.inf
    return loss


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
    n_rows, _ = X.shape
    I_ = [np.where(I == j) for j in range(k)]
    for i in range(n_rows):
        norms = np.array([np.linalg.norm(X[i][I_[j]] - C[j]) ** 2 / C[j].size for j in range(k)])
        idx = np.argmin(norms)
        updated_labels[i] = idx
    return updated_labels


def alternate_iteration(X: NDArray, I: NDArray, J: NDArray, k: int, eps: float, lamda: float) -> Tuple[float, NDArray]:
    continue_flag = True
    try:
        C = update_step(X, I, J, k)
        loss = get_loss(X, C, I, J, k)
    except IndexError: return np.inf, J

    while continue_flag:
        try:
            C = update_step(X, I, J, k)
            J = assignment_step(X, I, np.copy(J), C, k)
            new_loss = get_loss(X, C, I, J, k)
            if np.abs(loss - new_loss) < eps:
                continue_flag = False
            loss = new_loss
            # print(f"Loss inter: {loss}")
        except IndexError:
            continue_flag = False
    # print("End inter")
    loss = get_penalized_loss(X, C, I, J, k, lamda)
    return loss, J


def alternating_k_means_biclustering(
        data_matrix: NDArray,
        n_clusters: int,
        *,
        lamda=0.,
        eps=1.e-6
) -> Tuple[NDArray, NDArray, float]:
    total_loss = np.inf

    row_labels, _ = k_means(data_matrix, n_clusters)
    col_labels, _ = k_means(data_matrix.T, n_clusters)

    # try: row_labels = get_reordered_row_labels(data_matrix, row_labels, col_labels, n_clusters)
    # except IndexError: return row_labels, col_labels, total_loss, row_labels, col_labels

    # init_row_labels, init_col_labels = np.copy(row_labels), np.copy(col_labels)
    # try: init_row_labels = get_reordered_row_labels(data_matrix, init_row_labels, init_col_labels, n_clusters)
    # except IndexError: return row_labels, col_labels, total_loss

    while True:
        row_loss, row_labels = alternate_iteration(data_matrix, col_labels, np.copy(row_labels), n_clusters, eps / 2, lamda)
        col_loss, col_labels = alternate_iteration(data_matrix.T, row_labels, np.copy(col_labels), n_clusters, eps / 2, lamda)
        try: row_labels = get_reordered_row_labels(data_matrix, row_labels, col_labels, n_clusters)
        except: break

        if col_loss + row_loss == np.inf: break
        if np.abs(total_loss - (row_loss + col_loss)) < eps: break

        total_loss = row_loss + col_loss
    #     print(f"Loss total: {total_loss}")
    # print("End total")
    try: row_labels = get_reordered_row_labels(data_matrix, row_labels, col_labels, n_clusters)
    finally: return row_labels, col_labels, total_loss


def show(shape: Tuple[int, int], n_clusters: int, noise: int, n_runs: int):
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(11, 4), layout='constrained', dpi=200)

    # generate data matrix with cluster assignments
    # data, rows, cols = make_biclusters(
    #     shape=shape, n_clusters=n_clusters, noise=noise, shuffle=False)

    # data, rows, cols = make_biclusters_simulation(
    #     shape=shape,
    #     M=np.zeros((2, 2)),
    #     S=np.array([[1 + .3, 1.], [1., 1 + .3]]),
    #     n_clusters=2,
    #     sizes=np.array([.3, .7])
    # )

    b = .5
    data, rows, cols = make_biclusters_simulation(
        shape=shape,
        M=b * np.array([[.36, .90], [-.58, -.06]]),
        S=np.array([[1. + b, 1.], [1., 1. + b]]),
        n_clusters=2,
        sizes=np.array([.3, .7])
    )

    # show original dataset with visible clusters
    ax1.matshow(data, cmap=plt.cm.Reds)
    ax1.set_title("Original dataset")

    # shuffle data
    rng = np.random.RandomState(0)
    row_idx = rng.permutation(data.shape[0])
    col_idx = rng.permutation(data.shape[1])
    data = data[row_idx][:, col_idx]

    # show shuffled data
    ax2.matshow(data, cmap=plt.cm.Reds)
    ax2.set_title("Shuffled dataset")


    row_labels, col_labels = run_n_times(
        algorithm=alternating_k_means_biclustering,
        args={
            "data_matrix": data,
            "n_clusters": 2,
            "lamda": 0
        },
        n_runs=10
    )

    # reorder rows and cols of a data matrix to show clusters
    fit_data = data[np.argsort(row_labels)]
    fit_data = fit_data[:, np.argsort(col_labels)]

    # show data matrix with reordered rows and cols according to calculated cluster assignments
    ax3.matshow(fit_data, cmap=plt.cm.Reds)
    ax3.set_title(f"Alternating KMeans")

    # calculate consensus score between expected and actual biclusters
    score = consensus_score(get_biclusters_from_labels(shape, n_clusters, row_labels, col_labels),
                            (rows[:, row_idx], cols[:, col_idx]))

    print(f"Consensus Score: {score:.3f}")

    plt.show()





# def run(shape: Tuple[int, int], n_clusters: int, noise: int, n_runs: int):
#
#     data, rows, cols = make_biclusters(
#         shape=shape, n_clusters=n_clusters, noise=noise, shuffle=False)
#
#     # shuffle data
#     rng = np.random.RandomState(0)
#     row_idx = rng.permutation(data.shape[0])
#     col_idx = rng.permutation(data.shape[1])
#     data = data[row_idx][:, col_idx]
#
#     kms_scores, alt_scores = [0] * n_runs, [0] * n_runs
#
#     for i in range(n_runs):
#         row_labels, col_labels, loss = alternating_k_means_biclustering(data, n_clusters)
#
#         alt_scores[i] = consensus_score(get_biclusters_from_labels(shape, n_clusters, row_labels, col_labels),
#                                 (rows[:, row_idx], cols[:, col_idx]))
#
#         kms_scores[i] = consensus_score(get_biclusters_from_labels(shape, n_clusters, init_row_labels, init_col_labels),
#                                 (rows[:, row_idx], cols[:, col_idx]))
#
#     return {"ALT": alt_scores, "KMS": kms_scores}


show(
    shape=(400, 400),
    n_clusters=2,
    noise=10,
    n_runs=10
)
# df = pd.DataFrame(
#     run(
#         shape=(300, 300),
#         n_clusters=4,
#         noise=10,
#         n_runs=10
#     )
# )
# print(df)
