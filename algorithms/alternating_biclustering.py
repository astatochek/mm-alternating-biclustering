import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List
import pandas as pd

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
    n_rows, _ = X.shape
    loss = 0.
    I_ = [np.where(I == j) for j in range(k)]
    for i in range(n_rows):
        j = J[i]
        loss += np.linalg.norm(X[i][I_[j]] - C[j]) ** 2 / C[j].size
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
    n_rows, _ = X.shape
    I_ = [np.where(I == j) for j in range(k)]
    for i in range(n_rows):
        norms = np.array([np.linalg.norm(X[i][I_[j]] - C[j]) ** 2 / C[j].size for j in range(k)])
        idx = np.argmin(norms)
        updated_labels[i] = idx
    return updated_labels


def alternate_iteration(X: NDArray, I: NDArray, J: NDArray, k: int, eps: float) -> Tuple[float, NDArray]:
    loss = np.inf
    continue_flag = True
    while continue_flag:
        try:
            C = update_step(X, I, J, k)
            J = assignment_step(X, I, np.copy(J), C, k)
            new_loss = get_loss(X, C, I, J, k)
            if np.abs(loss - new_loss) < eps:
                continue_flag = False
            loss = new_loss
            print(f"Loss inter: {loss}")
        except IndexError:
            continue_flag = False
    print("End inter")
    return loss, J


def alternating_k_means_biclustering(
        data_matrix: NDArray,
        n_clusters: int,
        *,
        eps=1.e-6
) -> Tuple[NDArray, NDArray, float, NDArray, NDArray]:
    total_loss = np.inf

    row_labels, _ = k_means(data_matrix, n_clusters)
    # row_labels = np.random.randint(0, n_clusters, data_matrix.shape[0])
    col_labels, _ = k_means(data_matrix.T, n_clusters)

    # try: row_labels = get_reordered_row_labels(data_matrix, row_labels, col_labels, n_clusters)
    # except IndexError: return row_labels, col_labels, total_loss, row_labels, col_labels

    init_row_labels, init_col_labels = np.copy(row_labels), np.copy(col_labels)
    try: init_row_labels = get_reordered_row_labels(data_matrix, init_row_labels, init_col_labels, n_clusters)
    except IndexError: return row_labels, col_labels, total_loss, row_labels, col_labels

    while True:
        row_loss, row_labels = alternate_iteration(data_matrix, col_labels, np.copy(row_labels), n_clusters, eps / 2)
        col_loss, col_labels = alternate_iteration(data_matrix.T, row_labels, np.copy(col_labels), n_clusters, eps / 2)
        try: row_labels = get_reordered_row_labels(data_matrix, row_labels, col_labels, n_clusters)
        except: break

        if col_loss + row_loss == np.inf: break
        if np.abs(total_loss - (row_loss + col_loss)) < eps: break

        total_loss = row_loss + col_loss
        print(f"Loss total: {total_loss}")
    print("End total")
    try: row_labels = get_reordered_row_labels(data_matrix, row_labels, col_labels, n_clusters)
    finally: return row_labels, col_labels, total_loss, init_row_labels, init_col_labels


def show(shape: Tuple[int, int], n_clusters: int, noise: int, n_runs: int):
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(11, 4), layout='constrained', dpi=200)

    # generate data matrix with cluster assignments
    data, rows, cols = make_biclusters(
        shape=shape, n_clusters=n_clusters, noise=noise, shuffle=False)

    # show original dataset with visible clusters
    ax1.matshow(data, cmap=plt.cm.Blues)
    ax1.set_title("Original dataset")

    # shuffle data
    rng = np.random.RandomState(0)
    row_idx = rng.permutation(data.shape[0])
    col_idx = rng.permutation(data.shape[1])
    data = data[row_idx][:, col_idx]

    # show shuffled data
    ax2.matshow(data, cmap=plt.cm.Blues)
    ax2.set_title("Shuffled dataset")

    # calculate cluster assignments using an iterative algorithm
    labels: List[Tuple[NDArray, NDArray]] = []
    scores: List[float] = []
    for _ in range(n_runs):
        row_labels, col_labels, loss, __row_labels, __col_labels = alternating_k_means_biclustering(data, n_clusters)
        score = consensus_score(get_biclusters_from_labels(shape, n_clusters, row_labels, col_labels),
                        (rows[:, row_idx], cols[:, col_idx]))
        labels.append((row_labels, col_labels))
        scores.append(score)
    row_labels, col_labels = labels[np.argmax(scores)]
    print(scores)

    # reorder rows and cols of a data matrix to show clusters
    fit_data = data[np.argsort(row_labels)]
    fit_data = fit_data[:, np.argsort(col_labels)]

    # show data matrix with reordered rows and cols according to calculated cluster assignments
    ax3.matshow(fit_data, cmap=plt.cm.Blues)
    ax3.set_title(f"Alternating KMeans")

    # calculate consensus score between expected and actual biclusters
    score = consensus_score(get_biclusters_from_labels(shape, n_clusters, row_labels, col_labels),
                            (rows[:, row_idx], cols[:, col_idx]))

    print(f"Consensus Score: {score:.3f}")

    plt.show()


def run(shape: Tuple[int, int], n_clusters: int, noise: int, n_runs: int):

    data, rows, cols = make_biclusters(
        shape=shape, n_clusters=n_clusters, noise=noise, shuffle=False)

    # shuffle data
    rng = np.random.RandomState(0)
    row_idx = rng.permutation(data.shape[0])
    col_idx = rng.permutation(data.shape[1])
    data = data[row_idx][:, col_idx]

    kms_scores, alt_scores = [0] * n_runs, [0] * n_runs

    for i in range(n_runs):
        row_labels, col_labels, loss, init_row_labels, init_col_labels = alternating_k_means_biclustering(data, n_clusters)

        alt_scores[i] = consensus_score(get_biclusters_from_labels(shape, n_clusters, row_labels, col_labels),
                                (rows[:, row_idx], cols[:, col_idx]))

        kms_scores[i] = consensus_score(get_biclusters_from_labels(shape, n_clusters, init_row_labels, init_col_labels),
                                (rows[:, row_idx], cols[:, col_idx]))

    return {"ALT": alt_scores, "KMS": kms_scores}


show(
    shape=(1000, 1000),
    n_clusters=4,
    noise=10,
    n_runs=10
)
# df = pd.DataFrame(
#     run(
#         shape=(1000, 1000),
#         n_clusters=4,
#         noise=10,
#         n_runs=10
#     )
# )
# print(df)
