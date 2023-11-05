import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List

from sklearn.datasets import make_biclusters
from sklearn.metrics import consensus_score
from tqdm import trange
from generate import Dataset, make_checkerboard_with_custom_distribution, Distribution
from utils import get_biclusters_from_labels


def norm(x: NDArray, indices: NDArray) -> float:
    """dimensionality-reducing norm"""
    return np.sqrt(np.square(x[indices]) / indices.size)


def get_reordered_row_labels(
        data_matrix: NDArray,
        row_labels: NDArray,
        col_labels: NDArray
) -> NDArray:
    blocks = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            blocks[i][j] = np.mean([row[np.where(col_labels == j)] for row in data_matrix[np.where(row_labels == i)]])
    reordered_row_labels = np.array([np.argmax(row) for row in blocks])
    reorder = np.vectorize(lambda i: reordered_row_labels[i])
    return reorder(row_labels)


def k_means(data_matrix: NDArray, n_clusters: int, *, eps=1.e-6):
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
    loss = np.infty
    centroids = maximization_step(data_matrix, labels, n_clusters)

    while True:
        labels = expectation_step(data_matrix, centroids)
        centroids = maximization_step(data_matrix, labels, n_clusters)
        updated_loss = get_loss(data_matrix, labels, centroids)
        if loss - updated_loss < eps:
            break
        loss = updated_loss

    return labels


def alternating_k_means_biclustering(
        data_matrix: NDArray,
        n_clusters: int,
        *,
        eps=1.e-6
) -> Tuple[NDArray, NDArray, float]:
    def update_step(current_matrix: NDArray, preserved_labels: NDArray, processed_labels: NDArray) -> List[NDArray]:
        X = current_matrix
        I, J = preserved_labels, processed_labels
        centroids = []
        for j in range(n_clusters):
            rows_j = X[np.where(J == j)]
            if rows_j.size == 0:
                raise ValueError(f"Cluster {j} is left empty")
            I_j = np.where(I == j)
            centroid_j = np.mean([row[I_j] for row in rows_j], axis=0)
            centroids.append(centroid_j)
        return centroids

    def assignment_step(
            current_matrix: NDArray,
            preserved_labels: NDArray,
            processed_labels: NDArray,
            new_centroids: List[NDArray]
    ) -> NDArray:
        X, I, J, C = current_matrix, preserved_labels, processed_labels, new_centroids
        updated_labels = np.zeros(X.shape[0], dtype=np.int32)
        for i in range(X.shape[0]):
            updated_labels[i] = np.argmin(
                [np.linalg.norm(X[i][np.where(I == j)] - C[j]) ** 2 / I[np.where(I == j)].size for j in
                 range(n_clusters)])
        return updated_labels

    def alternate_iteration(
            current_matrix: NDArray,
            preserved_labels: NDArray,
            processed_labels: NDArray
    ) -> Tuple[float, NDArray]:
        I, J = preserved_labels, processed_labels
        X = current_matrix

        def get_loss(new_centroids: List[NDArray]) -> float:
            C = new_centroids
            loss = 0
            for i in range(X.shape[0]):
                row = X[i]
                norms = np.array([np.linalg.norm(row[np.where(I == j)] - C[j])**2 / I[np.where(I == j)].size
                                  for j in range(n_clusters)])
                idx = np.argmin(norms)
                loss += norms[idx]
            return loss / X.shape[0]

        alt_loss = np.infty

        while True:
            alt_centroids = update_step(X, I, J)
            J = assignment_step(X, I, J, alt_centroids)
            new_loss = get_loss(alt_centroids)
            if alt_loss - new_loss < eps:
                break
            alt_loss = new_loss

        return alt_loss, J

    n_rows, m_cols = data_matrix.shape
    total_loss = np.infty

    # row_labels = k_means(data_matrix, n_clusters)
    # col_labels = k_means(data_matrix.T, n_clusters)
    #
    # row_labels = get_reordered_row_labels(data_matrix, row_labels, col_labels)

    row_labels = np.random.randint(0, n_clusters, n_rows)
    col_labels = np.random.randint(0, n_clusters, m_cols)

    while True:
        try:
            row_loss, row_labels = alternate_iteration(data_matrix, col_labels, row_labels)
            col_loss, col_labels = alternate_iteration(data_matrix.T, row_labels, col_labels)
            if total_loss - (row_loss + col_loss) < eps * 2:
                break
            total_loss = row_loss + col_loss
            print(f"Loss: {total_loss}")
        except ValueError:
            break
    print("End")
    row_labels = get_reordered_row_labels(data_matrix, row_labels, col_labels)
    return row_labels, col_labels, total_loss


fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(11, 4), layout='constrained', dpi=200)

shape = (20, 20)
n_clusters = 3
noise = 5

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

n_runs = 5
labels: List[Tuple[NDArray, NDArray]] = []
losses: List[float] = []
for _ in range(n_runs):
    row_labels, col_labels, loss = alternating_k_means_biclustering(data, n_clusters)
    labels.append((row_labels, col_labels))
    losses.append(loss)
row_labels, col_labels = labels[np.argmin(losses)]

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
