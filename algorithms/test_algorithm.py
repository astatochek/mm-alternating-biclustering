from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from utils import get_submatrix_from_labels, get_reordered_row_labels


def calc_mu_and_sigma(x: NDArray, row_labels: NDArray, col_labels: NDArray, n_clusters: int) -> Tuple[
    NDArray, NDArray
]:
    mu = np.zeros((n_clusters, n_clusters))
    sigma = np.zeros((n_clusters, n_clusters))

    for i in range(n_clusters):
        for j in range(n_clusters):
            cluster_submatrix = get_submatrix_from_labels(x, row_labels, col_labels, i, j)
            mu[i][j] = np.mean(cluster_submatrix)
            sigma[i][j] = np.std(cluster_submatrix)

    return mu, sigma


def calc_vector_loss(vec: NDArray, labels: NDArray, n_clusters: int, mu_k: NDArray, sigma_k: NDArray):
    a, b = 1, 1

    vec_mu = np.zeros(n_clusters)
    vec_sigma = np.zeros(n_clusters)

    for i in range(n_clusters):
        sample = vec[labels == i]
        vec_mu[i] = np.mean(sample)
        vec_sigma[i] = np.std(sample)

    return a * np.linalg.norm(vec_mu - mu_k) ** 2 + b * np.linalg.norm(vec_sigma - sigma_k) ** 2


def get_new_labels(x: NDArray, labels: NDArray, n_clusters: int, mu: NDArray, sigma: NDArray, iter: int) -> NDArray:
    n, _ = x.shape
    new_labels = np.zeros(n, dtype=np.int8)
    for i in range(n):
        vec = x[i]
        loss_vec = np.array([calc_vector_loss(vec, labels, n_clusters, mu[k], sigma[k]) for k in range(n_clusters)])
        if np.random.rand() > iter / 10:
            inv = 1 / (loss_vec + .1)
            p = inv / np.sum(inv)
            new_labels[i] = np.argmax(np.random.multinomial(1, p))
        else:
            new_labels[i] = np.argmin(loss_vec)
    return new_labels


def calc_one_axis_loss(x: NDArray, assigned_labels: NDArray, preserved_labels: NDArray, mu: NDArray, sigma: NDArray, n_clusters: int):
    n, _ = x.shape
    return np.sum(
        [calc_vector_loss(x[i], preserved_labels, n_clusters, mu[assigned_labels[i]], sigma[assigned_labels[i]]) for i in
         range(n)])


def calc_total_loss(x: NDArray, row_labels: NDArray, col_labels: NDArray, n_clusters: int, mu: NDArray, sigma: NDArray):
    row_loss = calc_one_axis_loss(x, row_labels, col_labels, mu, sigma, n_clusters)
    col_loss = calc_one_axis_loss(x.T, col_labels, row_labels, mu.T, sigma.T, n_clusters)
    return row_loss + col_loss


def test_algo(data_matrix: NDArray, n_clusters: int, *, eps=1.e-6):
    n_rows, n_cols = data_matrix.shape
    row_labels = np.random.randint(0, n_clusters, n_rows)
    col_labels = np.random.randint(0, n_clusters, n_cols)
    mu, sigma = calc_mu_and_sigma(data_matrix, row_labels, col_labels, n_clusters)
    loss = calc_total_loss(data_matrix, row_labels, col_labels, n_clusters, mu, sigma)

    i = 1
    while True:
        _row_labels = get_new_labels(data_matrix, col_labels, n_clusters, mu, sigma, i)
        _col_labels = get_new_labels(data_matrix.T, row_labels, n_clusters, mu.T, sigma.T, i)
        try:
            _mu, _sigma = calc_mu_and_sigma(data_matrix, _row_labels, _col_labels, n_clusters)
            mu, sigma = _mu, _sigma
            row_labels, col_labels = _row_labels, _col_labels
        except IndexError:
            return row_labels, col_labels, loss

        _loss = calc_total_loss(data_matrix, row_labels, col_labels, n_clusters, mu, sigma)
        if np.abs(_loss - loss) <= eps or i > 50:
            break
        loss = _loss
        i += 1
        # print(f"Iter: {i}, Loss: {loss}")

    row_labels = get_reordered_row_labels(data_matrix, row_labels, col_labels, n_clusters)
    return row_labels, col_labels, loss
