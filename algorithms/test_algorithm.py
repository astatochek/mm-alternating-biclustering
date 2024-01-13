from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from utils import get_submatrix_from_labels


def calc_mu_and_sigma(x: NDArray, row_labels: NDArray, col_labels: NDArray, n_clusters: int) -> Tuple[NDArray, NDArray]:
    mu = np.zeros((n_clusters, n_clusters))
    sigma = np.zeros((n_clusters, n_clusters))

    for i in range(n_clusters):
        for j in range(n_clusters):
            cluster_submatrix = get_submatrix_from_labels(x, row_labels, col_labels, i, j)
            mu[i][j] = np.mean(cluster_submatrix)
            sigma[i][j] = np.std(cluster_submatrix)

    return mu, sigma


def calc_vector_loss(vec: NDArray, labels: NDArray, n_clusters: int, mu_i: NDArray, sigma_i: NDArray):
    a, b = 1, 1

    vec_mu = np.zeros(n_clusters)
    vec_sigma = np.zeros(n_clusters)

    for i in range(n_clusters):
        sample = vec[labels == i]
        vec_mu[i] = np.mean(sample)
        vec_sigma[i] = np.std(sample)

    return a * np.linalg.norm(vec_mu - mu_i) ** 2 + b * np.linalg.norm(vec_sigma - sigma_i) ** 2


def get_new_labels(x: NDArray, labels: NDArray, n_clusters: int, mu: NDArray, sigma: NDArray) -> NDArray:
    n, _ = x.shape
    new_labels = np.zeros(n, dtype=np.int8)
    for i in range(n):
        vec = x[i]
        loss_vec = [calc_vector_loss(vec, labels, n_clusters, mu[k], sigma[k]) for k in range(n_clusters)]
        new_labels[i] = np.argmin(loss_vec)
    return new_labels


def calc_one_axis_loss(x: NDArray, labels: NDArray, mu: NDArray, sigma: NDArray, n_clusters: int):
    n, _ = x.shape
    return np.sum([calc_vector_loss(x[i], labels, n_clusters, mu[labels[i]], sigma[labels[i]]) for i in range(n)])


def calc_total_loss(x: NDArray, row_labels: NDArray, col_labels: NDArray, n_clusters: int, mu: NDArray, sigma: NDArray):
    row_loss = calc_one_axis_loss(x, col_labels, mu, sigma, n_clusters)
    col_loss = calc_one_axis_loss(x.T, row_labels, mu.T, sigma.T, n_clusters)
    return row_loss + col_loss


def test_algo(data_matrix: NDArray, n_clusters: int, *, eps=1.e-6):
    n_rows, n_cols = data_matrix.shape
    row_labels = np.random.randint(0, n_clusters, n_rows)
    col_labels = np.random.randint(0, n_clusters, n_rows)
    mu, sigma = calc_mu_and_sigma(data_matrix, row_labels, col_labels, n_clusters)
    loss = calc_total_loss(data_matrix, row_labels, col_labels, n_clusters, mu, sigma)

    while True:
        _row_labels = get_new_labels(data_matrix, col_labels, n_clusters, mu, sigma)
        _col_labels = get_new_labels(data_matrix.T, row_labels, n_clusters, mu.T, sigma.T)
        try:
            _mu, _sigma = calc_mu_and_sigma(data_matrix, _row_labels, _col_labels, n_clusters)
            mu, sigma = _mu, _sigma
            row_labels, col_labels = _row_labels, _col_labels
        except IndexError:
            return row_labels, col_labels, loss

        _loss = calc_total_loss(data_matrix, row_labels, col_labels, n_clusters, mu, sigma)
        if np.abs(_loss - loss) <= eps:
            break
        loss = _loss

    return row_labels, col_labels, loss
