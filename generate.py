from typing import Tuple

import numpy as np
from enum import Enum


class Distribution(Enum):
    POISSONIAN = 1
    GAUSSIAN = 2
    UNIFORM = 3
    MULTINOMIAL = 4


class Dataset(Enum):
    BLOCK_DIAGONAL = 1
    CHECKERBOARD = 2


def make_checkerboard_with_custom_distribution(
        shape,
        n_clusters,
        *,
        noise=0.0,
        distribution: Distribution = Distribution.UNIFORM
):
    if hasattr(n_clusters, "__len__"):
        n_row_clusters, n_col_clusters = n_clusters
    else:
        n_row_clusters = n_col_clusters = n_clusters

    # row and column clusters of approximately equal sizes
    n_rows, n_cols = shape
    row_sizes = np.random.multinomial(
        n_rows, np.repeat(1.0 / n_row_clusters, n_row_clusters)
    )
    col_sizes = np.random.multinomial(
        n_cols, np.repeat(1.0 / n_col_clusters, n_col_clusters)
    )

    row_labels = np.hstack(
        [np.repeat(val, rep)
         for val, rep in zip(range(n_row_clusters), row_sizes)]
    )
    col_labels = np.hstack(
        [np.repeat(val, rep)
         for val, rep in zip(range(n_col_clusters), col_sizes)]
    )

    if distribution == Distribution.MULTINOMIAL:
        means = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            dist = np.random.randint(0, n_clusters, n_clusters) + 1.e-1
            dist = dist / np.sum(dist)
            means[i] = np.random.multinomial(100 * n_clusters, dist)

    result = np.zeros(shape, dtype=np.float64)
    for i in range(n_row_clusters):
        for j in range(n_col_clusters):
            selector = np.outer(row_labels == i, col_labels == j)
            match distribution:
                case Distribution.UNIFORM:
                    result[selector] += np.random.uniform(10, 100)
                case Distribution.GAUSSIAN:
                    result[selector] += np.random.normal(loc=100, scale=30)
                case Distribution.POISSONIAN:
                    result[selector] += np.random.poisson(lam=100)
                case Distribution.MULTINOMIAL:
                    result[selector] += means[i][j]

    if noise > 0:
        result += np.random.normal(scale=noise, size=result.shape)

    rows = np.vstack(
        [
            row_labels == label
            for label in range(n_row_clusters)
            for _ in range(n_col_clusters)
        ]
    )
    cols = np.vstack(
        [
            col_labels == label
            for _ in range(n_row_clusters)
            for label in range(n_col_clusters)
        ]
    )

    return result, rows, cols


def make_biclusters_simulation_1(shape: Tuple[int, int]):
    n_row_clusters, n_col_clusters = 2, 2

    n_rows, n_cols = shape

    row_sizes = np.random.multinomial(
        n_rows, np.array([.3, .7])
    )
    col_sizes = np.random.multinomial(
        n_cols, np.array([.3, .7])
    )

    row_labels = np.hstack(
        [np.repeat(val, rep)
         for val, rep in zip(range(n_row_clusters), row_sizes)]
    )
    col_labels = np.hstack(
        [np.repeat(val, rep)
         for val, rep in zip(range(n_col_clusters), col_sizes)]
    )
    b = 2.5
    result = np.zeros(shape, dtype=np.float64)
    for i_ in range(n_rows):
        for j_ in range(n_cols):
            i = row_labels[i_]
            j = col_labels[j_]
            if i == 0 and j == 0:
                result[i_][j_] += np.random.normal(loc=b * .36, scale=1)
            if i == 0 and j == 1:
                result[i_][j_] += np.random.normal(loc=b * .90, scale=1)
            if i == 1 and j == 0:
                result[i_][j_] += np.random.normal(loc=b * -.58, scale=1)
            if i == 1 and j == 1:
                result[i_][j_] -= np.random.normal(loc=b * -.06, scale=1)
    # for i in range(n_row_clusters):
    #     for j in range(n_col_clusters):
    #         selector = np.outer(row_labels == i, col_labels == j)
    #         if i == 0 and j == 0:
    #             result[selector] += np.random.normal(loc=.36, scale=1)
    #         if i == 0 and j == 1:
    #             result[selector] += np.random.normal(loc=.90, scale=1)
    #         if i == 1 and j == 0:
    #             result[selector] += np.random.normal(loc=-.58, scale=1)
    #         if i == 1 and j == 1:
    #             result[selector] -= np.random.normal(loc=-.06, scale=1)


    # if noise > 0:
    #     result += np.random.normal(scale=noise, size=result.shape)

    rows = np.vstack(
        [
            row_labels == label
            for label in range(n_row_clusters)
            for _ in range(n_col_clusters)
        ]
    )
    cols = np.vstack(
        [
            col_labels == label
            for _ in range(n_row_clusters)
            for label in range(n_col_clusters)
        ]
    )

    return result, rows, cols