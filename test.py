# from generate import Distribution
# from enum import Enum
#
#
# val = Distribution.POISSONIAN
#
# print(val == Distribution.POISSONIAN)
#
#
# class Color(Enum):
#     RED = 1
#     GREEN = 2
#     BLUE = 3
#
# color = Color.RED
#
#

import numpy as np
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist


def akmbiclust(X, k, lambda_val=0, nstart=1):
    n = X.shape[0]
    m = X.shape[1]
    best_row_labels = np.zeros(n)
    best_col_labels = np.zeros(m)
    min_loss = np.inf
    X0 = X.copy()

    for t in range(nstart):
        flag = 0
        row_perm = np.random.permutation(n)
        col_perm = np.random.permutation(m)
        row_perm_reverse = np.argsort(row_perm)
        col_perm_reverse = np.argsort(col_perm)
        X = X0[row_perm, col_perm]

        row_labels, _ = kmeans(X, k, iter=20)
        col_labels, _ = kmeans(X.T, k, iter=20)

        row_centers = [np.mean(X[row_labels == i, col_labels == i], axis=0) for i in range(k)]
        col_centers = [np.mean(X[row_labels == i, col_labels == i], axis=1) for i in range(k)]

        loss = compute_loss(X, row_labels, col_labels, lambda_val)['loss']

        if loss < min_loss:
            min_loss = loss
            best_row_labels = row_labels[row_perm_reverse]
            best_col_labels = col_labels[col_perm_reverse]

        total_iter = 0

        while total_iter < 50 and flag != 1:
            total_iter += 1
            col_iter = 0
            col_delta = 1

            while col_iter < 50 and col_delta > 0.001:
                col_iter += 1
                col_centers_vec = centers2vec(col_centers, col_labels)
                distance_matrix = np.inf * np.ones((n, k))
                X_mean = np.tile(col_centers_vec, (n, 1))
                X_temp = (X - X_mean) ** 2

                for i in range(k):
                    distance_matrix[:, i] = np.sum(X_temp[:, col_labels == i], axis=1) / np.sum(col_labels == i)

                row_labels = np.argmax(-distance_matrix, axis=1)

                if i not in row_labels:
                    flag = 1

                if flag == 1:
                    break

                new_col_centers = [np.mean(X[row_labels == i, col_labels == i], axis=0) for i in range(k)]
                col_delta = dist_centers(col_centers, new_col_centers) / norm_centers(col_centers)
                col_centers = new_col_centers

            if flag == 1:
                break

            row_centers = [np.mean(X[row_labels == i, col_labels == i], axis=1) for i in range(k)]
            X_tran = X.T
            row_iter = 0
            row_delta = 1

            while row_iter < 50 and row_delta > 0.001:
                row_iter += 1
                row_centers_vec = centers2vec(row_centers, row_labels)
                distance_matrix = np.inf * np.ones((m, k))
                X_mean = np.tile(row_centers_vec, (m, 1))
                X_temp = (X_tran - X_mean) ** 2

                for i in range(k):
                    distance_matrix[:, i] = np.sum(X_temp[:, row_labels == i], axis=1) / np.sum(row_labels == i)

                col_labels = np.argmax(-distance_matrix, axis=1)

                if i not in col_labels:
                    flag = 1

                if flag == 1:
                    break

                new_row_centers = [np.mean(X_tran[col_labels == i, row_labels == i], axis=0) for i in range(k)]
                row_delta = dist_centers(row_centers, new_row_centers) / norm_centers(row_centers)
                row_centers = new_row_centers

            if flag == 1:
                break

            col_centers = [np.mean(X[row_labels == i, col_labels == i], axis=1) for i in range(k)]
            new_loss = compute_loss(X, row_labels, col_labels, lambda_val)['loss']
            loss_delta = np.abs(loss - new_loss) / loss
            loss = new_loss

            if loss_delta < 0.01:
                flag = 1

        if loss < min_loss:
            min_loss = loss
            best_row_labels = row_labels[row_perm_reverse]
            best_col_labels = col_labels[col_perm_reverse]

    return {'row_labels': best_row_labels, 'col_labels': best_col_labels, 'loss': min_loss}



