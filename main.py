from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from sklearn.datasets import make_biclusters
from sklearn.metrics import consensus_score
from tqdm import tqdm, trange

from algorithms.alternating_biclustering import alternating_k_means_biclustering
from algorithms.kmeans import k_means_biclustering
from utils import run_n_times, get_biclusters_from_labels


def mean_scores(
        n_initializations: int,
        shape: Tuple[int, int],
        n_clusters: int,
        *,
        noise: int = 10
):
    scores: Dict[str, List[float]] = {'AKM': [], 'KM': []}
    data, rows, cols = make_biclusters(shape=shape, n_clusters=n_clusters, noise=noise, shuffle=False)

    rng = np.random.RandomState(0)
    row_idx = rng.permutation(data.shape[0])
    col_idx = rng.permutation(data.shape[1])
    data = data[row_idx][:, col_idx]

    for _ in trange(n_initializations):
        for key in scores:
            match key:
                case 'AKM':
                    row_labels, col_labels = run_n_times(
                        algorithm=alternating_k_means_biclustering,
                        args=(data, n_clusters),
                        n_runs=1
                    )
                case 'KM':
                    row_labels, col_labels = run_n_times(
                        algorithm=k_means_biclustering,
                        args=(data, n_clusters),
                        n_runs=1
                    )
            score = consensus_score(get_biclusters_from_labels(shape, n_clusters, row_labels, col_labels),
                                    (rows[:, row_idx], cols[:, col_idx]))
            scores[key].append(score)

    conclusion: Dict[str, Dict[str, float]] = {}
    for key in scores:
        conclusion[key] = {"MEAN": np.mean(scores[key]), "STD": np.std(scores[key])}
    return conclusion


# print(pd.DataFrame(mean_scores(
#     n_initializations=100,
#     shape=(300, 300),
#     n_clusters=2,
#     noise=10
# )))


