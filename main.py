from typing import Tuple, Dict, List, Callable, Any
import numpy as np
import pandas as pd
from sklearn.metrics import consensus_score
from tqdm import tqdm, trange

from algorithms.alternating_biclustering import alternating_k_means_biclustering
from algorithms.kmeans import k_means_biclustering
from generate import make_biclusters_simulation
from utils import get_biclusters_from_labels, run_n_times


def mean_scores(
        n_simulations: int,
        n_runs_per_simulation: int,
        shape: Tuple[int, int],
        generator: Callable,
        generator_args: Any,
        n_clusters: int,
        *,
        score_multiplier=1.,
):
    scores = {'AKM 0': [], 'AKM 0.1': [], 'AKM 1': [], 'KM': []}

    for _ in trange(n_simulations):
        data, rows, cols = generator(**generator_args)

        rng = np.random.RandomState(0)
        row_idx = rng.permutation(data.shape[0])
        col_idx = rng.permutation(data.shape[1])
        data = data[row_idx][:, col_idx]

        for key in scores:
            match key:
                case 'AKM 0':
                    row_labels, col_labels = run_n_times(
                        algorithm=alternating_k_means_biclustering,
                        args={
                            "data_matrix": data,
                            "n_clusters": n_clusters,
                            "lamda": 0.
                        },
                        n_runs=n_runs_per_simulation
                    )
                case 'AKM 0.1':
                    row_labels, col_labels = run_n_times(
                        algorithm=alternating_k_means_biclustering,
                        args={
                            "data_matrix": data,
                            "n_clusters": n_clusters,
                            "lamda": 0.1
                        },
                        n_runs=n_runs_per_simulation
                    )
                case 'AKM 1':
                    row_labels, col_labels = run_n_times(
                        algorithm=alternating_k_means_biclustering,
                        args={
                            "data_matrix": data,
                            "n_clusters": n_clusters,
                            "lamda": 1.
                        },
                        n_runs=n_runs_per_simulation
                    )
                case 'KM':
                    row_labels, col_labels = run_n_times(
                        algorithm=k_means_biclustering,
                        args={
                            "data_matrix": data,
                            "n_clusters": n_clusters,
                        },
                        n_runs=n_runs_per_simulation
                    )

            score = consensus_score(get_biclusters_from_labels(shape, n_clusters, row_labels, col_labels),
                                    (rows[:, row_idx], cols[:, col_idx])) * score_multiplier
            scores[key].append(score)

    conclusion: Dict[str, Dict[str, float]] = {}
    for key in scores:
        conclusion[key] = {"MEAN": np.mean(scores[key]), "STD": np.std(scores[key])}
    return conclusion


# b = 1.
# print(pd.DataFrame(mean_scores(
#     n_simulations=50,
#     n_runs_per_simulation=10,
#     shape=(100, 100),
#     generator=make_biclusters_simulation,
#     generator_args={
#         "shape": (100, 100),
#         "M": b * np.array([[.36, .90], [-.58, -.06]]),
#         "S": np.repeat(1., 4).reshape((2, 2)),
#         "n_clusters": 2,
#         "sizes": np.array([0.3, 0.7])
#     },
#     score_multiplier=2.,
#     n_clusters=2,
# )).to_markdown())

# b = 1.
# print(pd.DataFrame(mean_scores(
#     n_simulations=50,
#     n_runs_per_simulation=10,
#     shape=(100, 100),
#     generator=make_biclusters_simulation,
#     generator_args={
#         "shape": (100, 100),
#         "M": np.zeros((2, 2)),
#         "S": np.array([[1 + b, 1.], [1., 1 + b]]),
#         "n_clusters": 2,
#         "sizes": np.array([0.3, 0.7])
#     },
#     score_multiplier=2.,
#     n_clusters=2,
# )).to_markdown())

# b = .5
# print(pd.DataFrame(mean_scores(
#     n_simulations=50,
#     n_runs_per_simulation=10,
#     shape=(100, 100),
#     generator=make_biclusters_simulation,
#     generator_args={
#         "shape": (100, 100),
#         "M": b * np.array([[.36, .90], [-.58, -.06]]),
#         "S": np.array([[1. + b, 1.], [1., 1. + b]]),
#         "n_clusters": 2,
#         "sizes": np.array([0.3, 0.7])
#     },
#     score_multiplier=2.,
#     n_clusters=2,
# )).to_markdown())



