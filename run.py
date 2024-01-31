from typing import Dict, Callable, Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.metrics import consensus_score
from tqdm import trange

from algorithms.BBAC import BBAC
from algorithms.alternating_biclustering import alternating_k_means_biclustering
from algorithms.kmeans import k_means_biclustering
from algorithms.profile_likelihood import profile_likelihood_biclustering
from algorithms.test_algorithm import test_algo
from utils import get_biclusters_from_labels


def run_n_times(
        algorithm: Callable,
        args: Dict,
        n_runs: int
) -> Tuple[NDArray, NDArray]:
    labels: List[Tuple[NDArray, NDArray]] = []
    losses: List[float] = []

    for _ in range(n_runs):
        row_labels, col_labels, loss = algorithm(**args)
        labels.append((row_labels, col_labels))
        losses.append(loss)

    return labels[np.argmin(losses)]


def show(
        shape: Tuple[int, int],
        n_clusters: int,
        generate_data: Callable,
        algorithm: Callable,
        n_runs: int,
        algorithm_name: str,
        *,
        cm=plt.cm.Blues,
        score_multiplier=1.
):
    _data, _rows, _cols = generate_data()

    for _ in range(n_runs):
        data, rows, cols = np.copy(_data), np.copy(_rows), np.copy(_cols)

        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(11, 4), layout='constrained', dpi=200)

        # show original dataset with visible clusters
        ax1.matshow(data, cmap=cm)
        ax1.set_title("Original dataset")

        rng = np.random.RandomState(np.random.randint(100))
        row_idx = rng.permutation(data.shape[0])
        col_idx = rng.permutation(data.shape[1])
        data = data[row_idx][:, col_idx]

        ax2.matshow(data, cmap=cm)
        ax2.set_title("Shuffled dataset")

        row_labels, col_labels = run_n_times(
            algorithm=algorithm,
            args={
                "data_matrix": data,
                "n_clusters": n_clusters
            },
            n_runs=1
        )

        # reorder rows and cols of a data matrix to show clusters
        fit_data = data[np.argsort(row_labels)]
        fit_data = fit_data[:, np.argsort(col_labels)]

        # show data matrix with reordered rows and cols according to calculated cluster assignments
        ax3.matshow(fit_data, cmap=cm)

        # calculate consensus score between expected and actual biclusters
        score = consensus_score(get_biclusters_from_labels(shape, n_clusters, row_labels, col_labels),
                                (rows[:, row_idx], cols[:, col_idx]))

        ax3.set_title(f"{algorithm_name.title()}: {score * score_multiplier:.3f}")
        plt.show()


def sims_mean_scores(
        n_simulations: int,
        n_runs_per_simulation: int,
        shape: Tuple[int, int],
        generate_data: Callable,
        n_clusters: int,
        *,
        score_multiplier=1.,
):
    scores = {'AKM 0': [], 'AKM 0.1': [], 'AKM 1': [], 'KM': [], 'BBAC': [], 'ASAP': []}

    for _ in trange(n_simulations):
        data, rows, cols = generate_data()

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
                case 'BBAC':
                    row_labels, col_labels = run_n_times(
                        algorithm=BBAC,
                        args={
                            "data_matrix": data,
                            "n_clusters": n_clusters,
                        },
                        n_runs=n_runs_per_simulation
                    )
                case 'PL':
                    row_labels, col_labels = run_n_times(
                        algorithm=profile_likelihood_biclustering,
                        args={
                            "data_matrix": data,
                            "n_clusters": n_clusters,
                        },
                        n_runs=n_runs_per_simulation
                    )
                case 'ASAP':
                    row_labels, col_labels = run_n_times(
                        algorithm=test_algo,
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
