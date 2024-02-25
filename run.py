from typing import Dict, Callable, Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.metrics import consensus_score
from tqdm import trange
from math import ceil

from algorithms.BBAC import BBAC
from algorithms.alternating_biclustering import alternating_k_means_biclustering
from algorithms.kmeans import k_means_biclustering
from algorithms.profile_likelihood import profile_likelihood_biclustering
from algorithms.test_algorithm import ASAP
from utils import get_biclusters_from_labels


def run_n_times(
    algorithm: Callable, args: Dict, n_runs: int
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
    score_multiplier=1.0,
):
    _data, _rows, _cols = generate_data()

    for _ in range(n_runs):
        data, rows, cols = np.copy(_data), np.copy(_rows), np.copy(_cols)

        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(11, 4), layout="constrained", dpi=200
        )

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
            args={"data_matrix": data, "n_clusters": n_clusters},
            n_runs=1,
        )

        # reorder rows and cols of a data matrix to show clusters
        fit_data = data[np.argsort(row_labels)]
        fit_data = fit_data[:, np.argsort(col_labels)]

        # show data matrix with reordered rows and cols according to calculated cluster assignments
        ax3.matshow(fit_data, cmap=cm)

        # calculate consensus score between expected and actual biclusters
        score = consensus_score(
            get_biclusters_from_labels(shape, n_clusters, row_labels, col_labels),
            (rows[:, row_idx], cols[:, col_idx]),
        )

        ax3.set_title(f"{algorithm_name.title()}: {score * score_multiplier:.3f}")
        plt.show()


def sims_mean_scores(
    n_simulations: int,
    n_runs_per_simulation: int,
    shape: Tuple[int, int],
    generate_data: Callable,
    n_clusters: int,
    *,
    score_multiplier=1.0,
):
    scores = {"AKM 0": [], "AKM 0.1": [], "AKM 1": [], "KM": [], "BBAC": [], "ASAP": []}

    for _ in trange(n_simulations):
        data, rows, cols = generate_data()

        rng = np.random.RandomState(0)
        row_idx = rng.permutation(data.shape[0])
        col_idx = rng.permutation(data.shape[1])
        data = data[row_idx][:, col_idx]

        for key in scores:
            match key:
                case "AKM 0":
                    row_labels, col_labels = run_n_times(
                        algorithm=alternating_k_means_biclustering,
                        args={
                            "data_matrix": data,
                            "n_clusters": n_clusters,
                            "lamda": 0.0,
                        },
                        n_runs=n_runs_per_simulation,
                    )
                case "AKM 0.1":
                    row_labels, col_labels = run_n_times(
                        algorithm=alternating_k_means_biclustering,
                        args={
                            "data_matrix": data,
                            "n_clusters": n_clusters,
                            "lamda": 0.1,
                        },
                        n_runs=n_runs_per_simulation,
                    )
                case "AKM 1":
                    row_labels, col_labels = run_n_times(
                        algorithm=alternating_k_means_biclustering,
                        args={
                            "data_matrix": data,
                            "n_clusters": n_clusters,
                            "lamda": 1.0,
                        },
                        n_runs=n_runs_per_simulation,
                    )
                case "KM":
                    row_labels, col_labels = run_n_times(
                        algorithm=k_means_biclustering,
                        args={
                            "data_matrix": data,
                            "n_clusters": n_clusters,
                        },
                        n_runs=n_runs_per_simulation,
                    )
                case "BBAC":
                    row_labels, col_labels = run_n_times(
                        algorithm=BBAC,
                        args={
                            "data_matrix": data,
                            "n_clusters": n_clusters,
                        },
                        n_runs=n_runs_per_simulation,
                    )
                case "PL":
                    row_labels, col_labels = run_n_times(
                        algorithm=profile_likelihood_biclustering,
                        args={
                            "data_matrix": data,
                            "n_clusters": n_clusters,
                        },
                        n_runs=n_runs_per_simulation,
                    )
                case "ASAP":
                    row_labels, col_labels = run_n_times(
                        algorithm=ASAP,
                        args={
                            "data_matrix": data,
                            "n_clusters": n_clusters,
                        },
                        n_runs=n_runs_per_simulation,
                    )

            score = (
                consensus_score(
                    get_biclusters_from_labels(
                        shape, n_clusters, row_labels, col_labels
                    ),
                    (rows[:, row_idx], cols[:, col_idx]),
                )
                * score_multiplier
            )
            scores[key].append(score)

    conclusion: Dict[str, Dict[str, float]] = {}
    for key in scores:
        conclusion[key] = {"MEAN": np.mean(scores[key]), "STD": np.std(scores[key])}
    return conclusion


def show_real(
    data: NDArray, n_clusters: int, cmap="plasma", n_runs_per_simulation=1
) -> None:
    tags: List[str] = [
        "AKM 0",
        "AKM 0.1",
        "AKM 1",
        "KM",
        "PL",
        "BBAC",
        "ASAP 1:0",
        "ASAP 1:1",
        "ASAP 10:0.1",
        "ASAP 0.1:10",
    ]
    max_cols = 5
    num_cols = min(max_cols, len(tags))
    num_rows = ceil(len(tags) / num_cols)

    _, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(4 * num_cols, 4 * num_rows),
        layout="constrained",
        dpi=200,
    )

    # axs[0].matshow(data, cmap=cmap)
    # axs[0].set_title("Original Dataset")

    for i in range(len(tags)):
        tag = tags[i]
        match tag:
            case "AKM 0":
                row_labels, col_labels = run_n_times(
                    algorithm=alternating_k_means_biclustering,
                    args={"data_matrix": data, "n_clusters": n_clusters, "lamda": 0.0},
                    n_runs=n_runs_per_simulation,
                )
            case "AKM 0.1":
                row_labels, col_labels = run_n_times(
                    algorithm=alternating_k_means_biclustering,
                    args={"data_matrix": data, "n_clusters": n_clusters, "lamda": 0.1},
                    n_runs=n_runs_per_simulation,
                )
            case "AKM 1":
                row_labels, col_labels = run_n_times(
                    algorithm=alternating_k_means_biclustering,
                    args={"data_matrix": data, "n_clusters": n_clusters, "lamda": 1.0},
                    n_runs=n_runs_per_simulation,
                )
            case "KM":
                row_labels, col_labels = run_n_times(
                    algorithm=k_means_biclustering,
                    args={
                        "data_matrix": data,
                        "n_clusters": n_clusters,
                    },
                    n_runs=n_runs_per_simulation,
                )
            case "BBAC":
                row_labels, col_labels = run_n_times(
                    algorithm=BBAC,
                    args={
                        "data_matrix": data,
                        "n_clusters": n_clusters,
                    },
                    n_runs=n_runs_per_simulation,
                )
            case "PL":
                row_labels, col_labels = run_n_times(
                    algorithm=profile_likelihood_biclustering,
                    args={
                        "data_matrix": data,
                        "n_clusters": n_clusters,
                    },
                    n_runs=n_runs_per_simulation,
                )
            case "ASAP 1:1":
                row_labels, col_labels = run_n_times(
                    algorithm=ASAP,
                    args={
                        "data_matrix": data,
                        "n_clusters": n_clusters,
                        "a": 1,
                        "b": 1,
                    },
                    n_runs=n_runs_per_simulation,
                )
            case "ASAP 10:0.1":
                row_labels, col_labels = run_n_times(
                    algorithm=ASAP,
                    args={
                        "data_matrix": data,
                        "n_clusters": n_clusters,
                        "a": 10,
                        "b": 0.1,
                    },
                    n_runs=n_runs_per_simulation,
                )
            case "ASAP 0.1:10":
                row_labels, col_labels = run_n_times(
                    algorithm=ASAP,
                    args={
                        "data_matrix": data,
                        "n_clusters": n_clusters,
                        "a": 0.1,
                        "b": 10,
                    },
                    n_runs=n_runs_per_simulation,
                )
            case "ASAP 1:0":
                row_labels, col_labels = run_n_times(
                    algorithm=ASAP,
                    args={
                        "data_matrix": data,
                        "n_clusters": n_clusters,
                        "a": 1,
                        "b": 0,
                    },
                    n_runs=n_runs_per_simulation,
                )
        fit_data = data[np.argsort(row_labels)]
        fit_data = fit_data[:, np.argsort(col_labels)]

        if num_rows == 1:
            axs[i].set_title(tag)
            axs[i].matshow(fit_data, cmap=cmap)
        else:
            axs[int(i / num_cols)][i % num_cols].set_title(tag)
            axs[int(i / num_cols)][i % num_cols].matshow(fit_data, cmap=cmap)

    plt.show()
