import numpy as np
from matplotlib import pyplot as plt

from algorithms.BBAC import bregman_block_average_coclustering
from algorithms.alternating_biclustering import alternating_k_means_biclustering
from algorithms.profile_likelihood import profile_likelihood_biclustering
from generate import make_biclusters_simulation
from run import show

if __name__ == '__main__':
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

    b = 0.
    show(
        shape=(100, 100),
        n_clusters=2,
        generator=make_biclusters_simulation,
        generator_args={
            "shape": (100, 100),
            "M": np.array([[10, 0], [0, 10]]),
            "S": np.array([[1 + b, 1.], [1., 1 + b]]),
            "n_clusters": 2,
            "sizes": np.array([0.3, 0.7])
        },
        algorithm=profile_likelihood_biclustering,
        n_runs=10,
        algorithm_name='Profile Likelihood Biclustering',
        cm=plt.cm.Blues
    )

    # b = 1.
    # print(pd.DataFrame(sims_mean_scores(
    #     n_simulations=20,
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



