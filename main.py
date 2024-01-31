import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algorithms.BBAC import BBAC
from algorithms.profile_likelihood import profile_likelihood_biclustering
from algorithms.test_algorithm import test_algo
from generate import make_biclusters_simulation
from sklearn.datasets import make_biclusters, make_checkerboard
from run import show, sims_mean_scores

if __name__ == "__main__":
    # b = 1.
    # print(pd.DataFrame(sims_mean_scores(
    #     n_simulations=0,
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

    # n_clusters = 2
    # shape = (300, 300)
    # sizes = np.random.randint(6, 9, n_clusters)
    # sizes = sizes / np.sum(sizes)
    # show(
    #     shape=shape,
    #     n_clusters=n_clusters,
    #     generate_data=lambda: make_biclusters_simulation(
    #         shape=shape,
    #         n_clusters=n_clusters,
    #         M=10 * np.random.rand(*shape),
    #         S=np.random.rand(*shape) + .5,
    #         sizes=sizes
    #     ),
    #     algorithm=test_algo,
    #     n_runs=5,
    #     algorithm_name="ASAP",
    #     cm=plt.cm.Blues,
    #     score_multiplier=n_clusters
    # )

    # b = .5
    # n_clusters = 2
    # shape = (300, 300)
    # show(
    #     shape=shape,
    #     n_clusters=n_clusters,
    #     generate_data=lambda: make_biclusters_simulation(
    #         shape=shape,
    #         M=np.array([[.5, 0], [0, .5]]),
    #         S=np.array([[1. + b, 1.], [1., 1. + b]]),
    #         n_clusters=n_clusters,
    #         sizes=np.array([0.3, 0.7])
    #     ),
    #     algorithm=BBAC,
    #     n_runs=5,
    #     algorithm_name='BBAC',
    #     cm=plt.cm.Blues,
    #     score_multiplier=n_clusters
    # )

    # n_clusters = 3
    # shape = (300, 300)
    # show(
    #     shape=shape,
    #     n_clusters=n_clusters,
    #     generate_data=lambda: make_checkerboard(
    #         shape=shape,
    #         n_clusters=n_clusters,
    #         noise=10,
    #         shuffle=False,
    #     ),
    #     algorithm=test_algo,
    #     n_runs=5,
    #     algorithm_name='ASAP',
    #     cm=plt.cm.Blues,
    #     score_multiplier=n_clusters
    # )

    # n_clusters = 3
    # shape = (300, 300)
    # show(
    #     shape=shape,
    #     n_clusters=n_clusters,
    #     generate_data=lambda: make_checkerboard(
    #         shape=shape,
    #         n_clusters=n_clusters,
    #         noise=10,
    #         shuffle=False
    #     ),
    #     algorithm=test_algo,
    #     n_runs=5,
    #     score_multiplier=n_clusters,
    #     algorithm_name='ASAP',
    #     cm=plt.cm.Blues
    # )

    # n_clusters = 2
    # shape = (50, 50)
    # show(
    #     shape=shape,
    #     n_clusters=n_clusters,
    #     generator=make_biclusters,
    #     generator_args={
    #         "shape": shape,
    #         "n_clusters": n_clusters,
    #         "noise": 10,
    #         "shuffle": False
    #     },
    #     algorithm=test_algo,
    #     n_runs=1,
    #     algorithm_name='ASAP',
    #     cm=plt.cm.Blues
    # )

    b = .5
    shape = (100, 100)
    n_clusters = 3
    print(
        pd.DataFrame(
            sims_mean_scores(
                n_simulations=10,
                n_runs_per_simulation=5,
                shape=shape,
                generate_data=lambda: make_biclusters(
                    shape=shape,
                    n_clusters=n_clusters,
                    shuffle=False,
                    noise=10
                ),
                score_multiplier=1,
                n_clusters=n_clusters,
            )
        ).to_markdown()
    )

