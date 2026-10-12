"""Benchmark graph construction, solves, certificates and mixture gradients.

Run each storage choice in a separate process to measure peak RSS with
/usr/bin/time. Timing samples use one thread and identical synthetic inputs.
"""

import argparse
import json
import statistics
import time

import numpy as np
from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import ConvexSweep, SlopeDP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["slope", "sweep"], required=True)
    parser.add_argument(
        "--storage", choices=["flat", "tree", "augmented"], required=True
    )
    parser.add_argument("--positions", type=int, default=1000)
    parser.add_argument("--radius", type=float, default=8)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    xs = np.arange(args.positions, dtype=float)
    empirical = Distribution(xs[None, :], rng.integers(1, 8, len(xs)).astype(float))
    targets = [
        Distribution(
            (xs + offset)[None, :], rng.integers(1, 8, len(xs)).astype(float) / 2
        )
        for offset in [0.25, -0.25]
    ]
    p = 1 if args.solver == "slope" else 2
    point = np.ones(2)

    def build():
        config = (SlopeDP if p == 1 else ConvexSweep)(storage=args.storage)
        graph = WassersteinNetwork(
            empirical,
            targets,
            DistanceMetric.L1,
            split_distance=2 * args.radius + 1,
            solver=config,
            p=p,
            intensity_scale=1000,
            round_max_distance=False,
        )
        graph.set_cost_scaling(0)
        graph.add_independent_asymmetric_trash(args.radius**p / 2, args.radius**p / 2)
        graph.build()
        return graph

    def measure(fn):
        samples = []
        fn()  # Warmup.
        for _ in range(args.repetitions):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1000)
        return statistics.median(samples)

    construction = measure(build)
    graph = build()
    ordinary = measure(lambda: graph.solve(point))
    cost = graph.total_cost()

    def certified():
        graph.solve(point, with_duals=True)
        graph.dual_cut()

    certification = measure(certified)
    assert graph.total_cost() == cost
    cut_query = measure(graph.dual_cut)
    graph.solve(point)
    marginals = measure(graph.spectrum_proportion_derivatives)
    gradient = list(map(float, graph.spectrum_proportion_derivatives()))
    certified()
    cut = graph.dual_cut()
    assert cut["intercept"] + cut["gradient"] @ point <= cut["upper_bound"] + 1e-8
    print(
        json.dumps(
            dict(
                solver=args.solver,
                storage=args.storage,
                n=args.positions,
                radius=args.radius,
                repetitions=args.repetitions,
                construction_ms=construction,
                solve_ms=ordinary,
                solve_and_certificate_ms=certification,
                cached_cut_ms=cut_query,
                marginals_ms=marginals,
                cost=cost,
                gradient=gradient,
            )
        )
    )


if __name__ == "__main__":
    main()
