"""Rounded convex costs can violate Monge order; repair must be exact."""

import numpy as np
import pytest
from scipy.optimize import linprog
from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import ConvexSweep

CASES = [
    (1, 17, [0.337, 0.925, 1.503], [0.925, 1.503, 1.811], 47),
    (2, 10, [0.19, 0.392, 0.577], [0.577, 0.934, 1.017, 1.831], 29),
]


def dist(xs, mass=1):
    return Distribution(np.array([xs], float), np.full(len(xs), mass, float))


def build(p, tau, xs, ys, storage, scale=1):
    graph = WassersteinNetwork(
        dist(xs),
        [dist([y]) for y in ys],
        DistanceMetric.L1,
        split_distance=10,
        p=p,
        intensity_scale=scale,
        solver=ConvexSweep(storage=storage),
        round_max_distance=False,
    )
    graph.set_cost_scaling(3)
    graph.add_simple_trash(tau / 3)
    graph.build()
    return graph


def oracle(p, tau, xs, ys, weights, cost_scale=3):
    costs = np.floor(cost_scale * abs(np.array(xs)[:, None] - np.array(ys)) ** p + 0.5)
    reward = np.floor(cost_scale * tau / 3 + 0.5)
    n, m = costs.shape
    A = np.zeros((n + m, n * m))
    for i in range(n):
        for j in range(m):
            A[i, i * m + j] = A[n + j, i * m + j] = 1
    result = linprog(
        (costs - reward).ravel(),
        A_ub=A,
        b_ub=np.r_[np.ones(n), weights],
        bounds=(0, None),
        method="highs",
    )
    assert result.success
    return (reward * max(n, sum(weights)) + result.fun) / cost_scale


@pytest.mark.parametrize("p,tau,xs,ys,profit", CASES)
@pytest.mark.parametrize("storage", ["flat", "tree", "augmented"])
@pytest.mark.parametrize("scale", [1, 100000000])
def test_rounded_costs_solve_and_certify_global_lp(
    p, tau, xs, ys, profit, storage, scale
):
    graph = build(p, tau, xs, ys, storage, scale)
    weights = np.ones(len(ys))
    expected = (tau * max(len(xs), len(ys)) - profit) / 3
    graph.solve(weights)
    assert graph.total_cost() == pytest.approx(expected, abs=1e-10)
    graph.solve(weights, with_duals=True)
    cut = graph.dual_cut()
    assert graph.total_cost() == pytest.approx(expected, abs=1e-10)
    assert cut["upper_bound"] == pytest.approx(expected, abs=1e-10)
    assert cut["intercept"] + cut["gradient"] @ weights == pytest.approx(expected)
    for probe in [np.zeros(len(ys)), np.full(len(ys), 0.5), np.full(len(ys), 2)]:
        assert (
            cut["intercept"] + cut["gradient"] @ probe
            <= oracle(p, tau, xs, ys, probe) + 1e-9
        )
    # The geometry check must cover targets that currently have zero supply.
    for probe in [np.zeros(len(ys)), *np.eye(len(ys))]:
        graph.solve(probe, with_duals=True)
        expected_probe = oracle(p, tau, xs, ys, probe)
        assert graph.total_cost() == pytest.approx(expected_probe, abs=1e-10)
        assert graph.dual_cut()["upper_bound"] == pytest.approx(
            expected_probe, abs=1e-10
        )
    graph.solve(weights)
    if scale == 1:
        gradient = graph.spectrum_proportion_derivatives()
        for j in range(len(ys)):
            probe = weights.copy()
            probe[j] += 1
            graph.solve(probe)
            assert graph.total_cost() - expected == pytest.approx(
                gradient[j], abs=1e-10
            )
    graph.solve(weights, with_duals=True)
    graph.refine_intensity_precision(10)
    graph.solve(weights, with_duals=True)
    refined_expected = oracle(p, tau, xs, ys, weights, graph.scale_factor())
    assert graph.total_cost() == pytest.approx(refined_expected, abs=1e-10)
    assert graph.dual_cut()["upper_bound"] == pytest.approx(refined_expected, abs=1e-10)


@pytest.mark.parametrize("storage", ["flat", "tree", "augmented"])
def test_geometry_update_invalidates_cached_monge_check(storage):
    p, tau, xs, ys, profit = CASES[0]
    old_xs, old_ys = [0, 2, 4], [2, 4, 6]
    graph = build(p, tau, old_xs, old_ys, storage)
    graph.solve(np.ones(len(ys)), with_duals=True)
    graph.update_positions_and_solve(dist(xs), [dist([y]) for y in ys])
    expected = (tau * len(xs) - profit) / 3
    assert graph.total_cost() == pytest.approx(expected)
    assert graph.dual_cut()["upper_bound"] == pytest.approx(expected)
