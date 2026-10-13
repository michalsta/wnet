"""Quantize geometry before evaluating integer convex pair costs."""

import math
import numpy as np
import pytest
from scipy.optimize import linprog
from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import ConvexSweep

CASES = [
    (1, 17, [0.337, 0.925, 1.503], [0.925, 1.503, 1.811]),
    (2, 10, [0.19, 0.392, 0.577], [0.577, 0.934, 1.017, 1.831]),
]


def dist(xs):
    return Distribution(np.array([xs], float), np.ones(len(xs)))


def build(p, tau, xs, ys, storage, intensity_scale=1):
    graph = WassersteinNetwork(
        dist(xs),
        [dist([y]) for y in ys],
        DistanceMetric.L1,
        split_distance=10,
        p=p,
        intensity_scale=intensity_scale,
        solver=ConvexSweep(storage=storage),
        round_max_distance=False,
    )
    graph.set_cost_scaling(3)
    graph.add_simple_trash(tau / 3)
    graph.build()
    return graph


def oracle(p, tau, xs, ys, weights, scale):
    # These components start at the lowest empirical position and run right.
    # Independent reference: round each coordinate once, then use exact
    # integer linear/squared differences, never round each pair's real cost.
    denominator = scale if p == 1 else math.sqrt(scale)
    origin = min(xs + ys)
    qx = np.floor((np.array(xs) - origin) * denominator + 0.5).astype(np.int64)
    qy = np.floor((np.array(ys) - origin) * denominator + 0.5).astype(np.int64)
    costs = abs(qx[:, None] - qy) ** p
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            assert (
                costs[i, j] + costs[i + 1, j + 1] <= costs[i, j + 1] + costs[i + 1, j]
            )
    reward = math.floor(scale * tau / 3 + 0.5)
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
    return (reward * max(n, sum(weights)) + result.fun) / scale


@pytest.mark.parametrize("p,tau,xs,ys", CASES)
@pytest.mark.parametrize("storage", ["flat", "tree", "augmented"])
@pytest.mark.parametrize("intensity_scale", [1, 100000000])
def test_monge_grid_matches_unrestricted_lp(p, tau, xs, ys, storage, intensity_scale):
    graph = build(p, tau, xs, ys, storage, intensity_scale)
    scale = graph.scale_factor()
    assert scale == 3
    weights = np.ones(len(ys))
    expected = oracle(p, tau, xs, ys, weights, scale)
    graph.solve(weights)
    assert graph.total_cost() == pytest.approx(expected, abs=1e-10)
    assert graph.get_subgraph(0).convex_sweep_dual_stats() == (0, 0, 0, 0, 0)
    graph.solve(weights, with_duals=True)
    cut = graph.dual_cut()
    assert cut["rounded_cost"] == pytest.approx(expected, abs=1e-10)
    assert cut["intercept"] + cut["gradient"] @ weights == pytest.approx(expected)
    for probe in [np.zeros(len(ys)), np.full(len(ys), 0.5), np.full(len(ys), 2)]:
        assert (
            cut["intercept"] + cut["gradient"] @ probe
            <= oracle(p, tau, xs, ys, probe, scale) + 1e-9
        )
    for probe in [np.zeros(len(ys)), *np.eye(len(ys))]:
        graph.solve(probe, with_duals=True)
        assert graph.total_cost() == pytest.approx(
            oracle(p, tau, xs, ys, probe, scale), abs=1e-10
        )
    graph.solve(weights)
    if intensity_scale == 1:
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
    assert graph.total_cost() == pytest.approx(
        oracle(p, tau, xs, ys, weights, graph.scale_factor()), abs=1e-10
    )
    graph.dual_cut()


@pytest.mark.parametrize("storage", ["flat", "tree", "augmented"])
def test_geometry_update_rebuilds_shared_grid(storage):
    p, tau, xs, ys = CASES[0]
    graph = build(p, tau, [0, 2, 4], [2, 4, 6], storage)
    graph.solve(np.ones(len(ys)), with_duals=True)
    graph.update_positions_and_solve(dist(xs), [dist([y]) for y in ys])
    expected = oracle(p, tau, xs, ys, np.ones(len(ys)), graph.scale_factor())
    assert graph.total_cost() == pytest.approx(expected)
    assert graph.dual_cut()["rounded_cost"] == pytest.approx(expected)


@pytest.mark.parametrize("p", [1, 1.01, 1.5, 2, 2.5, 3, 5])
def test_fractional_geometry_duals_and_model_error(p):
    xs, ys = [0.01, 0.203, 0.521], [0.027, 0.251, 0.803]
    graph = build(p, 10, xs, ys, "flat", 1000)
    graph.solve(np.ones(3), with_duals=True)
    assert graph.dual_cut()["rounding_error"] < 1e-10
    assert graph.get_subgraph(0).convex_sweep_cost_quantization_error() > 0
    graph.solve(np.zeros(3), with_duals=True)
    cut = graph.dual_cut()
    graph.solve(np.ones(3), with_duals=True)
    assert cut["intercept"] + sum(cut["gradient"]) <= graph.total_cost() + 1e-9


@pytest.mark.parametrize("storage", ["flat", "tree", "augmented"])
def test_quadratic_scale_uses_profitable_cost_budget(storage):
    # Squared far-pair costs exceed int64 at the fine automatic scale. The
    # exact Monge model retains these wide squares; the solver only visits
    # profitable windows. Build and solve must use the same accumulator bound.
    xs, ys = [0, 10000], [0.01, 10000.01]
    graph = WassersteinNetwork(
        dist(xs),
        [dist(ys)],
        DistanceMetric.L1,
        split_distance=20000,
        p=2,
        intensity_scale=1e8,
        solver=ConvexSweep(storage=storage),
        round_max_distance=False,
    )
    graph.add_independent_asymmetric_trash(0.025**2 / 2, 0.025**2 / 2)
    graph.build()
    scale = graph.scale_factor()
    denominator = math.sqrt(scale)
    assert (10000 * denominator) ** 2 > np.iinfo(np.int64).max
    qx = [math.floor(x * denominator + 0.5) for x in xs]
    qy = [math.floor(y * denominator + 0.5) for y in ys]
    expected = sum((x - y) ** 2 for x, y in zip(qx, qy)) / scale
    graph.solve([1], with_duals=True)
    assert graph.total_cost() == pytest.approx(expected, abs=1e-12)
    assert graph.dual_cut()["rounded_cost"] == pytest.approx(expected, abs=1e-12)
    with pytest.raises(OverflowError, match="budget"):
        graph.solve([1000])
