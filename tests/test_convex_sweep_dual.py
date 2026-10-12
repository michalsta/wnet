"""Sweep certificates checked against independent continuous matching LPs."""

import numpy as np
import pytest
from scipy.optimize import linprog

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import ConvexSweep


def fixture(model, scale, p, seed=42, reverse_trash=False):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 6, 5).astype(float)
    e = rng.uniform(0.2, 2, 5)
    ys = [rng.integers(0, 6, 4).astype(float) for _ in range(2)]
    ts = [rng.uniform(0.2, 2, 4) for _ in range(2)]
    empirical = Distribution(x[None, :], e)
    targets = [Distribution(y[None, :], t) for y, t in zip(ys, ts)]
    graph = WassersteinNetwork(
        empirical,
        targets,
        DistanceMetric.L1,
        split_distance=20,
        intensity_scale=scale,
        solver=ConvexSweep(),
        p=p,
        round_max_distance=False,
    )
    graph.set_cost_scaling(0)
    ce, ct = 3, 5
    if reverse_trash:
        ce, ct = ct, ce
    if model == "simple":
        graph.add_simple_trash(ce)
        ct = ce
    elif model == "independent":
        graph.add_independent_asymmetric_trash(ce, ct)
    elif model == "both":
        graph.add_experimental_trash(ce)
        graph.add_theoretical_trash(ct)
    elif model == "experimental":
        graph.add_experimental_trash(ce)
        ct = 0
    else:
        graph.add_theoretical_trash(ct)
        ce = 0
    graph.build()
    distances = abs(x[:, None] - np.concatenate(ys)[None, :]) ** p
    cost_scale = graph.scale_factor()
    costs = (
        np.floor(distances)
        if p == 1
        else np.floor(distances * cost_scale + 0.5) / cost_scale
    )
    tau = (
        ce + ct
        if model == "independent"
        else min(ce, ct) if model in ("simple", "both") else max(ce, ct)
    )
    A = np.zeros((13, 40))
    for i in range(5):
        for j in range(8):
            A[i, i * 8 + j] = A[5 + j, i * 8 + j] = 1

    def objective(weights):
        supply = np.concatenate([t * w for t, w in zip(ts, weights)])
        E, T = sum(e), sum(supply)
        if model in ("simple", "both"):
            trash = (
                ce * E + ct * max(T - E, 0) if ce <= ct else ct * T + ce * max(E - T, 0)
            )
        else:
            trash = ce * E + ct * T
        result = linprog(
            (costs - tau).ravel(),
            A_ub=A,
            b_ub=np.concatenate([e, supply]),
            method="highs",
        )
        assert result.success
        return trash + result.fun

    return graph, objective


@pytest.mark.parametrize("p", [1, 1.5, 2, 3])
@pytest.mark.parametrize(
    "model", ["simple", "both", "independent", "experimental", "theoretical"]
)
@pytest.mark.parametrize("scale", [1, 17.3, 100000])
def test_global_bounds(p, model, scale):
    for seed in range(3):
        graph, objective = fixture(model, scale, p, seed)
        probes = np.vstack(
            [
                np.zeros(2),
                np.ones(2),
                [0, 2],
                [2, 0],
                np.random.default_rng(seed).uniform(0, 3, (6, 2)),
            ]
        )
        truths = [objective(w) for w in probes]
        with pytest.raises(RuntimeError, match="solve"):
            graph.dual_cut()
        for point in probes[:6]:
            graph.solve(point)
            cut = graph.dual_cut()
            assert objective(point) <= cut["upper_bound"] + 1e-8
            assert (
                abs(objective(point) - graph.total_cost())
                <= cut["rounding_error"] + 1e-8
            )
            for probe, value in zip(probes, truths):
                assert cut["intercept"] + cut["gradient"] @ probe <= value + 1e-8
            if scale == 100000:
                assert (
                    objective(point) - (cut["intercept"] + cut["gradient"] @ point)
                    <= 2 * cut["rounding_error"] + 1e-8
                )


def test_recovery_is_lazy_cached_and_invalidated():
    graph, _ = fixture("independent", 100000, 2)
    graph.solve([1, 1])
    sg = graph.get_subgraph(0)
    assert sg.convex_sweep_dual_stats() == (0, 0)
    first = graph.dual_cut()
    assert sg.convex_sweep_dual_stats()[0] == 1
    np.testing.assert_array_equal(first["gradient"], graph.dual_cut()["gradient"])
    assert sg.convex_sweep_dual_stats()[0] == 1
    graph.solve([0, 0.7])
    graph.dual_cut()
    assert sg.convex_sweep_dual_stats()[0] == 2
    graph.refine_intensity_precision(10)
    with pytest.raises(RuntimeError, match="solve"):
        graph.dual_cut()
    graph.solve([0, 0.7])
    graph.dual_cut()


@pytest.mark.parametrize("p", [1.5, 2, 3])
def test_reversed_trash_prices_across_mass_balance(p):
    graph, objective = fixture("both", 17.3, p, reverse_trash=True)
    probes = [[0, 0], [0, 3], [3, 0], [3, 3], [0.2, 0.7]]
    for point in probes:
        graph.solve(point)
        cut = graph.dual_cut()
        assert objective(point) <= cut["upper_bound"] + 1e-8
        for probe in probes:
            assert cut["intercept"] + cut["gradient"] @ probe <= objective(probe) + 1e-8


@pytest.mark.parametrize("integer", [False, True])
def test_position_updates_zero_reward_and_disconnected_peaks(integer):
    dtype = np.int64 if integer else float

    def dist(xs, values):
        return Distribution(np.array([xs], float), np.array(values, dtype=dtype))

    graph = WassersteinNetwork(
        dist([0, 2, 100], [1, 1, 1]),
        [dist([1, 200], [2, 1])],
        DistanceMetric.L1,
        split_distance=10,
        intensity_scale=100,
        p=2,
        solver=ConvexSweep(),
    )
    graph.set_cost_scaling(0)
    graph.add_independent_asymmetric_trash(3, 5)
    graph.build()
    graph.solve([1])
    cut = graph.dual_cut()
    assert cut["upper_bound"] == pytest.approx(10)
    assert cut["intercept"] + cut["gradient"] @ [1] == pytest.approx(10)
    graph.update_positions_and_solve(
        dist([0, 2, 100], [1, 1, 1]), [dist([0, 200], [2, 1])]
    )
    cut = graph.dual_cut()
    assert cut["upper_bound"] == pytest.approx(12)
    assert cut["intercept"] + cut["gradient"] @ [1] == pytest.approx(12)
    assert graph.get_subgraph(0).convex_sweep_dual_stats()[0] == 2
    zero = WassersteinNetwork(
        dist([0], [1]),
        [dist([0], [1])],
        DistanceMetric.L1,
        split_distance=10,
        p=2,
        solver=ConvexSweep(),
    )
    zero.add_independent_asymmetric_trash(0, 0)
    zero.build()
    zero.solve([1])
    assert zero.dual_cut()["upper_bound"] == pytest.approx(0)
