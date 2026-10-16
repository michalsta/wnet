"""Scaled p=1 paths must retain distance across arbitrarily fine samples."""

import numpy as np
import pytest
from scipy.optimize import linprog

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import ConvexSweep, NetworkSimplex, SlopeDP

BACKENDS = [
    "simplex",
    "slope-flat",
    "slope-tree",
    "slope-augmented",
    "sweep-flat",
    "sweep-tree",
    "sweep-augmented",
]


def solver(backend):
    if backend == "simplex":
        return NetworkSimplex()
    kind, storage = backend.split("-")
    return (SlopeDP if kind == "slope" else ConvexSweep)(storage=storage)


def dist(xs, ys):
    return Distribution(np.array([xs], dtype=float), np.array(ys, dtype=float))


def network(emp, refs, backend, scale=7, model="simple", intensity_scale=100):
    graph = WassersteinNetwork(
        dist(*emp),
        [dist(*ref) for ref in refs],
        DistanceMetric.L1,
        split_distance=100,
        round_max_distance=False,
        intensity_scale=intensity_scale,
        solver=solver(backend),
    )
    if scale is not None:
        graph.set_cost_scaling(scale)
    if model == "independent":
        graph.add_independent_asymmetric_trash(0.31, 0.47)
    else:
        graph.add_simple_trash(0.8)
    graph.build()
    return graph


def objective(emp, refs, weights, scale, model="simple", origin=0):
    """Unrestricted matching LP on independently rounded coordinates."""
    q = lambda x: np.floor(np.abs(np.asarray(x) - origin) * scale + 0.5)
    xs, ys = q(emp[0]), np.concatenate([q(ref[0]) for ref in refs])
    e = np.array(emp[1])
    t = np.concatenate([np.array(ref[1]) * w for ref, w in zip(refs, weights)])
    costs = np.abs(xs[:, None] - ys) / scale
    n, m = costs.shape
    A = np.zeros((n + m, n * m))
    for i in range(n):
        for j in range(m):
            A[i, i * m + j] = A[n + j, i * m + j] = 1
    if model == "independent":
        ce, ct = np.floor(np.array([0.31, 0.47]) * scale + 0.5) / scale
        reward, baseline = ce + ct, ce * e.sum() + ct * t.sum()
    else:
        reward = np.floor(0.8 * scale + 0.5) / scale
        baseline = reward * max(e.sum(), t.sum())
    result = linprog(
        (costs - reward).ravel(), A_ub=A, b_ub=np.r_[e, t], bounds=(0, None)
    )
    assert result.success
    return baseline + result.fun


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("offset", [0, -1e9, 1e9])
def test_inserting_zero_supply_samples_preserves_distance(backend, offset):
    # Each of 1,000 gaps rounds to zero at S=10 under independent rounding.
    for n in [2, 1001]:
        xs = np.linspace(0, 1, n) + offset
        mass = np.zeros(n)
        mass[0] = 1
        graph = network((xs, mass), [([1 + offset], [1])], backend, scale=10)
        # Trash is deliberately cheaper than the true distance: the preserved
        # geometry must choose trash, not the old free transport across zeros.
        graph.solve([1], with_duals=True)
        assert graph.total_cost() == pytest.approx(0.8, abs=1e-12)
        assert graph.dual_cut()["cost_rounding_error"] <= 0.1 + 1e-10


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model", ["simple", "independent"])
def test_costs_and_duals_match_coordinate_lp(backend, model):
    if backend == "simplex" and model == "independent":
        pytest.skip("Independent chain trash requires an analytic backend")
    emp = ([0, 0.029, 0.151, 0.283], [2, 1, 2, 1])
    refs = [([0.018, 0.127, 0.241], [1, 2, 1]), ([0.063, 0.193], [2, 1])]
    graph = network(emp, refs, backend, model=model)
    probes = [np.array(w) for w in [[0, 0], [0.25, 0.75], [1, 1], [2, 0.5]]]
    expected = [objective(emp, refs, w, 7, model) for w in probes]
    for weights, value in zip(probes, expected):
        graph.solve(weights, with_duals=True)
        assert graph.total_cost() == pytest.approx(value, abs=1e-9)
        cut = graph.dual_cut()
        assert cut["intercept"] + cut["gradient"] @ weights == pytest.approx(
            value, abs=1e-9
        )
        for probe, reference in zip(probes, expected):
            assert cut["intercept"] + cut["gradient"] @ probe <= reference + 1e-9


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("direction", [-1, 1])
def test_position_updates_and_precision_refinement_use_same_geometry(
    backend, direction
):
    emp = ([0, 0.029, 0.151, 0.283], [2, 1, 2, 1])
    refs = [([0.018, 0.127, 0.241], [1, 2, 1]), ([0.063, 0.193], [2, 1])]
    graph = network(emp, refs, backend)
    weights = np.array([0.25, 0.75])
    graph.solve(weights, with_duals=True)
    graph.dual_cut()
    for dilation in [0.73, 0.41, 1.13]:
        move = lambda data: (
            direction * np.array(data[0]) * dilation + 0.031,
            data[1],
        )
        moved_emp, moved_refs = move(emp), [move(ref) for ref in refs]
        graph.update_positions_and_solve(
            dist(*moved_emp), [dist(*r) for r in moved_refs]
        )
        expected = objective(moved_emp, moved_refs, weights, 7, origin=0.031)
        assert graph.total_cost() == pytest.approx(expected, abs=1e-9)
        cut = graph.dual_cut()
        assert cut["intercept"] + cut["gradient"] @ weights == pytest.approx(
            expected, abs=1e-9
        )
        graph.solve(weights, with_duals=True)
        assert graph.total_cost() == pytest.approx(expected, abs=1e-9)
    # Reconstruct from the updated real geometry, not stale factory gaps.
    graph.refine_intensity_precision(10)
    graph.solve(weights, with_duals=True)
    expected = objective(
        moved_emp, moved_refs, weights, graph.scale_factor(), origin=0.031
    )
    assert graph.total_cost() == pytest.approx(expected, abs=1e-9)
    cut = graph.dual_cut()
    assert cut["intercept"] + cut["gradient"] @ weights == pytest.approx(
        expected, abs=1e-9
    )


@pytest.mark.parametrize("backend", BACKENDS[:4])
def test_unscaled_legacy_chain_still_truncates_each_gap(backend):
    graph = network(([0, 0.4, 0.8], [1, 0, 0]), [([1.2], [1])], backend, scale=None)
    graph.solve([1])
    assert graph.scale_factor() == 1
    assert graph.total_cost() == 0


@pytest.mark.parametrize("backend", ["simplex", "slope-flat"])
def test_cost_error_bounds_use_coordinate_derived_arc_costs(backend):
    # A 0.002 gap rounds UP to 1 at S=10 because its endpoints straddle a
    # coordinate boundary, although independently rounding that gap gives 0.
    emp = ([0, 0.049, 0.051], [0, 1, 0])
    refs = [([0.051], [1])]
    graph = network(emp, refs, backend, scale=10)
    graph.solve([1])
    assert graph.total_cost() == pytest.approx(0.1)
    cut = graph.dual_cut()
    upper, rho, lower = (
        cut["cost_rounding_error"],
        cut["cost_round_up_ratio"],
        cut["cost_lower_slack"],
    )
    assert graph.total_cost() - lower <= 0.002 + upper + 1e-12
    assert upper >= 0.098 - 1e-12
    assert rho == 0
    assert lower >= 0.098 - 1e-12
    # Bounds must use the new real gap, as well as the new integer cost.
    graph.update_positions_and_solve(
        dist([0, 0.048, 0.052], [0, 1, 0]), [dist([0.052], [1])]
    )
    cut = graph.dual_cut()
    upper, rho, lower = (
        cut["cost_rounding_error"],
        cut["cost_round_up_ratio"],
        cut["cost_lower_slack"],
    )
    assert upper >= 0.096 - 1e-12
    assert rho == 0
    assert lower >= 0.096 - 1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_real_cost_lower_allowance_covers_mass_at_other_weights(backend):
    graph = network(([0, 0.049, 0.051], [0, 1, 0]), [([0.051], [1])], backend, scale=10)
    graph.solve([0], with_duals=True)
    cut = graph.dual_cut()
    # At the evaluated point nothing can match, but this is a GLOBAL cut:
    # a probe with positive theoretical mass needs the endpoint allowance.
    assert cut["cost_lower_slack"] >= 0.1 - 1e-12
    for weight in [0, 0.5, 1, 2]:
        lower = (cut["intercept"] + cut["gradient"] @ [weight]) / (
            1 + cut["cost_round_up_ratio"]
        ) - cut["cost_lower_slack"]
        real_cost = 0.8 * max(1, weight) + (0.002 - 0.8) * min(1, weight)
        assert lower <= real_cost + 1e-12


def test_supply_refinement_with_coarse_automatic_scale_retains_span():
    xs = np.linspace(0, 1, 2001)
    mass = np.zeros(len(xs))
    mass[0] = 1
    graph = network(
        (xs, mass), [([1], [1])], "slope-flat", scale=0, intensity_scale=1e6
    )
    graph.solve([1])
    previous_scale = graph.scale_factor()
    graph.refine_intensity_precision(1e9)
    graph.solve([1])
    assert graph.scale_factor() < previous_scale
    assert graph.scale_factor() < 1000
    assert graph.total_cost() == pytest.approx(
        np.floor(0.8 * graph.scale_factor() + 0.5) / graph.scale_factor()
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_random_coarse_coordinate_grids_against_unrestricted_lp(backend):
    model = "simple" if backend == "simplex" else "independent"
    for seed in range(8):
        rng = np.random.default_rng(seed)
        emp = (np.r_[0, np.sort(rng.uniform(0, 1, 4))], rng.integers(1, 4, 5))
        refs = [(np.sort(rng.uniform(0, 1, n)), rng.integers(1, 4, n)) for n in [3, 4]]
        scale = [3, 7, 19][seed % 3]
        graph = network(emp, refs, backend, scale=scale, model=model)
        for point in [[0.25, 0.75], [1, 1], [2, 0.5]]:
            graph.solve(point, with_duals=True)
            expected = objective(emp, refs, point, scale, model)
            assert graph.total_cost() == pytest.approx(expected, abs=1e-9)
            cut = graph.dual_cut()
            assert cut["intercept"] + cut["gradient"] @ point == pytest.approx(
                expected, abs=1e-9
            )
            for probe in [[0, 0], [0.5, 1.5], [1.25, 0.25]]:
                assert (
                    cut["intercept"] + cut["gradient"] @ probe
                    <= objective(emp, refs, probe, scale, model) + 1e-9
                )
