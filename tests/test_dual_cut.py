"""Global cuts checked against independent continuous matching LPs."""
import numpy as np
import pytest
from scipy.optimize import linprog

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex, WarmMode, CostScaling


def fixture(model, scale, mode, seed=42, chain=False, backend=None):
    rng = np.random.default_rng(seed)
    dim = 1 if chain else 2
    pos = rng.integers(0, 6, (dim, 5)).astype(float)
    e = rng.uniform(0.2, 2, 5)
    targets = [rng.integers(0, 6, (dim, 4)).astype(float) for _ in range(2)]
    intensities = [rng.uniform(0.2, 2, 4) for _ in range(2)]
    empirical = Distribution(pos, e)
    theoretical = [Distribution(p, i) for p, i in zip(targets, intensities)]
    solver = NetworkSimplex()
    solver.warm = mode
    kwargs = {"split_distance": 10.0} if chain else {"max_distance": 2.0}
    graph = WassersteinNetwork(empirical, theoretical, DistanceMetric.L1,
                               intensity_scale=scale, solver=solver if backend is None else backend,
                               force_dense_1d=not chain, round_max_distance=False,
                               **kwargs)
    ce, ct = 3, 5
    if model == "simple":
        graph.add_simple_trash(3)
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
    target_pos = np.column_stack(targets)
    dist = np.abs(pos[:, :, None] - target_pos[:, None, :]).sum(axis=0)
    pairs = np.argwhere(dist <= (np.inf if chain else 2))
    A = np.zeros((13, len(pairs)))
    for k, (i, j) in enumerate(pairs):
        A[i, k] = A[5 + j, k] = 1
    tau = ce + ct if model == "independent" else min(ce, ct) if model in ("both", "simple") else max(ce, ct)
    costs = np.array([dist[i, j] - tau for i, j in pairs])

    def objective(w):
        supply = np.concatenate([i * weight for i, weight in zip(intensities, w)])
        E, T = sum(e), sum(supply)
        if model == "independent":
            trash = ce * E + ct * T
        elif model in ("both", "simple"):
            trash = ce * E + ct * max(T - E, 0) if ce <= ct else ct * T + ce * max(E - T, 0)
        else:
            trash = ce * E + ct * T
        if not len(pairs):
            return trash
        result = linprog(costs, A_ub=A, b_ub=np.concatenate([e, supply]), method="highs")
        assert result.success
        return trash + result.fun

    return graph, objective


@pytest.mark.parametrize("model", ["simple", "both", "independent", "experimental", "theoretical"])
@pytest.mark.parametrize("scale", [1, 17.3, 100000])
@pytest.mark.parametrize("mode", [WarmMode.NONE, WarmMode.Dual, WarmMode.LinkCut])
def test_global_bounds_across_supply_and_mass_balance_changes(model, scale, mode):
    graph, objective = fixture(model, scale, mode)
    rng = np.random.default_rng(123)
    probes = [np.zeros(2), np.ones(2), *rng.uniform(0, 3, (12, 2))]
    truths = [objective(w) for w in probes]
    for w in probes[:6]:
        graph.solve(w)
        cut = graph.dual_cut()
        continuous = objective(w)
        assert abs(continuous - graph.total_cost()) <= cut["rounding_error"] + 1e-9
        assert continuous <= cut["upper_bound"] + 1e-9
        if scale >= 100000:
            assert continuous - (cut["intercept"] + cut["gradient"] @ w) <= 2 * cut["rounding_error"] + 1e-8
        for y, value in zip(probes, truths):
            assert cut["intercept"] + cut["gradient"] @ y <= value + 1e-9
        for k in range(graph.no_subgraphs()):
            data = graph.get_subgraph(k).dual_values()
            assert np.array_equal(data["upper_bound_multipliers"], np.minimum(data["reduced_costs"], 0))


@pytest.mark.parametrize("model", ["simple", "both", "experimental", "theoretical"])
def test_chain_cuts(model):
    graph, objective = fixture(model, 100000, WarmMode.Dual, chain=True)
    for w in ([0, 0], [0.1, 0.2], [1, 1], [2, 3]):
        graph.solve(w)
        cut = graph.dual_cut()
        for y in ([0, 0], [0.5, 0.5], [1, 1], [3, 2]):
            assert cut["intercept"] + cut["gradient"] @ y <= objective(y) + 1e-8


def test_dual_queries_require_solve_and_supported_backend():
    graph, _ = fixture("both", 10, WarmMode.Dual)
    with pytest.raises(RuntimeError, match="solve"):
        graph.dual_cut()
    graph.solve([1, 1])
    # Unsupported solvers must not silently emit an uncertified cut.
    other, _ = fixture("both", 10, WarmMode.Dual, backend=CostScaling())
    other.solve([1, 1])
    with pytest.raises(RuntimeError, match="NetworkSimplex"):
        other.dual_cut()


@pytest.mark.parametrize("model", ["simple", "both", "independent", "experimental", "theoretical"])
def test_unlimited_matching_caps_preserve_integer_objective(model):
    for seed in range(10):
        simplex, _ = fixture(model, 17.3, WarmMode.Dual, seed=seed)
        independent, _ = fixture(model, 17.3, WarmMode.Dual, seed=seed, backend=CostScaling())
        for point in np.random.default_rng(seed).uniform(0, 3, (5, 2)):
            simplex.solve(point)
            independent.solve(point)
            assert simplex.wnet.total_cost() == independent.wnet.total_cost()


def test_precision_rebuild_invalidates_solution_and_rejects_unsafe_scale():
    graph, _ = fixture("both", 17.3, WarmMode.Dual)
    graph.solve([0.5, 0.5])
    before = graph.total_cost()
    with pytest.raises(OverflowError, match="flow budget"):
        graph.refine_intensity_precision(1e30)
    assert graph.total_cost() == before
    graph.refine_intensity_precision(100)
    assert graph.intensity_scale_factor() == pytest.approx(1730)
    with pytest.raises(RuntimeError, match="solve"):
        graph.dual_cut()
    graph.solve([0.5, 0.5])
    assert graph.dual_cut()["rounding_error"] >= 0


def test_integer_valued_inputs_expose_certificates_and_can_refine_supplies():
    empirical = Distribution(np.array([[0.0], [0.0]]), np.array([2], dtype=np.int64))
    theoretical = Distribution(np.array([[0.0], [0.0]]), np.array([3], dtype=np.int64))
    graph = WassersteinNetwork(empirical, [theoretical], DistanceMetric.L1,
                               intensity_scale=1, max_distance=1)
    graph.add_simple_trash(3)
    graph.build()
    graph.solve([0.7])
    cut = graph.dual_cut()
    assert cut["intercept"] + cut["gradient"] @ [0.7] <= 0.3
    assert cut["upper_bound"] >= 0.3
    assert graph.get_subgraph(0).dual_values()["potentials"].dtype == np.int64
    graph.solve([1.0])
    exact = graph.dual_cut()
    assert exact["intercept"] + exact["gradient"] @ [1.0] == pytest.approx(3.0)
    graph.refine_intensity_precision(100)
    graph.solve([0.7])
    refined = graph.dual_cut()
    assert refined["intercept"] + refined["gradient"] @ [0.7] == pytest.approx(0.3)
    assert 0.3 <= refined["upper_bound"] < cut["upper_bound"]
