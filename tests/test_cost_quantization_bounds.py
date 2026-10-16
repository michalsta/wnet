"""dual_cut()'s cost-quantization allowances, checked against real-cost LPs.

dual_cut()'s own intercept/gradient/upper_bound certify the network's fixed
quantized costs.  cost_rounding_error, cost_round_up_ratio and cost_lower_slack
widen them into bounds on the REAL-cost objective:

    lower(y) = (intercept + gradient . y) / (1 + cost_round_up_ratio) - cost_lower_slack
    upper    = upper_bound + cost_rounding_error

Positions are real-valued and the cost grid is deliberately coarse, so the raw
quantized bounds are not valid for the real problem; the widened ones must be.
"""
import numpy as np
import pytest
from scipy.optimize import linprog

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex, SlopeDP, ConvexSweep

N_EMP, N_THEO = 6, 5


def fixture(model, kind, bits, seed=7):
    rng = np.random.default_rng(seed)
    dim = 2 if kind == "dense" else 1
    p = 2.0 if kind == "sweep" else 1.0
    pos = rng.uniform(0, 6, (dim, N_EMP))
    e = rng.uniform(0.2, 2, N_EMP)
    targets = [rng.uniform(0, 6, (dim, N_THEO)) for _ in range(2)]
    intensities = [rng.uniform(0.2, 2, N_THEO) for _ in range(2)]
    if kind == "dense":
        solver, kwargs = NetworkSimplex(), {"max_distance": 2.0, "force_dense_1d": True}
    elif kind == "chain":
        solver, kwargs = SlopeDP(), {"split_distance": 100.0}
    else:
        solver, kwargs = ConvexSweep(), {"split_distance": 100.0}
    graph = WassersteinNetwork(Distribution(pos, e),
                               [Distribution(t, i) for t, i in zip(targets, intensities)],
                               DistanceMetric.L1, p=p, solver=solver, intensity_scale=1e6,
                               round_max_distance=False, cost_precision_bits=bits, **kwargs)
    ce, ct = 1.3, 2.1
    if model == "simple":
        graph.add_simple_trash(ce)
        ct = ce
    elif model == "independent":
        graph.add_independent_asymmetric_trash(ce, ct)
    else:
        graph.add_experimental_trash(ce)
        graph.add_theoretical_trash(ct)
    graph.build()

    target_pos = np.column_stack(targets)
    dist = np.abs(pos[:, :, None] - target_pos[:, None, :]).sum(axis=0) ** p
    # Same candidate pairs as the network: max_distance applies to the L1
    # distance itself (before the power), and only to the dense factory.
    l1 = np.abs(pos[:, :, None] - target_pos[:, None, :]).sum(axis=0)
    pairs = np.argwhere(l1 <= (2.0 if kind == "dense" else np.inf))
    A = np.zeros((N_EMP + 2 * N_THEO, len(pairs)))
    for k, (i, j) in enumerate(pairs):
        A[i, k] = A[N_EMP + j, k] = 1
    tau = ce + ct if model == "independent" else min(ce, ct)
    costs = np.array([dist[i, j] - tau for i, j in pairs])

    def objective(w):
        supply = np.concatenate([i * weight for i, weight in zip(intensities, w)])
        E, T = e.sum(), supply.sum()
        if model == "independent":
            trash = ce * E + ct * T
        else:
            trash = ce * E + ct * max(T - E, 0) if ce <= ct else ct * T + ce * max(E - T, 0)
        res = linprog(costs, A_ub=A, b_ub=np.concatenate([e, supply]), method="highs")
        assert res.success
        return trash + res.fun

    return graph, objective


def bounds(graph, w, y):
    # ConvexSweep recovers its dual only from a traced solve.
    graph.solve(w, with_duals=isinstance(graph._solver, ConvexSweep))
    cut = graph.dual_cut()
    raw_lower = cut["intercept"] + cut["gradient"] @ y
    rho = cut["cost_round_up_ratio"]
    lower = raw_lower / (1 + rho) - cut["cost_lower_slack"]
    upper = cut["upper_bound"] + cut["cost_rounding_error"]
    return cut, raw_lower, lower, upper


@pytest.mark.parametrize("kind", ["dense", "chain", "sweep"])
@pytest.mark.parametrize("model", ["simple", "both", "independent"])
@pytest.mark.parametrize("bits", [3, 6, 40])
def test_widened_bounds_hold_for_real_costs(kind, model, bits):
    if kind == "sweep" and model == "independent":
        pytest.skip("ConvexSweep with independent trash is covered by the sweep suites")
    graph, objective = fixture(model, kind, bits)
    rng = np.random.default_rng(11)
    probes = [np.ones(2), np.array([0.3, 1.7]), *rng.uniform(0, 2.5, (5, 2))]
    truths = [objective(y) for y in probes]
    for w in probes[:4]:
        cut, _, _, upper = bounds(graph, w, w)
        assert cut["cost_rounding_error"] >= 0
        assert 0 <= cut["cost_round_up_ratio"] <= 1
        assert cut["cost_lower_slack"] >= 0
        tol = 1e-7 * max(1.0, abs(objective(w)))
        assert upper >= objective(w) - tol
        for y, truth in zip(probes, truths):
            _, _, lower, _ = bounds(graph, w, y)
            assert lower <= truth + 1e-7 * max(1.0, abs(truth))


@pytest.mark.parametrize("kind", ["dense", "chain"])
def test_raw_quantized_bounds_are_not_real_cost_bounds_on_a_coarse_grid(kind):
    # The reason the allowances exist: with a 3-bit cost grid the quantized
    # certificate misprices the real problem, in one direction or the other.
    graph, objective = fixture("both", kind, 3)
    rng = np.random.default_rng(3)
    violated = False
    for w in [np.ones(2), *rng.uniform(0, 2.5, (8, 2))]:
        cut, raw_lower, _, _ = bounds(graph, w, w)
        truth = objective(w)
        if cut["upper_bound"] < truth - 1e-9 or raw_lower > truth + 1e-9:
            violated = True
    assert violated


def test_fine_grid_allowances_are_negligible():
    graph, objective = fixture("both", "dense", 40)
    cut, _, _, _ = bounds(graph, np.ones(2), np.ones(2))
    assert cut["cost_rounding_error"] < 1e-9
    assert cut["cost_round_up_ratio"] < 1e-9
