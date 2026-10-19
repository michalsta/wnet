"""dual_cut()'s transfer bounds between two cost quantizations.

The transfer_* keys relate the quantized-cost objective Q of a network to the
real-cost objective R at every proportion vector w:

    (1 - sigma) R(w) - A  <=  Q(w)  <=  (1 + rho) R(w) + A

with rho = transfer_round_up_ratio, sigma = transfer_round_down_ratio and
A = transfer_absolute_error.  A cut valid for one quantization then transfers
to any other:

    Q_new(w) >= (1 - sigma_new) (cut_old(w) - A_old) / (1 + rho_old) - A_new

R comes from an independent LP.  For the dense factory Q is replicated
exactly (each arc and trash cost rounded to the network's cost grid); for
chain and ConvexSweep networks Q is bracketed by the network's own cut and
upper bound at a very fine supply precision.
"""
import numpy as np
import pytest
from scipy.optimize import linprog

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex, SlopeDP, ConvexSweep

N_EMP, N_THEO = 6, 5
CE, CT = 1.3, 2.1


def llround(x):
    """std::llround for nonnegative costs: halves round away from zero."""
    return np.floor(np.asarray(x, float) + 0.5)


def _has_transfer_keys():
    graph, *_ = build("both", "dense", 40)
    graph.solve(np.ones(2))
    return "transfer_round_down_ratio" in graph.dual_cut()


def build(model, kind, bits, seed=7, intensity_scale=1e6):
    rng = np.random.default_rng(seed)
    dim = 2 if kind == "dense" else 1
    p = 2.0 if kind in ("sweep", "dense_p2") else 1.0
    if kind == "dense_p2":
        dim = 2
    pos = rng.uniform(0, 6, (dim, N_EMP))
    e = rng.uniform(0.2, 2, N_EMP)
    targets = [rng.uniform(0, 6, (dim, N_THEO)) for _ in range(2)]
    intensities = [rng.uniform(0.2, 2, N_THEO) for _ in range(2)]
    if kind in ("dense", "dense_p2"):
        solver, kwargs = NetworkSimplex(), {"max_distance": 2.0, "force_dense_1d": True}
    elif kind == "chain":
        solver, kwargs = SlopeDP(), {"split_distance": 100.0}
    else:
        solver, kwargs = ConvexSweep(), {"split_distance": 100.0}
    graph = WassersteinNetwork(Distribution(pos, e),
                               [Distribution(t, i) for t, i in zip(targets, intensities)],
                               DistanceMetric.L1, p=p, solver=solver,
                               intensity_scale=intensity_scale, round_max_distance=False,
                               cost_precision_bits=bits, **kwargs)
    ce, ct = CE, CT
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
    l1 = np.abs(pos[:, :, None] - target_pos[:, None, :]).sum(axis=0)
    dist = l1 ** p
    dense = kind in ("dense", "dense_p2")
    pairs = np.argwhere(l1 <= (2.0 if dense else np.inf))
    A = np.zeros((N_EMP + 2 * N_THEO, len(pairs)))
    for k, (i, j) in enumerate(pairs):
        A[i, k] = A[N_EMP + j, k] = 1

    def objective(w, quantized=False):
        """R(w), or for the dense factory with quantized=True the exact Q(w)."""
        if quantized:
            S = graph.scale_factor()
            q = lambda c: llround(np.asarray(c) * S) / S  # noqa: E731
            d, c_e, c_t = q(dist), float(q(ce)), float(q(ct))
        else:
            d, c_e, c_t = dist, ce, ct
        tau = c_e + c_t if model == "independent" else min(c_e, c_t)
        costs = np.array([d[i, j] - tau for i, j in pairs])
        supply = np.concatenate([i * weight for i, weight in zip(intensities, w)])
        E, T = e.sum(), supply.sum()
        if model == "independent":
            trash = c_e * E + c_t * T
        elif c_e <= c_t:
            trash = c_e * E + c_t * max(T - E, 0)
        else:
            trash = c_t * T + c_e * max(E - T, 0)
        res = linprog(costs, A_ub=A, b_ub=np.concatenate([e, supply]), method="highs")
        assert res.success
        return trash + res.fun

    return graph, objective, (dist[tuple(pairs.T)], ce, ct)


pytestmark = pytest.mark.skipif(not _has_transfer_keys(),
                                reason="wnet does not report transfer bounds")

KINDS = ["dense", "dense_p2", "chain", "sweep"]
MODELS = ["simple", "both", "independent"]
BITS = [2, 3, 6, 40]
RNG_PROBES = np.random.default_rng(11).uniform(0, 2.5, (6, 2))
PROBES = [np.ones(2), np.array([0.3, 1.7]), np.array([0.0, 0.0]), *RNG_PROBES]


def _skip_unsupported(kind, model):
    if kind == "sweep" and model == "independent":
        pytest.skip("ConvexSweep with independent trash is covered by the sweep suites")


def solve_cut(graph, w):
    graph.solve(w, with_duals=isinstance(graph._solver, ConvexSweep))
    return graph.dual_cut()


def transfer(graph):
    cut = solve_cut(graph, np.ones(2))
    return (cut["transfer_round_up_ratio"], cut["transfer_round_down_ratio"],
            cut["transfer_absolute_error"])


def q_bracket(graph, objective, kind, y):
    """[lo, hi] containing Q(y): exact for dense, network bracket otherwise."""
    cut = solve_cut(graph, y)
    lo = cut["intercept"] + cut["gradient"] @ y
    hi = cut["upper_bound"]
    if kind.startswith("dense"):
        exact = objective(y, quantized=True)
        tol = 1e-6 * max(1.0, abs(exact))
        # The replica must agree with the network's own certificate.
        assert lo <= exact + tol and exact <= hi + tol, (lo, exact, hi)
        return exact, exact
    return lo, hi


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("bits", BITS)
def test_sandwich_holds_at_every_probe(kind, model, bits):
    _skip_unsupported(kind, model)
    graph, objective, _ = build(model, kind, bits)
    rho, sigma, A = transfer(graph)
    assert rho >= 0 and A >= 0 and 0 <= sigma <= 1
    for y in PROBES:
        R = objective(y)
        lo, hi = q_bracket(graph, objective, kind, y)
        tol = 1e-7 * max(1.0, abs(R))
        assert (1 - sigma) * R - A <= hi + tol, (y, R, lo, hi, sigma, A)
        assert lo <= (1 + rho) * R + A + tol, (y, R, lo, hi, rho, A)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("model", MODELS)
def test_transfer_bounds_do_not_depend_on_the_solve_point(kind, model):
    _skip_unsupported(kind, model)
    graph, _, _ = build(model, kind, 3)
    seen = set()
    for w in PROBES:
        cut = solve_cut(graph, w)
        seen.add((cut["transfer_round_up_ratio"], cut["transfer_round_down_ratio"],
                  cut["transfer_absolute_error"]))
    assert len(seen) == 1, seen


@pytest.mark.parametrize("kind", ["dense", "dense_p2"])
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("bits", BITS)
def test_dense_ratios_match_the_rounded_costs(kind, model, bits):
    # rho and sigma are the extreme relative round-up/round-down over every
    # positive matching and trash cost; recompute them from the replica.
    graph, _, (pair_costs, ce, ct) = build(model, kind, bits)
    rho, sigma, A = transfer(graph)
    S = graph.scale_factor()
    real = np.concatenate([pair_costs[pair_costs > 0], [ce, ct] if model != "simple" else [ce]])
    rel = (llround(real * S) - real * S) / (real * S)
    assert A == 0
    assert rho == pytest.approx(max(0.0, rel.max()), rel=1e-9, abs=1e-15)
    assert sigma == pytest.approx(max(0.0, -rel.min()), rel=1e-9, abs=1e-15)


def _transferred(cut_old, t_old, t_new, y):
    rho_old, _, a_old = t_old
    _, sigma_new, a_new = t_new
    raw = cut_old["intercept"] + cut_old["gradient"] @ y
    return (1 - sigma_new) * (raw - a_old) / (1 + rho_old) - a_new, raw


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("bits_old, bits_new", [(40, 2), (40, 3), (2, 40), (3, 6), (6, 3), (2, 3)])
def test_cuts_transfer_between_quantizations(kind, model, bits_old, bits_new):
    _skip_unsupported(kind, model)
    old, _, _ = build(model, kind, bits_old)
    new, objective, _ = build(model, kind, bits_new)
    t_old, t_new = transfer(old), transfer(new)
    brackets = [q_bracket(new, objective, kind, y) for y in PROBES]
    for w in PROBES:
        cut = solve_cut(old, w)
        for y, (lo, hi) in zip(PROBES, brackets):
            moved, _ = _transferred(cut, t_old, t_new, y)
            assert moved <= hi + 1e-7 * max(1.0, abs(hi)), (w, y, moved, lo, hi)


@pytest.mark.parametrize("kind", ["dense", "dense_p2", "chain"])
def test_untransferred_cuts_are_invalid_on_a_coarser_grid(kind):
    # Sensitivity check for the test above: on these instances a raw cut from
    # a fine grid lies strictly above the coarse grid's objective somewhere,
    # so dropping the transfer correction would be caught.
    violated = 0
    for model in MODELS:
        for seed in range(4):
            old, _, _ = build(model, kind, 40, seed=seed)
            new, objective, _ = build(model, kind, 2, seed=seed)
            t_old, t_new = transfer(old), transfer(new)
            for w in PROBES:
                cut = solve_cut(old, w)
                for y in PROBES:
                    lo, hi = q_bracket(new, objective, kind, y)
                    moved, raw = _transferred(cut, t_old, t_new, y)
                    assert moved <= hi + 1e-7 * max(1.0, abs(hi))
                    if raw > hi + 1e-6 * max(1.0, abs(hi)):
                        violated += 1
    assert violated > 0


def test_cost_rounded_to_zero_gives_sigma_one():
    # A positive cost far below one grid step rounds to zero, so the only
    # valid relative lower bound is Q >= 0 * R.
    emp = Distribution(np.array([[0.0, 5.0]]), np.array([1.0, 1.0]))
    theo = Distribution(np.array([[0.01, 5.0]]), np.array([1.0, 1.0]))
    graph = WassersteinNetwork(emp, [theo], DistanceMetric.L1, p=2.0,
                               solver=NetworkSimplex(), max_distance=1.0,
                               force_dense_1d=True, intensity_scale=1e3,
                               round_max_distance=False, cost_precision_bits=1)
    graph.add_experimental_trash(3.0)
    graph.add_theoretical_trash(3.0)
    graph.build()
    graph.solve(np.ones(1))
    cut = graph.dual_cut()
    assert cut["transfer_round_down_ratio"] == pytest.approx(1.0)
    assert graph.total_cost() >= 0


def test_fine_grid_transfer_bounds_are_negligible():
    graph, _, _ = build("both", "dense", 40)
    rho, sigma, A = transfer(graph)
    assert rho < 1e-9 and sigma < 1e-9 and A == 0
