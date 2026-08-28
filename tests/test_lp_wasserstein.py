"""
Tests for order-p (Lp) Wasserstein support.

The network operates in W_p**p units (each matching edge costs ground_distance**p),
so WassersteinNetwork.total_cost() is sum d**p * flow; the high-level
WassersteinDistance() returns its p-th root. p == 1 must reproduce the legacy
1-Wasserstein behaviour exactly.
"""

import numpy as np
import pytest

from wnet.distribution import Distribution, Distribution_1D
from wnet.distances import DistanceMetric
from wnet.wasserstein_network import WassersteinNetwork
from wnet.wasserstein import TruncatedWassersteinDistance
from wnet.wnet_cpp import CWassersteinNetworkFactory


BIG = 10**12
# solve() requires trash edges; this per-unit cost is far above any matching
# cost in these tests, so trash is never used and total_cost is the pure
# matching cost.
HUGE_TRASH = 10**9


def _build_solved(base, targets, metric, p, max_distance=BIG, trash=HUGE_TRASH):
    W = WassersteinNetwork(base, targets, metric, max_distance, p=p)
    if trash is not None:
        W.add_simple_trash(trash)
    W.build()
    W.solve()
    return W


# --------------------------------------------------------------------------
# Single-pair analytic: moving mass m from 0 to a costs m * a**p (W_p**p),
# and the rooted distance is (m * a**p)**(1/p).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("a, m", [(3.0, 10), (5.0, 4), (7.0, 1)])
@pytest.mark.parametrize(
    "metric", [DistanceMetric.L1, DistanceMetric.L2, DistanceMetric.LINF]
)
def test_single_pair_analytic(p, a, m, metric):
    base = Distribution_1D(np.array([0.0]), np.array([m]))
    target = Distribution_1D(np.array([a]), np.array([m]))
    W = _build_solved(base, [target], metric, p)
    assert W.total_cost() == m * int(a) ** p
    # Rooted distance via the high-level API (trash cap well above the pair).
    got = TruncatedWassersteinDistance(base, target, metric, 1000, p=p)
    assert np.isclose(got, (m * a**p) ** (1.0 / p))


# --------------------------------------------------------------------------
# 1D monotone coupling: optimal transport in 1D matches sorted peaks, so the
# W_p**p cost is sum_i mass_i * |x_i - y_i|**p over the sorted pairing.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("p", [1, 2, 3])
def test_1d_monotone_cost(p):
    base = Distribution_1D(np.array([0.0, 10.0]), np.array([1, 1]))
    target = Distribution_1D(np.array([3.0, 12.0]), np.array([1, 1]))
    W = _build_solved(base, [target], DistanceMetric.L1, p)
    # sorted pairing: 0->3 (gap 3), 10->12 (gap 2)
    assert W.total_cost() == 3**p + 2**p


# --------------------------------------------------------------------------
# p != 1 must use the dense factory in 1D (chain gap costs are not additive
# under exponentiation); p == 1 still uses the chain factory.
# --------------------------------------------------------------------------
def test_p_neq_1_forces_dense_in_1d():
    base = Distribution_1D(np.array([0.0, 5.0]), np.array([1, 1]))
    target = Distribution_1D(np.array([2.0, 7.0]), np.array([1, 1]))

    W2 = WassersteinNetwork(base, [target], DistanceMetric.L1, BIG, p=2)
    W2.build()
    assert W2.count_chain_edges() == 0
    assert W2.count_matching_edges() > 0

    # No cap (dense semantics trivially preserved) => chain factory in 1D for
    # p == 1.  A finite max_distance without both-side trash now guarantees
    # dense per-pair semantics, so it would use the dense factory instead.
    W1 = WassersteinNetwork(base, [target], DistanceMetric.L1, None, p=1)
    W1.build()
    assert W1.count_chain_edges() > 0  # chain factory in 1D for p == 1


def test_create_1d_rejects_p_neq_1():
    base = Distribution_1D(np.array([0.0]), np.array([1]))
    target = Distribution_1D(np.array([1.0]), np.array([1]))
    with pytest.raises(Exception):
        CWassersteinNetworkFactory.create_1d(
            base.vecdist, [target.vecdist], DistanceMetric.L1, BIG, 2
        )


# --------------------------------------------------------------------------
# Non-additivity: a chain (additive d**p gaps) would mis-price transport that
# skips an intermediate peak; the dense factory prices it correctly as a
# single |a-c|**p hop.  Mass at 0 must reach 10; the only theoretical peak is
# at 10, with a spectator empirical peak at 4.  Correct W_2**2 cost = 100,
# NOT 4**2 + 6**2 = 52 (what additive chain gaps would give).
# --------------------------------------------------------------------------
def test_non_additive_dense_cost():
    base = Distribution_1D(np.array([0.0]), np.array([1]))
    target = Distribution_1D(np.array([10.0]), np.array([1]))
    W = _build_solved(base, [target], DistanceMetric.L2, 2)
    assert W.total_cost() == 100


# --------------------------------------------------------------------------
# Gradient: for L2 with p, d(d**p)/dx = p * d**(p-1) * (Δ/d).  We compare the
# C++ position gradient against the same quantity recomputed in Python from the
# solved flows (no finite differences, so no truncation noise).
# --------------------------------------------------------------------------
def _expected_position_grads(base_pos, theo_pos, flows, p):
    n_emp = base_pos.shape[1]
    n_theo = theo_pos.shape[1]
    emp = np.zeros((n_emp, 2))
    theo = np.zeros((n_theo, 2))
    e_idx, t_idx, f = flows
    for e, t, fl in zip(e_idx.tolist(), t_idx.tolist(), f.tolist()):
        if fl == 0:
            continue
        delta = base_pos[:, e] - theo_pos[:, t]  # (2,)
        d = np.linalg.norm(delta)
        if d == 0:
            continue
        g = delta / d                       # grad_x of L2
        factor = p * d ** (p - 1)
        contrib = fl * factor * g
        emp[e] += contrib
        theo[t] -= contrib
    return emp, theo


@pytest.mark.parametrize("p", [1, 2, 3])
def test_position_gradient_matches_flows(p):
    base_pos = np.array([[0.0, 10.0], [0.0, 0.0]])
    theo_pos = np.array([[2.0, 8.0], [1.0, 2.0]])
    base = Distribution(base_pos, np.array([3, 2]))
    target = Distribution(theo_pos, np.array([3, 2]))

    W = _build_solved(base, [target], DistanceMetric.L2, p)
    flows = W.flows_for_target(0)
    exp_emp, exp_theo = _expected_position_grads(base_pos, theo_pos, flows, p)

    emp_grad, theo_grads = W.update_positions_and_get_gradient(base, [target])
    assert np.allclose(emp_grad, exp_emp, atol=1e-6)
    assert np.allclose(theo_grads[0], exp_theo, atol=1e-6)


# --------------------------------------------------------------------------
# signal_part_derivatives is the marginal W_p**p cost of +1 unit of signal;
# with d**p edge costs it must equal the brute-force cost delta.
# --------------------------------------------------------------------------
def test_signal_part_derivative_bruteforce():
    p = 2
    trash = 50**p
    base = Distribution_1D(np.array([0.0, 6.0]), np.array([5, 5]))
    target = Distribution_1D(np.array([1.0, 9.0]), np.array([5, 4]))

    W = _build_solved(base, [target], DistanceMetric.L1, p, trash=trash)
    base_cost = W.total_cost()
    derivs = W.signal_part_derivatives()

    # signal_part_derivatives is the marginal cost of +1 theoretical signal,
    # i.e. raising one Theoretical->Sink demand by 1 (empirical side unchanged).
    target_bumped = Distribution_1D(np.array([1.0, 9.0]), np.array([6, 4]))
    Wb = _build_solved(base, [target_bumped], DistanceMetric.L1, p, trash=trash)
    expected_delta = Wb.total_cost() - base_cost
    assert derivs[0][0] == expected_delta


# ==========================================================================
# Fractional p (auto cost scaling).  For p != 1 the integer solver works in
# scaled units (round(scale_factor() * d**p)); the public API divides scale
# back out.  p == 1 stays at scale 1 with truncation (bit-exact legacy).
# ==========================================================================

def test_scale_factor_is_one_for_p1():
    base = Distribution_1D(np.array([0.0, 10.0]), np.array([1, 1]))
    target = Distribution_1D(np.array([3.0, 12.0]), np.array([1, 1]))
    W = _build_solved(base, [target], DistanceMetric.L1, 1)
    assert W.scale_factor() == 1


def test_scale_factor_upscales_small_costs():
    # Small real costs (<= a few) get a large scale so precision is preserved.
    base = Distribution_1D(np.array([0.0]), np.array([1]))
    target = Distribution_1D(np.array([2.0]), np.array([1]))
    W = _build_solved(base, [target], DistanceMetric.L1, 1.5, trash=10)
    assert W.scale_factor() > 1


def test_p1_float_equals_p1_int():
    base = Distribution_1D(np.array([0.0, 10.0]), np.array([2, 3]))
    target = Distribution_1D(np.array([3.0, 12.0]), np.array([2, 3]))
    Wi = _build_solved(base, [target], DistanceMetric.L1, 1)
    Wf = _build_solved(base, [target], DistanceMetric.L1, 1.0)
    assert Wf.scale_factor() == 1
    assert Wf.total_cost() == Wi.total_cost()


@pytest.mark.parametrize(
    "p, g1, g2, expected",
    [
        (1.5, 4.0, 9.0, 4.0**1.5 + 9.0**1.5),   # 8 + 27 = 35
        (2.5, 4.0, 9.0, 4.0**2.5 + 9.0**2.5),   # 32 + 243 = 275
    ],
)
def test_half_integer_exact_oracle(p, g1, g2, expected):
    # Perfect-square gaps make d**p integer-exact, so total_cost() (real units)
    # is exact regardless of the chosen scale.  1D monotone (convex cost p>1).
    base = Distribution_1D(np.array([0.0, 100.0]), np.array([1, 1]))
    target = Distribution_1D(np.array([g1, 100.0 + g2]), np.array([1, 1]))
    W = _build_solved(base, [target], DistanceMetric.L1, p)
    assert W.total_cost() == expected


def test_fractional_precision_beats_truncation():
    # The whole point of pre-round scaling: a single d=2, p=1.5 pair must yield
    # 2**1.5 = 2.828..., NOT 3 (which naive truncation of round(d**p) would give).
    base = Distribution_1D(np.array([0.0]), np.array([1]))
    target = Distribution_1D(np.array([2.0]), np.array([1]))
    W = _build_solved(base, [target], DistanceMetric.L1, 1.5, trash=10)
    assert np.isclose(W.total_cost(), 2.0**1.5, rtol=1e-6)
    assert abs(W.total_cost() - 3.0) > 0.1


@pytest.mark.parametrize("bad_p", [0.0, 0.5, 0.999, -2.0, float("inf"), float("nan")])
def test_invalid_p_rejected(bad_p):
    base = Distribution_1D(np.array([0.0]), np.array([1]))
    target = Distribution_1D(np.array([1.0]), np.array([1]))
    # Python entry point.
    with pytest.raises(ValueError):
        WassersteinNetwork(base, [target], DistanceMetric.L1, BIG, p=bad_p)
    # C++ factory directly (incl. +inf, which must not slip through to build).
    with pytest.raises(Exception):
        CWassersteinNetworkFactory.create(
            base.vecdist, [target.vecdist], DistanceMetric.L1, BIG, bad_p
        )


def test_fractional_p_forces_dense_and_rejects_chain():
    base = Distribution_1D(np.array([0.0, 5.0]), np.array([1, 1]))
    target = Distribution_1D(np.array([2.0, 7.0]), np.array([1, 1]))
    W = WassersteinNetwork(base, [target], DistanceMetric.L1, BIG, p=1.5)
    W.build()
    assert W.count_chain_edges() == 0
    assert W.count_matching_edges() > 0
    with pytest.raises(Exception):
        CWassersteinNetworkFactory.create_1d(
            base.vecdist, [target.vecdist], DistanceMetric.L1, BIG, 1.5
        )


@pytest.mark.parametrize("p", [1.5, 2.5])
def test_position_gradient_fractional_p(p):
    base_pos = np.array([[0.0, 10.0], [0.0, 0.0]])
    theo_pos = np.array([[2.0, 8.0], [1.0, 2.0]])
    base = Distribution(base_pos, np.array([3, 2]))
    target = Distribution(theo_pos, np.array([3, 2]))

    W = _build_solved(base, [target], DistanceMetric.L2, p)
    flows = W.flows_for_target(0)
    exp_emp, exp_theo = _expected_position_grads(base_pos, theo_pos, flows, p)
    emp_grad, theo_grads = W.update_positions_and_get_gradient(base, [target])
    assert np.allclose(emp_grad, exp_emp, atol=1e-6)
    assert np.allclose(theo_grads[0], exp_theo, atol=1e-6)


def test_position_gradient_independent_of_scale():
    # Gradients are computed in real units; changing the scale (via trash size,
    # which moves C_max) must not change them.
    base_pos = np.array([[0.0, 10.0], [0.0, 0.0]])
    theo_pos = np.array([[2.0, 8.0], [1.0, 2.0]])
    base = Distribution(base_pos, np.array([3, 2]))
    target = Distribution(theo_pos, np.array([3, 2]))

    Wa = _build_solved(base, [target], DistanceMetric.L2, 1.5, trash=10)
    Wb = _build_solved(base, [target], DistanceMetric.L2, 1.5, trash=10**6)
    assert Wa.scale_factor() != Wb.scale_factor()
    ga, _ = Wa.update_positions_and_get_gradient(base, [target])
    gb, _ = Wb.update_positions_and_get_gradient(base, [target])
    assert np.allclose(ga, gb, atol=1e-9)
