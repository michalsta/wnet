"""ConvexSweep: chain-native exact W_p (p >= 1) solver.

Validation strategy (see docs/wp_chain_design.md):

* independent trash is partition-invariant, so sweep-vs-dense parity holds on
  any span;
* annihilating trash couples through the per-subgraph max(E, T) budget, so
  chain runs and dense components must coincide for parity — tested on spans
  smaller than the cap;
* at p == 1 the sweep must agree bit-exactly with SlopeDP on identical chain
  partitions (any span) — two independent algorithms, same semantics;
* gradients additionally checked by finite differences.
"""

import numpy as np
import pytest

from wnet import WassersteinNetwork
from wnet.distribution import Distribution_1D
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import ConvexSweep, SlopeDP

C_EXP = 0.5
C_THEO = 0.8125  # binary-exact; C_EXP + C_THEO off the 0.25 grid (no ties)


def d1(pos, intens):
    return Distribution_1D(np.array(pos, dtype=float), np.array(intens, dtype=float))


def build(emp, theos, kind, independent, p, cap, intensity_scale=64.0):
    kw = dict(round_max_distance=False, intensity_scale=intensity_scale, p=p)
    if kind == "sweep":
        W = WassersteinNetwork(emp, theos, DistanceMetric.LINF,
                               split_distance=cap, solver=ConvexSweep(), **kw)
    elif kind == "slope":
        W = WassersteinNetwork(emp, theos, DistanceMetric.LINF,
                               split_distance=cap, solver=SlopeDP(), **kw)
    else:
        W = WassersteinNetwork(emp, theos, DistanceMetric.LINF, cap,
                               force_dense_1d=True, **kw)
    W.set_cost_scaling(0)
    if independent:
        W.add_independent_asymmetric_trash(C_EXP, C_THEO)
    else:
        W.add_experimental_trash(C_EXP)
        W.add_theoretical_trash(C_THEO)
    W.build()
    return W


def gen(rng, span, n_theos=None):
    def side(maxb=7, maxc=4):
        n = rng.integers(1, maxb + 1)
        pos = np.unique(rng.integers(0, span, n)) * 0.25
        return d1(pos, rng.integers(1, maxc + 1, len(pos)).astype(float))
    emp = side()
    k = n_theos if n_theos is not None else int(rng.integers(1, 4))
    return emp, [side() for _ in range(k)]


def test_threshold_toy():
    # match iff c(d) <= tau: d^2 vs 1.3125 -> R ~ 1.1456
    for d, matched in [(1.0, True), (1.25, False)]:
        W = build(d1([0.0], [1.0]), [d1([d], [1.0])], "sweep", True, 2.0, 5.0,
                  intensity_scale=1.0)
        W.solve([1.0])
        expect = d * d if matched else C_EXP + C_THEO
        assert W.total_cost() == pytest.approx(expect, rel=1e-9)


def test_selection_regression():
    # Design-note F4: pendings x1=0, x2=1, one theo at 1.05: match the nearer
    # x2 (d=0.05), trash x1 (C_EXP).
    W = build(d1([0.0, 1.0], [1.0, 1.0]), [d1([1.05], [1.0])],
              "sweep", True, 2.0, 5.0, intensity_scale=1.0)
    W.solve([1.0])
    # x2 matched at cost 0.05^2, x1 trashed at C_EXP, theo fully filled.
    assert W.total_cost() == pytest.approx(0.05 ** 2 + C_EXP, rel=1e-9)


@pytest.mark.parametrize("p", [2.0, 3.0, 1.5])
def test_independent_matches_dense_any_span(p):
    rng = np.random.default_rng(101)
    for trial in range(40):
        emp, theos = gen(rng, 80)
        point = list(rng.integers(6, 160, len(theos)) / 64.0)
        Wc = build(emp, theos, "sweep", True, p, 5.0)
        Wd = build(emp, theos, "dense", True, p, 5.0)
        Wc.solve(point)
        Wd.solve(point)
        assert Wc.total_cost() == pytest.approx(Wd.total_cost(), rel=1e-9), (
            f"p={p} trial {trial}")
        gc = np.asarray(Wc.spectrum_proportion_derivatives())
        gd = np.asarray(Wd.spectrum_proportion_derivatives())
        np.testing.assert_allclose(gc, gd, rtol=1e-9, atol=1e-6,
                                   err_msg=f"p={p} trial {trial}")


def test_annihilating_matches_dense_on_parity_spans():
    # Chain runs == dense components requires span < cap for the budget-
    # coupled annihilating model (partitions coincide, single run).
    rng = np.random.default_rng(202)
    for trial in range(40):
        emp, theos = gen(rng, 16)   # span <= 4 < cap 5
        point = list(rng.integers(6, 160, len(theos)) / 64.0)
        Wc = build(emp, theos, "sweep", False, 2.0, 5.0)
        Wd = build(emp, theos, "dense", False, 2.0, 5.0)
        Wc.solve(point)
        Wd.solve(point)
        assert Wc.total_cost() == pytest.approx(Wd.total_cost(), rel=1e-9), (
            f"trial {trial}")
        gc = np.asarray(Wc.spectrum_proportion_derivatives())
        gd = np.asarray(Wd.spectrum_proportion_derivatives())
        np.testing.assert_allclose(gc, gd, rtol=1e-9, atol=1e-6,
                                   err_msg=f"trial {trial}")


@pytest.mark.parametrize("independent", [True, False])
def test_p1_matches_slopedp_bitexact(independent):
    # p == 1: the sweep must reproduce SlopeDP on identical chain partitions
    # (any span) — two independent algorithms, identical semantics.
    rng = np.random.default_rng(303)
    for trial in range(40):
        emp, theos = gen(rng, 80)
        point = list(rng.integers(6, 160, len(theos)) / 64.0)
        Wc = build(emp, theos, "sweep", independent, 1.0, 5.0)
        Ws = build(emp, theos, "slope", independent, 1.0, 5.0)
        Wc.solve(point)
        Ws.solve(point)
        assert Wc.total_cost() == Ws.total_cost(), f"trial {trial}"
        gc = np.asarray(Wc.spectrum_proportion_derivatives())
        gs = np.asarray(Ws.spectrum_proportion_derivatives())
        np.testing.assert_array_equal(gc, gs, err_msg=f"trial {trial}")


def test_gradient_matches_finite_difference():
    emp = d1([100.0, 100.4, 107.0], [2.0, 2.0, 1.0])
    theos = [d1([100.0], [3.0]), d1([104.0], [1.0])]
    W = WassersteinNetwork(
        emp, theos, DistanceMetric.LINF, split_distance=8.0,
        solver=ConvexSweep(), p=2.0, round_max_distance=False,
        intensity_scale=1e6,
    )
    W.set_cost_scaling(0)
    W.add_independent_asymmetric_trash(C_EXP, C_THEO)
    W.build()

    def cost(point):
        W.solve(list(point))
        return W.total_cost()

    base = [0.7, 1.3]
    W.solve(base)
    grads = np.asarray(W.spectrum_proportion_derivatives())
    eps = 1e-4
    for k in range(2):
        up = list(base)
        up[k] += eps
        fd = (cost(up) - cost(base)) / eps
        # Tolerance bounded by the integer-supply staircase inside the finite
        # difference, not by the gradient.
        assert fd == pytest.approx(grads[k], rel=5e-3, abs=1e-6), (
            f"component {k}: fd={fd} vs grad={grads[k]}"
        )


def test_non_sweep_solver_on_p2_chain_rejected():
    emp, theos = d1([0.0], [1.0]), [d1([1.0], [1.0])]
    W = WassersteinNetwork(emp, theos, DistanceMetric.LINF,
                           split_distance=5.0, solver=SlopeDP(), p=2.0,
                           round_max_distance=False)
    # the wrapper's explicit-split validation rejects p != 1 without
    # ConvexSweep before the C++ gate is even reached
    W.add_simple_trash(1.0)
    with pytest.raises(ValueError, match="ConvexSweep"):
        W.build()
