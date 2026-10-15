"""ConvexSweep: chain-native exact solver for a Monge quantized W_p model.

Validation strategy (see docs/wp_chain_design.md):

* independent trash is partition-invariant, so sweep-vs-dense parity holds on
  any span, within the cost-model error allowance;
* annihilating trash couples through the per-subgraph max(E, T) budget, so
  chain runs and dense components must coincide for parity — tested on spans
  smaller than the cap;
* at p == 1 compare the coordinate-grid sweep with the additive-gap SlopeDP
  model within the two quantization errors;
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


def build(emp, theos, kind, independent, p, cap, intensity_scale=64.0, storage="flat"):
    kw = dict(round_max_distance=False, intensity_scale=intensity_scale, p=p)
    if kind == "sweep":
        W = WassersteinNetwork(
            emp,
            theos,
            DistanceMetric.LINF,
            split_distance=cap,
            solver=ConvexSweep(storage=storage),
            **kw,
        )
    elif kind == "slope":
        W = WassersteinNetwork(
            emp,
            theos,
            DistanceMetric.LINF,
            split_distance=cap,
            solver=SlopeDP(storage=storage),
            **kw,
        )
    else:
        W = WassersteinNetwork(
            emp, theos, DistanceMetric.LINF, cap, force_dense_1d=True, **kw
        )
    W.set_cost_scaling(0)
    if independent:
        W.add_independent_asymmetric_trash(C_EXP, C_THEO)
    else:
        W.add_experimental_trash(C_EXP)
        W.add_theoretical_trash(C_THEO)
    W.build()
    return W


def model_error(graph, mass):
    return mass * (
        max(
            (sg.convex_sweep_cost_quantization_error() for sg in graph.subgraphs()),
            default=0,
        )
        + 0.5 / graph.scale_factor()
    )


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
        W = build(
            d1([0.0], [1.0]),
            [d1([d], [1.0])],
            "sweep",
            True,
            2.0,
            5.0,
            intensity_scale=1.0,
        )
        W.solve([1.0])
        expect = d * d if matched else C_EXP + C_THEO
        assert W.total_cost() == pytest.approx(expect, abs=model_error(W, 1))


def test_selection_regression():
    # Design-note F4: pendings x1=0, x2=1, one theo at 1.05: match the nearer
    # x2 (d=0.05), trash x1 (C_EXP).
    W = build(
        d1([0.0, 1.0], [1.0, 1.0]),
        [d1([1.05], [1.0])],
        "sweep",
        True,
        2.0,
        5.0,
        intensity_scale=1.0,
    )
    W.solve([1.0])
    # x2 matched at cost 0.05^2, x1 trashed at C_EXP, theo fully filled.
    assert W.total_cost() == pytest.approx(0.05**2 + C_EXP, abs=model_error(W, 2))


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
        assert Wc.total_cost() == pytest.approx(
            Wd.total_cost(), rel=1e-9, abs=model_error(Wc, emp.sum_intensities)
        ), f"p={p} trial {trial}"
        gc = np.asarray(Wc.spectrum_proportion_derivatives())
        gd = np.asarray(Wd.spectrum_proportion_derivatives())
        np.testing.assert_allclose(
            gc,
            gd,
            rtol=1e-9,
            atol=max(
                1e-6,
                4
                * model_error(Wc, emp.sum_intensities)
                * Wc.intensity_scale_factor()
                * sum(t.sum_intensities for t in theos),
            ),
            err_msg=f"p={p} trial {trial}",
        )


def test_annihilating_matches_dense_on_parity_spans():
    # Chain runs == dense components requires span < cap for the budget-
    # coupled annihilating model (partitions coincide, single run).
    rng = np.random.default_rng(202)
    for trial in range(40):
        emp, theos = gen(rng, 16)  # span <= 4 < cap 5
        point = list(rng.integers(6, 160, len(theos)) / 64.0)
        Wc = build(emp, theos, "sweep", False, 2.0, 5.0)
        Wd = build(emp, theos, "dense", False, 2.0, 5.0)
        Wc.solve(point)
        Wd.solve(point)
        assert Wc.total_cost() == pytest.approx(
            Wd.total_cost(), rel=1e-9, abs=model_error(Wc, emp.sum_intensities)
        ), f"trial {trial}"
        gc = np.asarray(Wc.spectrum_proportion_derivatives())
        gd = np.asarray(Wd.spectrum_proportion_derivatives())
        np.testing.assert_allclose(
            gc,
            gd,
            rtol=1e-9,
            atol=max(
                1e-6,
                4
                * model_error(Wc, emp.sum_intensities)
                * Wc.intensity_scale_factor()
                * sum(t.sum_intensities for t in theos),
            ),
            err_msg=f"trial {trial}",
        )


@pytest.mark.parametrize("independent", [True, False])
def test_p1_matches_slopedp_with_quantization_bound(independent):
    # Coordinate-grid costs and sums of rounded gap costs can differ slightly.
    rng = np.random.default_rng(303)
    for trial in range(40):
        emp, theos = gen(rng, 80)
        point = list(rng.integers(6, 160, len(theos)) / 64.0)
        Wc = build(emp, theos, "sweep", independent, 1.0, 5.0)
        Ws = build(emp, theos, "slope", independent, 1.0, 5.0)
        Wc.solve(point)
        Ws.solve(point)
        gap_error = (
            emp.sum_intensities
            * (len(emp.intensities) + sum(len(t.intensities) for t in theos))
            / Ws.scale_factor()
        )
        assert Wc.total_cost() == pytest.approx(
            Ws.total_cost(), abs=model_error(Wc, emp.sum_intensities) + gap_error
        ), f"trial {trial}"
        gc = np.asarray(Wc.spectrum_proportion_derivatives())
        gs = np.asarray(Ws.spectrum_proportion_derivatives())
        np.testing.assert_allclose(
            gc, gs, atol=1e-9, rtol=1e-9, err_msg=f"trial {trial}"
        )


@pytest.mark.parametrize("storage", ["flat", "tree", "augmented"])
def test_gradient_matches_finite_difference(storage):
    emp = d1([100.0, 100.4, 107.0], [2.0, 2.0, 1.0])
    theos = [d1([100.0], [3.0]), d1([104.0], [1.0])]
    W = WassersteinNetwork(
        emp,
        theos,
        DistanceMetric.LINF,
        split_distance=8.0,
        solver=ConvexSweep(storage=storage),
        p=2.0,
        round_max_distance=False,
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
        assert fd == pytest.approx(
            grads[k], rel=5e-3, abs=1e-6
        ), f"component {k}: fd={fd} vs grad={grads[k]}"


@pytest.mark.parametrize("storage", ["flat", "tree", "augmented"])
def test_non_sweep_solver_on_p2_chain_rejected(storage):
    emp, theos = d1([0.0], [1.0]), [d1([1.0], [1.0])]
    W = WassersteinNetwork(
        emp,
        theos,
        DistanceMetric.LINF,
        split_distance=5.0,
        solver=SlopeDP(storage=storage),
        p=2.0,
        round_max_distance=False,
    )
    # the wrapper's explicit-split validation rejects p != 1 without
    # ConvexSweep before the C++ gate is even reached
    W.add_simple_trash(1.0)
    with pytest.raises(ValueError, match="ConvexSweep"):
        W.build()


# ---------------------------------------------------------------------------
# Position updates
#
# The sweep prices pairs from real positions cached in _sweep_real_pos, which
# the build reconstructs from the (immutable) ChainEdge costs. A position
# update therefore has to hand the new positions down explicitly; getting that
# wrong would silently re-solve at the old geometry, so parity against a fresh
# build at the new positions is the property that matters.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [1.5, 2.0, 3.0])
@pytest.mark.parametrize("independent", [True, False])
def test_update_positions_matches_fresh_build(p, independent):
    emp = d1([0.0, 1.0, 4.0], [3.0, 2.0, 5.0])
    theos = [d1([0.5, 3.0, 8.0], [2.0, 4.0, 3.0])]
    moved_emp = d1([0.25, 1.75, 4.5], [3.0, 2.0, 5.0])
    moved_theos = [d1([0.75, 3.25, 7.5], [2.0, 4.0, 3.0])]

    W = build(emp, theos, "sweep", independent, p, cap=100.0)
    W.solve()
    W.update_positions_and_solve(moved_emp, moved_theos)

    fresh = build(moved_emp, moved_theos, "sweep", independent, p, cap=100.0)
    fresh.solve()
    dense = build(moved_emp, moved_theos, "dense", independent, p, cap=100.0)
    dense.solve()

    assert W.total_cost() == pytest.approx(
        fresh.total_cost(),
        rel=1e-9,
        abs=model_error(W, emp.sum_intensities)
        + model_error(fresh, emp.sum_intensities),
    )
    assert W.total_cost() == pytest.approx(
        dense.total_cost(), rel=1e-9, abs=model_error(W, emp.sum_intensities)
    )


def test_update_positions_uses_new_geometry():
    # Guards the specific failure mode: reconstructing positions from the
    # build-time ChainEdge costs would leave the cost unchanged after a move
    # that provably alters it.
    emp = d1([0.0], [2.0])
    theos = [d1([1.0], [2.0])]
    W = build(emp, theos, "sweep", True, 2.0, cap=100.0)
    W.solve()
    before = W.total_cost()
    W.update_positions_and_solve(d1([0.0], [2.0]), [d1([3.0], [2.0])])
    after = W.total_cost()
    assert after != pytest.approx(before, rel=1e-9)
    fresh = build(d1([0.0], [2.0]), [d1([3.0], [2.0])], "sweep", True, 2.0, cap=100.0)
    fresh.solve()
    assert after == pytest.approx(fresh.total_cost(), rel=1e-9)


def test_update_positions_and_get_gradient_not_implemented():
    # The chain position gradient is built on per-gap fluxes, which the sweep
    # neither reports nor could price additively for p != 1.  It must say so
    # rather than fall through to an unrelated error.
    emp = d1([0.0, 1.0], [2.0, 1.0])
    theos = [d1([0.5, 2.0], [1.0, 2.0])]
    W = build(emp, theos, "sweep", True, 2.0, cap=100.0)
    W.solve()
    with pytest.raises(NotImplementedError, match="ConvexSweep"):
        W.update_positions_and_get_gradient(emp, theos)


@pytest.mark.parametrize("storage", ["flat", "tree", "augmented"])
@pytest.mark.parametrize("with_duals", [False, True])
def test_grouped_grid_ties_keep_exact_marginals_and_peak_mapping(storage, with_duals):
    # Different native positions collapse onto one cost-grid coordinate. The
    # target peaks bracket empirical peaks in the original tie order; reversing
    # that order alone cannot ensure a leftward partner for the extra unit.
    emp = d1([100.0, 100.0 + 2e-12], [2.0, 3.0])
    theos = [d1([100.0 - 1e-12], [1.0]), d1([100.0 + 3e-12], [1.0])]
    graph = WassersteinNetwork(
        emp,
        theos,
        DistanceMetric.LINF,
        split_distance=4.0,
        solver=ConvexSweep(storage=storage),
        p=2,
        round_max_distance=False,
        intensity_scale=64,
    )
    graph.set_cost_scaling(0)
    graph.add_independent_asymmetric_trash(C_EXP, C_THEO)
    graph.build()
    # Exercise invalidation, disappearance and reactivation of grouped supply.
    for weights in ([3.0, 1.0], [0.0, 0.0], [0.0, 4.0], [4.0, 0.0], [3.0, 1.0]):
        graph.solve(weights, with_duals=with_duals)
        original = graph.total_cost()
        gradients = np.asarray(graph.spectrum_proportion_derivatives())
        np.testing.assert_allclose(gradients, [-C_EXP, -C_EXP], atol=1e-12)
        signals = graph.signal_part_derivatives()
        assert set(signals) == {0, 1}
        for spectrum in signals.values():
            assert set(spectrum) == {0}
            assert spectrum[0] == pytest.approx(-C_EXP, abs=1e-12)
        if with_duals:
            cut = graph.dual_cut()
            assert cut["intercept"] + cut["gradient"] @ weights <= original + 1e-12
            assert cut["upper_bound"] >= original - 1e-12
        # Exactly one integer capacity unit, avoiding finite-difference tolerance.
        for j in range(2):
            more = list(weights)
            more[j] += 1 / 64
            graph.solve(more)
            assert (graph.total_cost() - original) * 64 == pytest.approx(
                gradients[j],
                abs=1e-10,
            )
