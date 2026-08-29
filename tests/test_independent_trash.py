"""Tests for add_independent_asymmetric_trash (dualdeconv4 semantics).

Every discarded empirical unit pays C_exp and every phantom-filled theoretical
unit pays C_theo, charged independently: an (empirical, theoretical) excess
pair costs C_exp + C_theo, never the annihilating model's min(C_exp, C_theo),
and the match-vs-dump threshold is C_exp + C_theo.
"""

import numpy as np
import pytest

from wnet import WassersteinNetwork
from wnet.distribution import Distribution_1D
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex, SlopeDP

C_EXP = 0.5
C_THEO = 1.0


def d1(pos, intens):
    return Distribution_1D(np.array(pos, dtype=float), np.array(intens, dtype=float))


def make(emp, theos, independent=True, max_distance=1.0):
    W = WassersteinNetwork(
        emp,
        theos,
        DistanceMetric.LINF,
        max_distance,
        force_dense_1d=True,
        round_max_distance=False,
        intensity_scale=1.0,
    )
    W.set_cost_scaling(0)
    if independent:
        W.add_independent_asymmetric_trash(C_EXP, C_THEO)
    else:
        W.add_experimental_trash(C_EXP)
        W.add_theoretical_trash(C_THEO)
    W.build()
    return W


def make_chain(emp, theos, split_distance=1.0, intensity_scale=1.0,
               c_exp=C_EXP, c_theo=C_THEO):
    """Chain factory + SlopeDP backend with independent trash."""
    W = WassersteinNetwork(
        emp,
        theos,
        DistanceMetric.LINF,
        split_distance=split_distance,
        solver=SlopeDP(),
        round_max_distance=False,
        intensity_scale=intensity_scale,
    )
    W.set_cost_scaling(0)
    W.add_independent_asymmetric_trash(c_exp, c_theo)
    W.build()
    return W


def toy():
    # The diagram example: E1 (100.0, 2u), E2 (100.4, 2u), noise N (107.0, 1u);
    # spectrum 0 = one peak at 100.0 (3u), spectrum 1 = one peak at 104.0 (1u).
    # With max_distance 1: matches E1->T1 (d=0, 2u) + E2->T1 (d=0.4, 1u);
    # excess after matching: Xe = 2, Xt = 1.
    emp = d1([100.0, 100.4, 107.0], [2.0, 2.0, 1.0])
    return emp, [d1([100.0], [3.0]), d1([104.0], [1.0])]


def test_toy_excess_charged_in_full():
    # matching cost 1u * 0.4; independent excess: 2*C_exp + 1*C_theo = 2.0.
    W = make(*toy(), independent=True)
    W.solve([1.0, 1.0])
    assert W.total_cost() == pytest.approx(0.4 + 2 * C_EXP + 1 * C_THEO, rel=1e-9)


def test_toy_annihilating_reference():
    # Same network, annihilating model. In this toy both excesses sit on
    # dead-end nodes (N and T2 have no matching arc within max_distance), and
    # isolated nodes are charged in full at the network level in BOTH models
    # — the annihilating pair discount only operates inside a subgraph (see
    # test_match_vs_dump_threshold_is_sum for the in-subgraph contrast).  So
    # the annihilating total here equals the independent one.
    W = make(*toy(), independent=False)
    W.solve([1.0, 1.0])
    assert W.total_cost() == pytest.approx(0.4 + 2 * C_EXP + 1 * C_THEO, rel=1e-9)


def test_match_vs_dump_threshold_is_sum():
    # One empirical unit at 100, one theoretical unit at 100.7: d = 0.7 sits
    # between min(C_exp, C_theo) = 0.5 and C_exp + C_theo = 1.5.
    emp = d1([100.0], [1.0])
    theos = [d1([100.7], [1.0])]
    W_ind = make(emp, theos, independent=True)
    W_ind.solve([1.0])
    assert W_ind.total_cost() == pytest.approx(0.7, rel=1e-9)  # matches
    W_ann = make(emp, theos, independent=False)
    W_ann.solve([1.0])
    assert W_ann.total_cost() == pytest.approx(0.5, rel=1e-9)  # dumps the pair


def test_subgraph_decomposition_is_exact():
    # Two clusters far apart (separate subgraphs): surplus empirical in one,
    # deficit in the other. Independent cost = sum of per-cluster costs.
    emp = d1([100.0, 200.0], [2.0, 1.0])
    theos = [d1([100.0], [1.0]), d1([200.0], [2.0])]
    W = make(emp, theos, independent=True)
    assert W.no_subgraphs() == 2
    W.solve([1.0, 1.0])
    # cluster A: 1 emp surplus -> C_exp; cluster B: 1 theo deficit -> C_theo.
    assert W.total_cost() == pytest.approx(C_EXP + C_THEO, rel=1e-9)


def test_scaling_with_point():
    # Doubling a component's proportion doubles its unfillable capacity and
    # the phantom charge.
    emp = d1([100.0], [1.0])
    theos = [d1([100.0], [1.0])]
    W = make(emp, theos, independent=True)
    W.solve([3.0])
    # 1u matched at d=0; 2u phantom-filled at C_theo each.
    assert W.total_cost() == pytest.approx(2 * C_THEO, rel=1e-9)


def test_gradient_matches_finite_difference():
    emp = d1([100.0, 100.4, 107.0], [2.0, 2.0, 1.0])
    theos = [d1([100.0], [3.0]), d1([104.0], [1.0])]
    # Fine intensity grid so the finite difference sees the slope rather than
    # the flat tread of the integer-supply staircase (trunc(w * t * scale)).
    W = WassersteinNetwork(
        emp, theos, DistanceMetric.LINF, 1.0,
        force_dense_1d=True, round_max_distance=False, intensity_scale=1e6,
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
        # Tolerance bounded by the integer-supply staircase riding inside the
        # finite difference (~0.3% at this eps/scale), not by the gradient.
        assert fd == pytest.approx(grads[k], rel=5e-3, abs=1e-6), (
            f"component {k}: fd={fd} vs grad={grads[k]}"
        )


def test_exclusive_with_other_trash_models():
    emp, theos = toy()
    W = WassersteinNetwork(emp, theos, DistanceMetric.LINF, 1.0,
                           force_dense_1d=True, round_max_distance=False)
    W.add_experimental_trash(C_EXP)
    with pytest.raises(RuntimeError, match="exclusive"):
        W.add_independent_asymmetric_trash(C_EXP, C_THEO)
    W2 = WassersteinNetwork(emp, theos, DistanceMetric.LINF, 1.0,
                            force_dense_1d=True, round_max_distance=False)
    W2.add_independent_asymmetric_trash(C_EXP, C_THEO)
    with pytest.raises(RuntimeError, match="exclusive"):
        W2.add_simple_trash(C_EXP)


def test_forces_dense_factory():
    # Without force_dense_1d the wrapper must still not pick the chain.
    emp, theos = toy()
    W = WassersteinNetwork(emp, theos, DistanceMetric.LINF, 1.0,
                           round_max_distance=False, intensity_scale=1.0)
    W.set_cost_scaling(0)
    W.add_independent_asymmetric_trash(C_EXP, C_THEO)
    W.build()
    W.solve([1.0, 1.0])
    assert W.count_chain_edges() == 0
    assert W.total_cost() == pytest.approx(0.4 + 2 * C_EXP + 1 * C_THEO, rel=1e-9)


# --------------------------------------------------------------------------- #
# Chain factory + SlopeDP backend
# --------------------------------------------------------------------------- #


def test_chain_slopedp_toy_matches_dense():
    W = make_chain(*toy(), split_distance=1.0)
    assert W.count_chain_edges() > 0
    W.solve([1.0, 1.0])
    assert W.total_cost() == pytest.approx(0.4 + 2 * C_EXP + 1 * C_THEO, rel=1e-9)


def test_chain_threshold_is_sum():
    # d = 0.7 sits between min(C_exp, C_theo) = 0.5 and C_exp + C_theo = 1.5:
    # independent trash must match (0.7), where the annihilating chain dumps.
    emp = d1([100.0], [1.0])
    theos = [d1([100.7], [1.0])]
    W = make_chain(emp, theos, split_distance=1.0)
    W.solve([1.0])
    assert W.total_cost() == pytest.approx(0.7, rel=1e-9)


def test_chain_decomposition_and_point_scaling():
    emp = d1([100.0, 200.0], [2.0, 1.0])
    theos = [d1([100.0], [1.0]), d1([200.0], [2.0])]
    W = make_chain(emp, theos, split_distance=1.0)
    assert W.no_subgraphs() == 2
    W.solve([1.0, 1.0])
    assert W.total_cost() == pytest.approx(C_EXP + C_THEO, rel=1e-9)
    # Tripling component 1's proportion: 6 theoretical units against 1
    # empirical -> 1 matched, 5 phantom-filled.
    W.solve([1.0, 3.0])
    assert W.total_cost() == pytest.approx(C_EXP + 5 * C_THEO, rel=1e-9)


def test_chain_gradient_matches_dense_and_fd():
    emp = d1([100.0, 100.4, 107.0], [2.0, 2.0, 1.0])
    theos = [d1([100.0], [3.0]), d1([104.0], [1.0])]
    Wc = make_chain(emp, theos, split_distance=8.0, intensity_scale=1e6)
    Wd = WassersteinNetwork(
        emp, theos, DistanceMetric.LINF, 8.0,
        force_dense_1d=True, round_max_distance=False, intensity_scale=1e6,
    )
    Wd.set_cost_scaling(0)
    Wd.add_independent_asymmetric_trash(C_EXP, C_THEO)
    Wd.build()

    base = [0.7, 1.3]
    Wc.solve(base)
    Wd.solve(base)
    assert Wc.total_cost() == pytest.approx(Wd.total_cost(), rel=1e-9)
    gc = np.asarray(Wc.spectrum_proportion_derivatives())
    gd = np.asarray(Wd.spectrum_proportion_derivatives())
    np.testing.assert_allclose(gc, gd, rtol=1e-9)

    def cost(point):
        Wc.solve(list(point))
        return Wc.total_cost()

    eps = 1e-4
    for k in range(2):
        up = list(base)
        up[k] += eps
        fd = (cost(up) - cost(base)) / eps
        # Tolerance bounded by the integer-supply staircase inside the finite
        # difference, not by the gradient (see the dense FD test above).
        assert fd == pytest.approx(gc[k], rel=5e-3, abs=1e-6), (
            f"component {k}: fd={fd} vs grad={gc[k]}"
        )


def test_chain_fuzz_matches_dense():
    # Random 1-D instances: chain+SlopeDP totals and gradients must agree
    # with dense+NetworkSimplex at random points.
    #
    # Factory parity needs split_distance >= C_exp + C_theo: any transport
    # actually used costs at most the dump-both threshold C_exp + C_theo, so
    # with the cap at or above it the dense per-pair pruning and the chain run
    # partition admit exactly the same useful matches.  Positions land on a
    # 0.25 grid, costs are binary-exact (0.5, 0.8125) with an off-grid sum
    # (1.3125), so quantization is exact and no match-vs-dump ties arise.
    c_exp, c_theo = 0.5, 0.8125
    rng = np.random.default_rng(1234)
    for trial in range(20):
        n_e = rng.integers(2, 12)
        emp = d1(
            np.sort(rng.integers(0, 80, n_e)) * 0.25,
            rng.integers(1, 6, n_e).astype(float),
        )
        theos = []
        for _ in range(int(rng.integers(1, 4))):
            n_t = rng.integers(1, 10)
            theos.append(
                d1(
                    np.sort(rng.integers(0, 80, n_t)) * 0.25,
                    rng.integers(1, 6, n_t).astype(float),
                )
            )
        split = float(rng.choice([1.5, 2.0, 5.0]))
        Wc = make_chain(emp, theos, split_distance=split, intensity_scale=64.0,
                        c_exp=c_exp, c_theo=c_theo)
        Wd = WassersteinNetwork(
            emp, theos, DistanceMetric.LINF, split,
            force_dense_1d=True, round_max_distance=False, intensity_scale=64.0,
        )
        Wd.set_cost_scaling(0)
        Wd.add_independent_asymmetric_trash(c_exp, c_theo)
        Wd.build()
        # Points on the 1/64 grid keep every theo supply integral, so the
        # subgraph convention (truncate supply, then price) and the
        # isolated-node convention (price the real supply, then truncate)
        # coincide — the two factories put different peaks in those buckets.
        for point in (
            [1.0] * len(theos),
            list(rng.integers(6, 160, len(theos)) / 64.0),
        ):
            Wc.solve(point)
            Wd.solve(point)
            assert Wc.total_cost() == pytest.approx(Wd.total_cost(), rel=1e-9), (
                f"trial {trial}, point {point}: chain {Wc.total_cost()} "
                f"vs dense {Wd.total_cost()}"
            )
            gc = np.asarray(Wc.spectrum_proportion_derivatives())
            gd = np.asarray(Wd.spectrum_proportion_derivatives())
            np.testing.assert_allclose(
                gc, gd, rtol=1e-9, atol=1e-9,
                err_msg=f"trial {trial}, point {point}",
            )


def test_chain_non_slopedp_solver_rejected():
    emp, theos = toy()
    W = WassersteinNetwork(
        emp, theos, DistanceMetric.LINF,
        split_distance=1.0, solver=NetworkSimplex(),
        round_max_distance=False, intensity_scale=1.0,
    )
    with pytest.raises(ValueError, match="SlopeDP"):
        W.add_independent_asymmetric_trash(C_EXP, C_THEO)
