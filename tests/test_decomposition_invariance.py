"""Subgraph decomposition must not change the answer.

Splitting a network into connected components is an efficiency device: the
reported cost and derivatives must be exactly what the undecomposed network
would report.  That is not automatic for the annihilating trash models
(``add_simple_trash``, and ``add_experimental_trash`` combined with
``add_theoretical_trash``), whose bill budgets ``max(E, T)`` units:
``sum_c max(E_c, T_c)`` exceeds ``max(sum_c E_c, sum_c T_c)`` whenever two
components hold their excess on opposite sides, so a per-component bill
charges for excess that would have annihilated against excess elsewhere.

The tests below perturb an instance in a way that provably cannot change its
optimum — inserting peaks of zero intensity, which carry no supply and open no
capacity — but which does merge the components into one.  Every reported
quantity must be unmoved.
"""

import numpy as np
import pytest

from wnet import Distribution_1D, WassersteinNetwork
from wnet.distances import DistanceMetric

C_EXP = 5.0
C_THEO = 13.0

# Two clusters 1000 apart: cluster A has empirical excess, cluster B
# theoretical excess, so the two annihilate only if priced together.
EMP_POS, EMP_INT = [0.0, 1000.0], [7.0, 3.0]
THEO_POS, THEO_INT = [1.0, 1001.0], [4.0, 6.0]
MAX_DISTANCE = 10.0

ALL_MODES = ["simple", "both", "experimental", "theoretical", "independent"]
# The one-sided asymmetric models have a separate, pre-existing defect in the
# marginals themselves (they disagree with a re-solve finite difference), so
# only their costs are pinned here.
ANNIHILATING_OR_INDEPENDENT = ["simple", "both", "independent"]


def _add_trash(net, mode):
    if mode == "simple":
        net.add_simple_trash(C_EXP)
    elif mode == "both":
        net.add_experimental_trash(C_EXP)
        net.add_theoretical_trash(C_THEO)
    elif mode == "experimental":
        net.add_experimental_trash(C_EXP)
    elif mode == "theoretical":
        net.add_theoretical_trash(C_THEO)
    elif mode == "independent":
        net.add_independent_asymmetric_trash(C_EXP, C_THEO)
    else:  # pragma: no cover - guarded by the parametrisation
        raise AssertionError(f"unknown trash mode {mode}")


def _bridge(positions, intensities, step):
    """Append zero-intensity peaks on a grid spanning the whole range.

    Zero intensity means no supply and no arc capacity, so the optimum is
    untouched; the peaks exist only to connect the components.
    """
    grid = list(np.arange(min(positions), max(positions) + step, step))
    return list(positions) + grid, list(intensities) + [0.0] * len(grid)


def _solve(mode, bridged):
    emp_pos, emp_int = EMP_POS, EMP_INT
    theo_pos, theo_int = THEO_POS, THEO_INT
    if bridged:
        emp_pos, emp_int = _bridge(emp_pos, emp_int, MAX_DISTANCE / 2)
        theo_pos, theo_int = _bridge(theo_pos, theo_int, MAX_DISTANCE / 2)
    net = WassersteinNetwork(
        Distribution_1D(np.array(emp_pos, float), np.array(emp_int, float)),
        [Distribution_1D(np.array(theo_pos, float), np.array(theo_int, float))],
        DistanceMetric.L1,
        MAX_DISTANCE,
        force_dense_1d=True,
    )
    _add_trash(net, mode)
    net.build()
    net.solve()
    return net


@pytest.mark.parametrize("mode", ALL_MODES)
def test_zero_mass_peaks_do_not_change_cost(mode):
    split, bridged = _solve(mode, False), _solve(mode, True)
    assert split.no_subgraphs() > 1
    assert bridged.no_subgraphs() == 1
    assert split.total_cost() == pytest.approx(bridged.total_cost(), rel=1e-9)


@pytest.mark.parametrize("mode", ANNIHILATING_OR_INDEPENDENT)
def test_zero_mass_peaks_do_not_change_derivatives(mode):
    split, bridged = _solve(mode, False), _solve(mode, True)
    split_sig = split.signal_part_derivatives()[0]
    bridged_sig = bridged.signal_part_derivatives()[0]
    # The bridge peaks are appended, so the original peaks keep their indices.
    for peak in range(len(THEO_POS)):
        assert split_sig[peak] == pytest.approx(bridged_sig[peak], rel=1e-9)
    assert split.spectrum_proportion_derivatives()[0] == pytest.approx(
        bridged.spectrum_proportion_derivatives()[0], rel=1e-9
    )


def test_simple_trash_prices_opposite_excesses_together():
    # 4 units matched in cluster A and 3 in cluster B; the leftover 3
    # empirical and 3 theoretical units annihilate against each other, so the
    # bill is max(10, 10) - 7 = 3 escaping units rather than six charged
    # separately.
    net = _solve("simple", False)
    assert net.no_subgraphs() == 2
    assert net.total_cost() == pytest.approx(4 * 1 + 3 * 1 + 3 * C_EXP)


def test_independent_trash_still_charges_both_sides():
    # The independent model is unaffected: its bill is per side by definition,
    # which is additive over components and so was never decomposition-
    # dependent in the first place.
    net = _solve("independent", False)
    assert net.total_cost() == pytest.approx(4 * 1 + 3 * 1 + 3 * C_EXP + 3 * C_THEO)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_dead_end_only_network_matches_connected_pricing(mode):
    # Every peak is a dead-end (no pair within the cap), so the whole bill
    # comes from nodes excluded from every subgraph.  Those must be priced by
    # the same network-wide rule as nodes inside one.
    net = WassersteinNetwork(
        Distribution_1D(np.array([0.0]), np.array([5.0])),
        [Distribution_1D(np.array([500.0]), np.array([8.0]))],
        DistanceMetric.L1,
        1.0,
        force_dense_1d=True,
    )
    _add_trash(net, mode)
    net.build()
    net.solve()
    expected = {
        # Annihilating: max(5, 8) = 8 units escape by the cheaper route.
        "simple": 8 * C_EXP,
        "both": 5 * C_EXP + 3 * C_THEO,
        # One-sided: only the side with an escape route is budgeted at all.
        "experimental": 5 * C_EXP,
        "theoretical": 8 * C_THEO,
        # Independent: both sides charged in full.
        "independent": 5 * C_EXP + 8 * C_THEO,
    }[mode]
    assert net.no_subgraphs() == 0
    assert net.total_cost() == pytest.approx(expected, rel=1e-9)
