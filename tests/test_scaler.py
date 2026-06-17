"""
Tests for wnet.Scaler — the shared integer-scaling advisor used by wnetdeconv
and wnetalign.

Covers: the four accessors, the wnetdeconv (precision-driven, two-factor) mode,
the explicit-scale-factor override, the wnetalign (tie_factors single-cap) mode,
the int64-overflow cap, the distance-resolution guard, the per-spectrum
intensity-loss guard, dimension dispatch, and input validation.

The C++ Scaler sums intensities with std::accumulate while these references use
numpy; the results agree only to floating-point round-off, so comparisons use a
relative tolerance rather than exact equality.
"""

import numpy as np
import pytest

from wnet import Distribution, Scaler
from wnet.distribution import Distribution_1D
from wnet.distances import DistanceMetric

RTOL = 1e-9
MAX_INT = float(1 << 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def d1(pos, intens):
    return Distribution_1D(np.array(pos, dtype=float), np.array(intens, dtype=float))


def dN(positions, intens):
    """positions: (dim, n) array-like."""
    return Distribution(np.array(positions, dtype=float), np.array(intens, dtype=float))


def expected_auto(emp_sum, theo_sums, max_distance, trash_costs,
                  precision=1e-3, max_int=MAX_INT):
    """Reference re-implementation of the auto (wnetdeconv) factor computation."""
    max_sum = max(emp_sum, sum(theo_sums))
    costs = [max_distance] + list(trash_costs)
    min_c, max_c = min(costs), max(costs)
    sfd = 1.0 / (precision * min_c)
    sfi = 1.0 / (precision * max_sum)
    cap = max_int / (max_c * max_sum)
    prod = sfd * sfi
    if prod > cap:
        shrink = np.sqrt(cap / prod)
        sfd *= shrink
        sfi *= shrink
    return sfd, sfi


def expected_tie(emp_sum, theo_sums, max_distance, trash_costs, max_int=MAX_INT):
    max_sum = max(emp_sum, sum(theo_sums))
    max_c = max([max_distance] + list(trash_costs))
    return float(np.sqrt(max_int / (max_sum * max_c)))


# A pair of well-resolved spectra (large integer-ish intensities → no flooring),
# so formula tests aren't disturbed by the intensity guard.
def big_pair():
    emp = d1([1.0, 2.0, 3.0], [100.0, 200.0, 300.0])           # sum 600
    theo = [d1([1.0, 2.5], [50.0, 150.0])]                     # sum 200
    return emp, theo


# ---------------------------------------------------------------------------
# Accessors & basic invariants
# ---------------------------------------------------------------------------

def test_accessors_return_floats():
    emp, theo = big_pair()
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0], max_dropped_fraction=1.0)
    for v in (s.sf_distance(), s.sf_intensity(), s.scale_factor(), s.ftol()):
        assert isinstance(v, float) and v > 0.0


def test_scale_factor_is_geometric_mean():
    emp, theo = big_pair()
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.5, 0.7], max_dropped_fraction=1.0)
    assert s.scale_factor() == pytest.approx(np.sqrt(s.sf_distance() * s.sf_intensity()), rel=RTOL)


def test_ftol_is_inverse_product():
    emp, theo = big_pair()
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.5], max_dropped_fraction=1.0)
    assert s.ftol() == pytest.approx(1.0 / (s.sf_distance() * s.sf_intensity()), rel=RTOL)


# ---------------------------------------------------------------------------
# Auto (wnetdeconv) mode — matches the reference formula
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("precision", [1e-2, 1e-3, 1e-4, 5e-3])
def test_auto_matches_formula_over_precision(precision):
    emp, theo = big_pair()
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.25, 0.22],
               precision=precision, max_dropped_fraction=1.0)
    sfd, sfi = expected_auto(600.0, [200.0], 1.0, [0.25, 0.22], precision=precision)
    assert s.sf_distance() == pytest.approx(sfd, rel=RTOL)
    assert s.sf_intensity() == pytest.approx(sfi, rel=RTOL)


@pytest.mark.parametrize("max_distance,costs", [
    (1.0, [0.25, 0.22]),
    (0.1, [2.0]),
    (5.0, [5.0, 5.0]),
    (0.25, [0.25, 0.25]),
])
def test_auto_cost_selection(max_distance, costs):
    emp, theo = big_pair()
    s = Scaler(emp, theo, DistanceMetric.LINF, max_distance, costs, max_dropped_fraction=1.0)
    sfd, sfi = expected_auto(600.0, [200.0], max_distance, costs)
    assert s.sf_distance() == pytest.approx(sfd, rel=RTOL)
    assert s.sf_intensity() == pytest.approx(sfi, rel=RTOL)


def test_sf_distance_uses_min_cost():
    # min cost-per-unit-flow drives sf_distance; here the trash cost (0.1) < max_distance (1.0)
    emp, theo = big_pair()
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.1], max_dropped_fraction=1.0)
    assert s.sf_distance() == pytest.approx(1.0 / (1e-3 * 0.1), rel=RTOL)


def test_sf_intensity_uses_max_sum_theoretical_dominant():
    # theoretical sum (200+500=700) dominates empirical (600)
    emp = d1([1.0, 2.0, 3.0], [100.0, 200.0, 300.0])
    theo = [d1([1.0], [200.0]), d1([2.0], [500.0])]
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0], max_dropped_fraction=1.0)
    assert s.sf_intensity() == pytest.approx(1.0 / (1e-3 * 700.0), rel=RTOL)


def test_sf_intensity_uses_max_sum_empirical_dominant():
    emp = d1([1.0, 2.0], [900.0, 900.0])          # sum 1800
    theo = [d1([1.0], [100.0])]                     # sum 100
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0], max_dropped_fraction=1.0)
    assert s.sf_intensity() == pytest.approx(1.0 / (1e-3 * 1800.0), rel=RTOL)


def test_precision_scales_factors_inversely():
    emp, theo = big_pair()
    a = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.5], precision=1e-3, max_dropped_fraction=1.0)
    b = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.5], precision=1e-4, max_dropped_fraction=1.0)
    # 10x finer precision → 10x larger factors
    assert b.sf_distance() == pytest.approx(10.0 * a.sf_distance(), rel=RTOL)
    assert b.sf_intensity() == pytest.approx(10.0 * a.sf_intensity(), rel=RTOL)


def test_multiple_theoretical_sum():
    emp = d1([1.0], [10.0])
    theo = [d1([1.0], [30.0]), d1([2.0], [40.0]), d1([3.0], [50.0])]   # sum 120
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0], max_dropped_fraction=1.0)
    assert s.sf_intensity() == pytest.approx(1.0 / (1e-3 * 120.0), rel=RTOL)


# ---------------------------------------------------------------------------
# Explicit scale_factor override
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sf", [1.0, 100.0, 1e6, 12345.0])
def test_explicit_sets_both_factors_equal(sf):
    emp, theo = big_pair()
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0],
               explicit_scale_factor=sf, max_dropped_fraction=1.0)
    assert s.sf_distance() == pytest.approx(sf, rel=RTOL)
    assert s.sf_intensity() == pytest.approx(sf, rel=RTOL)
    assert s.scale_factor() == pytest.approx(sf, rel=RTOL)


def test_explicit_takes_precedence_over_tie_factors():
    emp, theo = big_pair()
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0],
               explicit_scale_factor=500.0, tie_factors=True, max_dropped_fraction=1.0)
    assert s.sf_distance() == pytest.approx(500.0, rel=RTOL)
    assert s.sf_intensity() == pytest.approx(500.0, rel=RTOL)


def test_explicit_zero_falls_back_to_auto():
    emp, theo = big_pair()
    explicit = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.5],
                      explicit_scale_factor=0.0, max_dropped_fraction=1.0)
    auto = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.5], max_dropped_fraction=1.0)
    assert explicit.sf_distance() == pytest.approx(auto.sf_distance(), rel=RTOL)
    assert explicit.sf_intensity() == pytest.approx(auto.sf_intensity(), rel=RTOL)


# ---------------------------------------------------------------------------
# tie_factors (wnetalign) mode
# ---------------------------------------------------------------------------

def test_tie_factors_single_value():
    emp, theo = big_pair()
    s = Scaler(emp, theo, DistanceMetric.LINF, 2.0, [1.0, 1.0],
               tie_factors=True, max_dropped_fraction=1.0)
    assert s.sf_distance() == pytest.approx(s.sf_intensity(), rel=RTOL)
    assert s.scale_factor() == pytest.approx(s.sf_distance(), rel=RTOL)


def test_tie_factors_matches_overflow_cap_formula():
    emp, theo = big_pair()
    s = Scaler(emp, theo, DistanceMetric.LINF, 2.0, [1.0, 1.0],
               tie_factors=True, max_dropped_fraction=1.0)
    assert s.scale_factor() == pytest.approx(expected_tie(600.0, [200.0], 2.0, [1.0, 1.0]), rel=RTOL)


@pytest.mark.parametrize("precision", [1e-2, 1e-6, 1e-12])
def test_tie_factors_ignores_precision(precision):
    emp, theo = big_pair()
    s = Scaler(emp, theo, DistanceMetric.LINF, 2.0, [1.0],
               precision=precision, tie_factors=True, max_dropped_fraction=1.0)
    assert s.scale_factor() == pytest.approx(expected_tie(600.0, [200.0], 2.0, [1.0]), rel=RTOL)


# ---------------------------------------------------------------------------
# int64 overflow cap
# ---------------------------------------------------------------------------

def test_cap_binds_at_fine_precision():
    emp, theo = big_pair()
    # absurdly fine precision → uncapped product would blow past max_int → cap binds
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.5],
               precision=1e-12, max_dropped_fraction=1.0)
    cap_product = MAX_INT / (max(1.0, 0.5) * 600.0)   # max_c * max_sum
    assert s.sf_distance() * s.sf_intensity() == pytest.approx(cap_product, rel=1e-6)


def test_cap_preserves_factor_ratio():
    emp, theo = big_pair()
    # uniform shrink keeps sf_distance/sf_intensity == max_sum/min_cost
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.5],
               precision=1e-12, max_dropped_fraction=1.0)
    expected_ratio = 600.0 / 0.5    # max_sum / min_cost
    assert s.sf_distance() / s.sf_intensity() == pytest.approx(expected_ratio, rel=1e-6)


def test_no_cap_at_coarse_precision():
    emp, theo = big_pair()
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.5],
               precision=1e-3, max_dropped_fraction=1.0)
    sfd, sfi = expected_auto(600.0, [200.0], 1.0, [0.5], precision=1e-3)
    assert s.sf_distance() == pytest.approx(sfd, rel=RTOL)
    assert s.sf_intensity() == pytest.approx(sfi, rel=RTOL)


def test_smaller_max_int_shrinks_more():
    emp, theo = big_pair()
    big = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.5],
                 precision=1e-9, max_dropped_fraction=1.0, max_int=float(1 << 60))
    small = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.5],
                   precision=1e-9, max_dropped_fraction=1.0,
                   max_int=float(1 << 40), enforce_distance_resolution=False)
    assert small.sf_distance() < big.sf_distance()


# ---------------------------------------------------------------------------
# Intensity-loss guard
# ---------------------------------------------------------------------------

def test_intensity_guard_fires_on_flooring():
    # explicit sf=1.0, intensities 0.4 → trunc(0.4)=0 → 100% dropped
    emp = d1([1.0, 2.0], [0.4, 0.4])
    theo = [d1([1.0], [0.4])]
    with pytest.raises(ValueError, match="quantization"):
        Scaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0], explicit_scale_factor=1.0)


def test_intensity_guard_passes_when_resolved():
    emp = d1([1.0, 2.0], [0.4, 0.4])
    theo = [d1([1.0], [0.4])]
    # sf=1000 → trunc(400)=400, negligible loss
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0], explicit_scale_factor=1000.0)
    assert s.scale_factor() == pytest.approx(1000.0, rel=RTOL)


def test_intensity_guard_disabled_by_threshold_one():
    emp = d1([1.0, 2.0], [0.4, 0.4])
    theo = [d1([1.0], [0.4])]
    # would drop 100% but guard disabled
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0],
               explicit_scale_factor=1.0, max_dropped_fraction=1.0)
    assert s.sf_intensity() == pytest.approx(1.0, rel=RTOL)


def test_intensity_guard_threshold_boundary():
    # one peak 1.5 at sf=1 → trunc=1, drop = 0.5/1.5 = 33.3%
    emp = d1([1.0], [1.5])
    theo = [d1([1.0], [1.5])]
    # 33% < 50% limit → passes
    Scaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0],
           explicit_scale_factor=1.0, max_dropped_fraction=0.5)
    # 33% > 30% limit → fires
    with pytest.raises(ValueError, match="quantization"):
        Scaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0],
               explicit_scale_factor=1.0, max_dropped_fraction=0.3)


def test_intensity_guard_names_worst_theoretical():
    emp = d1([1.0], [1000.0])                      # well resolved
    theo = [d1([1.0], [1000.0]), d1([1.0, 2.0], [0.4, 0.4])]   # second floors
    with pytest.raises(ValueError, match=r"theoretical_spectra\[1\]"):
        Scaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0], explicit_scale_factor=1.0)


def test_intensity_guard_names_empirical():
    emp = d1([1.0, 2.0], [0.4, 0.4])               # empirical floors
    theo = [d1([1.0], [1000.0])]
    with pytest.raises(ValueError, match="empirical_spectrum"):
        Scaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0], explicit_scale_factor=1.0)


def test_intensity_guard_default_threshold_is_five_percent():
    # ~6.7% drop (one peak 1.5 at sf=1, but make total smaller so frac>5%)
    emp = d1([1.0], [1.06])     # trunc(1.06)=1, drop=0.06/1.06=5.66% > 5%
    theo = [d1([1.0], [1000.0])]
    with pytest.raises(ValueError, match="quantization"):
        Scaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0], explicit_scale_factor=1.0)


# ---------------------------------------------------------------------------
# Distance-resolution guard
# ---------------------------------------------------------------------------

def test_distance_guard_fires_under_aggressive_cap():
    # tiny max_int forces a huge shrink → int(min_cost * sf_distance) < 1
    emp, theo = big_pair()
    with pytest.raises(ValueError, match="positive integer|no edges"):
        Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.5],
               precision=1e-3, max_int=1.0, max_dropped_fraction=1.0)


def test_distance_guard_can_be_disabled():
    emp, theo = big_pair()
    # same aggressive cap, guard off → builds
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.5],
               precision=1e-3, max_int=1.0, max_dropped_fraction=1.0,
               enforce_distance_resolution=False)
    assert s.sf_distance() > 0.0


def test_distance_guard_not_applied_in_explicit_mode():
    # explicit tiny factor → int(min_cost*sf) < 1, but explicit mode skips the guard
    emp, theo = big_pair()
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.5],
               explicit_scale_factor=1e-6, max_dropped_fraction=1.0)
    assert s.sf_distance() == pytest.approx(1e-6, rel=RTOL)


def test_distance_guard_not_applied_in_tie_mode():
    emp, theo = big_pair()
    # tie mode with tiny max_int → small factor, but tie skips distance guard
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.5],
               tie_factors=True, max_int=1.0, max_dropped_fraction=1.0)
    assert s.sf_distance() > 0.0


# ---------------------------------------------------------------------------
# Dimension dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [1, 2, 3, 5, 10])
def test_dispatch_across_dimensions(dim):
    rng = np.random.default_rng(dim)
    emp = dN(rng.random((dim, 6)) * 10, rng.uniform(50, 100, 6))
    theo = [dN(rng.random((dim, 5)) * 10, rng.uniform(50, 100, 5))]
    s = Scaler(emp, theo, DistanceMetric.L2, 2.0, [1.0], max_dropped_fraction=1.0)
    assert s.scale_factor() == pytest.approx(np.sqrt(s.sf_distance() * s.sf_intensity()), rel=RTOL)


def test_dimension_mismatch_raises():
    emp = d1([1.0], [1.0])                         # dim 1
    theo = [dN([[1.0], [2.0]], [1.0])]             # dim 2
    with pytest.raises(ValueError, match="dimension"):
        Scaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0], max_dropped_fraction=1.0)


@pytest.mark.parametrize("metric", [DistanceMetric.L1, DistanceMetric.L2, DistanceMetric.LINF])
def test_all_metrics_accepted(metric):
    emp, theo = big_pair()
    s = Scaler(emp, theo, metric, 1.0, [1.0], max_dropped_fraction=1.0)
    # metric does not affect the factor computation today, but must be accepted
    assert s.sf_distance() > 0.0


def test_metric_does_not_change_factors():
    emp, theo = big_pair()
    a = Scaler(emp, theo, DistanceMetric.L1, 1.0, [1.0], max_dropped_fraction=1.0)
    b = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0], max_dropped_fraction=1.0)
    assert a.sf_distance() == pytest.approx(b.sf_distance(), rel=RTOL)
    assert a.sf_intensity() == pytest.approx(b.sf_intensity(), rel=RTOL)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_empty_trash_costs_raises():
    emp, theo = big_pair()
    with pytest.raises(ValueError):
        Scaler(emp, theo, DistanceMetric.LINF, 1.0, [], max_dropped_fraction=1.0)


def test_zero_intensity_spectra_raises():
    emp = d1([1.0, 2.0], [0.0, 0.0])
    theo = [d1([1.0], [0.0])]
    with pytest.raises(ValueError, match="positive"):
        Scaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0], max_dropped_fraction=1.0)


def test_negative_cost_raises():
    emp, theo = big_pair()
    with pytest.raises(ValueError, match="positive"):
        Scaler(emp, theo, DistanceMetric.LINF, 1.0, [-1.0], max_dropped_fraction=1.0)


def test_zero_max_distance_with_zero_cost_raises():
    emp, theo = big_pair()
    with pytest.raises(ValueError, match="positive"):
        Scaler(emp, theo, DistanceMetric.LINF, 0.0, [0.0], max_dropped_fraction=1.0)


def test_trash_costs_accepts_list_and_ndarray():
    emp, theo = big_pair()
    a = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.25, 0.22], max_dropped_fraction=1.0)
    b = Scaler(emp, theo, DistanceMetric.LINF, 1.0, np.array([0.25, 0.22]), max_dropped_fraction=1.0)
    assert a.sf_distance() == pytest.approx(b.sf_distance(), rel=RTOL)
    assert a.sf_intensity() == pytest.approx(b.sf_intensity(), rel=RTOL)


def test_single_trash_cost_symmetric():
    emp, theo = big_pair()
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.7], max_dropped_fraction=1.0)
    sfd, sfi = expected_auto(600.0, [200.0], 1.0, [0.7])
    assert s.sf_distance() == pytest.approx(sfd, rel=RTOL)


# ---------------------------------------------------------------------------
# Cross-mode consistency
# ---------------------------------------------------------------------------

def test_auto_vs_explicit_geometric_mean_consistency():
    # explicit at the auto scale_factor does NOT reproduce the unequal auto
    # factors (explicit ties them), but its scale_factor equals the input.
    emp, theo = big_pair()
    auto = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.25], max_dropped_fraction=1.0)
    sf = auto.scale_factor()
    explicit = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [0.25],
                      explicit_scale_factor=sf, max_dropped_fraction=1.0)
    assert explicit.scale_factor() == pytest.approx(sf, rel=RTOL)
    # auto factors are unequal here (max_sum != min_cost), explicit ties them
    assert auto.sf_distance() != pytest.approx(auto.sf_intensity(), rel=RTOL)
    assert explicit.sf_distance() == pytest.approx(explicit.sf_intensity(), rel=RTOL)
