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

from wnet import Distribution, Scaler, WassersteinNetwork
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

def test_empty_trash_costs_allowed():
    # Trash is optional (the pure-distance path has none); cost bounds then come
    # from max_distance alone.
    emp, theo = big_pair()
    s = Scaler(emp, theo, DistanceMetric.LINF, 2.0, [], max_dropped_fraction=1.0)
    # min_cost == max_cost == max_distance == 2.0
    assert s.sf_distance() == pytest.approx(1.0 / (1e-3 * 2.0), rel=RTOL)
    assert s.sf_intensity() == pytest.approx(1.0 / (1e-3 * 600.0), rel=RTOL)


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


# ---------------------------------------------------------------------------
# End-to-end scaling invariance — the property the whole machinery exists for:
# solving the SAME real problem at different sensible scales must recover (very
# nearly) the same real transport cost.  Quantization error shrinks as the
# scale grows, so coarse scales sit slightly off and fine scales converge.
#
# Wiring mirrors the consumers exactly: positions pre-scaled by sf_distance,
# intensities quantized via intensity_scale=sf_intensity, trash costs scaled by
# sf_distance, and the real cost recovered as total_cost() / sf_distance (for
# p == 1 the network's own cost scale_factor() is 1).
# ---------------------------------------------------------------------------

def _scaled(spec, sfd):
    return Distribution(np.asarray(spec.positions) * sfd, np.asarray(spec.intensities))


def _solve_real_cost(emp, theo, max_distance, exp_cost, theo_cost,
                     *, precision=1e-3, explicit=0.0, point=None):
    """Recover the real transport cost solving at the scale the Scaler picks."""
    sc = Scaler(emp, [theo] if isinstance(theo, Distribution) else theo,
                DistanceMetric.LINF, max_distance,
                [exp_cost, theo_cost], precision=precision,
                explicit_scale_factor=explicit, max_dropped_fraction=1.0)
    sfd, sfi = sc.sf_distance(), sc.sf_intensity()
    targets = [theo] if isinstance(theo, Distribution) else theo
    net = WassersteinNetwork(
        _scaled(emp, sfd), [_scaled(t, sfd) for t in targets],
        DistanceMetric.LINF, max_distance=int(round(max_distance * sfd)),
        intensity_scale=sfi,
    )
    net.add_experimental_trash(int(exp_cost * sfd))
    net.add_theoretical_trash(int(theo_cost * sfd))
    net.build()
    net.solve(point if point is not None else [1.0] * len(targets))
    return net.total_cost() / sfd


def _emp_theo_1d():
    emp = Distribution_1D(np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                          np.array([0.30, 0.10, 0.20, 0.20, 0.20]))
    theo = Distribution_1D(np.array([0.2, 1.1, 2.0, 3.3, 4.1]),
                           np.array([0.25, 0.15, 0.20, 0.20, 0.20]))
    return emp, theo


def _emp_theo_2d():
    rng = np.random.default_rng(7)
    emp = Distribution(rng.uniform(0, 5, (2, 8)), rng.uniform(0.05, 0.3, 8))
    theo = Distribution(rng.uniform(0, 5, (2, 7)), rng.uniform(0.05, 0.3, 7))
    return emp, theo


def test_cost_stable_across_precision_sweep_1d():
    emp, theo = _emp_theo_1d()
    costs = [_solve_real_cost(emp, theo, 1.0, 1.0, 1.0, precision=p)
             for p in (1e-3, 1e-4, 1e-5)]
    spread = (max(costs) - min(costs)) / np.mean(costs)
    assert spread < 5e-3, f"costs {costs} spread {spread:.2%} too large"


def test_cost_stable_across_explicit_scale_sweep_1d():
    # CLAUDE.md's "safe" intensity scale range 1e4–1e6 (plus 1e3) must agree.
    emp, theo = _emp_theo_1d()
    costs = [_solve_real_cost(emp, theo, 1.0, 1.0, 1.0, explicit=K)
             for K in (1e3, 1e4, 1e5, 1e6)]
    spread = (max(costs) - min(costs)) / np.mean(costs)
    assert spread < 5e-3, f"costs {costs} spread {spread:.2%} too large"


def test_coarse_scale_within_two_percent_of_fine_1d():
    emp, theo = _emp_theo_1d()
    coarse = _solve_real_cost(emp, theo, 1.0, 1.0, 1.0, precision=1e-2)
    fine = _solve_real_cost(emp, theo, 1.0, 1.0, 1.0, precision=1e-4)
    assert coarse == pytest.approx(fine, rel=2e-2)


def test_cost_stable_across_explicit_scale_sweep_2d():
    emp, theo = _emp_theo_2d()
    costs = [_solve_real_cost(emp, theo, 2.0, 1.0, 1.0, explicit=K)
             for K in (1e3, 1e4, 1e5, 1e6)]
    spread = (max(costs) - min(costs)) / np.mean(costs)
    assert spread < 1e-2, f"costs {costs} spread {spread:.2%} too large"


@pytest.mark.parametrize("point", [[0.5], [1.0], [1.5]])
def test_cost_stable_at_various_points_across_scale(point):
    # at any fixed proportion, the recovered cost is scale-stable
    emp, theo = _emp_theo_1d()
    lo = _solve_real_cost(emp, theo, 1.0, 1.0, 1.0, explicit=1e4, point=point)
    hi = _solve_real_cost(emp, theo, 1.0, 1.0, 1.0, explicit=1e6, point=point)
    assert lo == pytest.approx(hi, rel=5e-3)


def test_all_sensible_scales_near_fine_reference():
    # Quantization error is bounded (not monotone) in the scale: every sensible
    # scale stays within a small band of a fine-scale reference.  (A coarse
    # scale may coincidentally hit the exact value, so we assert closeness, not
    # monotone convergence.)
    emp, theo = _emp_theo_1d()
    ref = _solve_real_cost(emp, theo, 1.0, 1.0, 1.0, explicit=1e7)
    for K in (1e3, 1e4, 1e5, 1e6):
        assert _solve_real_cost(emp, theo, 1.0, 1.0, 1.0, explicit=K) == pytest.approx(ref, rel=1e-2)


# ---------------------------------------------------------------------------
# p-awareness: fractional intensities are finely scaled regardless of p (the
# old "pin to 1 for p != 1" hack is gone); only integer-valued data returns 1.
# ---------------------------------------------------------------------------

def test_fine_grid_integer_valued_p_not_one_returns_one():
    # big_pair has integer-valued intensities → scale 1 (bit-compatible),
    # independent of p.
    emp, theo = big_pair()
    s = Scaler(emp, theo, DistanceMetric.LINF, 1.0, [], p=2.0, fine_grid_intensity=True)
    assert s.sf_intensity() == pytest.approx(1.0, rel=RTOL)


def test_fine_grid_fractional_p_not_one_scales():
    # fractional intensities at p != 1 are now finely scaled (no longer pinned
    # to 1): the cap uses cost_bound**p so the network's cost scale stays >= 1.
    emp = d1([0.0, 5.0], [0.5, 0.5])
    theo = d1([1.0, 6.0], [0.5, 0.5])
    s = Scaler(emp, [theo], DistanceMetric.L2, 10.0, [], p=2.0, fine_grid_intensity=True)
    assert s.sf_intensity() > 1.0
    # cap = max_int / (cost_bound**p * total); cost_bound = min(span=6, md=10) = 6,
    # so cap = 2**60 / (6**2 * 2.0) ≈ huge; target = 2**30 / 2.0 dominates.
    assert s.sf_intensity() == pytest.approx(2.0 ** 30 / 2.0, rel=1e-9)


def test_fine_grid_p_not_one_cost_bound_cap_can_bind():
    # a large ground span at p=2 makes cost_bound**p huge → the overflow cap
    # (not the fine-grid target) limits the intensity scale, but it stays >= 1.
    emp = d1([0.0, 1e7], [0.5, 0.5])
    theo = d1([1.0, 1e7 + 1], [0.5, 0.5])
    s = Scaler(emp, [theo], DistanceMetric.L2, 2e7, [], p=2.0, fine_grid_intensity=True)
    assert s.sf_intensity() >= 1.0


# ---------------------------------------------------------------------------
# Fine-grid intensity policy (the WassersteinNetwork/WassersteinDistance path)
# ---------------------------------------------------------------------------

def test_fine_grid_sf_distance_is_one():
    # fine-grid never pre-scales positions
    emp = d1([0.0, 5.0], [0.5, 0.5])
    theo = d1([1.0, 6.0], [0.5, 0.5])
    s = Scaler(emp, [theo], DistanceMetric.L1, 10.0, [], fine_grid_intensity=True)
    assert s.sf_distance() == pytest.approx(1.0, rel=RTOL)


def test_fine_grid_integer_valued_returns_one():
    # already-integer intensities → scale 1 (bit-compatible with the int backend)
    emp = d1([0.0, 5.0], [3.0, 2.0])
    theo = d1([1.0, 6.0], [2.0, 3.0])
    s = Scaler(emp, [theo], DistanceMetric.L1, 10.0, [], fine_grid_intensity=True)
    assert s.sf_intensity() == pytest.approx(1.0, rel=RTOL)


def test_fine_grid_fractional_uses_fine_grid():
    # fractional intensities → ~2**30 total-flow grid (here total = 1.0+1.0 = 2)
    emp = d1([0.0, 5.0], [0.5, 0.5])
    theo = d1([1.0, 6.0], [0.5, 0.5])
    s = Scaler(emp, [theo], DistanceMetric.L1, 10.0, [], fine_grid_intensity=True)
    assert s.sf_intensity() == pytest.approx(2.0 ** 30 / 2.0, rel=1e-9)


def test_fine_grid_matches_wassersteinnetwork():
    # WassersteinNetwork must derive its intensity scale from this exact policy.
    emp = d1([0.0, 5.0], [0.5, 0.5])
    theo = d1([1.0, 6.0], [0.5, 0.5])
    net = WassersteinNetwork(emp, [theo], DistanceMetric.L1, max_distance=10.0)
    net.build()
    scaler_val = Scaler(emp, [theo], DistanceMetric.L1, 10.0, [],
                        fine_grid_intensity=True).sf_intensity()
    assert net.intensity_scale_factor() == pytest.approx(scaler_val, rel=RTOL)


def test_fine_grid_zero_intensity_returns_one():
    emp = d1([0.0, 1.0], [0.0, 0.0])
    theo = d1([1.0], [0.0])
    s = Scaler(emp, [theo], DistanceMetric.L1, 10.0, [], fine_grid_intensity=True)
    assert s.sf_intensity() == pytest.approx(1.0, rel=RTOL)


# ---------------------------------------------------------------------------
# Integer backend = no scaling at all, p == 1 only (raw C++ network).
# ---------------------------------------------------------------------------

def _int_dist(positions, intens):
    import wnet.wnet_cpp as cpp
    return cpp.CVectorDistribution1(
        np.array(positions, dtype=float), np.array(intens, dtype=np.int64)
    )


def test_integer_backend_forbids_p_not_one():
    import wnet.wnet_cpp as cpp
    D = _int_dist([[0.0, 1.0, 5.0]], [3, 2, 5])
    T = _int_dist([[1.0, 6.0]], [5, 5])
    with pytest.raises(Exception, match="p == 1|double-intensity"):
        cpp.CWassersteinNetworkFactory.create(D, [T], DistanceMetric.L1, 100, 2.0)


def test_integer_backend_rejects_intensity_scaling():
    import wnet.wnet_cpp as cpp
    D = _int_dist([[0.0, 1.0, 5.0]], [3, 2, 5])
    T = _int_dist([[1.0, 6.0]], [5, 5])
    net = cpp.CWassersteinNetworkFactory.create(D, [T], DistanceMetric.L1, 100, 1.0)
    with pytest.raises(Exception, match="does not support intensity scaling"):
        net.set_intensity_scale(2.0)
    net.set_intensity_scale(1.0)  # the identity scale is fine


def test_integer_backend_identity_cost():
    # p == 1, scale 1: total_cost is the exact integer transport cost.
    import wnet.wnet_cpp as cpp
    D = _int_dist([[0.0, 5.0]], [5, 5])
    T = _int_dist([[1.0, 6.0]], [5, 5])
    net = cpp.CWassersteinNetworkFactory.create(D, [T], DistanceMetric.L1, 100, 1.0)
    assert net.intensity_scale_factor() == 1.0
    assert net.scale_factor() == 1
    net.build()
    net.solve([1.0])
    # move 5 units a distance 1 + 5 units a distance 1 = 10
    assert net.total_cost() == 10.0


def test_float_backend_allows_p_not_one_and_scaling():
    import wnet.wnet_cpp as cpp
    D = cpp.CVectorDistributionFloat1(np.array([[0.0, 5.0]]), np.array([0.5, 0.5]))
    T = cpp.CVectorDistributionFloat1(np.array([[1.0, 6.0]]), np.array([0.5, 0.5]))
    net = cpp.CWassersteinNetworkFactory.create(D, [T], DistanceMetric.L2, 100, 2.0)
    net.set_intensity_scale(8.0)  # must not raise on the float backend
    assert net.intensity_scale_factor() == 8.0


# ---------------------------------------------------------------------------
# End-to-end: fractional intensities at p != 1 are now accurate (the pin is
# gone), while integer-intensity p != 1 stays bit-identical and large masses
# don't overflow.
# ---------------------------------------------------------------------------

def test_wd_fractional_p2_accurate():
    from wnet.wasserstein import WassersteinDistance
    # mass 0.5 moves 0->1 (d=1) and 0.5 moves 5->6 (d=1); W2 = (0.5+0.5)^(1/2) = 1
    F1 = Distribution(np.array([[0.0, 5.0]]), np.array([0.5, 0.5]))
    F2 = Distribution(np.array([[1.0, 6.0]]), np.array([0.5, 0.5]))
    assert WassersteinDistance(F1, F2, DistanceMetric.L2, p=2) == pytest.approx(1.0, rel=1e-6)


def test_wd_integer_p2_bit_identical():
    # integer intensities → intensity scale 1 regardless of p (unchanged)
    from wnet.wasserstein import WassersteinDistance
    S1 = Distribution(np.array([[0, 1, 5, 10], [0, 0, 0, 3]]), np.array([10, 5, 5, 5]))
    S2 = Distribution(np.array([[1, 10], [0, 0]]), np.array([20, 5]))
    assert WassersteinDistance(S1, S2, DistanceMetric.L2, p=2) == 11.61895003862225


def test_wd_large_mass_p2_no_overflow():
    from wnet.wasserstein import WassersteinDistance
    G1 = Distribution(np.array([[0.0, 5.0]]), np.array([1234.5, 8765.5]))
    G2 = Distribution(np.array([[1.0, 6.0]]), np.array([1234.5, 8765.5]))
    v = WassersteinDistance(G1, G2, DistanceMetric.L2, p=2)
    # total cost = 1^2*1234.5 + 1^2*8765.5 = 10000; W2 = sqrt(10000) = 100
    assert np.isfinite(v) and v == pytest.approx(100.0, rel=1e-6)
