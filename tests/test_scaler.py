"""
Tests for the scaler hierarchy in wnet.scaling.

Covers each named class (WNetDeconvScaler, WNetAlignScaler, FineGridScaler,
GenericScaler), the shared guard logic, dimension dispatch, input validation,
and end-to-end transport-cost stability.
"""

import numpy as np
import pytest

from wnet import Distribution, WassersteinNetwork
from wnet import WNetAlignScaler, WNetDeconvScaler, FineGridScaler, GenericScaler
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


def big_pair():
    """Spectra with large integer intensities — no flooring noise.
    emp: peaks [100, 200, 300] sum=600; p95 of emp = 100 (last peak to reach 100%).
    theo[0]: peaks [50, 150] sum=200; p95 of theo = 50 (last peak).
    min_p95 = 50  →  sf_intensity = 1/(0.10*50) = 0.2
    """
    emp = d1([1.0, 2.0, 3.0], [100.0, 200.0, 300.0])
    theo = [d1([1.0, 2.5], [50.0, 150.0])]
    return emp, theo


def intensities_of(d):
    return list(d.vecdist.py_get_intensities())


def min_quantile_peak_ref(all_dists, frac):
    """Python reference: minimum p-frac-quantile peak across distributions."""

    def qpeak(intens, frac):
        s = sorted(intens, reverse=True)
        total = sum(s)
        if not total:
            return 0.0
        c = 0.0
        for v in s:
            c += v
            if c >= frac * total:
                return v
        return s[-1] if s else 0.0

    return min(qpeak(intensities_of(d), frac) for d in all_dists)


def max_int_cap_sf_at(target_sf, max_cost, max_sum):
    """Return max_int s.t. overflow_cap = target_sf (forces sf to ≤ target_sf)."""
    return target_sf * max_cost * max_sum


# ---------------------------------------------------------------------------
# Shared accessor invariants (hold for all classes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (WNetDeconvScaler, {"max_dropped_fraction": 1.0}),
        (WNetAlignScaler, {}),
        (FineGridScaler, {}),
        (GenericScaler, {"max_dropped_frac": 1.0}),
    ],
)
def test_accessors_return_positive_floats(cls, kwargs):
    emp, theo = big_pair()
    s = cls(emp, theo, DistanceMetric.LINF, 1.0, [1.0], **kwargs)
    for v in (s.sf_distance(), s.sf_intensity(), s.scale_factor(), s.ftol()):
        assert isinstance(v, float) and v > 0.0


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (WNetDeconvScaler, {"max_dropped_fraction": 1.0}),
        (WNetAlignScaler, {}),
        (FineGridScaler, {}),
        (GenericScaler, {"max_dropped_frac": 1.0}),
    ],
)
def test_scale_factor_is_geometric_mean(cls, kwargs):
    emp, theo = big_pair()
    s = cls(emp, theo, DistanceMetric.LINF, 1.0, [0.5, 0.7], **kwargs)
    assert s.scale_factor() == pytest.approx(
        np.sqrt(s.sf_distance() * s.sf_intensity()), rel=RTOL
    )


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (WNetDeconvScaler, {"max_dropped_fraction": 1.0}),
        (WNetAlignScaler, {}),
        (FineGridScaler, {}),
        (GenericScaler, {"max_dropped_frac": 1.0}),
    ],
)
def test_ftol_is_inverse_product(cls, kwargs):
    emp, theo = big_pair()
    s = cls(emp, theo, DistanceMetric.LINF, 1.0, [0.5], **kwargs)
    assert s.ftol() == pytest.approx(
        1.0 / (s.sf_distance() * s.sf_intensity()), rel=RTOL
    )


# ---------------------------------------------------------------------------
# WNetDeconvScaler — p95-anchored intensity-only scale
# ---------------------------------------------------------------------------


def test_deconv_sf_distance_is_always_one():
    emp, theo = big_pair()
    s = WNetDeconvScaler(
        emp, theo, DistanceMetric.LINF, 1.0, [0.25, 0.22], max_dropped_fraction=1.0
    )
    assert s.sf_distance() == 1.0


def test_deconv_sf_intensity_matches_p95_formula():
    # big_pair: min_p95 = 50 → sf = 1/(0.10*50) = 0.2
    emp, theo = big_pair()
    s = WNetDeconvScaler(
        emp, theo, DistanceMetric.LINF, 1.0, [1.0], max_dropped_fraction=1.0
    )
    min_p95 = min_quantile_peak_ref([emp] + theo, 0.95)  # = 50
    assert s.sf_intensity() == pytest.approx(1.0 / (0.10 * min_p95), rel=RTOL)


def test_deconv_min_p95_takes_minimum_across_spectra():
    # theo has a smaller p95 than emp → sf is set by theo
    emp = d1([1.0], [200.0])  # p95 = 200 (only peak)
    theo = [d1([1.0], [1.0])]  # p95 = 1   → min_p95 = 1
    s = WNetDeconvScaler(
        emp, theo, DistanceMetric.LINF, 1.0, [1.0], max_dropped_fraction=1.0
    )
    assert s.sf_intensity() == pytest.approx(1.0 / (0.10 * 1.0), rel=RTOL)


def test_deconv_overflow_cap_applies():
    # Force the overflow cap to bind below the p95-computed sf.
    emp, theo = big_pair()  # max_sum=600, sf from p95 = 0.2
    max_c, max_sum = 1.0, 600.0
    tiny_max_int = 0.05  # cap = 0.05/(1.0*600) ≈ 8.3e-5 << 0.2
    s = WNetDeconvScaler(
        emp,
        theo,
        DistanceMetric.LINF,
        1.0,
        [max_c],
        max_int=tiny_max_int,
        max_dropped_fraction=1.0,
    )
    expected_cap = tiny_max_int / (max_c * max_sum)
    assert s.sf_intensity() == pytest.approx(expected_cap, rel=1e-6)


def test_deconv_smaller_max_int_lowers_cap():
    emp, theo = big_pair()
    big_sf = WNetDeconvScaler(
        emp,
        theo,
        DistanceMetric.LINF,
        1.0,
        [0.5],
        max_int=float(1 << 60),
        max_dropped_fraction=1.0,
    )
    small_sf = WNetDeconvScaler(
        emp,
        theo,
        DistanceMetric.LINF,
        1.0,
        [0.5],
        max_int=float(1 << 40),
        max_dropped_fraction=1.0,
    )
    assert small_sf.sf_intensity() <= big_sf.sf_intensity()


def test_deconv_empty_trash_costs_allowed():
    emp, theo = big_pair()
    s = WNetDeconvScaler(
        emp, theo, DistanceMetric.LINF, 2.0, [], max_dropped_fraction=1.0
    )
    assert s.sf_distance() == 1.0
    assert s.sf_intensity() > 0.0


# ---------------------------------------------------------------------------
# WNetAlignScaler — tied sqrt(max_int / (max_sum * max_cost)) factor
# ---------------------------------------------------------------------------


def test_align_sf_distance_equals_sf_intensity():
    emp, theo = big_pair()
    s = WNetAlignScaler(emp, theo, DistanceMetric.LINF, 2.0, [1.0, 1.0])
    assert s.sf_distance() == pytest.approx(s.sf_intensity(), rel=RTOL)


def test_align_scale_factor_equals_sf():
    emp, theo = big_pair()
    s = WNetAlignScaler(emp, theo, DistanceMetric.LINF, 2.0, [1.0])
    assert s.scale_factor() == pytest.approx(s.sf_distance(), rel=RTOL)


def test_align_matches_sqrt_cap_formula():
    emp, theo = big_pair()  # max_sum = max(600, 200) = 600
    s = WNetAlignScaler(emp, theo, DistanceMetric.LINF, 2.0, [1.0, 1.0])
    expected = np.sqrt(MAX_INT / (600.0 * 2.0))
    assert s.sf_distance() == pytest.approx(expected, rel=RTOL)
    assert s.sf_intensity() == pytest.approx(expected, rel=RTOL)


def test_align_max_int_ratio_follows_sqrt():
    emp, theo = big_pair()
    a = WNetAlignScaler(
        emp, theo, DistanceMetric.LINF, 2.0, [1.0], max_int=float(1 << 60)
    )
    b = WNetAlignScaler(
        emp, theo, DistanceMetric.LINF, 2.0, [1.0], max_int=float(1 << 40)
    )
    # ratio should match sqrt(2**60) / sqrt(2**40) = 2**10
    assert a.sf_distance() == pytest.approx(b.sf_distance() * (2**10), rel=RTOL)


# ---------------------------------------------------------------------------
# FineGridScaler — ~2^30 total-flow grid for WassersteinNetwork path
# ---------------------------------------------------------------------------


def test_fine_grid_sf_distance_is_one():
    emp = d1([0.0, 5.0], [0.5, 0.5])
    theo = d1([1.0, 6.0], [0.5, 0.5])
    s = FineGridScaler(emp, [theo], DistanceMetric.L1, 10.0, [])
    assert s.sf_distance() == pytest.approx(1.0, rel=RTOL)


def test_fine_grid_integer_valued_returns_one():
    emp = d1([0.0, 5.0], [3.0, 2.0])
    theo = d1([1.0, 6.0], [2.0, 3.0])
    s = FineGridScaler(emp, [theo], DistanceMetric.L1, 10.0, [])
    assert s.sf_intensity() == pytest.approx(1.0, rel=RTOL)


def test_fine_grid_fractional_targets_2_30_grid():
    # total = sum of all intensities across emp + theo = 0.5+0.5 + 0.5+0.5 = 2.0
    # target = 2**30 / total = 2**30 / 2.0 = 2**29
    emp = d1([0.0, 5.0], [0.5, 0.5])
    theo = d1([1.0, 6.0], [0.5, 0.5])
    s = FineGridScaler(emp, [theo], DistanceMetric.L1, 10.0, [])
    assert s.sf_intensity() == pytest.approx(2.0**30 / 2.0, rel=1e-9)


def test_fine_grid_integer_valued_p_not_one_returns_one():
    emp, theo = big_pair()  # integer intensities
    s = FineGridScaler(emp, theo, DistanceMetric.LINF, 1.0, [], p=2.0)
    assert s.sf_intensity() == pytest.approx(1.0, rel=RTOL)


def test_fine_grid_zero_intensity_returns_one():
    emp = d1([0.0, 1.0], [0.0, 0.0])
    theo = d1([1.0], [0.0])
    s = FineGridScaler(emp, [theo], DistanceMetric.L1, 10.0, [])
    assert s.sf_intensity() == pytest.approx(1.0, rel=RTOL)


def test_fine_grid_matches_wassersteinnetwork():
    emp = d1([0.0, 5.0], [0.5, 0.5])
    theo = d1([1.0, 6.0], [0.5, 0.5])
    net = WassersteinNetwork(emp, [theo], DistanceMetric.L1, max_distance=10.0)
    net.build()
    scaler_val = FineGridScaler(emp, [theo], DistanceMetric.L1, 10.0, []).sf_intensity()
    assert net.intensity_scale_factor() == pytest.approx(scaler_val, rel=RTOL)


# ---------------------------------------------------------------------------
# GenericScaler — parameterized p-quantile policy
# ---------------------------------------------------------------------------


def test_generic_rounding_tol_scales_intensity_inversely():
    emp, theo = big_pair()
    a = GenericScaler(
        emp,
        theo,
        DistanceMetric.LINF,
        1.0,
        [0.5],
        rounding_tol=0.10,
        max_dropped_frac=1.0,
    )
    b = GenericScaler(
        emp,
        theo,
        DistanceMetric.LINF,
        1.0,
        [0.5],
        rounding_tol=0.01,
        max_dropped_frac=1.0,
    )
    # 10× finer rounding_tol → 10× larger sf_intensity
    assert b.sf_intensity() == pytest.approx(10.0 * a.sf_intensity(), rel=RTOL)
    assert a.sf_distance() == 1.0 and b.sf_distance() == 1.0


def test_generic_p95_frac_changes_quantile_peak():
    # emp: [400, 300, 200, 100]; total=1000
    # p50: cumsum=400→40%<50%, +300=700→70%≥50% → qpeak=300
    # p95: cumsum=400+300+200=900→90%<95%, +100=1000→100%≥95% → qpeak=100
    # min (theo=[1000]) → min_p50=300, min_p95=100
    # sf(p50)=1/(0.10*300)≈0.033 < sf(p95)=1/(0.10*100)=0.1
    emp = d1([1.0, 2.0, 3.0, 4.0], [400.0, 300.0, 200.0, 100.0])
    theo = [d1([1.0], [1000.0])]
    s50 = GenericScaler(
        emp,
        theo,
        DistanceMetric.LINF,
        1.0,
        [1.0],
        p95_frac=0.50,
        rounding_tol=0.10,
        max_dropped_frac=1.0,
    )
    s95 = GenericScaler(
        emp,
        theo,
        DistanceMetric.LINF,
        1.0,
        [1.0],
        p95_frac=0.95,
        rounding_tol=0.10,
        max_dropped_frac=1.0,
    )
    assert s50.sf_intensity() < s95.sf_intensity()


def test_generic_sf_distance_is_one():
    emp, theo = big_pair()
    s = GenericScaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0], max_dropped_frac=1.0)
    assert s.sf_distance() == 1.0


# ---------------------------------------------------------------------------
# Rounding-loss guard (applies to WNetDeconvScaler and GenericScaler)
# ---------------------------------------------------------------------------


def test_guard_fires_when_rounding_loss_too_high():
    # intensities 0.4, sf forced to 1 → trunc(0.4)=0, 100% dropped
    emp = d1([1.0, 2.0], [0.4, 0.4])  # sum=0.8
    theo = [d1([1.0], [0.4])]  # sum=0.4; max_sum=0.8, max_cost=1.0
    mi = max_int_cap_sf_at(1.0, 1.0, 0.8)
    with pytest.raises(ValueError, match="quantization"):
        WNetDeconvScaler(
            emp,
            theo,
            DistanceMetric.LINF,
            1.0,
            [1.0],
            max_int=mi,
            max_dropped_fraction=0.05,
        )


def test_guard_passes_when_auto_sf_resolves_peaks():
    # WNetDeconvScaler auto-computes a good sf; large integer peaks floor exactly
    emp, theo = big_pair()
    WNetDeconvScaler(emp, theo, DistanceMetric.LINF, 1.0, [1.0])  # no raise


def test_guard_disabled_by_fraction_one():
    emp = d1([1.0, 2.0], [0.4, 0.4])
    theo = [d1([1.0], [0.4])]
    mi = max_int_cap_sf_at(1.0, 1.0, 0.8)
    # guard disabled → no error even though 100% would be dropped
    WNetDeconvScaler(
        emp, theo, DistanceMetric.LINF, 1.0, [1.0], max_int=mi, max_dropped_fraction=1.0
    )


def test_guard_threshold_boundary():
    # one peak 1.5 at sf=1 → trunc=1, drop = 0.5/1.5 ≈ 33.3%
    emp = d1([1.0], [1.5])
    theo = [d1([1.0], [1.5])]
    max_c, max_sum = 1.0, 1.5  # max_sum = max(1.5, 1.5) = 1.5
    mi = max_int_cap_sf_at(1.0, max_c, max_sum)
    # 33% < 50% → passes
    WNetDeconvScaler(
        emp,
        theo,
        DistanceMetric.LINF,
        1.0,
        [max_c],
        max_int=mi,
        max_dropped_fraction=0.50,
    )
    # 33% > 30% → fires
    with pytest.raises(ValueError, match="quantization"):
        WNetDeconvScaler(
            emp,
            theo,
            DistanceMetric.LINF,
            1.0,
            [max_c],
            max_int=mi,
            max_dropped_fraction=0.30,
        )


def test_guard_names_worst_theoretical():
    emp = d1([1.0], [1000.0])
    theo = [d1([1.0], [1000.0]), d1([1.0, 2.0], [0.4, 0.4])]
    max_sum = max(1000.0, 1000.0 + 0.8)  # ≈ 1000.8
    mi = max_int_cap_sf_at(1.0, 1.0, max_sum)
    with pytest.raises(ValueError, match=r"theoretical_spectra\[1\]"):
        WNetDeconvScaler(
            emp,
            theo,
            DistanceMetric.LINF,
            1.0,
            [1.0],
            max_int=mi,
            max_dropped_fraction=0.05,
        )


def test_guard_names_empirical():
    emp = d1([1.0, 2.0], [0.4, 0.4])  # will floor
    theo = [d1([1.0], [1000.0])]
    max_sum = max(0.8, 1000.0)  # = 1000
    mi = max_int_cap_sf_at(1.0, 1.0, max_sum)
    with pytest.raises(ValueError, match="empirical_spectrum"):
        WNetDeconvScaler(
            emp,
            theo,
            DistanceMetric.LINF,
            1.0,
            [1.0],
            max_int=mi,
            max_dropped_fraction=0.05,
        )


def test_deconv_default_guard_threshold_is_twenty_percent():
    # loss ≈ 23.1% > 20% (default) → fires; loss ≈ 9.1% < 20% → passes
    theo = [d1([1.0], [1.0])]

    # 23.1% case: emp=[1.3], sf forced to 1 → trunc=1, loss=0.3/1.3≈23.1%
    max_sum_fires = max(1.3, 1.0)
    mi_fires = max_int_cap_sf_at(1.0, 1.0, max_sum_fires)
    with pytest.raises(ValueError, match="quantization"):
        WNetDeconvScaler(
            d1([1.0], [1.3]), theo, DistanceMetric.LINF, 1.0, [1.0], max_int=mi_fires
        )

    # 9.1% case: emp=[1.1], sf forced to 1 → trunc=1, loss=0.1/1.1≈9.1%
    max_sum_passes = max(1.1, 1.0)
    mi_passes = max_int_cap_sf_at(1.0, 1.0, max_sum_passes)
    WNetDeconvScaler(
        d1([1.0], [1.1]), theo, DistanceMetric.LINF, 1.0, [1.0], max_int=mi_passes
    )  # no raise


# ---------------------------------------------------------------------------
# Dimension dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [1, 2, 3, 5, 10])
def test_dispatch_across_dimensions(dim):
    rng = np.random.default_rng(dim)
    emp = dN(rng.random((dim, 6)) * 10, rng.uniform(50, 100, 6))
    theo = [dN(rng.random((dim, 5)) * 10, rng.uniform(50, 100, 5))]
    s = WNetDeconvScaler(
        emp, theo, DistanceMetric.L2, 2.0, [1.0], max_dropped_fraction=1.0
    )
    assert s.scale_factor() == pytest.approx(
        np.sqrt(s.sf_distance() * s.sf_intensity()), rel=RTOL
    )


def test_dimension_mismatch_raises():
    emp = d1([1.0], [1.0])  # dim 1
    theo = [dN([[1.0], [2.0]], [1.0])]  # dim 2
    with pytest.raises(ValueError, match="dimension"):
        WNetDeconvScaler(
            emp, theo, DistanceMetric.LINF, 1.0, [1.0], max_dropped_fraction=1.0
        )


@pytest.mark.parametrize(
    "metric", [DistanceMetric.L1, DistanceMetric.L2, DistanceMetric.LINF]
)
def test_all_metrics_accepted(metric):
    emp, theo = big_pair()
    s = WNetDeconvScaler(emp, theo, metric, 1.0, [1.0], max_dropped_fraction=1.0)
    assert s.sf_distance() > 0.0


def test_metric_does_not_change_factors():
    emp, theo = big_pair()
    a = WNetDeconvScaler(
        emp, theo, DistanceMetric.L1, 1.0, [1.0], max_dropped_fraction=1.0
    )
    b = WNetDeconvScaler(
        emp, theo, DistanceMetric.LINF, 1.0, [1.0], max_dropped_fraction=1.0
    )
    assert a.sf_intensity() == pytest.approx(b.sf_intensity(), rel=RTOL)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_zero_intensity_spectra_raises():
    emp = d1([1.0, 2.0], [0.0, 0.0])
    theo = [d1([1.0], [0.0])]
    with pytest.raises(ValueError, match="positive"):
        WNetDeconvScaler(
            emp, theo, DistanceMetric.LINF, 1.0, [1.0], max_dropped_fraction=1.0
        )


def test_negative_cost_raises():
    emp, theo = big_pair()
    with pytest.raises(ValueError, match="positive"):
        WNetDeconvScaler(
            emp, theo, DistanceMetric.LINF, 1.0, [-1.0], max_dropped_fraction=1.0
        )


def test_zero_max_distance_and_cost_raises():
    emp, theo = big_pair()
    with pytest.raises(ValueError, match="positive"):
        WNetDeconvScaler(
            emp, theo, DistanceMetric.LINF, 0.0, [0.0], max_dropped_fraction=1.0
        )


def test_trash_costs_accepts_list_and_ndarray():
    emp, theo = big_pair()
    a = WNetDeconvScaler(
        emp, theo, DistanceMetric.LINF, 1.0, [0.25, 0.22], max_dropped_fraction=1.0
    )
    b = WNetDeconvScaler(
        emp,
        theo,
        DistanceMetric.LINF,
        1.0,
        np.array([0.25, 0.22]),
        max_dropped_fraction=1.0,
    )
    assert a.sf_intensity() == pytest.approx(b.sf_intensity(), rel=RTOL)


def test_empty_trash_costs_and_single_cost_both_work():
    emp, theo = big_pair()
    WNetDeconvScaler(emp, theo, DistanceMetric.LINF, 2.0, [], max_dropped_fraction=1.0)
    WNetDeconvScaler(
        emp, theo, DistanceMetric.LINF, 1.0, [0.7], max_dropped_fraction=1.0
    )


def test_generic_invalid_p95_frac_raises():
    emp, theo = big_pair()
    with pytest.raises(ValueError, match="p95_frac"):
        GenericScaler(
            emp,
            theo,
            DistanceMetric.LINF,
            1.0,
            [1.0],
            p95_frac=0.0,
            max_dropped_frac=1.0,
        )


def test_generic_invalid_rounding_tol_raises():
    emp, theo = big_pair()
    with pytest.raises(ValueError, match="rounding_tol"):
        GenericScaler(
            emp,
            theo,
            DistanceMetric.LINF,
            1.0,
            [1.0],
            rounding_tol=0.0,
            max_dropped_frac=1.0,
        )


# ---------------------------------------------------------------------------
# Cross-class invariants
# ---------------------------------------------------------------------------


def test_deconv_sf_distance_one_align_sf_equal():
    emp, theo = big_pair()
    deconv = WNetDeconvScaler(
        emp, theo, DistanceMetric.LINF, 1.0, [0.25], max_dropped_fraction=1.0
    )
    align = WNetAlignScaler(emp, theo, DistanceMetric.LINF, 1.0, [0.25])
    assert deconv.sf_distance() == 1.0
    assert align.sf_distance() == pytest.approx(align.sf_intensity(), rel=RTOL)


def test_generic_at_default_params_equals_deconv():
    # GenericScaler(p95_frac=0.95, rounding_tol=0.10) must match WNetDeconvScaler.
    emp, theo = big_pair()
    g = GenericScaler(
        emp,
        theo,
        DistanceMetric.LINF,
        1.0,
        [0.25],
        p95_frac=0.95,
        rounding_tol=0.10,
        max_dropped_frac=1.0,
    )
    d = WNetDeconvScaler(
        emp, theo, DistanceMetric.LINF, 1.0, [0.25], max_dropped_fraction=1.0
    )
    assert g.sf_intensity() == pytest.approx(d.sf_intensity(), rel=RTOL)


# ---------------------------------------------------------------------------
# End-to-end scaling invariance
# ---------------------------------------------------------------------------


def _emp_theo_1d():
    emp = Distribution_1D(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([0.30, 0.10, 0.20, 0.20, 0.20])
    )
    theo = Distribution_1D(
        np.array([0.2, 1.1, 2.0, 3.3, 4.1]), np.array([0.25, 0.15, 0.20, 0.20, 0.20])
    )
    return emp, theo


def _emp_theo_2d():
    rng = np.random.default_rng(7)
    emp = Distribution(rng.uniform(0, 5, (2, 8)), rng.uniform(0.05, 0.3, 8))
    theo = Distribution(rng.uniform(0, 5, (2, 7)), rng.uniform(0.05, 0.3, 7))
    return emp, theo


def _solve_real_cost(
    emp, theo, max_distance, exp_cost, theo_cost, *, explicit=0.0, point=None
):
    """Solve OT with real positions + network cost scaling."""
    targets = [theo] if isinstance(theo, Distribution) else theo
    if explicit:
        sfi = float(explicit)
        cost_scale = int(explicit)
    else:
        sc = WNetDeconvScaler(
            emp,
            targets,
            DistanceMetric.LINF,
            max_distance,
            [exp_cost, theo_cost],
            max_dropped_fraction=1.0,
        )
        sfi = sc.sf_intensity()
        cost_scale = 0
    net = WassersteinNetwork(
        emp,
        targets,
        DistanceMetric.LINF,
        max_distance=max_distance,
        intensity_scale=sfi,
        round_max_distance=False,
    )
    net.set_cost_scaling(cost_scale)
    net.add_experimental_trash(exp_cost)
    net.add_theoretical_trash(theo_cost)
    net.build()
    net.solve(point if point is not None else [1.0] * len(targets))
    return net.total_cost()


def test_cost_stable_across_explicit_scale_sweep_1d():
    emp, theo = _emp_theo_1d()
    costs = [
        _solve_real_cost(emp, theo, 1.0, 1.0, 1.0, explicit=K)
        for K in (1e3, 1e4, 1e5, 1e6)
    ]
    spread = (max(costs) - min(costs)) / np.mean(costs)
    assert spread < 5e-3, f"costs {costs} spread {spread:.2%}"


def test_cost_stable_across_explicit_scale_sweep_2d():
    emp, theo = _emp_theo_2d()
    costs = [
        _solve_real_cost(emp, theo, 2.0, 1.0, 1.0, explicit=K)
        for K in (1e3, 1e4, 1e5, 1e6)
    ]
    spread = (max(costs) - min(costs)) / np.mean(costs)
    assert spread < 1e-2, f"costs {costs} spread {spread:.2%}"


def test_auto_scale_cost_near_fine_explicit_1d():
    emp, theo = _emp_theo_1d()
    auto_cost = _solve_real_cost(emp, theo, 1.0, 1.0, 1.0)
    ref_cost = _solve_real_cost(emp, theo, 1.0, 1.0, 1.0, explicit=1e5)
    assert auto_cost == pytest.approx(ref_cost, rel=2e-2)


@pytest.mark.parametrize("point", [[0.5], [1.0], [1.5]])
def test_cost_stable_at_various_points_across_scale(point):
    emp, theo = _emp_theo_1d()
    lo = _solve_real_cost(emp, theo, 1.0, 1.0, 1.0, explicit=1e4, point=point)
    hi = _solve_real_cost(emp, theo, 1.0, 1.0, 1.0, explicit=1e6, point=point)
    assert lo == pytest.approx(hi, rel=5e-3)


def test_all_sensible_explicit_scales_near_fine_reference_1d():
    emp, theo = _emp_theo_1d()
    ref = _solve_real_cost(emp, theo, 1.0, 1.0, 1.0, explicit=1e7)
    for K in (1e3, 1e4, 1e5, 1e6):
        assert _solve_real_cost(emp, theo, 1.0, 1.0, 1.0, explicit=K) == pytest.approx(
            ref, rel=1e-2
        )


# ---------------------------------------------------------------------------
# Auto intensity scale must see trash costs declared after __init__
# ---------------------------------------------------------------------------


def test_auto_intensity_scale_sees_late_trash_cost():
    """The auto FineGridScaler runs at build() so it sees add_simple_trash()
    costs declared after __init__.  A huge trash cost must shrink the
    intensity grid to keep the int64 cost accumulator safe; sizing the grid
    blind (trash_costs=[]) overflows the budget at solve()."""
    # Fractional intensities -> the auto grid engages.  Both sides carry the
    # SAME per-peak intensities so truncation quantizes them identically and
    # the supplies balance exactly at any scale — otherwise a 1-unit quantized
    # imbalance would ride the 1e10 trash edge and swamp the cost comparison
    # (an inherent truncation artifact, not what this test is about).
    emp = d1([0.0, 5.0], [0.3, 0.7])
    theo = d1([1.0, 6.0], [0.3, 0.7])
    trash = 1e10

    # The blind scale (what __init__-time sizing used to pick) busts the
    # solve()-time accumulator guard once the trash cost exists.
    blind_scale = FineGridScaler(
        emp, [theo], DistanceMetric.L1, 100.0, trash_costs=[], p=1.0
    ).sf_intensity()
    W_blind = WassersteinNetwork(
        emp, [theo], DistanceMetric.L1, 100, intensity_scale=blind_scale
    )
    W_blind.add_simple_trash(trash)
    W_blind.build()
    with pytest.raises(OverflowError):
        W_blind.solve([1.0])

    # Auto sizing (build-time, trash-aware) must succeed.
    W_auto = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W_auto.add_simple_trash(trash)
    W_auto.build()
    W_auto.solve([1.0])
    auto_cost = W_auto.total_cost()

    # And match an explicitly chosen safe grid.
    W_safe = WassersteinNetwork(
        emp, [theo], DistanceMetric.L1, 100, intensity_scale=1e6
    )
    W_safe.add_simple_trash(trash)
    W_safe.build()
    W_safe.solve([1.0])
    assert auto_cost == pytest.approx(W_safe.total_cost(), rel=1e-4)
