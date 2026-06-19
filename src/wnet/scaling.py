from typing import Optional, Sequence, Union

import numpy as np

from wnet.wnet_cpp import *  # noqa: F401,F403  (CScalerFloat{DIM} etc., DistanceMetric)
from wnet.distances import DistanceMetric
from wnet.distribution import Distribution


def _cpp_class(prefix: str, dim: int):
    """Return the C++ binding class ``C{prefix}Float{dim}``, or raise."""
    cls = globals().get(f"C{prefix}Float{dim}")
    if cls is None:
        raise ValueError(
            f"Dimension {dim} is unavailable in this build of the wnet C++ "
            f"extension (compiled with a smaller WNET_MAX_DIM)."
        )
    return cls


def _check_dims(empirical: Distribution, theoretical: Sequence[Distribution]):
    dim = empirical.dimension
    for t in theoretical:
        if t.dimension != dim:
            raise ValueError(
                f"all spectra must share the same dimension "
                f"(got {t.dimension} and {dim})."
            )
    return dim


def _trash_array(trash_costs) -> np.ndarray:
    arr = np.ascontiguousarray(trash_costs, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("trash_costs must be a 1D array (may be empty).")
    return arr


class _ScalerBase:
    """
    Shared read-only interface for all scaler variants.

    Subclasses set ``self._cpp`` in their constructor and inherit these methods.
    """

    def sf_distance(self) -> float:
        return self._cpp.sf_distance()

    def sf_intensity(self) -> float:
        return self._cpp.sf_intensity()

    def scale_factor(self) -> float:
        """Geometric mean: ``scale_factor**2 == sf_distance * sf_intensity``."""
        return self._cpp.scale_factor()

    def ftol(self) -> float:
        """Suggested optimiser tolerance: ``1 / (sf_distance * sf_intensity)``."""
        return self._cpp.ftol()


# =============================================================================
# WNetAlignScaler
# =============================================================================


class WNetAlignScaler(_ScalerBase):
    """
    Tied single-factor scaler for wnetalign.

    Computes ``sf = sqrt(max_int / (max_sum * max_cost))`` and sets both
    ``sf_distance`` and ``sf_intensity`` equal to it.  No rounding guard —
    the caller is expected to pre-scale positions using this factor.

    Parameters
    ----------
    empirical : Distribution
    theoretical : Sequence[Distribution]
    metric : DistanceMetric
    max_distance : float
    trash_costs : array-like
        Active trash costs (may be empty).
    max_int : float, optional
        int64 overflow cap.  Default ``2**60``.
    """

    def __init__(
        self,
        empirical: Distribution,
        theoretical: Sequence[Distribution],
        metric: DistanceMetric,
        max_distance: Union[int, float],
        trash_costs: Sequence[float],
        max_int: float = float(1 << 60),
    ) -> None:
        theoretical = list(theoretical)
        dim = _check_dims(empirical, theoretical)
        self._cpp = _cpp_class("WNetAlignScaler", dim)(
            empirical.vecdist,
            [t.vecdist for t in theoretical],
            metric,
            float(max_distance),
            _trash_array(trash_costs),
            float(max_int),
        )


# =============================================================================
# WNetDeconvScaler
# =============================================================================


class WNetDeconvScaler(_ScalerBase):
    """
    Intensity-only scaler for wnetdeconv solvers.

    ``sf_distance`` is always 1; ``sf_intensity`` is anchored to the p95 peak
    (the least-intense peak still inside the top-95 % of signal mass, sorted
    most→least intense).  That peak and all larger ones round with at most 10 %
    relative error.  The bottom 5 % tail rounds more freely; the rounding-loss
    guard is set to 0.20 by default.

    Parameters
    ----------
    empirical : Distribution
    theoretical : Sequence[Distribution]
    metric : DistanceMetric
    max_distance : float
    trash_costs : array-like
    max_int : float, optional
        int64 overflow cap.  Default ``2**60``.
    max_dropped_fraction : float, optional
        Total rounding loss guard per spectrum.  ``>= 1.0`` disables it.
        Default 0.20.
    """

    def __init__(
        self,
        empirical: Distribution,
        theoretical: Sequence[Distribution],
        metric: DistanceMetric,
        max_distance: Union[int, float],
        trash_costs: Sequence[float],
        max_int: float = float(1 << 60),
        max_dropped_fraction: float = 0.20,
    ) -> None:
        theoretical = list(theoretical)
        dim = _check_dims(empirical, theoretical)
        self._cpp = _cpp_class("WNetDeconvScaler", dim)(
            empirical.vecdist,
            [t.vecdist for t in theoretical],
            metric,
            float(max_distance),
            _trash_array(trash_costs),
            float(max_int),
            float(max_dropped_fraction),
        )


# =============================================================================
# FineGridScaler
# =============================================================================


class FineGridScaler(_ScalerBase):
    """
    Fine-grid intensity scaler for WassersteinNetwork / WassersteinDistance.

    ``sf_distance`` is always 1 (positions are kept as real ground distances).
    ``sf_intensity`` maps real intensities onto a fine integer supply grid
    targeting ~2\ :sup:`30` total flow without overflowing the int64 cost
    accumulator.  Returns ``sf_intensity == 1`` when all intensities are
    already integer-valued.

    Parameters
    ----------
    empirical : Distribution
    theoretical : Sequence[Distribution]
    metric : DistanceMetric
    max_distance : float
    trash_costs : array-like
    p : float, optional
        Wasserstein order.  Default 1.0.
    max_int : float, optional
        int64 overflow cap.  Default ``2**60``.
    """

    def __init__(
        self,
        empirical: Distribution,
        theoretical: Sequence[Distribution],
        metric: DistanceMetric,
        max_distance: Union[int, float],
        trash_costs: Sequence[float],
        p: float = 1.0,
        max_int: float = float(1 << 60),
    ) -> None:
        theoretical = list(theoretical)
        dim = _check_dims(empirical, theoretical)
        self._cpp = _cpp_class("FineGridScaler", dim)(
            empirical.vecdist,
            [t.vecdist for t in theoretical],
            metric,
            float(max_distance),
            _trash_array(trash_costs),
            float(p),
            float(max_int),
        )


# =============================================================================
# GenericScaler
# =============================================================================


class GenericScaler(_ScalerBase):
    """
    General-purpose scaler with no backward-compatibility constraints.

    Uses the same p-quantile intensity policy as :class:`WNetDeconvScaler` but
    exposes ``p95_frac`` and ``rounding_tol`` as parameters, making it suitable
    for any context.  ``sf_distance`` is always 1.

    Parameters
    ----------
    empirical : Distribution
    theoretical : Sequence[Distribution]
    metric : DistanceMetric
    max_distance : float
    trash_costs : array-like
    p95_frac : float, optional
        Mass fraction defining the "signal" band, in (0, 1].  Default 0.95.
    rounding_tol : float, optional
        Max relative rounding error on the quantile peak, in (0, 1].
        Default 0.10.
    max_dropped_frac : float, optional
        Total rounding-loss guard per spectrum.  ``>= 1.0`` disables it.
        Default 0.20.
    max_int : float, optional
        int64 overflow cap.  Default ``2**60``.
    """

    def __init__(
        self,
        empirical: Distribution,
        theoretical: Sequence[Distribution],
        metric: DistanceMetric,
        max_distance: Union[int, float],
        trash_costs: Sequence[float],
        p95_frac: float = 0.95,
        rounding_tol: float = 0.10,
        max_dropped_frac: float = 0.20,
        max_int: float = float(1 << 60),
    ) -> None:
        theoretical = list(theoretical)
        dim = _check_dims(empirical, theoretical)
        self._cpp = _cpp_class("GenericScaler", dim)(
            empirical.vecdist,
            [t.vecdist for t in theoretical],
            metric,
            float(max_distance),
            _trash_array(trash_costs),
            float(p95_frac),
            float(rounding_tol),
            float(max_dropped_frac),
            float(max_int),
        )
