from typing import Optional, Sequence, Union

import numpy as np

from wnet.wnet_cpp import *  # noqa: F401,F403  (CScalerFloat{DIM}, DistanceMetric)
from wnet.distances import DistanceMetric
from wnet.distribution import Distribution


class Scaler:
    """
    Single source of truth for the distance/intensity scale factors that turn
    real-valued spectra into the scaled-integer min-cost-flow network wnet
    solves.

    Defaults reproduce wnetdeconv's precision-driven behaviour (two independent
    factors from a relative ``precision`` target, an int64-overflow cap, a
    distance-resolution guard and a per-spectrum intensity-loss guard).  The
    ``tie_factors`` / ``max_dropped_fraction`` / ``enforce_distance_resolution``
    knobs reproduce wnetalign's single-overflow-cap behaviour.

    Pure advisor: it computes the factors at construction and exposes them via
    :meth:`sf_distance`, :meth:`sf_intensity`, :meth:`scale_factor`, :meth:`ftol`.
    The caller still applies them (scale positions, ``set_intensity_scale``,
    unscale ``total_cost``).  Construction may raise ``ValueError`` (guards).

    Parameters
    ----------
    empirical : Distribution
    theoretical : Sequence[Distribution]
        All spectra must share ``empirical``'s dimension.
    metric : DistanceMetric
        Stored for (future) metric-aware decisions.
    max_distance : float
    trash_costs : array-like
        Active trash costs (1 or 2 entries).
    precision : float, optional
        Relative precision target driving the two factors.  Default 1e-3.
    explicit_scale_factor : float, optional
        ``> 0`` overrides the auto computation, setting both factors equal.
    tie_factors : bool, optional
        ``True`` => single overflow-cap factor for both, ignoring ``precision``
        (wnetalign mode).  Default False.
    max_dropped_fraction : float, optional
        Per-spectrum intensity-loss limit; ``>= 1.0`` disables the guard.
        Default 0.05.
    enforce_distance_resolution : bool, optional
        Distance-resolution guard (auto mode only).  Default True.
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
        precision: float = 1e-3,
        explicit_scale_factor: float = 0.0,
        tie_factors: bool = False,
        max_dropped_fraction: float = 0.05,
        enforce_distance_resolution: bool = True,
        max_int: float = float(1 << 60),
    ) -> None:
        theoretical = list(theoretical)
        dim = empirical.dimension
        for t in theoretical:
            if t.dimension != dim:
                raise ValueError(
                    f"all spectra must share the same dimension "
                    f"(got {t.dimension} and {dim})."
                )
        cls = globals().get(f"CScalerFloat{dim}")
        if cls is None:
            raise ValueError(
                f"Dimension {dim} is unavailable in this build of the wnet C++ "
                f"extension (compiled with a smaller WNET_MAX_DIM)."
            )
        trash = np.ascontiguousarray(trash_costs, dtype=np.float64)
        if trash.ndim != 1 or trash.shape[0] == 0:
            raise ValueError("trash_costs must be a non-empty 1D array.")
        self._cpp = cls(
            empirical.vecdist,
            [t.vecdist for t in theoretical],
            metric,
            float(max_distance),
            trash,
            float(precision),
            float(explicit_scale_factor),
            bool(tie_factors),
            float(max_dropped_fraction),
            bool(enforce_distance_resolution),
            float(max_int),
        )

    def sf_distance(self) -> float:
        return self._cpp.sf_distance()

    def sf_intensity(self) -> float:
        return self._cpp.sf_intensity()

    def scale_factor(self) -> float:
        return self._cpp.scale_factor()

    def ftol(self) -> float:
        return self._cpp.ftol()
