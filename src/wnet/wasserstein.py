import math

from .wasserstein_network import WassersteinNetwork
from .distribution import Distribution
from .distances import DistanceMetric

import numpy as np


def _has_fractional_positions(*distributions: Distribution) -> bool:
    """True if any distribution has a non-integer position coordinate.

    With p == 1 the solver's legacy mode truncates each edge cost to an
    integer, which is exact for integer positions but silently loses up to one
    distance unit per unit of flow otherwise.  Callers use this to decide
    whether to opt into cost scaling (real-valued costs) for p == 1.
    """
    for d in distributions:
        pos = d.positions
        if not np.all(pos == np.round(pos)):
            return True
    return False


def WassersteinDistance(
    distribution1: Distribution,
    distribution2: Distribution,
    distance: DistanceMetric,
    p: float = 1.0,
    force_dense_1d: bool = False,
    solver=None,
    method: str = None,
) -> float:
    """
    Computes the p-Wasserstein distance between two distributions using the provided ground metric.

    The optimal transport cost is computed with per-pair cost ground_distance**p,
    and the returned value is its p-th root: W_p = (min_pi sum d**p pi)**(1/p).

    With p == 1 and all-integer positions the computation is bit-exact with the
    legacy integer solver.  Fractional positions automatically enable cost
    scaling so that real-valued ground distances are priced exactly (instead of
    being truncated to integers).

    Args:
        distribution1 (Distribution): The first distribution.
        distribution2 (Distribution): The second distribution.
        distance (DistanceMetric): The ground metric to use (e.g. DistanceMetric.L1, DistanceMetric.L2).
        p (float): Wasserstein transport order, any real number >= 1. p=1 is the classic 1-Wasserstein; p=2 is the quadratic W_2; fractional p (e.g. 1.5) is supported via automatic cost scaling. p != 1 forces the dense factory.
        force_dense_1d (bool): In 1D, force the dense factory. See WassersteinNetwork for details.
        solver: Solver config object (e.g. NetworkSimplex(), CostScaling()). Defaults to NetworkSimplex().
        method (str): Deprecated. Use solver= instead.

    Returns:
        float: The p-Wasserstein distance between the two distributions.

    Raises:
        RuntimeError: If the distributions do not have the same total intensity.
        InfeasibleError: If the total intensities are equal as reals but land on
            unequal integers after per-peak quantisation (possible with
            fractional intensities). The distributions are never silently
            re-quantised to force balance; pass integer intensities that
            balance exactly, or use TruncatedWassersteinDistance (whose trash
            edges absorb the imbalance).
    """
    if not np.isclose(distribution1.sum_intensities, distribution2.sum_intensities):
        raise RuntimeError("Distributions must have the same total intensity")
    # p == 1 defaults to the legacy integer-truncation cost mode, which is
    # bit-exact for integer positions but truncates fractional ground
    # distances (e.g. two unit masses 0.6 apart would yield distance 0).
    # Opt into cost scaling whenever positions are fractional; p != 1
    # already cost-scales automatically.
    enable_cost_scaling = p == 1.0 and _has_fractional_positions(
        distribution1, distribution2
    )
    W = WassersteinNetwork(
        distribution1,
        [distribution2],
        distance,
        None,
        # Cost scaling needs the dense factory here: on a trash-less network
        # the 1D chain factory never picks an automatic cost scale, so its
        # gap costs would be llround()ed at scale 1 (wrong for fractional
        # positions).  With integer positions the chain stays available.
        force_dense_1d=force_dense_1d or enable_cost_scaling,
        p=p,
        solver=solver,
        method=method,
    )
    if enable_cost_scaling:
        W.set_cost_scaling()
    W.build()
    W.solve()
    # total_cost() is in W_p**p units (sum of d**p * flow); take the p-th root.
    return W.total_cost() ** (1.0 / p)


def TruncatedWassersteinDistance(
    distribution1: Distribution,
    distribution2: Distribution,
    distance: DistanceMetric,
    max_distance: float,
    p: float = 1.0,
    force_dense_1d: bool = False,
    solver=None,
    method: str = None,
) -> float:
    """
    Computes the truncated p-Wasserstein distance between two distributions, limiting the per-pair ground distance to max_distance.

    max_distance is expressed in *ground-distance* units (the per-pair matching
    filter keeps pairs with d <= max_distance). The trash edge therefore costs
    max_distance**p per unit, to stay comparable to a matched pair at the cap.

    With p == 1 and all-integer positions and cap the computation is bit-exact
    with the legacy integer solver.  A fractional cap or fractional positions
    automatically enable cost scaling and keep the exact (un-rounded) matching
    threshold, so real-valued distances are priced exactly and no pair beyond
    the requested cap can be matched.

    Args:
        distribution1 (Distribution): The first distribution.
        distribution2 (Distribution): The second distribution.
        distance (DistanceMetric): The ground metric to use (e.g. DistanceMetric.L1, DistanceMetric.L2).
        max_distance (float): The maximum allowed per-pair ground distance (in distance units).
        p (float): Wasserstein transport order, any real number >= 1. p != 1 forces the dense factory (via automatic cost scaling).
        force_dense_1d (bool): In 1D, force the dense factory. See WassersteinNetwork for details.
        solver: Solver config object (e.g. NetworkSimplex(), CostScaling()). Defaults to NetworkSimplex().
        method (str): Deprecated. Use solver= instead.

    Returns:
        float: The truncated p-Wasserstein distance between the two distributions.

    Raises:
        AssertionError: If the distributions do not have the same total intensity.
    """
    if not np.isclose(distribution1.sum_intensities, distribution2.sum_intensities):
        raise RuntimeError("Distributions must have the same total intensity")
    max_distance = float(max_distance)
    # p == 1 defaults to the legacy integer-truncation cost mode: exact for
    # all-integer inputs, wrong otherwise (fractional distances truncate;
    # a fractional cap would also let pairs slightly beyond the requested
    # max_distance be matched at understated cost).  Opt into cost scaling
    # when positions or the cap are fractional; p != 1 cost-scales
    # automatically.  Whenever costs are real-valued, keep the real
    # (un-ceiled) matching threshold too.
    enable_cost_scaling = p == 1.0 and (
        max_distance != math.floor(max_distance)
        or _has_fractional_positions(distribution1, distribution2)
    )
    keep_real_threshold = enable_cost_scaling or p != 1.0
    W = WassersteinNetwork(
        distribution1,
        [distribution2],
        distance,
        max_distance,
        force_dense_1d=force_dense_1d,
        p=p,
        solver=solver,
        method=method,
        round_max_distance=not keep_real_threshold,
    )
    if enable_cost_scaling:
        W.set_cost_scaling()
    # Trash cost is in W_p**p units, matching the d**p matching-edge costs.
    W.add_simple_trash(max_distance ** p)
    W.build()
    W.solve()
    return W.total_cost() ** (1.0 / p)
