from .wasserstein_network import WassersteinNetwork
from .distribution import Distribution
from .distances import DistanceMetric

import numpy as np


def WassersteinDistance(
    distribution1: Distribution,
    distribution2: Distribution,
    distance: DistanceMetric,
    p: int = 1,
    force_dense_1d: bool = False,
    solver=None,
    method: str = None,
) -> float:
    """
    Computes the p-Wasserstein distance between two distributions using the provided ground metric.

    The optimal transport cost is computed with per-pair cost ground_distance**p,
    and the returned value is its p-th root: W_p = (min_pi sum d**p pi)**(1/p).

    Args:
        distribution1 (Distribution): The first distribution.
        distribution2 (Distribution): The second distribution.
        distance (DistanceMetric): The ground metric to use (e.g. DistanceMetric.L1, DistanceMetric.L2).
        p (int): Wasserstein transport order (integer >= 1). p=1 is the classic 1-Wasserstein; p=2 is the quadratic W_2. p != 1 forces the dense factory.
        force_dense_1d (bool): In 1D, force the dense factory. See WassersteinNetwork for details.
        solver: Solver config object (e.g. NetworkSimplex(), CostScaling()). Defaults to NetworkSimplex().
        method (str): Deprecated. Use solver= instead.

    Returns:
        float: The p-Wasserstein distance between the two distributions.

    Raises:
        RuntimeError: If the distributions do not have the same total intensity.
    """
    if not np.isclose(distribution1.sum_intensities, distribution2.sum_intensities):
        raise RuntimeError("Distributions must have the same total intensity")
    W = WassersteinNetwork(
        distribution1,
        [distribution2],
        distance,
        None,
        force_dense_1d=force_dense_1d,
        p=p,
        solver=solver,
        method=method,
    )
    W.build()
    W.solve()
    # total_cost() is in W_p**p units (sum of d**p * flow); take the p-th root.
    return W.total_cost() ** (1.0 / p)


def TruncatedWassersteinDistance(
    distribution1: Distribution,
    distribution2: Distribution,
    distance: DistanceMetric,
    max_distance: float,
    p: int = 1,
    force_dense_1d: bool = False,
    solver=None,
    method: str = None,
) -> float:
    """
    Computes the truncated p-Wasserstein distance between two distributions, limiting the per-pair ground distance to max_distance.

    max_distance is expressed in *ground-distance* units (the per-pair matching
    filter keeps pairs with d <= max_distance). The trash edge therefore costs
    max_distance**p per unit, to stay comparable to a matched pair at the cap.

    Args:
        distribution1 (Distribution): The first distribution.
        distribution2 (Distribution): The second distribution.
        distance (DistanceMetric): The ground metric to use (e.g. DistanceMetric.L1, DistanceMetric.L2).
        max_distance (float): The maximum allowed per-pair ground distance (in distance units).
        p (int): Wasserstein transport order (integer >= 1). p != 1 forces the dense factory.
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
    W = WassersteinNetwork(
        distribution1,
        [distribution2],
        distance,
        max_distance,
        force_dense_1d=force_dense_1d,
        p=p,
        solver=solver,
        method=method,
    )
    # Trash cost is in W_p**p units, matching the d**p matching-edge costs.
    W.add_simple_trash(max_distance ** p)
    W.build()
    W.solve()
    return W.total_cost() ** (1.0 / p)
