from .wasserstein_network import WassersteinNetwork
from .distribution import Distribution
from .distances import Distance

import numpy as np

def WassersteinDistance(
    distribution1: Distribution,
    distribution2: Distribution,
    distance: Distance,
    force_dense_1d: bool = False,
    method: str = "network_simplex",
) -> float:
    """
    Computes the Wasserstein distance between two distributions using the provided distance metric.

    Args:
        distribution1 (Distribution): The first distribution.
        distribution2 (Distribution): The second distribution.
        distance (Distance): The distance metric to use. Must be a subclass of wnet.distances.Distance
        force_dense_1d (bool): In 1D, force the dense factory. See WassersteinNetwork for details.
        method (str): Min-cost flow algorithm. "network_simplex" (default) or "cycle_canceling".

    Returns:
        float: The Wasserstein distance between the two distributions.

    Raises:
        RuntimeError: If the distributions do not have the same total intensity.
    """
    if not np.isclose(distribution1.sum_intensities, distribution2.sum_intensities):
        raise RuntimeError("Distributions must have the same total intensity")
    W = WassersteinNetwork(
        distribution1, [distribution2], distance, None,
        force_dense_1d=force_dense_1d, method=method)
    W.build()
    W.solve()
    return W.total_cost()


def TruncatedWassersteinDistance(
    distribution1: Distribution,
    distribution2: Distribution,
    distance: Distance,
    max_distance: float,
    force_dense_1d: bool = False,
    method: str = "network_simplex",
) -> float:
    """
    Computes the truncated Wasserstein distance between two distributions, limiting the transport cost to max_distance.

    Args:
        distribution1 (Distribution): The first distribution.
        distribution2 (Distribution): The second distribution.
        distance (Distance): The distance metric to use. Must be a subclass of wnet.distances.Distance
        max_distance (float): The maximum allowed transport cost.
        force_dense_1d (bool): In 1D, force the dense factory. See WassersteinNetwork for details.
        method (str): Min-cost flow algorithm. "network_simplex" (default) or "cycle_canceling".

    Returns:
        float: The truncated Wasserstein distance between the two distributions.

    Raises:
        AssertionError: If the distributions do not have the same total intensity.
    """
    if not np.isclose(distribution1.sum_intensities, distribution2.sum_intensities):
        raise RuntimeError("Distributions must have the same total intensity")
    W = WassersteinNetwork(
        distribution1, [distribution2], distance, max_distance,
        force_dense_1d=force_dense_1d, method=method)
    W.add_simple_trash(max_distance)
    W.build()
    W.solve()
    return W.total_cost()
