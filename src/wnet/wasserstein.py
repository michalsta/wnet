from .wasserstein_network import WassersteinNetwork
from .distribution import Distribution
from .distances import Distance


def WassersteinDistance(
    distribution1: Distribution, distribution2: Distribution, distance: Distance
) -> float:
    """
    Computes the Wasserstein distance between two distributions using the provided distance metric.

    Args:
        distribution1 (Distribution): The first distribution.
        distribution2 (Distribution): The second distribution.
        distance (Distance): The distance metric to use. Must be a subclass of wnet.distances.Distance

    Returns:
        float: The Wasserstein distance between the two distributions.

    Raises:
        AssertionError: If the distributions do not have the same total intensity.
    """
    assert (
        distribution1.sum_intensities == distribution2.sum_intensities
    ), "Distributions must have the same total intensity"
    W = WassersteinNetwork(distribution1, [distribution2], distance, None)
    W.build()
    W.solve()
    return W.total_cost()


def TruncatedWassersteinDistance(
    distribution1: Distribution,
    distribution2: Distribution,
    distance: Distance,
    max_distance: float,
) -> float:
    """
    Computes the truncated Wasserstein distance between two distributions, limiting the transport cost to max_distance.

    Args:
        distribution1 (Distribution): The first distribution.
        distribution2 (Distribution): The second distribution.
        distance (Distance): The distance metric to use. Must be a subclass of wnet.distances.Distance
        max_distance (float): The maximum allowed transport cost.

    Returns:
        float: The truncated Wasserstein distance between the two distributions.

    Raises:
        AssertionError: If the distributions do not have the same total intensity.
    """
    assert (
        distribution1.sum_intensities == distribution2.sum_intensities
    ), "Distributions must have the same total intensity"
    W = WassersteinNetwork(distribution1, [distribution2], distance, max_distance)
    W.build()
    W.add_simple_trash(max_distance)
    W.solve()
    return W.total_cost()
