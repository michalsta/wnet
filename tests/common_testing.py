import numpy as np
from wnet import Distribution, WassersteinNetwork
from wnet.distances import L1Distance


def create_large_distribution(seed, dimension, pos_range, int_range, size):
    rng = np.random.default_rng(seed)
    data = rng.uniform(size=(dimension, size)) * pos_range
    intensities = rng.uniform(size=(size,)) * int_range
    return Distribution(data, intensities)


def create_large_wsdflow_instance(
    seed, dimension, pos_range, int_range, size, no_theoretical_distributions, point
):
    experimental_distribution = create_large_distribution(
        seed, dimension, pos_range, int_range, size
    )
    theoretical_distributions = [
        create_large_distribution(seed + i + 1, dimension, pos_range, int_range, size)
        for i in range(no_theoretical_distributions)
    ]
    wsdflow_instance = WassersteinNetwork(
        experimental_distribution,
        theoretical_distributions,
        distance=L1Distance(),
        max_distance=int(pos_range * dimension / 10),
    )
    wsdflow_instance.build()
    return (wsdflow_instance, point)


def solve_large_wsdflow_instance(wsdflow_instance_point):
    wsdflow_instance, point = wsdflow_instance_point
    if point is not None:
        wsdflow_instance.solve(point)
    else:
        wsdflow_instance.solve()
    return wsdflow_instance.total_cost()
