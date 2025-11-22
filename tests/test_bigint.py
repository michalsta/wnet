import numpy as np

from wnet import WassersteinNetwork, Distribution_1D
from wnet.distances import DistanceMetric


def test_scale():
    for exponent in range(0, 18):
        scale_factor = 10**exponent
        empirical_spectrum = Distribution_1D(np.array([1]), np.array([1])).scaled(
            scale_factor
        )
        theoretical_spectrum = Distribution_1D(np.array([2]), np.array([1])).scaled(
            scale_factor
        )
        print(empirical_spectrum)
        max_distance = 10

        DG = WassersteinNetwork(
            empirical_spectrum, [theoretical_spectrum], DistanceMetric.L2, max_distance
        )
        DG.add_simple_trash(10)
        DG.build()
        DG.solve()
        # print(DG.total_cost())
        assert DG.total_cost() == scale_factor


if __name__ == "__main__":
    test_scale()
    print("Everything passed")
