from wnet import WassersteinNetwork, Distribution_1D
import numpy as np


def test_scale():
    for exponent in range(0, 18):
        scale_factor = 10 ** exponent
        empirical_spectrum = Distribution_1D(np.array([1]), np.array([1])).scaled(scale_factor)
        theoretical_spectrum = Distribution_1D(np.array([2]), np.array([1])).scaled(scale_factor)
        print(empirical_spectrum)
        dist_fun = lambda x, y: np.linalg.norm(x - y, axis=0)
        max_distance = 10
        def wrapped_dist(p, y):
            print(p, p.index, p.positions)
            i = p.index
            x = p.positions[:, i : i + 1]
            return dist_fun(x[: np.newaxis], y)

        DG = WassersteinNetwork(
            empirical_spectrum, [theoretical_spectrum], wrapped_dist, max_distance
        )
        DG.add_simple_trash(10)
        DG.build()
        DG.set_point([1.0])
        #print(DG.total_cost())
        assert DG.total_cost() == scale_factor

if __name__ == "__main__":
    test_scale()
    print("Everything passed")

