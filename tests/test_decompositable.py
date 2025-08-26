from wnet import WassersteinNetwork, Distribution
from wnet.distances import wrap_distance_function
import numpy as np


def test_simple():
    S1 = Distribution(np.array([[1, 2, 30]]), np.array([1, 4, 3]))
    S2 = Distribution(np.array([[1, 4, 30, 31]]), np.array([1, 1, 1, 1]))
    S3 = Distribution(np.array([[3, 4, 32]]), np.array([1, 1, 3]))

    G = WassersteinNetwork(
        S1, [S2, S3], wrap_distance_function(lambda x, y: np.linalg.norm(x - y, axis=0)), 5
    )
    # G.show()
    # G.show_cgraph()
    G.add_simple_trash(10)
    G.build()
    # G.show()
    print(G.set_point([1, 1]))

    # for subgr in G.subgraphs:
    #    subgr.show()

    # for csubgraph in G.csubgraph_objs():
    #    csubgraph.show()


if __name__ == "__main__":
    test_simple()
