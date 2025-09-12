from wnet import Distribution, WassersteinNetwork
from wnet.distances import L1Distance, wrap_distance_function
import numpy as np

positions1 = np.array(
    [[0, 1, 5, 10],
     [0, 0, 0, 3]]
)
intensities1 = np.array([10, 5, 5, 5])

positions2 = np.array(
    [[1,10],
    [0, 0]])
intensities2 = np.array([20, 5])

S1 = Distribution(positions1, intensities1)
S2 = Distribution(positions2, intensities2)

W = WassersteinNetwork(S1, [S2], L1Distance(), 10)
W.add_simple_trash(15)
W.build()
W.solve()
print("Total cost:", W.total_cost())
print(W.subgraphs()[0].as_netowkrx())
print(W.flows_for_target(0))
W.subgraphs()[0].show()