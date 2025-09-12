from wnet import Distribution, WassersteinNetwork
from wnet.distances import L1Distance, wrap_distance_function
import numpy as np

positions1 = np.array([[0, 1, 5, 15], [0, 0, 0, 0]])
masses1 = np.array([10, 5, 5, 5])

positions2 = np.array([[1,10], [0, 0]])
masses2 = np.array([20, 5])

S1 = Distribution(positions1, masses1)
S2 = Distribution(positions2, masses2)

W = WassersteinNetwork(S1, [S2], L1Distance(), None)
W.add_simple_trash(5)
W.build()
W.set_point([1.0])
print("Total cost:", W.total_cost())
print(W.subgraphs())
print(W.subgraphs()[0].as_netowkrx())
W.subgraphs()[0].show()