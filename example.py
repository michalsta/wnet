from wnet import Distribution, WassersteinNetwork
from wnet.distances import L1Distance, wrap_distance_function
import numpy as np

S1 = Distribution(np.array([[0], [0]]), np.array([1]))
S2 = Distribution(np.array([[1], [0]]), np.array([1]))

W = WassersteinNetwork(S1, [S2], L1Distance(), 10)
W.build()
W.set_point([1.0])
print("Total cost:", W.total_cost())
print(W.subgraphs())
print(W.subgraphs()[0].as_netowkrx())