from .wasserstein_network import WassersteinNetwork

def WassersteinDistance(distribution1, distribution2, distance):
    assert distribution1.sum_intensities == distribution2.sum_intensities, "Distributions must have the same total intensity"
    W = WassersteinNetwork(distribution1, [distribution2], distance, None)
    W.build()
    W.solve()
    return W.total_cost()

def TruncatedWassersteinDistance(distribution1, distribution2, distance, max_distance):
    assert distribution1.sum_intensities == distribution2.sum_intensities, "Distributions must have the same total intensity"
    W = WassersteinNetwork(distribution1, [distribution2], distance, max_distance)
    W.build()
    W.add_simple_trash(max_distance)
    W.solve()
    return W.total_cost()