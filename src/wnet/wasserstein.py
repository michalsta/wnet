from wasserstein_network import WassersteinNetwork

def WassersteinDistance(distribution1, distribution2, distance):
    assert distribution1.sum_intensities() == distribution2.sum_intensities(), "Distributions must have the same total intensity"
    W = WassersteinNetwork(distribution1, [distribution2], distance, None)
    W.build()
    return W.solve()

def TruncatedWassersteinDistance(distribution1, distribution2, distance):
    assert distribution1.sum_intensities() == distribution2.sum_intensities(), "Distributions must have the same total intensity"
    W = WassersteinNetwork(distribution1, [distribution2], distance, None)
    W.build()
    return W.solve()