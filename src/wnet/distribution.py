import numpy as np
from functools import cached_property
from wnet.wnet_cpp import CDistribution

class Distribution(CDistribution):
    def __init__(self, positions, intensities):
        super().__init__(positions, intensities)

    def scaled(self, scale_factor):
        new_positions = self.positions
        new_intensities = self.intensities * scale_factor
        return Distribution(new_positions, new_intensities)

    @property
    def positions(self):
        return self.get_positions()

    @property
    def intensities(self):
        return self.get_intensities()

    @cached_property
    def sum_intensities(self):
        return np.sum(self.intensities)

def Distribution_1D(positions, intensities):
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions)
    if not isinstance(intensities, np.ndarray):
        intensities = np.array(intensities)
    assert len(positions.shape) == 1
    assert len(intensities.shape) == 1
    assert positions.shape[0] == intensities.shape[0]
    return Distribution(positions[np.newaxis, :], intensities)
