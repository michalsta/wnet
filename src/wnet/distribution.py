from functools import cached_property
from typing import Optional
import numpy as np

from wnet.wnet_cpp import *


class Distribution:
    """
    Distribution represents a collection of points and their associated intensities. Meant to be immutable.
    Inherits from:
        CDistribution (wnet.wnet_cpp): A C++ extension class that provides core functionality for handling distributions.
    Args:
        positions (array-like): The spatial positions of the distribution.
        intensities (array-like): The intensity values corresponding to each position.
    Methods:
        scaled(scale_factor):
            Returns a new Distribution instance with intensities scaled by the given factor.
    Properties:
        positions:
            Returns the positions of the distribution.
        intensities:
            Returns the intensities of the distribution.
        sum_intensities:
            Returns the sum of all intensities in the distribution (cached).
    """

    def __init__(
        self,
        positions: np.ndarray,
        intensities: np.ndarray,
        label: Optional[str] = None,
    ) -> None:
        """
        Initialize the distribution with given positions and intensities.

        Args:
            positions (np.ndarray): Array of positions.
            intensities (np.ndarray): Array of intensities corresponding to each position.
            label (str | None): Optional label for the distribution.
        """
        # super().__init__(positions, intensities.astype(np.int64, copy=False))
        self.positions = positions
        self.intensities = intensities
        dimension = positions.shape[0]
        if dimension < 1 or dimension > 20:
            raise ValueError(
                f"Unsupported dimension: {dimension}. Must be between 1 and 20."
            )
        cfun = globals()[f"CVectorDistribution{dimension}"]

        self.vecdist = lambda: cfun(
            positions.astype(np.float64), intensities.astype(np.int64)
        )
        self.label = label

    def scaled(self, scale_factor: float) -> "Distribution":
        """
        Creates a new Distribution instance with intensities scaled by the given factor.

        Args:
            scale_factor (float): The factor by which to scale the intensities.

        Returns:
            Distribution: A new Distribution instance with scaled intensities and unchanged positions.
        """
        new_positions = self.positions
        new_intensities = self.intensities * scale_factor
        return Distribution(new_positions, new_intensities, label=self.label)

    def positions_intensities_scaled(self, scale_factor: float) -> "Distribution":
        """
        Creates a new Distribution instance with both positions and intensities scaled by the given factor.

        Args:
            scale_factor (float): The factor by which to scale the positions and intensities.

        Returns:
            Distribution: A new Distribution instance with scaled positions and intensities.
        """
        new_positions = self.positions.astype(np.float64, copy=False) * scale_factor
        new_intensities = self.intensities.astype(np.float64, copy=False) * scale_factor
        return Distribution(new_positions, new_intensities, label=self.label)

    def normalized(self) -> "Distribution":
        """
        Creates a new Distribution instance with intensities normalized to sum to 1.

        Returns:
            Distribution: A new Distribution instance with normalized intensities and unchanged positions.
        """
        total_intensity = self.sum_intensities
        if total_intensity == 0:
            raise ValueError(
                "Cannot normalize a distribution with zero total intensity."
            )
        new_positions = self.positions
        new_intensities = self.intensities / total_intensity
        return Distribution(new_positions, new_intensities, label=self.label)

    @cached_property
    def sum_intensities(self) -> float:
        return np.sum(self.intensities)

    @property
    def dimension(self) -> int:
        return self.positions.shape[0]

    def cpp_repr(self) -> str:
        return f"VectorDistribution<{self.dimension}> distribution(\n{{{self.positions.tolist()}}},\n{{{self.intensities.tolist()}}}\n);"

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the axis-aligned bounding box of the distribution.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                - The first array represents the minimum coordinates of the bounding box.
                - The second array represents the maximum coordinates of the bounding box.
        """
        min_coords = np.min(self.positions, axis=1)
        max_coords = np.max(self.positions, axis=1)
        return min_coords, max_coords


def Distribution_1D(
    positions: np.ndarray, intensities: np.ndarray, label: Optional[str] = None
) -> Distribution:
    """
    Creates a 1D distribution from given positions and intensities.

    Parameters
    ----------
    positions : np.ndarray or array-like
        1D array of position values.
    intensities : np.ndarray or array-like
        1D array of intensity values corresponding to each position.

    Returns
    -------
    Distribution
        A Distribution object representing the 1D distribution.

    Raises
    ------
    AssertionError
        If positions or intensities are not 1D arrays, or their lengths do not match.
    """
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions)
    if not isinstance(intensities, np.ndarray):
        intensities = np.array(intensities)
    assert len(positions.shape) == 1
    assert len(intensities.shape) == 1
    assert positions.shape[0] == intensities.shape[0]
    return Distribution(positions[np.newaxis, :], intensities)
