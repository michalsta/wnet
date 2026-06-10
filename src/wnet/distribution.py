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
        positions = np.asarray(positions)
        intensities = np.asarray(intensities)
        # positions is (dimension, n_points); a bare 1D array is the classic
        # footgun (its length gets read as the dimension), so reject it early
        # and point at the helper instead of failing cryptically downstream.
        if positions.ndim != 2:
            raise ValueError(
                f"positions must be a 2D array of shape (dimension, n_points), "
                f"got shape {positions.shape}. For 1D data use "
                f"Distribution_1D(positions, intensities) or pass "
                f"positions[np.newaxis, :]."
            )
        if intensities.ndim != 1 or intensities.shape[0] != positions.shape[1]:
            raise ValueError(
                f"intensities must be a 1D array with one value per point "
                f"(n_points={positions.shape[1]}), got shape {intensities.shape}."
            )
        self.positions = positions
        self.intensities = intensities
        dimension = positions.shape[0]
        if dimension < 1 or dimension > 20:
            raise ValueError(
                f"Unsupported dimension: {dimension}. Must be between 1 and 20."
            )
        self.label = label

    @cached_property
    def vecdist(self):
        # Real (double) intensities — the network quantizes to int64 supplies
        # internally at build/solve, so we no longer truncate here. (Previously
        # this built the int64 CVectorDistribution, silently flooring fractional
        # intensities; see CVectorDistributionFloat.)
        cfun = globals()[f"CVectorDistributionFloat{self.dimension}"]
        return cfun(
            self.positions.astype(np.float64), self.intensities.astype(np.float64)
        )

    def n_highest(self, n: int) -> "Distribution":
        """
        Returns a new Distribution containing the n peaks with the highest intensities.

        Args:
            n (int): Number of top peaks to retain.

        Returns:
            Distribution: New distribution with at most n peaks, ordered by intensity descending.
        """
        result = self.vecdist.n_highest(n)
        return Distribution(
            result.py_get_positions(), result.py_get_intensities(), label=self.label
        )

    def p_trim(self, p: float) -> "Distribution":
        """
        Returns a new Distribution keeping the fewest highest-intensity peaks whose
        combined intensity is at least p * total_intensity.

        Args:
            p (float): Fraction of total signal to retain (0.0–1.0).

        Returns:
            Distribution: New distribution covering at least fraction p of the signal.
        """
        result = self.vecdist.p_trim(p)
        return Distribution(
            result.py_get_positions(), result.py_get_intensities(), label=self.label
        )

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
        # Polymorphic: subclasses (e.g. Spectrum) get back their own type.
        return type(self)(new_positions, new_intensities, label=self.label)

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
        return type(self)(new_positions, new_intensities, label=self.label)

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
        return type(self)(new_positions, new_intensities, label=self.label)

    def as_distribution(self) -> "Distribution":
        """
        Returns a plain Distribution with the same positions and intensities.

        For a bare Distribution this is a typed copy; for subclasses (e.g.
        Spectrum) it strips the subclass-specific state, yielding a base
        Distribution. Always returns a Distribution, never ``type(self)``.

        Returns:
            Distribution: A base Distribution with the same data.
        """
        return Distribution(self.positions, self.intensities, label=self.label)

    @cached_property
    def sum_intensities(self) -> float:
        return np.sum(self.intensities)

    @property
    def dimension(self) -> int:
        return self.positions.shape[0]

    def cpp_repr(self) -> str:
        return f"VectorDistribution<{self.dimension}> distribution(\n{{{self.positions.tolist()}}},\n{{{self.intensities.tolist()}}}\n);"

    def __len__(self):
        return self.intensities.shape[0]

    def plot(self, filename: Optional[str] = None) -> None:
        """
        Visualize the distribution as an N×N plot matrix.

        Diagonal (i,i): stem plot of dimension i vs intensity.
        Off-diagonal (i,j): 2D scatter of dim i vs dim j, intensity as color.

        Args:
            filename: if given, save to file instead of showing interactively.
        """
        import matplotlib.pyplot as plt

        pos = self.positions
        intensities = self.intensities.astype(float)
        D = self.dimension
        title = self.label if self.label is not None else "Distribution"

        i_max = intensities.max()
        norm_int = intensities / i_max if i_max > 0 else intensities

        fig, axes = plt.subplots(D, D, figsize=(3 * D, 3 * D), squeeze=False)
        fig.suptitle(title)

        sc = None
        for i in range(D):
            for j in range(D):
                ax = axes[i][j]
                if i == j:
                    ax.vlines(pos[i], 0, intensities, linewidth=1)
                    ax.plot(pos[i], intensities, "o", markersize=3)
                    ax.set_ylabel("Intensity")
                else:
                    sc = ax.scatter(
                        pos[j],
                        pos[i],
                        c=intensities,
                        s=10 + 100 * norm_int,
                        cmap="viridis",
                        alpha=0.7,
                    )
                if i == D - 1:
                    ax.set_xlabel(f"dim {j}")
                if j == 0:
                    ax.set_ylabel(f"dim {i}" if i != j else "Intensity")

        if sc is not None:
            fig.colorbar(sc, ax=axes.ravel().tolist(), label="Intensity", shrink=0.6)

        fig.tight_layout()

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

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

    def __getstate__(self) -> dict:
        """
        Prepares the state of the Distribution object for pickling.

        Returns:
            dict: A dictionary containing the state of the object, including positions, intensities, and label.
        """
        state = self.__dict__.copy()
        # Remove cached properties to avoid pickling them
        if "vecdist" in state:
            del state["vecdist"]
        return state

    # def __setstate__(self, state: dict) -> None: default is fine

    def __str__(self) -> str:
        return f"Distribution(label={self.label}, dimension={self.dimension}, num_peaks={len(self)}, total_intensity={self.sum_intensities:.4f})"

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
    ValueError
        If positions or intensities are not 1D arrays, or their lengths do not match.
    """
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions)
    if not isinstance(intensities, np.ndarray):
        intensities = np.array(intensities)
    if positions.ndim != 1:
        raise ValueError(f"positions must be 1D, got shape {positions.shape}")
    if intensities.ndim != 1:
        raise ValueError(f"intensities must be 1D, got shape {intensities.shape}")
    if positions.shape[0] != intensities.shape[0]:
        raise ValueError(
            f"positions and intensities must have the same length, got {positions.shape[0]} and {intensities.shape[0]}"
        )
    return Distribution(positions[np.newaxis, :], intensities, label=label)
