from functools import cached_property
from typing import Optional
import numpy as np

from wnet.wnet_cpp import *


class Distribution:
    """
    Distribution represents a collection of points and their associated intensities. Meant to be immutable.

    Positions and intensities are owned by a C++ ``CVectorDistributionFloat``
    object (``vecdist``); the ``positions``/``intensities`` properties are
    read-only numpy views over its buffers.
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
        dimension = positions.shape[0]
        if dimension < 1 or dimension > 20:
            raise ValueError(
                f"Unsupported dimension: {dimension}. Must be between 1 and 20."
            )
        cls = globals().get(f"CVectorDistributionFloat{dimension}")
        if cls is None:
            raise ValueError(
                f"Dimension {dimension} is unavailable in this build of the wnet "
                f"C++ extension (compiled with a smaller WNET_MAX_DIM)."
            )
        # The C++ VectorDistribution owns positions/intensities (as real float64)
        # — the single source of truth.  The `positions`/`intensities` properties
        # below are read-only numpy views over its buffers (zero-copy).
        # Use np.array (always copies) so the C++ ctor receives a writable,
        # C-contiguous float64 array — passing a read-only view (e.g. another
        # Distribution's `.positions`) would otherwise be rejected by nanobind.
        self._cpp = cls(
            np.array(positions, dtype=np.float64, order="C"),
            np.array(intensities, dtype=np.float64, order="C"),
        )
        self._dimension = dimension
        self.label = label

    @property
    def vecdist(self):
        # The C++ distribution object (CVectorDistributionFloat{dim}) — the
        # single source of truth, consumed directly by the network.
        return self._cpp

    @property
    def positions(self) -> np.ndarray:
        # Read-only (dimension, n_points) view over the C++ buffer (zero-copy;
        # the view keeps the C++ object alive). Copy it if you need to mutate.
        # The binding already returns the view read-only (const-qualified
        # nanobind ndarray); re-clearing the flag here is defense in depth.
        v = self._cpp.positions_view().T
        v.flags.writeable = False
        return v

    @property
    def intensities(self) -> np.ndarray:
        # Read-only (n_points,) view over the C++ buffer (zero-copy).
        # Read-only at the binding level too; see `positions`.
        v = self._cpp.intensities_view()
        v.flags.writeable = False
        return v

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

    def sorted_by_positions(self) -> "Distribution":
        """
        Returns a new Distribution with peaks sorted lexicographically by
        position (dimension 0 first, then 1, ...). Peaks are not merged.

        Mirrors masserstein's ``Spectrum.sort_confs``, generalised to N
        dimensions. The sort is performed in C++.

        Returns:
            Distribution: A new, position-sorted distribution.
        """
        result = self.vecdist.sorted_by_positions()
        return type(self)(
            result.py_get_positions(), result.py_get_intensities(), label=self.label
        )

    def binned(self, bin_width: float) -> "Distribution":
        """
        Returns a new Distribution with each position coordinate rounded to the
        nearest multiple of ``bin_width``, then peaks falling in the same bin
        merged (intensities summed) and the result sorted lexicographically.

        The wnet analogue of masserstein's ``coarse_bin``, generalised to N
        dimensions and parameterised by bin width rather than decimal digits.
        Performed in C++.

        Args:
            bin_width (float): Width of each bin; must be positive.

        Returns:
            Distribution: A new, binned distribution.
        """
        result = self.vecdist.binned(float(bin_width))
        return type(self)(
            result.py_get_positions(), result.py_get_intensities(), label=self.label
        )

    def __add__(self, other: "Distribution") -> "Distribution":
        """
        Returns the sum of two distributions: the union of their peaks, with
        peaks sharing an identical position merged (intensities summed) and the
        result sorted lexicographically.

        Mirrors masserstein's ``Spectrum.__add__``. The merge is performed in
        C++.

        Args:
            other (Distribution): The distribution to add; must have the same
                dimension as ``self``.

        Returns:
            Distribution: A new distribution holding the merged peaks.
        """
        if not isinstance(other, Distribution):
            return NotImplemented
        if other.dimension != self.dimension:
            raise ValueError(
                f"Cannot add distributions of differing dimension "
                f"({self.dimension} and {other.dimension})."
            )
        result = self.vecdist.add(other.vecdist)
        label = self._combined_label(self.label, other.label)
        return type(self)(
            result.py_get_positions(), result.py_get_intensities(), label=label
        )

    def __mul__(self, scalar: float) -> "Distribution":
        """
        Returns a new Distribution with intensities multiplied by ``scalar``
        and positions unchanged (the C++ analogue of :meth:`scaled`).

        Mirrors masserstein's ``Spectrum.__mul__``. Performed in C++.

        Args:
            scalar (float): The factor by which to scale the intensities.

        Returns:
            Distribution: A new distribution with scaled intensities.
        """
        result = self.vecdist.scaled(float(scalar))
        return type(self)(
            result.py_get_positions(), result.py_get_intensities(), label=self.label
        )

    def __rmul__(self, scalar: float) -> "Distribution":
        # Scalar multiplication is commutative.
        return self.__mul__(scalar)

    @staticmethod
    def LinearCombination(
        distributions: "list[Distribution]",
        weights: np.ndarray,
        label: Optional[str] = None,
    ) -> "Distribution":
        """
        Returns the linear combination ``sum_i weights[i] *
        distributions[i]``: the union of every peak of every input (each
        intensity scaled by that input's weight), with peaks sharing an
        identical position merged (intensities summed) and the result sorted
        lexicographically.

        Mirrors masserstein's ``Spectrum.ScalarProduct``. The combine is
        performed in C++ in a single pass.

        Args:
            distributions (list[Distribution]): The distributions to combine;
                all must share the same dimension.
            weights (array-like): One weight per distribution.
            label (str | None): Optional label for the result.

        Returns:
            Distribution: A new distribution holding the combined peaks.
        """
        distributions = list(distributions)
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        if not distributions:
            raise ValueError("LinearCombination requires at least one distribution.")
        if len(distributions) != weights.shape[0]:
            raise ValueError(
                f"number of distributions ({len(distributions)}) must match "
                f"number of weights ({weights.shape[0]})."
            )
        dim = distributions[0].dimension
        for d in distributions:
            if d.dimension != dim:
                raise ValueError(
                    "all distributions must have the same dimension "
                    f"(got {d.dimension} and {dim})."
                )
        cpp_cls = type(distributions[0].vecdist)
        result = cpp_cls.linear_combination([d.vecdist for d in distributions], weights)
        return Distribution(
            result.py_get_positions(), result.py_get_intensities(), label=label
        )

    @staticmethod
    def _combined_label(a: Optional[str], b: Optional[str]) -> Optional[str]:
        if a is None and b is None:
            return None
        return f"{a} + {b}"

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
        return float(np.sum(self.intensities))

    @property
    def dimension(self) -> int:
        return self._dimension

    def cpp_repr(self) -> str:
        return f"VectorDistribution<{self.dimension}> distribution(\n{{{self.positions.tolist()}}},\n{{{self.intensities.tolist()}}}\n);"

    def __len__(self):
        return self._cpp.size()

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
        Prepares the state for pickling.  The C++ object isn't directly
        picklable, so serialize writable copies of the (read-only) array views
        and rebuild on unpickle.
        """
        return {
            "positions": np.array(self.positions),
            "intensities": np.array(self.intensities),
            "label": self.label,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["positions"], state["intensities"], label=state["label"])

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
