"""
The raw C++ buffer views (positions_view / intensities_view) must be
read-only at the binding level: writing through them would desync the
distribution's sorted_indices and corrupt later matching.

NOTE: requires a wnet_cpp build that includes the const-qualified view
bindings in register_dim.hpp.
"""

import numpy as np
import pytest

from wnet import Distribution
from wnet.distribution import Distribution_1D


def _spectra():
    d1 = Distribution_1D(
        np.array([0.0, 1.5, 3.0]), np.array([1.0, 2.0, 3.0]), label="a"
    )
    d2 = Distribution(
        np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([0.5, 0.7]), label="b"
    )
    return d1, d2


@pytest.mark.parametrize("idx", [0, 1])
def test_raw_views_are_not_writeable(idx):
    d = _spectra()[idx]
    pv = d.vecdist.positions_view()
    iv = d.vecdist.intensities_view()
    assert not pv.flags.writeable
    assert not iv.flags.writeable


@pytest.mark.parametrize("idx", [0, 1])
def test_writing_through_raw_views_raises(idx):
    d = _spectra()[idx]
    pv = d.vecdist.positions_view()
    iv = d.vecdist.intensities_view()
    with pytest.raises((ValueError, RuntimeError)):
        pv[0, 0] = 123.0
    with pytest.raises((ValueError, RuntimeError)):
        iv[0] = 123.0
    with pytest.raises((ValueError, RuntimeError)):
        pv[...] = 0.0


@pytest.mark.parametrize("idx", [0, 1])
def test_properties_are_not_writeable(idx):
    d = _spectra()[idx]
    assert not d.positions.flags.writeable
    assert not d.intensities.flags.writeable
    with pytest.raises((ValueError, RuntimeError)):
        d.positions[0, 0] = 42.0
    with pytest.raises((ValueError, RuntimeError)):
        d.intensities[0] = 42.0


def test_views_still_expose_correct_data():
    d1, d2 = _spectra()
    np.testing.assert_array_equal(
        d1.vecdist.positions_view(), np.array([[0.0], [1.5], [3.0]])
    )
    np.testing.assert_array_equal(
        d1.vecdist.intensities_view(), np.array([1.0, 2.0, 3.0])
    )
    # Property view is the transpose of the raw (n_points, dim) view.
    np.testing.assert_array_equal(d2.positions, np.array([[0.0, 1.0], [2.0, 3.0]]))
    np.testing.assert_array_equal(d2.intensities, np.array([0.5, 0.7]))


def test_writable_copy_is_still_available():
    # Copying out must remain the supported mutation path.
    d1, _ = _spectra()
    p = np.array(d1.positions)
    p[0, 0] = 99.0  # no raise
    assert d1.positions[0, 0] == 0.0
