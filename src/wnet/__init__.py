#! /usr/bin/env python
# -*- coding: utf-8 -*-

from .distribution import Distribution_1D, Distribution
from . import wnet_cpp
from .wnet_cpp import InfeasibleError
from .wasserstein_network import WassersteinNetwork
from .wasserstein import WassersteinDistance, TruncatedWassersteinDistance
from .scaling import WNetAlignScaler, WNetDeconvScaler, FineGridScaler, GenericScaler


def is_nanobind_split() -> bool:
    """True when wnet_cpp was built in nanobind split mode. See pylmcf.nanobind_mode."""
    from pylmcf.nanobind_mode import extension_is_split

    return extension_is_split(wnet_cpp)


def _check_nanobind_modes() -> None:
    # wnet consumes pylmcf's headers and both are loaded into one process; a
    # mode mismatch here means the two nanobind runtimes cannot see each other.
    import pylmcf.pylmcf_cpp
    from pylmcf.nanobind_mode import check_consistent

    check_consistent([("pylmcf", pylmcf.pylmcf_cpp), ("wnet", wnet_cpp)])


_check_nanobind_modes()
