"""Tests for the DistanceMetric enum in wnet.distances."""

from wnet.distances import DistanceMetric


def test_distance_metric_members():
    """DistanceMetric exposes L1, L2, and LINF."""
    assert hasattr(DistanceMetric, "L1")
    assert hasattr(DistanceMetric, "L2")
    assert hasattr(DistanceMetric, "LINF")


def test_distance_metric_distinct():
    """DistanceMetric members are distinct values."""
    assert DistanceMetric.L1 != DistanceMetric.L2
    assert DistanceMetric.L1 != DistanceMetric.LINF
    assert DistanceMetric.L2 != DistanceMetric.LINF
