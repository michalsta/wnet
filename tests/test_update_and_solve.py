"""Tests for WassersteinNetwork.update_positions_and_solve().

Verifies that updating positions without rebuilding the graph gives the same
result as a fresh solve, that the warm-restart counter advances, and that
the chain-order validation fires when peaks cross.
"""

import numpy as np
import pytest

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric


def make_1d(positions, intensities):
    return Distribution(
        np.array([positions], dtype=np.float64), np.array(intensities, dtype=np.int64)
    )


def make_nd(positions_2d, intensities):
    return Distribution(
        np.array(positions_2d, dtype=np.float64), np.array(intensities, dtype=np.int64)
    )


def build_and_solve(base, targets, metric=DistanceMetric.L2, trash=10):
    W = WassersteinNetwork(base, targets, metric)
    W.add_simple_trash(trash)
    W.build()
    W.solve()
    return W


# ---------------------------------------------------------------------------
# 1D chain: update_positions_and_solve matches a fresh solve
# ---------------------------------------------------------------------------


class TestUpdateAndSolve1DChain:
    def test_parity_with_fresh_solve(self):
        """Moving peaks slightly: updated cost == cost from a fresh network."""
        base = make_1d([1.0, 3.0, 5.0], [10, 10, 10])
        target = make_1d([2.0, 4.0, 6.0], [10, 10, 10])

        W = build_and_solve(base, [target])

        # Perturb positions
        base2 = make_1d([1.1, 3.1, 5.1], [10, 10, 10])
        target2 = make_1d([2.1, 4.1, 6.1], [10, 10, 10])

        W.update_positions_and_solve(base2, [target2])
        cost_updated = W.total_cost()

        W_ref = build_and_solve(base2, [target2])
        cost_ref = W_ref.total_cost()

        assert cost_updated == cost_ref

    def test_warm_restart_increments(self):
        """Warm-start counter increases after update_positions_and_solve (NetworkSimplex)."""
        base = make_1d([1.0, 3.0, 5.0], [10, 10, 10])
        target = make_1d([2.0, 4.0, 6.0], [10, 10, 10])
        W = build_and_solve(base, [target])
        warm_before = W.wnet.warm_start_count()

        base2 = make_1d([1.2, 3.2, 5.2], [10, 10, 10])
        target2 = make_1d([2.2, 4.2, 6.2], [10, 10, 10])
        W.update_positions_and_solve(base2, [target2])

        assert W.wnet.warm_start_count() > warm_before

    def test_chain_order_violation_raises(self):
        """Peaks that have crossed raise an exception (option B)."""
        base = make_1d([1.0, 3.0, 5.0], [10, 10, 10])
        target = make_1d([2.0, 4.0, 6.0], [10, 10, 10])
        W = build_and_solve(base, [target])

        # Swap two base peaks so they cross
        base_crossed = make_1d([3.0, 1.0, 5.0], [10, 10, 10])
        with pytest.raises(Exception):
            W.update_positions_and_solve(base_crossed, [target])

    def test_idempotent_same_positions(self):
        """update_positions_and_solve with identical positions gives the same cost."""
        base = make_1d([1.0, 3.0, 5.0], [10, 10, 10])
        target = make_1d([2.0, 4.0, 6.0], [10, 10, 10])
        W = build_and_solve(base, [target])
        cost_before = W.total_cost()

        W.update_positions_and_solve(base, [target])
        assert W.total_cost() == cost_before

    def test_multiple_updates(self):
        """Repeated updates converge to the correct cost each time."""
        base = make_1d([1.0, 3.0, 5.0], [10, 10, 10])
        target = make_1d([2.0, 4.0, 6.0], [10, 10, 10])
        W = build_and_solve(base, [target])

        for delta in [0.1, 0.2, 0.3, 0.5]:
            b = make_1d([1.0 + delta, 3.0 + delta, 5.0 + delta], [10, 10, 10])
            t = make_1d([2.0 + delta, 4.0 + delta, 6.0 + delta], [10, 10, 10])
            W.update_positions_and_solve(b, [t])
            W_ref = build_and_solve(b, [t])
            assert W.total_cost() == W_ref.total_cost()


# ---------------------------------------------------------------------------
# Dense (N-D): update_positions_and_solve matches a fresh solve
# ---------------------------------------------------------------------------


class TestUpdateAndSolveDense:
    def test_2d_warm_matches_cold_sequence(self):
        """Randomized dense 2-D: a warm-restarting solver must track an
        always-cold solver bit-exactly through a sequence of position updates.

        Regression test for the costs_changed warm fast path: warmRun used to
        return the OLD flows priced at the new costs (no reoptimization) when
        only costs changed, which is invisible on the tiny fixtures above.
        """
        from wnet.wnet_cpp import NetworkSimplex, WarmMode

        rng = np.random.default_rng(7)
        N = 80
        pos_base = rng.uniform(0, 60, size=(2, N))
        pos_theo = pos_base + rng.uniform(-3, 3, size=(2, N))
        inten_b = rng.integers(1, 1000, N)
        inten_t = rng.integers(1, 1000, N)
        base = make_nd(pos_base, inten_b)

        def build(warm):
            ns = NetworkSimplex()
            ns.warm = warm
            W = WassersteinNetwork(
                base, [make_nd(pos_theo, inten_t)], DistanceMetric.L2,
                max_distance=8, solver=ns,
            )
            W.add_simple_trash(8)
            W.build()
            W.solve()
            return W

        W_warm = build(WarmMode.DualRatio)
        W_cold = build(WarmMode.NONE)
        assert W_warm.total_cost() == W_cold.total_cost()

        for _ in range(6):
            theo2 = make_nd(pos_theo + rng.uniform(-2, 2, size=(2, N)), inten_t)
            W_warm.update_positions_and_solve(base, [theo2])
            W_cold.update_positions_and_solve(base, [theo2])
            assert W_warm.total_cost() == W_cold.total_cost()
    def test_1d_dense_parity(self):
        """Dense 1D: updated cost matches a fresh solve."""
        base = make_1d([0.0, 5.0], [100, 100])
        target = make_1d([2.0, 7.0], [100, 100])
        W = WassersteinNetwork(base, [target], DistanceMetric.L2, force_dense_1d=True)
        W.add_simple_trash(20)
        W.build()
        W.solve()

        base2 = make_1d([0.5, 5.5], [100, 100])
        target2 = make_1d([2.5, 7.5], [100, 100])
        W.update_positions_and_solve(base2, [target2])

        W_ref = WassersteinNetwork(
            base2, [target2], DistanceMetric.L2, force_dense_1d=True
        )
        W_ref.add_simple_trash(20)
        W_ref.build()
        W_ref.solve()

        assert W.total_cost() == W_ref.total_cost()

    def test_2d_parity(self):
        """2D dense: updated cost matches a fresh solve."""
        pos_b = np.array([[0.0, 4.0], [0.0, 4.0]])  # shape [2, 2]
        pos_t = np.array([[1.0, 5.0], [1.0, 5.0]])
        base = Distribution(pos_b, np.array([50, 50], dtype=np.int64))
        target = Distribution(pos_t, np.array([50, 50], dtype=np.int64))

        W = build_and_solve(base, [target], metric=DistanceMetric.L2, trash=20)

        pos_b2 = pos_b + 0.3
        pos_t2 = pos_t + 0.3
        base2 = Distribution(pos_b2, np.array([50, 50], dtype=np.int64))
        target2 = Distribution(pos_t2, np.array([50, 50], dtype=np.int64))
        W.update_positions_and_solve(base2, [target2])

        W_ref = build_and_solve(base2, [target2], metric=DistanceMetric.L2, trash=20)
        assert W.total_cost() == W_ref.total_cost()

    def test_l1_metric(self):
        """update_positions_and_solve works with L1 metric."""
        base = make_1d([0.0, 5.0], [100, 100])
        target = make_1d([1.0, 6.0], [100, 100])
        W = WassersteinNetwork(base, [target], DistanceMetric.L1, force_dense_1d=True)
        W.add_simple_trash(20)
        W.build()
        W.solve()

        base2 = make_1d([0.5, 5.5], [100, 100])
        target2 = make_1d([1.5, 6.5], [100, 100])
        W.update_positions_and_solve(base2, [target2])

        W_ref = WassersteinNetwork(
            base2, [target2], DistanceMetric.L1, force_dense_1d=True
        )
        W_ref.add_simple_trash(20)
        W_ref.build()
        W_ref.solve()

        assert W.total_cost() == W_ref.total_cost()


# ---------------------------------------------------------------------------
# Multi-spectrum: update_positions_and_solve with multiple targets
# ---------------------------------------------------------------------------


class TestUpdateAndSolveMultiSpectrum:
    def test_two_targets_parity(self):
        base = make_1d([1.0, 3.0, 5.0], [10, 10, 10])
        target1 = make_1d([2.0, 4.0, 6.0], [10, 10, 10])
        target2 = make_1d([0.5, 2.5, 4.5], [10, 10, 10])

        W = build_and_solve(base, [target1, target2])

        base2 = make_1d([1.1, 3.1, 5.1], [10, 10, 10])
        target1b = make_1d([2.1, 4.1, 6.1], [10, 10, 10])
        target2b = make_1d([0.6, 2.6, 4.6], [10, 10, 10])
        W.update_positions_and_solve(base2, [target1b, target2b])

        W_ref = build_and_solve(base2, [target1b, target2b])
        assert W.total_cost() == W_ref.total_cost()
