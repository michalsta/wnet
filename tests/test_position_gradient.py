"""Tests for WassersteinNetwork.update_positions_and_get_gradient().

Positions are scaled ×1000 so that distances are large integers and integer
truncation in total_cost() is negligible.  A perturbation of EPS=100 changes
distances by ~100; truncation error per FD step is bounded by 1/(2*EPS)=0.005,
so FD of total_cost() reliably matches the analytical gradient with ATOL=1.0.
"""

import numpy as np
import pytest

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric

EPS = 100.0
ATOL = 1.0
TRASH = 100_000


def make_dist(positions_2d, intensities):
    return Distribution(
        np.array(positions_2d, dtype=np.float64),
        np.array(intensities, dtype=np.int64),
    )


def build_dense(base, targets, metric, trash=TRASH):
    W = WassersteinNetwork(base, targets, metric, force_dense_1d=True)
    W.add_simple_trash(trash)
    W.build()
    W.solve()
    return W


def cost_at(
    base_pos,
    tgt_pos_list,
    base_int,
    tgt_int_list,
    metric,
    trash=TRASH,
    force_dense_1d=True,
):
    base = make_dist(base_pos, base_int)
    targets = [make_dist(tp, ti) for tp, ti in zip(tgt_pos_list, tgt_int_list)]
    W = WassersteinNetwork(base, targets, metric, force_dense_1d=force_dense_1d)
    W.add_simple_trash(trash)
    W.build()
    W.solve()
    return W.total_cost()


# ---------------------------------------------------------------------------
# 2D dense: gradient vs FD of total_cost() for all three metrics
# ---------------------------------------------------------------------------


class TestGradient2D:
    @pytest.fixture(params=[DistanceMetric.L1, DistanceMetric.L2, DistanceMetric.LINF])
    def metric(self, request):
        return request.param

    def setup_method(self):
        # Positions scaled ×1000 so distances are large; truncation negligible.
        self.base_pos = np.array(
            [[300.0, 4100.0, 8700.0], [200.0, 3500.0, 6800.0]], dtype=np.float64
        )
        self.tgt_pos = np.array(
            [[1900.0, 5300.0, 9100.0], [1400.0, 4200.0, 7600.0]], dtype=np.float64
        )
        self.base_int = np.array([100, 100, 100], dtype=np.int64)
        self.tgt_int = np.array([100, 100, 100], dtype=np.int64)

    def _get_gradient(self, metric):
        base = make_dist(self.base_pos, self.base_int)
        target = make_dist(self.tgt_pos, self.tgt_int)
        W = build_dense(base, [target], metric)
        emp_grad, theo_grads = W.update_positions_and_get_gradient(base, [target])
        return emp_grad, theo_grads

    def test_emp_grad_fd(self, metric):
        emp_grad, _ = self._get_gradient(metric)
        n_peaks = self.base_pos.shape[1]
        dim = self.base_pos.shape[0]
        assert emp_grad.shape == (n_peaks, dim)

        for i in range(n_peaks):
            for d in range(dim):
                pos_p = self.base_pos.copy()
                pos_p[d, i] += EPS
                pos_m = self.base_pos.copy()
                pos_m[d, i] -= EPS
                cp = cost_at(
                    pos_p, [self.tgt_pos], self.base_int, [self.tgt_int], metric
                )
                cm = cost_at(
                    pos_m, [self.tgt_pos], self.base_int, [self.tgt_int], metric
                )
                fd = (cp - cm) / (2.0 * EPS)
                assert abs(emp_grad[i, d] - fd) < ATOL, (
                    f"metric={metric} emp peak {i} dim {d}: "
                    f"grad={emp_grad[i,d]:.4f}, fd={fd:.4f}"
                )

    def test_theo_grad_fd(self, metric):
        _, theo_grads = self._get_gradient(metric)
        n_peaks = self.tgt_pos.shape[1]
        dim = self.tgt_pos.shape[0]
        assert len(theo_grads) == 1
        assert theo_grads[0].shape == (n_peaks, dim)

        for i in range(n_peaks):
            for d in range(dim):
                pos_p = self.tgt_pos.copy()
                pos_p[d, i] += EPS
                pos_m = self.tgt_pos.copy()
                pos_m[d, i] -= EPS
                cp = cost_at(
                    self.base_pos, [pos_p], self.base_int, [self.tgt_int], metric
                )
                cm = cost_at(
                    self.base_pos, [pos_m], self.base_int, [self.tgt_int], metric
                )
                fd = (cp - cm) / (2.0 * EPS)
                assert abs(theo_grads[0][i, d] - fd) < ATOL, (
                    f"metric={metric} theo peak {i} dim {d}: "
                    f"grad={theo_grads[0][i,d]:.4f}, fd={fd:.4f}"
                )

    def test_output_dtype_and_shape(self, metric):
        emp_grad, theo_grads = self._get_gradient(metric)
        assert emp_grad.dtype == np.float64
        assert theo_grads[0].dtype == np.float64
        assert emp_grad.shape == (self.base_pos.shape[1], self.base_pos.shape[0])
        assert theo_grads[0].shape == (self.tgt_pos.shape[1], self.tgt_pos.shape[0])


# ---------------------------------------------------------------------------
# Multiple targets
# ---------------------------------------------------------------------------


class TestGradientMultiTarget:
    def test_two_targets_emp_and_theo_grad(self):
        base_pos = np.array([[300.0, 5700.0], [200.0, 5100.0]], dtype=np.float64)
        t1_pos = np.array([[1900.0, 6800.0], [1400.0, 6300.0]], dtype=np.float64)
        t2_pos = np.array([[-1200.0, 4100.0], [-1300.0, 3900.0]], dtype=np.float64)
        base_int = np.array([200, 200], dtype=np.int64)  # supply covers both targets
        t1_int = np.array([100, 100], dtype=np.int64)
        t2_int = np.array([100, 100], dtype=np.int64)

        base = make_dist(base_pos, base_int)
        target1 = make_dist(t1_pos, t1_int)
        target2 = make_dist(t2_pos, t2_int)
        W = build_dense(base, [target1, target2], DistanceMetric.L2)
        emp_grad, theo_grads = W.update_positions_and_get_gradient(
            base, [target1, target2]
        )

        assert emp_grad.shape == (2, 2)
        assert len(theo_grads) == 2
        assert theo_grads[0].shape == (2, 2)
        assert theo_grads[1].shape == (2, 2)

        metric = DistanceMetric.L2
        tgt_pos_list = [t1_pos, t2_pos]
        tgt_int_list = [t1_int, t2_int]

        for i in range(2):
            for d in range(2):
                pos_p = base_pos.copy()
                pos_p[d, i] += EPS
                pos_m = base_pos.copy()
                pos_m[d, i] -= EPS
                cp = cost_at(pos_p, tgt_pos_list, base_int, tgt_int_list, metric)
                cm = cost_at(pos_m, tgt_pos_list, base_int, tgt_int_list, metric)
                fd = (cp - cm) / (2.0 * EPS)
                assert abs(emp_grad[i, d] - fd) < ATOL

        for s, (tgt_pos, tgt_int) in enumerate(zip(tgt_pos_list, tgt_int_list)):
            for i in range(2):
                for d in range(2):
                    pos_p = tgt_pos.copy()
                    pos_p[d, i] += EPS
                    pos_m = tgt_pos.copy()
                    pos_m[d, i] -= EPS
                    tgt_p = tgt_pos_list[:s] + [pos_p] + tgt_pos_list[s + 1 :]
                    tgt_m = tgt_pos_list[:s] + [pos_m] + tgt_pos_list[s + 1 :]
                    cp = cost_at(base_pos, tgt_p, base_int, tgt_int_list, metric)
                    cm = cost_at(base_pos, tgt_m, base_int, tgt_int_list, metric)
                    fd = (cp - cm) / (2.0 * EPS)
                    assert abs(theo_grads[s][i, d] - fd) < ATOL


# ---------------------------------------------------------------------------
# Dense 1D (force_dense_1d=True) — gradients should work
# ---------------------------------------------------------------------------


class TestGradient1DDense:
    def test_1d_dense_emp_grad_l2(self):
        base_pos = np.array([[300.0, 5700.0]], dtype=np.float64)
        tgt_pos = np.array([[2100.0, 7400.0]], dtype=np.float64)
        base_int = np.array([100, 100], dtype=np.int64)
        tgt_int = np.array([100, 100], dtype=np.int64)

        base = make_dist(base_pos, base_int)
        target = make_dist(tgt_pos, tgt_int)
        W = build_dense(base, [target], DistanceMetric.L2)
        emp_grad, theo_grads = W.update_positions_and_get_gradient(base, [target])

        assert emp_grad.shape == (2, 1)
        for i in range(2):
            pos_p = base_pos.copy()
            pos_p[0, i] += EPS
            pos_m = base_pos.copy()
            pos_m[0, i] -= EPS
            cp = cost_at(pos_p, [tgt_pos], base_int, [tgt_int], DistanceMetric.L2)
            cm = cost_at(pos_m, [tgt_pos], base_int, [tgt_int], DistanceMetric.L2)
            fd = (cp - cm) / (2.0 * EPS)
            assert abs(emp_grad[i, 0] - fd) < ATOL


# ---------------------------------------------------------------------------
# 1D chain network — gradient via FD against chain-network total_cost()
# ---------------------------------------------------------------------------


def _build_chain(base, targets, trash=TRASH):
    W = WassersteinNetwork(base, targets, DistanceMetric.L2)
    W.add_simple_trash(trash)
    W.build()
    W.solve()
    return W


class TestGradient1DChain:
    """Gradient tests for the default 1D chain factory (force_dense_1d=False)."""

    def setup_method(self):
        self.base_pos = np.array([[300.0, 5700.0]], dtype=np.float64)
        self.tgt_pos = np.array([[2100.0, 7400.0]], dtype=np.float64)
        self.base_int = np.array([100, 100], dtype=np.int64)
        self.tgt_int = np.array([100, 100], dtype=np.int64)

    def _get_gradient(self):
        base = make_dist(self.base_pos, self.base_int)
        target = make_dist(self.tgt_pos, self.tgt_int)
        W = _build_chain(base, [target])
        emp_grad, theo_grads = W.update_positions_and_get_gradient(base, [target])
        return emp_grad, theo_grads

    def test_emp_grad_fd(self):
        emp_grad, _ = self._get_gradient()
        assert emp_grad.shape == (2, 1)

        for i in range(2):
            pos_p = self.base_pos.copy()
            pos_p[0, i] += EPS
            pos_m = self.base_pos.copy()
            pos_m[0, i] -= EPS
            cp = cost_at(
                pos_p,
                [self.tgt_pos],
                self.base_int,
                [self.tgt_int],
                DistanceMetric.L2,
                force_dense_1d=False,
            )
            cm = cost_at(
                pos_m,
                [self.tgt_pos],
                self.base_int,
                [self.tgt_int],
                DistanceMetric.L2,
                force_dense_1d=False,
            )
            fd = (cp - cm) / (2.0 * EPS)
            assert (
                abs(emp_grad[i, 0] - fd) < ATOL
            ), f"chain emp peak {i}: grad={emp_grad[i,0]:.4f}, fd={fd:.4f}"

    def test_theo_grad_fd(self):
        _, theo_grads = self._get_gradient()
        assert len(theo_grads) == 1
        assert theo_grads[0].shape == (2, 1)

        for i in range(2):
            pos_p = self.tgt_pos.copy()
            pos_p[0, i] += EPS
            pos_m = self.tgt_pos.copy()
            pos_m[0, i] -= EPS
            cp = cost_at(
                self.base_pos,
                [pos_p],
                self.base_int,
                [self.tgt_int],
                DistanceMetric.L2,
                force_dense_1d=False,
            )
            cm = cost_at(
                self.base_pos,
                [pos_m],
                self.base_int,
                [self.tgt_int],
                DistanceMetric.L2,
                force_dense_1d=False,
            )
            fd = (cp - cm) / (2.0 * EPS)
            assert (
                abs(theo_grads[0][i, 0] - fd) < ATOL
            ), f"chain theo peak {i}: grad={theo_grads[0][i,0]:.4f}, fd={fd:.4f}"

    def test_output_dtype_and_shape(self):
        emp_grad, theo_grads = self._get_gradient()
        assert emp_grad.dtype == np.float64
        assert theo_grads[0].dtype == np.float64
        assert emp_grad.shape == (2, 1)
        assert theo_grads[0].shape == (2, 1)

    def test_chain_matches_dense(self):
        """Chain and dense networks must produce identical gradients."""
        base = make_dist(self.base_pos, self.base_int)
        target = make_dist(self.tgt_pos, self.tgt_int)

        W_chain = _build_chain(base, [target])
        emp_chain, theo_chain = W_chain.update_positions_and_get_gradient(
            base, [target]
        )

        W_dense = build_dense(base, [target], DistanceMetric.L2)
        emp_dense, theo_dense = W_dense.update_positions_and_get_gradient(
            base, [target]
        )

        np.testing.assert_allclose(emp_chain, emp_dense, atol=ATOL)
        np.testing.assert_allclose(theo_chain[0], theo_dense[0], atol=ATOL)

    def test_two_targets(self):
        """Chain gradient with two targets, each base peak matches one target peak."""
        base_pos = np.array([[300.0, 5700.0]], dtype=np.float64)
        t1_pos = np.array([[2100.0, 7400.0]], dtype=np.float64)
        t2_pos = np.array([[-500.0, 4000.0]], dtype=np.float64)
        base_int = np.array([200, 200], dtype=np.int64)
        t1_int = np.array([100, 100], dtype=np.int64)
        t2_int = np.array([100, 100], dtype=np.int64)

        base = make_dist(base_pos, base_int)
        target1 = make_dist(t1_pos, t1_int)
        target2 = make_dist(t2_pos, t2_int)
        W = _build_chain(base, [target1, target2])
        emp_grad, theo_grads = W.update_positions_and_get_gradient(
            base, [target1, target2]
        )

        assert emp_grad.shape == (2, 1)
        assert theo_grads[0].shape == (2, 1)
        assert theo_grads[1].shape == (2, 1)

        tgt_pos_list = [t1_pos, t2_pos]
        tgt_int_list = [t1_int, t2_int]

        for i in range(2):
            pos_p = base_pos.copy()
            pos_p[0, i] += EPS
            pos_m = base_pos.copy()
            pos_m[0, i] -= EPS
            cp = cost_at(
                pos_p,
                tgt_pos_list,
                base_int,
                tgt_int_list,
                DistanceMetric.L2,
                force_dense_1d=False,
            )
            cm = cost_at(
                pos_m,
                tgt_pos_list,
                base_int,
                tgt_int_list,
                DistanceMetric.L2,
                force_dense_1d=False,
            )
            fd = (cp - cm) / (2.0 * EPS)
            assert abs(emp_grad[i, 0] - fd) < ATOL

        for s, (tgt_pos, tgt_int) in enumerate(zip(tgt_pos_list, tgt_int_list)):
            for i in range(2):
                pos_p = tgt_pos.copy()
                pos_p[0, i] += EPS
                pos_m = tgt_pos.copy()
                pos_m[0, i] -= EPS
                tgt_p = tgt_pos_list[:s] + [pos_p] + tgt_pos_list[s + 1 :]
                tgt_m = tgt_pos_list[:s] + [pos_m] + tgt_pos_list[s + 1 :]
                cp = cost_at(
                    base_pos,
                    tgt_p,
                    base_int,
                    tgt_int_list,
                    DistanceMetric.L2,
                    force_dense_1d=False,
                )
                cm = cost_at(
                    base_pos,
                    tgt_m,
                    base_int,
                    tgt_int_list,
                    DistanceMetric.L2,
                    force_dense_1d=False,
                )
                fd = (cp - cm) / (2.0 * EPS)
                assert abs(theo_grads[s][i, 0] - fd) < ATOL
