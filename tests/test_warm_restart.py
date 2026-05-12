"""
Regression tests for warm-restart of the network simplex solver.

These tests verify that re-solving a network — either at the same point or
after cycling through other points — produces results identical to a fresh
cold solve.  When the warm-restart feature is implemented, update warm_solve()
below to call the actual API; the assertions here define the correctness
contract that warm-restart must preserve.

Observable quantities under test:
  * total_cost()                    — optimal transport cost
  * flows_for_target(i)             — flow decomposition per target
  * signal_part_derivatives()       — per-peak marginal costs
  * spectrum_proportion_derivatives() — per-spectrum gradient
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import dataclass

from wnet import WassersteinNetwork
from wnet.distribution import Distribution, Distribution_1D
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import CWassersteinNetworkFactory, NetworkSimplex


# ---------------------------------------------------------------------------
# Warm-restart shim — replace the body when the API is ready
# ---------------------------------------------------------------------------

def warm_solve(net, point=None):
    """Re-solve using a warm restart.

    NetworkSimplex has warm=True by default, so calling solve() after a prior
    solve reuses the existing spanning-tree basis via warmRun().  The cold
    solve is the first call inside _build_and_solve_cpp, where ns_solver has
    no value yet and the cold branch is taken regardless of the warm flag.
    """
    if point is None:
        net.solve()
    else:
        net.solve(point)


# ---------------------------------------------------------------------------
# Result capture
# ---------------------------------------------------------------------------

@dataclass
class SolveResults:
    """Snapshot of every observable quantity after a solve."""
    total_cost: int
    flows: dict        # {target_id: sorted list of (emp_idx, theo_idx, flow)}
    peak_derivs: dict  # {(spec_id, peak_idx): deriv}
    spec_derivs: dict  # {spec_id: deriv}


def _capture_cpp(net, n_targets: int) -> SolveResults:
    """Capture results from a raw CWassersteinNetwork."""
    flows = {}
    for t in range(n_targets):
        emp, theo, flow = net.flows_for_target(t)
        flows[t] = sorted(
            (int(e), int(t_), int(f))
            for e, t_, f in zip(emp.tolist(), theo.tolist(), flow.tolist())
            if f != 0
        )

    peak_derivs: dict[tuple[int, int], int] = {}
    spec_derivs: dict[int, int] = {}
    for i in range(net.no_subgraphs()):
        sg = net.get_subgraph(i)
        for spec_id, pk, d in sg.signal_part_derivatives():
            peak_derivs[(int(spec_id), int(pk))] = int(d)
        for spec_id, d in sg.spectrum_proportion_derivatives():
            spec_derivs[int(spec_id)] = spec_derivs.get(int(spec_id), 0) + int(d)

    return SolveResults(
        total_cost=net.total_cost(),
        flows=flows,
        peak_derivs=peak_derivs,
        spec_derivs=spec_derivs,
    )


def _capture_W(W, n_targets: int) -> SolveResults:
    """Capture results from a WassersteinNetwork wrapper."""
    flows = {}
    for t in range(n_targets):
        emp, theo, flow = W.flows_for_target(t)
        flows[t] = sorted(
            (int(e), int(t_), int(f))
            for e, t_, f in zip(emp.tolist(), theo.tolist(), flow.tolist())
            if f != 0
        )

    # Flatten nested dicts to (spec_id, peak_idx) → deriv
    peak_derivs: dict[tuple[int, int], int] = {}
    for spec_id, peaks in W.signal_part_derivatives().items():
        for pk, d in peaks.items():
            peak_derivs[(int(spec_id), int(pk))] = int(d)

    spec_derivs = {int(k): int(v) for k, v in W.spectrum_proportion_derivatives().items()}

    return SolveResults(
        total_cost=W.total_cost(),
        flows=flows,
        peak_derivs=peak_derivs,
        spec_derivs=spec_derivs,
    )


def _assert_exact(fresh: SolveResults, warm: SolveResults, tag: str = "") -> None:
    """All quantities must match exactly (same point, same basis expected)."""
    prefix = f"[{tag}] " if tag else ""
    assert fresh.total_cost == warm.total_cost, (
        f"{prefix}cost: fresh={fresh.total_cost} warm={warm.total_cost}"
    )
    assert fresh.flows == warm.flows, f"{prefix}flows differ"
    assert fresh.peak_derivs == warm.peak_derivs, f"{prefix}peak_derivs differ"
    assert fresh.spec_derivs == warm.spec_derivs, f"{prefix}spec_derivs differ"


def _marginals(flow_list):
    """Row/column sums for a flow list, for cross-point marginal checks."""
    row: dict[int, int] = {}
    col: dict[int, int] = {}
    for e, t, f in flow_list:
        row[e] = row.get(e, 0) + f
        col[t] = col.get(t, 0) + f
    return row, col


def _assert_cost_and_marginals(r1: SolveResults, r2: SolveResults, tag: str = "") -> None:
    """Cost and flow marginals must match; exact decomposition may differ."""
    prefix = f"[{tag}] " if tag else ""
    assert r1.total_cost == r2.total_cost, (
        f"{prefix}cost: {r1.total_cost} vs {r2.total_cost}"
    )
    for t_id in set(r1.flows) | set(r2.flows):
        row1, col1 = _marginals(r1.flows.get(t_id, []))
        row2, col2 = _marginals(r2.flows.get(t_id, []))
        assert row1 == row2, f"{prefix}target={t_id} emp marginals: {row1} vs {row2}"
        assert col1 == col2, f"{prefix}target={t_id} theo marginals: {col1} vs {col2}"
    assert r1.peak_derivs == r2.peak_derivs, f"{prefix}peak_derivs differ"
    assert r1.spec_derivs == r2.spec_derivs, f"{prefix}spec_derivs differ"


# ---------------------------------------------------------------------------
# Helpers: build and solve (C++ layer)
# ---------------------------------------------------------------------------

def _build_and_solve_cpp(factory_fn, base, targets, trash_cost, max_dist, point=None):
    vec_base = base.vecdist
    vec_targets = [t.vecdist for t in targets]
    net = factory_fn(vec_base, vec_targets, DistanceMetric.L1, max_dist)
    if trash_cost is not None:
        net.add_simple_trash(trash_cost)
    net.build(NetworkSimplex())
    if point is None:
        net.solve()
    else:
        net.solve(np.asarray(point, dtype=np.float64))
    return net


def _build_and_solve_W(base, targets, trash_cost, max_dist, point=None, distance=DistanceMetric.L1, **kw):
    W = WassersteinNetwork(base, targets, distance, max_distance=max_dist, **kw)
    if trash_cost is not None:
        W.add_simple_trash(trash_cost)
    W.build()
    if point is None:
        W.solve()
    else:
        W.solve(np.asarray(point, dtype=np.float64))
    return W


# ===========================================================================
# 1.  Idempotency: solve() → warm_solve() at same point
# ===========================================================================

class TestIdempotency:
    """warm_solve() at the same point must reproduce all results exactly."""

    def test_1d_chain_balanced_masses(self):
        """Balanced 1D — trash is required by the solver but stays at zero flow."""
        base = Distribution_1D(np.array([0.0, 5.0, 10.0]), np.array([3, 6, 3]))
        target = Distribution_1D(np.array([1.0, 6.0, 9.0]), np.array([3, 6, 3]))
        net = _build_and_solve_cpp(CWassersteinNetworkFactory.create_1d, base, [target], 1000, 1000)
        fresh = _capture_cpp(net, 1)
        warm_solve(net)
        _assert_exact(fresh, _capture_cpp(net, 1), "1d_chain_balanced_masses")

    def test_1d_chain_simple_trash(self):
        base = Distribution_1D(np.array([0.0, 5.0, 10.0]), np.array([4, 6, 2]))
        target = Distribution_1D(np.array([1.0, 6.0]), np.array([4, 6]))
        net = _build_and_solve_cpp(CWassersteinNetworkFactory.create_1d, base, [target], 50, 1000)
        fresh = _capture_cpp(net, 1)
        warm_solve(net)
        _assert_exact(fresh, _capture_cpp(net, 1), "1d_chain_simple_trash")

    def test_1d_dense_simple_trash(self):
        base = Distribution_1D(np.array([0.0, 5.0, 10.0]), np.array([4, 6, 2]))
        target = Distribution_1D(np.array([1.0, 6.0]), np.array([4, 6]))
        net = _build_and_solve_cpp(CWassersteinNetworkFactory.create, base, [target], 50, 1000)
        fresh = _capture_cpp(net, 1)
        warm_solve(net)
        _assert_exact(fresh, _capture_cpp(net, 1), "1d_dense_simple_trash")

    def test_2d_dense_simple_trash(self):
        pos_e = np.array([[0.0, 1.0, 5.0], [0.0, 0.0, 3.0]])
        pos_t = np.array([[1.0, 4.0], [0.0, 3.0]])
        E = Distribution(pos_e, np.array([5, 3, 4]))
        T = Distribution(pos_t, np.array([6, 6]))
        W = _build_and_solve_W(E, [T], 50, 100)
        fresh = _capture_W(W, 1)
        warm_solve(W)
        _assert_exact(fresh, _capture_W(W, 1), "2d_dense_simple_trash")

    def test_1d_chain_excess_base(self):
        """More empirical than theoretical — excess goes to trash."""
        base = Distribution_1D(np.array([0.0]), np.array([10]))
        target = Distribution_1D(np.array([10.0]), np.array([5]))
        net = _build_and_solve_cpp(CWassersteinNetworkFactory.create_1d, base, [target], 100, 1000)
        fresh = _capture_cpp(net, 1)
        warm_solve(net)
        _assert_exact(fresh, _capture_cpp(net, 1), "1d_chain_excess_base")

    def test_1d_chain_excess_theo(self):
        """More theoretical than empirical — excess goes to trash."""
        base = Distribution_1D(np.array([0.0]), np.array([5]))
        target = Distribution_1D(np.array([10.0]), np.array([10]))
        net = _build_and_solve_cpp(CWassersteinNetworkFactory.create_1d, base, [target], 100, 1000)
        fresh = _capture_cpp(net, 1)
        warm_solve(net)
        _assert_exact(fresh, _capture_cpp(net, 1), "1d_chain_excess_theo")

    def test_1d_chain_with_truncation(self):
        """max_distance splits the problem into components."""
        base = Distribution_1D(np.array([0.0, 1.0, 100.0, 101.0]), np.array([3, 3, 5, 5]))
        target = Distribution_1D(np.array([2.0]), np.array([6]))
        net = _build_and_solve_cpp(CWassersteinNetworkFactory.create_1d, base, [target], 20, 5)
        fresh = _capture_cpp(net, 1)
        warm_solve(net)
        _assert_exact(fresh, _capture_cpp(net, 1), "1d_chain_truncation")

    def test_wnet_wrapper_idempotency(self):
        """WassersteinNetwork wrapper: warm_solve() reproduces all results."""
        bp = [0.0, 50.0]
        bi = [8, 8]
        tp = [10.0, 40.0, 60.0]
        ti = [3, 2, 5]
        base = Distribution_1D(np.array(bp), np.array(bi, dtype=np.int64))
        target = Distribution_1D(np.array(tp), np.array(ti, dtype=np.int64))
        W = _build_and_solve_W(base, [target], 100, 200)
        fresh = _capture_W(W, 1)
        warm_solve(W)
        _assert_exact(fresh, _capture_W(W, 1), "wnet_wrapper_idempotency")

    @pytest.mark.parametrize("n_repeats", [2, 5, 10])
    def test_repeated_warm_solve_stays_stable(self, n_repeats):
        """Multiple successive warm restarts must not drift from the first result."""
        base = Distribution_1D(np.array([0.0, 5.0, 10.0, 15.0]), np.array([4, 6, 2, 3]))
        target = Distribution_1D(np.array([1.0, 6.0, 14.0]), np.array([4, 6, 5]))
        net = _build_and_solve_cpp(CWassersteinNetworkFactory.create_1d, base, [target], 50, 1000)
        fresh = _capture_cpp(net, 1)
        for _ in range(n_repeats):
            warm_solve(net)
        _assert_exact(fresh, _capture_cpp(net, 1), f"n_repeats={n_repeats}")


# ===========================================================================
# 2.  Asymmetric trash: idempotency under warm restart
# ===========================================================================

class TestAsymmetricTrashIdempotency:
    """Results with experimental/theoretical trash must be stable after warm restart."""

    def test_experimental_trash_idempotency(self):
        base = Distribution_1D(np.array([0.0, 50.0]), np.array([10, 10]))
        target = Distribution_1D(np.array([5.0]), np.array([10]))
        vec_base = base.vecdist
        vec_target = [target.vecdist]
        net = CWassersteinNetworkFactory.create_1d(vec_base, vec_target, DistanceMetric.L1, 1000)
        net.add_experimental_trash(30)
        net.build(NetworkSimplex())
        net.solve()
        fresh = _capture_cpp(net, 1)
        warm_solve(net)
        _assert_exact(fresh, _capture_cpp(net, 1), "experimental_trash")

    def test_theoretical_trash_idempotency(self):
        base = Distribution_1D(np.array([0.0]), np.array([5]))
        target = Distribution_1D(np.array([10.0, 20.0]), np.array([5, 5]))
        vec_base = base.vecdist
        vec_target = [target.vecdist]
        net = CWassersteinNetworkFactory.create_1d(vec_base, vec_target, DistanceMetric.L1, 1000)
        net.add_theoretical_trash(30)
        net.build(NetworkSimplex())
        net.solve()
        fresh = _capture_cpp(net, 1)
        warm_solve(net)
        _assert_exact(fresh, _capture_cpp(net, 1), "theoretical_trash")


# ===========================================================================
# 3.  Multi-spectrum idempotency
# ===========================================================================

class TestMultiSpectrumIdempotency:
    """Warm restart must preserve results for networks with multiple target spectra."""

    def test_two_spectra_no_point(self):
        """solve() (point=None means equal proportions) is stable."""
        base = Distribution_1D(np.array([0.0, 5.0, 10.0]), np.array([4, 6, 2]))
        t1 = Distribution_1D(np.array([1.0, 6.0]), np.array([4, 6]))
        t2 = Distribution_1D(np.array([9.5]), np.array([2]))
        net = _build_and_solve_cpp(CWassersteinNetworkFactory.create_1d, base, [t1, t2], 50, 1000)
        fresh = _capture_cpp(net, 2)
        warm_solve(net)
        _assert_exact(fresh, _capture_cpp(net, 2), "two_spectra_no_point")

    def test_two_spectra_equal_point(self):
        """solve([0.5, 0.5]) is stable."""
        base = Distribution_1D(np.array([0.0, 5.0, 10.0]), np.array([4, 6, 2]))
        t1 = Distribution_1D(np.array([1.0, 6.0]), np.array([4, 6]))
        t2 = Distribution_1D(np.array([9.5]), np.array([2]))
        net = _build_and_solve_cpp(CWassersteinNetworkFactory.create_1d, base, [t1, t2], 50, 1000)
        net.solve(np.array([0.5, 0.5]))
        fresh = _capture_cpp(net, 2)
        warm_solve(net, np.array([0.5, 0.5]))
        _assert_exact(fresh, _capture_cpp(net, 2), "two_spectra_equal_point")

    def test_two_spectra_asymmetric_point(self):
        """solve([0.7, 0.3]) is stable."""
        base = Distribution_1D(np.array([0.0, 5.0, 10.0]), np.array([10, 10, 10]))
        t1 = Distribution_1D(np.array([1.0, 6.0]), np.array([5, 5]))
        t2 = Distribution_1D(np.array([9.5, 11.0]), np.array([5, 5]))
        net = _build_and_solve_cpp(CWassersteinNetworkFactory.create_1d, base, [t1, t2], 50, 1000)
        pt = np.array([0.7, 0.3])
        net.solve(pt)
        fresh = _capture_cpp(net, 2)
        warm_solve(net, pt)
        _assert_exact(fresh, _capture_cpp(net, 2), "two_spectra_asymmetric_point")

    def test_three_spectra(self):
        base = Distribution_1D(np.array([0.0, 5.0, 10.0, 15.0]), np.array([6, 6, 6, 6]))
        t1 = Distribution_1D(np.array([1.0, 6.0]), np.array([4, 6]))
        t2 = Distribution_1D(np.array([9.5]), np.array([4]))
        t3 = Distribution_1D(np.array([14.0, 16.0]), np.array([3, 3]))
        pt = np.array([0.4, 0.3, 0.3])
        net = _build_and_solve_cpp(CWassersteinNetworkFactory.create_1d, base, [t1, t2, t3], 80, 1000)
        net.solve(pt)
        fresh = _capture_cpp(net, 3)
        warm_solve(net, pt)
        _assert_exact(fresh, _capture_cpp(net, 3), "three_spectra")


# ===========================================================================
# 4.  Point-cycling: solve(p1) → solve(p2) → … → warm_solve(p1)
#     Results must match a fresh cold solve at p1.
# ===========================================================================

class TestPointCycling:
    """
    After solving at one point, then cycling through other points, then
    warm-restarting back to the original point, results must equal a fresh
    cold solve at that point.

    Because network simplex may reach a different optimal basis when coming
    from a warm start, we only require cost and flow-marginal equality here
    (not exact flow-tuple equality), but do require exact derivative equality.
    """

    def _fresh_at_point(self, base, targets, trash_cost, max_dist, point):
        """Build a brand-new network and solve at point."""
        net = _build_and_solve_cpp(
            CWassersteinNetworkFactory.create_1d, base, targets, trash_cost, max_dist,
            point=point,
        )
        return net, _capture_cpp(net, len(targets))

    def test_single_spectrum_cycle(self):
        base = Distribution_1D(np.array([0.0, 5.0, 10.0]), np.array([4, 6, 2]))
        target = Distribution_1D(np.array([1.0, 6.0]), np.array([4, 6]))
        orig_pt = np.array([1.0])
        other_pts = [np.array([0.5]), np.array([2.0]), np.array([0.1]), np.array([3.0])]

        fresh_net, fresh = self._fresh_at_point(base, [target], 50, 1000, orig_pt)

        net = _build_and_solve_cpp(
            CWassersteinNetworkFactory.create_1d, base, [target], 50, 1000, point=orig_pt
        )
        for pt in other_pts:
            net.solve(pt)
        warm_solve(net, orig_pt)

        warm = _capture_cpp(net, 1)
        _assert_cost_and_marginals(fresh, warm, "single_spectrum_cycle")

    def test_two_spectra_cycle(self):
        base = Distribution_1D(np.array([0.0, 5.0, 10.0]), np.array([10, 10, 10]))
        t1 = Distribution_1D(np.array([1.0, 6.0]), np.array([5, 5]))
        t2 = Distribution_1D(np.array([9.5, 11.0]), np.array([5, 5]))
        orig_pt = np.array([0.5, 0.5])

        fresh_net, fresh = self._fresh_at_point(base, [t1, t2], 50, 1000, orig_pt)

        net = _build_and_solve_cpp(
            CWassersteinNetworkFactory.create_1d, base, [t1, t2], 50, 1000, point=orig_pt
        )
        for pt in [np.array([0.3, 0.7]), np.array([1.5, 0.5]), np.array([0.8, 1.2])]:
            net.solve(pt)
        warm_solve(net, orig_pt)

        warm = _capture_cpp(net, 2)
        _assert_cost_and_marginals(fresh, warm, "two_spectra_cycle")

    def test_wnet_wrapper_cycle(self):
        """WassersteinNetwork wrapper: point cycling via warm_solve()."""
        bp, bi = [0.0, 50.0], [10, 10]
        tp, ti = [10.0, 40.0], [5, 5]
        md = 100

        base = Distribution_1D(np.array(bp), np.array(bi, dtype=np.int64))
        target = Distribution_1D(np.array(tp), np.array(ti, dtype=np.int64))

        W_fresh = _build_and_solve_W(base, [target], md, md, point=[1.0])
        fresh = _capture_W(W_fresh, 1)

        W = _build_and_solve_W(base, [target], md, md, point=[1.0])
        for pt in [[0.5], [2.0], [0.1], [3.0]]:
            W.solve(np.array(pt))
        warm_solve(W, np.array([1.0]))

        warm = _capture_W(W, 1)
        _assert_cost_and_marginals(fresh, warm, "wnet_wrapper_cycle")

    @pytest.mark.long
    @pytest.mark.parametrize("seed", range(20))
    def test_random_single_spectrum_cycle(self, seed):
        """Randomized: point cycling must restore cost and marginals."""
        rng = np.random.default_rng(seed)
        m = int(rng.integers(2, 12))
        n = int(rng.integers(2, 12))
        e_pos = rng.integers(0, 200, size=m).astype(np.float64)
        t_pos = rng.integers(0, 200, size=n).astype(np.float64)
        e_int = rng.integers(1, 10, size=m).astype(np.int64)
        t_int = rng.integers(1, 10, size=n).astype(np.int64)
        base = Distribution_1D(e_pos, e_int)
        target = Distribution_1D(t_pos, t_int)
        trash_cost = 50
        max_dist = int(rng.integers(50, 300))
        orig_pt = np.array([1.0])

        fresh_net, fresh = self._fresh_at_point(base, [target], trash_cost, max_dist, orig_pt)

        net = _build_and_solve_cpp(
            CWassersteinNetworkFactory.create_1d, base, [target], trash_cost, max_dist, point=orig_pt
        )
        n_other = int(rng.integers(3, 8))
        for pt_val in rng.uniform(0.2, 3.0, size=n_other):
            net.solve(np.array([pt_val]))
        warm_solve(net, orig_pt)

        warm = _capture_cpp(net, 1)
        _assert_cost_and_marginals(
            fresh, warm,
            f"seed={seed} m={m} n={n} max_dist={max_dist}"
        )

    @pytest.mark.long
    @pytest.mark.parametrize("seed", range(10))
    def test_random_two_spectra_cycle(self, seed):
        """Randomized two-spectrum point cycling."""
        rng = np.random.default_rng(seed)
        m = int(rng.integers(2, 10))
        n1 = int(rng.integers(2, 8))
        n2 = int(rng.integers(2, 8))
        e_pos = rng.integers(0, 100, size=m).astype(np.float64)
        e_int = rng.integers(1, 10, size=m).astype(np.int64)
        t1_pos = rng.integers(0, 100, size=n1).astype(np.float64)
        t1_int = rng.integers(1, 10, size=n1).astype(np.int64)
        t2_pos = rng.integers(0, 100, size=n2).astype(np.float64)
        t2_int = rng.integers(1, 10, size=n2).astype(np.int64)
        base = Distribution_1D(e_pos, e_int)
        t1 = Distribution_1D(t1_pos, t1_int)
        t2 = Distribution_1D(t2_pos, t2_int)
        trash_cost = 100
        max_dist = 300
        orig_pt = np.array([0.5, 0.5])

        fresh_net, fresh = self._fresh_at_point(base, [t1, t2], trash_cost, max_dist, orig_pt)

        net = _build_and_solve_cpp(
            CWassersteinNetworkFactory.create_1d, base, [t1, t2], trash_cost, max_dist, point=orig_pt
        )
        for _ in range(5):
            p = rng.uniform(0.2, 1.5, size=2)
            net.solve(p)
        warm_solve(net, orig_pt)

        warm = _capture_cpp(net, 2)
        _assert_cost_and_marginals(
            fresh, warm,
            f"seed={seed} m={m} n1={n1} n2={n2}"
        )


# ===========================================================================
# 5.  Chain vs dense factory: both must be stable after warm restart
# ===========================================================================

class TestFactoryParity:
    """After warm restart, chain and dense factories must still agree on cost."""

    @pytest.mark.parametrize("trash_cost", [50, 200])
    def test_parity_after_warm_restart(self, trash_cost):
        base = Distribution_1D(np.array([0.0, 5.0, 10.0]), np.array([4, 6, 2]))
        target = Distribution_1D(np.array([1.0, 6.0]), np.array([4, 6]))

        chain_net = _build_and_solve_cpp(
            CWassersteinNetworkFactory.create_1d, base, [target], trash_cost, 1000
        )
        dense_net = _build_and_solve_cpp(
            CWassersteinNetworkFactory.create, base, [target], trash_cost, 1000
        )
        assert chain_net.total_cost() == dense_net.total_cost()

        warm_solve(chain_net)
        warm_solve(dense_net)
        assert chain_net.total_cost() == dense_net.total_cost()

    @pytest.mark.long
    @pytest.mark.parametrize("seed", range(10))
    def test_random_parity_after_warm_restart(self, seed):
        """Chain and dense must agree after warm restart on random inputs."""
        rng = np.random.default_rng(seed)
        m = int(rng.integers(1, 20))
        n = int(rng.integers(1, 20))
        e_pos = rng.integers(0, 200, size=m).astype(np.float64)
        t_pos = rng.integers(0, 200, size=n).astype(np.float64)
        e_int = rng.integers(1, 10, size=m).astype(np.int64)
        t_int = rng.integers(1, 10, size=n).astype(np.int64)
        base = Distribution_1D(e_pos, e_int)
        target = Distribution_1D(t_pos, t_int)
        trash_cost = 50
        max_dist = int(rng.integers(50, 300))

        chain_net = _build_and_solve_cpp(
            CWassersteinNetworkFactory.create_1d, base, [target], trash_cost, max_dist
        )
        dense_net = _build_and_solve_cpp(
            CWassersteinNetworkFactory.create, base, [target], trash_cost, max_dist
        )
        chain_cost_1 = chain_net.total_cost()
        dense_cost_1 = dense_net.total_cost()
        if chain_cost_1 != dense_cost_1:
            pytest.skip(f"seed={seed}: cost divergence before restart (max_dist truncation)")

        warm_solve(chain_net)
        warm_solve(dense_net)

        chain_cost_2 = chain_net.total_cost()
        dense_cost_2 = dense_net.total_cost()
        assert chain_cost_1 == chain_cost_2, f"seed={seed}: chain cost changed after warm restart"
        assert dense_cost_1 == dense_cost_2, f"seed={seed}: dense cost changed after warm restart"
        assert chain_cost_2 == dense_cost_2, f"seed={seed}: parity lost after warm restart"


# ===========================================================================
# 6.  Derivative stability
# ===========================================================================

class TestDerivativeStability:
    """Derivatives must be identical before and after warm restart."""

    def test_peak_derivatives_stable(self):
        base = Distribution_1D(np.array([0.0]), np.array([10]))
        target = Distribution_1D(np.array([10.0]), np.array([5]))
        net = _build_and_solve_cpp(CWassersteinNetworkFactory.create_1d, base, [target], 100, 1000)
        d_before = {(int(s), int(p)): int(d)
                    for i in range(net.no_subgraphs())
                    for s, p, d in net.get_subgraph(i).signal_part_derivatives()}
        warm_solve(net)
        d_after = {(int(s), int(p)): int(d)
                   for i in range(net.no_subgraphs())
                   for s, p, d in net.get_subgraph(i).signal_part_derivatives()}
        assert d_before == d_after

    def test_spectrum_proportion_derivatives_stable(self):
        base = Distribution_1D(np.array([0.0, 50.0]), np.array([8, 8]))
        target = Distribution_1D(np.array([10.0, 40.0, 60.0]), np.array([3, 2, 5]))
        W = _build_and_solve_W(base, [target], 100, 200)
        pd_before = W.spectrum_proportion_derivatives()
        warm_solve(W)
        pd_after = W.spectrum_proportion_derivatives()
        assert pd_before == pd_after

    def test_derivatives_stable_after_multi_point_cycle(self):
        """Derivatives at point=1 match a fresh solve after cycling through other points."""
        bp, bi = [0.0, 50.0], [10, 10]
        tp, ti = [10.0, 40.0], [5, 5]
        md = 100

        base = Distribution_1D(np.array(bp), np.array(bi, dtype=np.int64))
        target = Distribution_1D(np.array(tp), np.array(ti, dtype=np.int64))

        W_fresh = _build_and_solve_W(base, [target], md, md, point=[1.0])
        fresh_cost = W_fresh.total_cost()
        fresh_peak = W_fresh.signal_part_derivatives()
        fresh_spec = W_fresh.spectrum_proportion_derivatives()

        W = _build_and_solve_W(base, [target], md, md, point=[1.0])
        for pt in [[0.5], [2.0], [0.1], [3.0]]:
            W.solve(np.array(pt))
        warm_solve(W, np.array([1.0]))

        assert W.total_cost() == fresh_cost
        assert W.signal_part_derivatives() == fresh_peak
        assert W.spectrum_proportion_derivatives() == fresh_spec

    @pytest.mark.long
    @pytest.mark.parametrize("seed", range(20))
    def test_random_derivatives_stable_after_cycle(self, seed):
        """Randomized: derivatives at point=1 must survive point cycling."""
        rng = np.random.default_rng(seed)
        n_base = int(rng.integers(2, 8))
        n_target = int(rng.integers(2, 8))
        bp = (rng.uniform(0, 100, size=n_base) * 1000).astype(np.int64).astype(np.float64)
        bi = rng.integers(1, 20, size=n_base).astype(np.int64)
        tp = (rng.uniform(0, 100, size=n_target) * 1000).astype(np.int64).astype(np.float64)
        ti = rng.integers(1, 20, size=n_target).astype(np.int64)
        md = 50000

        base = Distribution_1D(bp, bi)
        target = Distribution_1D(tp, ti)

        W_fresh = _build_and_solve_W(base, [target], md, md, point=[1.0])
        fresh_cost = W_fresh.total_cost()
        fresh_peak = W_fresh.signal_part_derivatives()
        fresh_spec = W_fresh.spectrum_proportion_derivatives()

        W = _build_and_solve_W(base, [target], md, md, point=[1.0])
        points = [[p] for p in rng.uniform(0.3, 3.0, size=5)]
        for pt in points:
            W.solve(np.array(pt))
        warm_solve(W, np.array([1.0]))

        assert W.total_cost() == fresh_cost, f"seed={seed}: cost"
        assert W.signal_part_derivatives() == fresh_peak, f"seed={seed}: peak_derivs"
        assert W.spectrum_proportion_derivatives() == fresh_spec, f"seed={seed}: spec_derivs"


# ===========================================================================
# 7.  Large randomized idempotency
# ===========================================================================

class TestLargeIdempotency:
    """Warm-restart idempotency on larger random networks."""

    @pytest.mark.long
    @pytest.mark.parametrize("seed", range(10))
    def test_random_1d_idempotency(self, seed):
        rng = np.random.default_rng(seed)
        m = int(rng.integers(5, 50))
        n = int(rng.integers(5, 50))
        e_pos = rng.integers(0, 1000, size=m).astype(np.float64)
        t_pos = rng.integers(0, 1000, size=n).astype(np.float64)
        e_int = rng.integers(1, 20, size=m).astype(np.int64)
        t_int = rng.integers(1, 20, size=n).astype(np.int64)
        base = Distribution_1D(e_pos, e_int)
        target = Distribution_1D(t_pos, t_int)
        trash_cost = 100
        max_dist = int(rng.integers(100, 500))

        net = _build_and_solve_cpp(
            CWassersteinNetworkFactory.create_1d, base, [target], trash_cost, max_dist
        )
        fresh = _capture_cpp(net, 1)
        warm_solve(net)
        _assert_exact(fresh, _capture_cpp(net, 1), f"seed={seed} m={m} n={n}")

    @pytest.mark.long
    @pytest.mark.parametrize("seed", range(5))
    def test_random_2d_idempotency(self, seed):
        rng = np.random.default_rng(seed)
        m = int(rng.integers(3, 20))
        n = int(rng.integers(3, 20))
        e_pos = rng.integers(0, 100, size=(2, m)).astype(np.float64)
        t_pos = rng.integers(0, 100, size=(2, n)).astype(np.float64)
        e_int = rng.integers(1, 10, size=m).astype(np.int64)
        t_int = rng.integers(1, 10, size=n).astype(np.int64)
        E = Distribution(e_pos, e_int)
        T = Distribution(t_pos, t_int)
        W = _build_and_solve_W(E, [T], 100, 500)
        fresh = _capture_W(W, 1)
        warm_solve(W)
        _assert_exact(fresh, _capture_W(W, 1), f"seed={seed} m={m} n={n}")


# ===========================================================================
# 8.  Cold vs warm parity: successive solve() calls must agree exactly
# ===========================================================================

class TestColdVsWarmParity:
    """
    Builds two identical networks — one fully cold (NetworkSimplex(warm=False)),
    one warm-whenever-possible (NetworkSimplex() with warm=True default) — then
    drives both through the same sequence of points.  Cost, flow marginals, and
    derivatives must agree at every step.
    """

    @staticmethod
    def _cpp_pair(factory_fn, base, targets, trash_cost, max_dist):
        cold_cfg = NetworkSimplex()
        cold_cfg.warm = False

        def _make(cfg):
            net = factory_fn(
                base.vecdist, [t.vecdist for t in targets], DistanceMetric.L1, max_dist
            )
            if trash_cost is not None:
                net.add_simple_trash(trash_cost)
            net.build(cfg)
            return net

        return _make(cold_cfg), _make(NetworkSimplex())

    @staticmethod
    def _W_pair(base, targets, trash_cost, max_dist):
        cold_cfg = NetworkSimplex()
        cold_cfg.warm = False

        def _make(solver):
            W = WassersteinNetwork(
                base, targets, DistanceMetric.L1, max_distance=max_dist, solver=solver
            )
            if trash_cost is not None:
                W.add_simple_trash(trash_cost)
            W.build()
            return W

        return _make(cold_cfg), _make(NetworkSimplex())

    @staticmethod
    def _drive(cold, warm, points, n_targets, capture_fn):
        for pt in points:
            cold.solve(pt)
            warm.solve(pt)
            _assert_cost_and_marginals(
                capture_fn(cold, n_targets),
                capture_fn(warm, n_targets),
                f"pt={pt}",
            )

    def test_1d_chain_single_spectrum(self):
        base = Distribution_1D(np.array([0.0, 5.0, 10.0]), np.array([4, 6, 2]))
        target = Distribution_1D(np.array([1.0, 6.0]), np.array([4, 6]))
        cold, warm = self._cpp_pair(
            CWassersteinNetworkFactory.create_1d, base, [target], 50, 1000
        )
        pts = [np.array([0.5]), np.array([1.5]), np.array([2.0]), np.array([0.2]), np.array([1.0])]
        self._drive(cold, warm, pts, 1, _capture_cpp)

    def test_1d_dense_single_spectrum(self):
        base = Distribution_1D(np.array([0.0, 5.0, 10.0]), np.array([4, 6, 2]))
        target = Distribution_1D(np.array([1.0, 6.0]), np.array([4, 6]))
        cold, warm = self._cpp_pair(
            CWassersteinNetworkFactory.create, base, [target], 50, 1000
        )
        pts = [np.array([0.5]), np.array([1.5]), np.array([2.0]), np.array([0.2]), np.array([1.0])]
        self._drive(cold, warm, pts, 1, _capture_cpp)

    def test_1d_chain_two_spectra(self):
        base = Distribution_1D(np.array([0.0, 5.0, 10.0]), np.array([10, 10, 10]))
        t1 = Distribution_1D(np.array([1.0, 6.0]), np.array([5, 5]))
        t2 = Distribution_1D(np.array([9.5, 11.0]), np.array([5, 5]))
        cold, warm = self._cpp_pair(
            CWassersteinNetworkFactory.create_1d, base, [t1, t2], 50, 1000
        )
        pts = [
            np.array([0.3, 0.7]),
            np.array([0.5, 0.5]),
            np.array([0.8, 0.2]),
            np.array([1.2, 0.8]),
            np.array([0.1, 0.9]),
            np.array([0.5, 0.5]),
        ]
        self._drive(cold, warm, pts, 2, _capture_cpp)

    def test_wrapper_single_spectrum(self):
        base = Distribution_1D(np.array([0.0, 50.0]), np.array([8, 8]))
        target = Distribution_1D(np.array([10.0, 40.0, 60.0]), np.array([3, 2, 5]))
        cold, warm = self._W_pair(base, [target], 100, 200)
        pts = [np.array([0.5]), np.array([1.5]), np.array([2.0]), np.array([0.3]), np.array([1.0])]
        self._drive(cold, warm, pts, 1, _capture_W)

    def test_wrapper_two_spectra(self):
        base = Distribution_1D(np.array([0.0, 50.0]), np.array([10, 10]))
        t1 = Distribution_1D(np.array([10.0, 40.0]), np.array([5, 5]))
        t2 = Distribution_1D(np.array([20.0, 60.0]), np.array([5, 5]))
        cold, warm = self._W_pair(base, [t1, t2], 100, 200)
        pts = [
            np.array([0.4, 0.6]),
            np.array([0.6, 0.4]),
            np.array([1.0, 1.0]),
            np.array([0.2, 0.8]),
            np.array([0.5, 0.5]),
        ]
        self._drive(cold, warm, pts, 2, _capture_W)

    @pytest.mark.long
    @pytest.mark.parametrize("seed", range(15))
    def test_random_1d_chain(self, seed):
        rng = np.random.default_rng(seed)
        m = int(rng.integers(2, 12))
        n = int(rng.integers(2, 12))
        e_pos = rng.integers(0, 200, size=m).astype(np.float64)
        t_pos = rng.integers(0, 200, size=n).astype(np.float64)
        e_int = rng.integers(1, 10, size=m).astype(np.int64)
        t_int = rng.integers(1, 10, size=n).astype(np.int64)
        base = Distribution_1D(e_pos, e_int)
        target = Distribution_1D(t_pos, t_int)
        max_dist = int(rng.integers(50, 300))
        cold, warm = self._cpp_pair(
            CWassersteinNetworkFactory.create_1d, base, [target], 50, max_dist
        )
        pts = [np.array([v]) for v in rng.uniform(0.2, 3.0, size=6)]
        self._drive(cold, warm, pts, 1, _capture_cpp)

    @pytest.mark.long
    @pytest.mark.parametrize("seed", range(10))
    def test_random_1d_two_spectra(self, seed):
        rng = np.random.default_rng(seed)
        m = int(rng.integers(2, 10))
        n1 = int(rng.integers(2, 8))
        n2 = int(rng.integers(2, 8))
        e_pos = rng.integers(0, 100, size=m).astype(np.float64)
        e_int = rng.integers(1, 10, size=m).astype(np.int64)
        t1_pos = rng.integers(0, 100, size=n1).astype(np.float64)
        t1_int = rng.integers(1, 10, size=n1).astype(np.int64)
        t2_pos = rng.integers(0, 100, size=n2).astype(np.float64)
        t2_int = rng.integers(1, 10, size=n2).astype(np.int64)
        base = Distribution_1D(e_pos, e_int)
        t1 = Distribution_1D(t1_pos, t1_int)
        t2 = Distribution_1D(t2_pos, t2_int)
        cold, warm = self._cpp_pair(
            CWassersteinNetworkFactory.create_1d, base, [t1, t2], 100, 300
        )
        pts = [rng.uniform(0.2, 1.5, size=2) for _ in range(6)]
        self._drive(cold, warm, pts, 2, _capture_cpp)
