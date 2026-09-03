"""The free-threaded build must actually stay free-threaded.

On CPython 3.15t the extension is built in split mode with nanobind's
FREE_THREADED option, which declares ``Py_MOD_GIL_NOT_USED``. If that option is
ever dropped, or the build quietly falls back to a linked mode, importing the
extension re-enables the GIL and the wheel is free-threaded in name only --
nothing else in the suite would notice, because every test still passes.

The whole module is skipped unless the GIL is genuinely off *after* importing
the extension. That is the point: on 3.14t the linked fallback is expected to
turn the GIL back on, so this correctly stays quiet there rather than failing.
"""

import sys
import threading

import numpy as np
import pytest

from wnet import Distribution, Distribution_1D, WassersteinDistance
from wnet.distances import DistanceMetric

pytestmark = pytest.mark.skipif(
    not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled(),
    reason="needs a free-threaded interpreter with the GIL still disabled after import",
)

N_POINTS = 30
N_THREADS = 8
N_ROUNDS = 25


def _intensities(rng):
    # Whole numbers, and the same multiset on both sides: a solve without trash
    # edges needs the two totals to match *after* quantisation to scaled
    # integers, which balancing two float draws by a ratio does not guarantee.
    vals = rng.integers(1, 6, N_POINTS).astype(float)
    return vals, rng.permutation(vals)


def _clouds():
    # Seeded, so every call below has the same right answer.
    rng = np.random.default_rng(0)
    pos1 = rng.uniform(0.0, 10.0, (2, N_POINTS))
    pos2 = rng.uniform(0.0, 10.0, (2, N_POINTS))
    int1, int2 = _intensities(rng)
    return (pos1, int1), (pos2, int2)


def _chain():
    rng = np.random.default_rng(1)
    pos1 = np.sort(rng.uniform(0.0, 10.0, N_POINTS))
    pos2 = np.sort(rng.uniform(0.0, 10.0, N_POINTS))
    int1, int2 = _intensities(rng)
    return (pos1, int1), (pos2, int2)


def _solve_once(clouds, chain):
    # Fresh distributions and a fresh network per call: the claim under test is
    # that the module holds no *global* state, not that one network may be
    # driven from two threads at once.
    (p1, i1), (p2, i2) = clouds
    dense = WassersteinDistance(
        Distribution(p1.copy(), i1.copy()),
        Distribution(p2.copy(), i2.copy()),
        DistanceMetric.L2,
    )
    (q1, j1), (q2, j2) = chain
    # The 1-D path is a different solver (the slope DP), so exercise it too.
    line = WassersteinDistance(
        Distribution_1D(q1.copy(), j1.copy()),
        Distribution_1D(q2.copy(), j2.copy()),
        DistanceMetric.L1,
    )
    return dense, line


def test_gil_stays_disabled_after_import():
    assert not sys._is_gil_enabled()


def test_concurrent_distances_agree_with_serial():
    clouds, chain = _clouds(), _chain()
    expected = _solve_once(clouds, chain)

    results = []
    errors = []

    def worker():
        try:
            for _ in range(N_ROUNDS):
                results.append(_solve_once(clouds, chain))
        except BaseException as exc:  # noqa: BLE001 - re-raised in the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:3]
    assert len(results) == N_THREADS * N_ROUNDS
    assert all(r == expected for r in results)
    # If the GIL had been re-enabled behind our back, the run above proves nothing.
    assert not sys._is_gil_enabled()
