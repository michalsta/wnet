"""Stage-1 prototype for the W_p (p > 1) chain-native sweep solver.

See docs/wp_chain_design.md.  Three independently coded layers:

1. ``alignment_dp``   -- unit-granularity alignment DP.  Directly encodes the
   monotone-coupling structure (design note F2); trivially correct, O(N*M) in
   units.  Ground truth for small instances.
2. ``sweep_dp``       -- the candidate algorithm: sweep over merged sorted
   blocks; per side, the pending set of a state with k pending units is
   always the LAST k units of that side processed so far (suffix lemma:
   an optimal solution never matches a unit rightward past an unmatched
   same-side unit, so pendings are contiguous suffixes).  Both sides' value
   arrays are carried; matching consumes the OLDEST pendings first
   (monotonicity) and collapses to a windowed max via newest-rank prefix
   sums of pair profits.  Target complexity O(P * W) in blocks-in-window.
3. wnet dense factory -- the shipping reference, including quantization,
   trash bracket and independent-trash pricing (p = 2).

Kernel solved by 1 and 2: maximum-profit monotone partial matching of
empirical vs theoretical unit positions with per-pair profit
pi = tau - c(|x - y|); unmatched units are free here (their price lives in
the trash bracket outside the kernel).  For independent trash
(tau = C_exp + C_theo):  true total cost = C_exp*E + C_theo*T - profit.

Run:  python wp_sweep_prototype.py          # toys + fuzz + wnet cross-check
      python wp_sweep_prototype.py bench    # perfumes-scale timing
"""

import sys
import time

import numpy as np


# --------------------------------------------------------------------------- #
# Layer 1: unit-granularity alignment DP (ground truth)
# --------------------------------------------------------------------------- #

def alignment_dp(emp_units, theo_units, tau, cost):
    """Max-profit monotone partial matching, O(N*M) units."""
    N, M = len(emp_units), len(theo_units)
    dp = np.zeros(M + 1)
    for i in range(1, N + 1):
        prev = dp
        dp = np.empty(M + 1)
        dp[0] = prev[0]
        x = emp_units[i - 1]
        for j in range(1, M + 1):
            pi = tau - cost(abs(x - theo_units[j - 1]))
            dp[j] = max(dp[j - 1], prev[j], prev[j - 1] + pi)
    return float(dp[M]) if N else 0.0


# --------------------------------------------------------------------------- #
# Layer 2: block-granularity sweep DP (the candidate algorithm)
# --------------------------------------------------------------------------- #

class _Side:
    """One side's pending state: block deque (oldest first) + value array.

    V[k] = best total profit so far in a state with exactly k pending units
    of this side (the last k units of the deque).  V[0] is the side-neutral
    state and is kept equal across both sides by the caller.
    """

    def __init__(self):
        self.blocks = []          # (position, count), oldest first
        self.V = np.zeros(1)

    def units(self):
        return sum(c for _, c in self.blocks)

    def nonincreasing(self):
        # Free drops: a state with fewer pendings is reachable from any state
        # with more (drop the oldest), so V may be replaced by its suffix max.
        self.V = np.maximum.accumulate(self.V[::-1])[::-1]

    def prune(self, z, radius):
        """Drop pendings that can never match profitably again.

        States are suffixes anchored at the NEWEST end, so removing the
        oldest `drop` units leaves every state k <= K - drop untouched;
        states beyond that (which included pruned units) fold into the
        largest surviving state via the free-drop envelope.
        """
        if radius is None:
            return
        drop = 0
        while self.blocks and abs(z - self.blocks[0][0]) > radius:
            drop += self.blocks[0][1]
            self.blocks.pop(0)
        if drop:
            self.nonincreasing()
            keep = len(self.V) - drop  # = K_new + 1
            self.V = self.V[:max(keep, 1)].copy()

    def rank_prefix_profits(self, z, tau, cost):
        """T[r] = total profit of the NEWEST r pending units against z."""
        K = self.units()
        T = np.empty(K + 1)
        T[0] = 0.0
        r = 1
        for pos, cnt in reversed(self.blocks):
            pi = tau - cost(abs(z - pos))
            for _ in range(cnt):
                T[r] = T[r - 1] + pi
                r += 1
        return T

    def append_block(self, z, q):
        """Same-side transition: extend the suffix with up to q new units."""
        self.nonincreasing()
        K = len(self.V) - 1
        Vp = np.empty(K + q + 1)
        for k2 in range(K + q + 1):
            Vp[k2] = self.V[max(k2 - q, 0)]
        self.V = Vp
        self.blocks.append((z, q))


def sweep_dp(emp_blocks, theo_blocks, tau, cost, radius=None):
    """Max-profit monotone partial matching over position blocks."""
    events = []
    for pos, cnt in emp_blocks:
        if cnt > 0:
            events.append((pos, 0, cnt))
    for pos, cnt in theo_blocks:
        if cnt > 0:
            events.append((pos, 1, cnt))
    events.sort()

    sides = [_Side(), _Side()]

    def sync_neutral():
        best0 = max(sides[0].V[0], sides[1].V[0])
        sides[0].V[0] = best0
        sides[1].V[0] = best0

    for z, sigma, q in events:
        own, opp = sides[sigma], sides[1 - sigma]
        own.prune(z, radius)
        opp.prune(z, radius)

        # Opposite side: match the oldest j <= q of its pendings against
        # this block.  With T = newest-rank prefix profits, the oldest j of
        # a k-suffix are ranks k-j+1..k, worth T[k] - T[k-j]:
        #   V'(g) = max_{g <= k <= g+q} [ V(k) + T(k) ] - T(g)
        # Side flip: consume all opp pendings at state j (match all j),
        # leaving m = q - j fresh units of this block pending on `own`:
        #   flip(m) = V_opp(j) + T(j),  m = q - j.
        opp.nonincreasing()
        K = len(opp.V) - 1
        T = opp.rank_prefix_profits(z, tau, cost)
        VT = opp.V + T
        newVopp = np.empty(K + 1)
        for g in range(K + 1):
            hi = min(K, g + q)
            newVopp[g] = np.max(VT[g:hi + 1]) - T[g]
        flip = np.full(q + 1, -np.inf)
        for j in range(min(K, q) + 1):
            flip[q - j] = VT[j]

        # Own side: extend suffix with this block's units (append),
        # then merge the flip contributions (same state space: pendings =
        # last m units of `own`, which after append are units of this block).
        own.append_block(z, q)
        n = len(own.V)
        for m in range(min(q, n - 1) + 1):
            if flip[m] > own.V[m]:
                own.V[m] = flip[m]

        opp.V = newVopp
        sync_neutral()

    return float(max(sides[0].V.max(), sides[1].V.max()))


# --------------------------------------------------------------------------- #
# Utilities: blocks <-> units, random instances, wnet reference
# --------------------------------------------------------------------------- #

def blocks_to_units(blocks):
    out = []
    for pos, cnt in sorted(blocks):
        out.extend([pos] * cnt)
    return np.array(out)


def random_instance(rng, grid=0.25, span=80, max_blocks=10, max_cnt=5):
    def side():
        n = rng.integers(1, max_blocks + 1)
        pos = np.unique(rng.integers(0, span, n)) * grid
        return [(float(p), int(rng.integers(1, max_cnt + 1))) for p in pos]
    return side(), side()


def wnet_reference(emp_blocks, theo_blocks, c_exp, c_theo, p, max_distance):
    """True total cost from the wnet dense factory (independent trash)."""
    from wnet import WassersteinNetwork
    from wnet.distribution import Distribution_1D
    from wnet.distances import DistanceMetric

    def dist(blocks):
        pos = np.array([b[0] for b in blocks])
        cnt = np.array([float(b[1]) for b in blocks])
        return Distribution_1D(pos, cnt)

    W = WassersteinNetwork(
        dist(emp_blocks), [dist(theo_blocks)], DistanceMetric.LINF,
        max_distance, force_dense_1d=True, round_max_distance=False,
        intensity_scale=1.0, p=p,
    )
    W.set_cost_scaling(0)
    W.add_independent_asymmetric_trash(c_exp, c_theo)
    W.build()
    W.solve([1.0])
    return W.total_cost()


# --------------------------------------------------------------------------- #
# Toys and fuzz
# --------------------------------------------------------------------------- #

def run_toys():
    ok_all = True
    cost = lambda d: d * d
    tau = 1.5  # C_exp = 0.5, C_theo = 1.0; R = sqrt(1.5)
    for d, expect in [(1.0, tau - 1.0), (1.3, 0.0)]:
        got = sweep_dp([(0.0, 1)], [(d, 1)], tau, cost)
        ok = abs(got - expect) < 1e-12
        ok_all &= ok
        print(f"toy threshold d={d}: profit={got:.4f} expect={expect:.4f} "
              f"{'ok' if ok else 'FAIL'}")
    # F4 single-partner: pendings x1=0, x2=1, one theo at 1.4:
    # optimal matches x2 (d=0.4), trashes x1.
    got = sweep_dp([(0.0, 1), (1.0, 1)], [(1.4, 1)], tau, cost)
    expect = tau - 0.4 ** 2
    ok = abs(got - expect) < 1e-12
    ok_all &= ok
    print(f"toy F4 single: profit={got:.4f} expect={expect:.4f} "
          f"{'ok' if ok else 'FAIL'}")
    # F4 double: two theos; check against alignment DP.
    eb, tb = [(0.0, 1), (1.0, 1)], [(1.4, 1), (2.0, 1)]
    ref = alignment_dp(blocks_to_units(eb), blocks_to_units(tb), tau, cost)
    got = sweep_dp(eb, tb, tau, cost)
    ok = abs(got - ref) < 1e-12
    ok_all &= ok
    print(f"toy F4 double: sweep={got:.4f} alignment={ref:.4f} "
          f"{'ok' if ok else 'FAIL'}")
    return ok_all


def run_fuzz(n_trials=300, p=2.0, c_exp=0.5, c_theo=0.8125, seed=7,
             n_wnet=60):
    cost = lambda d: d ** p
    tau = c_exp + c_theo
    R = tau ** (1.0 / p)
    rng = np.random.default_rng(seed)
    worst = 0.0
    fails = 0
    for trial in range(n_trials):
        emp_b, theo_b = random_instance(rng)
        eu, tu = blocks_to_units(emp_b), blocks_to_units(theo_b)
        ref = alignment_dp(eu, tu, tau, cost)
        got = sweep_dp(emp_b, theo_b, tau, cost, radius=None)
        got_pruned = sweep_dp(emp_b, theo_b, tau, cost, radius=R)
        err = max(abs(ref - got), abs(ref - got_pruned))
        worst = max(worst, err)
        if err > 1e-9:
            fails += 1
            if fails <= 5:
                print(f"FAIL trial {trial}: alignment={ref:.6f} "
                      f"sweep={got:.6f} pruned={got_pruned:.6f}")
                print(f"  emp={emp_b}")
                print(f"  theo={theo_b}")
        if trial < n_wnet:
            E = sum(c for _, c in emp_b)
            T = sum(c for _, c in theo_b)
            true = wnet_reference(emp_b, theo_b, c_exp, c_theo, p,
                                  max_distance=2 * R)
            err2 = abs((c_exp * E + c_theo * T - ref) - true)
            if err2 > 1e-6:
                fails += 1
                print(f"FAIL trial {trial} vs wnet: kernel-total="
                      f"{c_exp * E + c_theo * T - ref:.6f} wnet={true:.6f}")
    print(f"fuzz p={p}: {n_trials} trials ({n_wnet} vs wnet), "
          f"{fails} failures, worst |err| = {worst:.2e}")
    return fails == 0


def run_bench():
    rng = np.random.default_rng(0)
    K = 12449
    grid = 9.64e-4
    pos = np.arange(K) * grid
    emp_b = [(float(p), int(c)) for p, c in zip(pos, rng.integers(1, 30, K))]
    theo_b = [(float(p), int(c)) for p, c in
              zip(pos + grid / 2, rng.integers(1, 30, K))]
    c_exp, c_theo, p = 0.0484, 0.0625, 2.0
    tau = c_exp + c_theo
    R = tau ** 0.5
    cost = lambda d: d * d
    t0 = time.time()
    profit = sweep_dp(emp_b, theo_b, tau, cost, radius=R)
    print(f"bench: K={K} R={R:.3f} window~{2 * R / grid:.0f} grid steps; "
          f"profit={profit:.3f}  [{time.time() - t0:.1f}s python]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        run_bench()
    else:
        ok = run_toys()
        ok &= run_fuzz()
        sys.exit(0 if ok else 1)
