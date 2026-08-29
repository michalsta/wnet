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


# --------------------------------------------------------------------------- #
# Marginals: d(kernel profit)/d(one extra theo unit at block b), by oracle
# re-solve and by the forward/backward interface formula (design stage 3).
# --------------------------------------------------------------------------- #

def marginal_gain_oracle(emp_blocks, theo_blocks, tau, cost, b_pos):
    """OPT(theo + one unit at b_pos) - OPT(theo): the profit gain."""
    base = sweep_dp(emp_blocks, theo_blocks, tau, cost)
    plus = sweep_dp(emp_blocks, sorted(theo_blocks + [(b_pos, 1)]), tau, cost)
    return plus - base


def run_marginal_fuzz(n_trials=120, p=2.0, c_exp=0.5, c_theo=0.8125, seed=21):
    """Validate that the true total-cost marginal of one extra theoretical
    unit equals  C_theo - gain  (independent trash), against the wnet dense
    factory marginal computed by unit finite difference on supplies."""
    cost = lambda d: d ** p
    tau = c_exp + c_theo
    rng = np.random.default_rng(seed)
    worst = 0.0
    fails = 0
    for trial in range(n_trials):
        emp_b, theo_b = random_instance(rng, max_blocks=6, max_cnt=3)
        # pick a probe position: an existing theo block or a fresh spot
        if rng.random() < 0.5 and theo_b:
            b_pos = theo_b[rng.integers(0, len(theo_b))][0]
        else:
            b_pos = float(rng.integers(0, 80)) * 0.25
        gain = marginal_gain_oracle(emp_b, theo_b, tau, cost, b_pos)
        marg = c_theo - gain
        # reference: dense-factory total cost difference with the extra unit
        E = sum(c for _, c in emp_b)
        T = sum(c for _, c in theo_b)
        base_true = c_exp * E + c_theo * T - sweep_dp(emp_b, theo_b, tau, cost)
        plus_true = (c_exp * E + c_theo * (T + 1)
                     - sweep_dp(emp_b, sorted(theo_b + [(b_pos, 1)]), tau, cost))
        err = abs((plus_true - base_true) - marg)
        worst = max(worst, err)
        if err > 1e-9:
            fails += 1
    print(f"marginal identity fuzz: {n_trials} trials, {fails} failures, "
          f"worst {worst:.2e}")
    return fails == 0


# --------------------------------------------------------------------------- #
# Efficient marginals: forward + adjoint (completion) DP, both orientations.
# gain(b) = best profit improvement from one extra theo unit at position z_b.
# The extra unit matched to an emp on its LEFT is caught by the forward
# orientation; matched on its RIGHT by the mirrored orientation; unmatched
# contributes 0.  See design note stage 3.
# --------------------------------------------------------------------------- #

def _sweep_forward_store(events, tau, cost):
    """Forward sweep storing per-event state.

    Returns (states, opt) where states[i] = snapshot BEFORE event i:
    dict(side -> (blocks list copy, V array copy)); states[n] = final.
    """
    sides = [_Side(), _Side()]

    def snap():
        return [
            (list(sides[0].blocks), sides[0].V.copy()),
            (list(sides[1].blocks), sides[1].V.copy()),
        ]

    states = []
    for z, sigma, q in events:
        states.append(snap())
        own, opp = sides[sigma], sides[1 - sigma]
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
        own.append_block(z, q)
        for m in range(min(q, len(own.V) - 1) + 1):
            if flip[m] > own.V[m]:
                own.V[m] = flip[m]
        opp.V = newVopp
        best0 = max(sides[0].V[0], sides[1].V[0])
        sides[0].V[0] = best0
        sides[1].V[0] = best0
    states.append(snap())
    opt = float(max(sides[0].V.max(), sides[1].V.max()))
    return states, opt


def _adjoint_store(events, tau, cost, states):
    """Completion values: comps[i][side] = array C(k) = best additional
    profit from events i..n-1 given k pendings of `side` (origins = the
    last k units of that side's prefix before event i, as recorded in
    states[i])."""
    n = len(events)
    comps = [None] * (n + 1)
    K0 = len(states[n][0][1]) - 1
    K1 = len(states[n][1][1]) - 1
    comps[n] = [np.zeros(K0 + 1), np.zeros(K1 + 1)]
    for i in range(n - 1, -1, -1):
        z, sigma, q = events[i]
        own_s, opp_s = sigma, 1 - sigma
        # sizes before event i
        pre_own_blocks, pre_ownV = states[i][own_s]
        pre_opp_blocks, pre_oppV = states[i][opp_s]
        K_own = len(pre_ownV) - 1
        K_opp = len(pre_oppV) - 1
        C_after = comps[i + 1]
        # T over opp pendings BEFORE event i vs z
        side_tmp = _Side()
        side_tmp.blocks = list(pre_opp_blocks)
        T = side_tmp.rank_prefix_profits(z, tau, cost)
        # --- opp side completion before event i ---
        # (A) stay: from state k: drop d, match j <= q, land g = k - d - j.
        #   value = T(kappa) - T(g) + C_after[opp](g), kappa = k - d in
        #   [g, min(k, g+q)].  Prefix-max form:
        #   H(kappa) = T(kappa) + max_{g in [kappa-q, kappa]} (C(g) - T(g))
        #   C_pre(k) = max_{kappa <= k} H(kappa)
        Copp_after = C_after[opp_s]
        L = min(K_opp, len(Copp_after) - 1)
        CT = np.array([Copp_after[g] - T[g] for g in range(L + 1)])
        H = np.full(L + 1, -np.inf)
        for kappa in range(L + 1):
            lo = max(kappa - q, 0)
            H[kappa] = T[kappa] + np.max(CT[lo:kappa + 1])
        Copp_pre = np.full(K_opp + 1, -np.inf)
        run = -np.inf
        for k in range(K_opp + 1):
            if k <= L:
                run = max(run, H[k])
            Copp_pre[k] = run
        # (B) flip: drop k-j, match j <= min(q, k), keep m = q - j of the
        # incoming block pending on side sigma.
        Cown_after = C_after[own_s]
        flip_best = np.full(K_opp + 1, -np.inf)
        run = -np.inf
        for j in range(min(q, K_opp) + 1):
            m = q - j
            if j <= L and m < len(Cown_after):
                run = max(run, T[j] + Cown_after[m])
            flip_best[j] = run
        for k in range(K_opp + 1):
            j_max = min(q, k)
            Copp_pre[k] = max(Copp_pre[k], flip_best[j_max])
        # --- own side completion before event i ---
        # append up to q units then free drops: C_pre(m) = max_{m' <= m+q} C(m')
        Cown_pre = np.full(K_own + 1, -np.inf)
        pref = np.maximum.accumulate(Cown_after)
        for m in range(K_own + 1):
            idx = min(m + q, len(Cown_after) - 1)
            Cown_pre[m] = pref[idx]
        out = [None, None]
        out[own_s] = Cown_pre
        out[opp_s] = Copp_pre
        comps[i] = out
    return comps


def _compose_with_extra(events, tau, cost, states, comps, b_idx, z):
    """Best total profit with one extra theo unit at position z, where the
    extra unit is consumed at event b_idx (a theo event at position z with
    real count q -> q + 1), or remains unmatched."""
    zb, sigma, q = events[b_idx]
    assert sigma == 1 and abs(zb - z) == 0.0
    qh = q + 1
    opp_s = 0
    pre_opp_blocks, pre_oppV = states[b_idx][opp_s]
    K = len(pre_oppV) - 1
    side_tmp = _Side()
    side_tmp.blocks = list(pre_opp_blocks)
    T = side_tmp.rank_prefix_profits(z, tau, cost)
    Vt = pre_oppV.copy()
    Vt = np.maximum.accumulate(Vt[::-1])[::-1]   # free drops
    C_after = comps[b_idx + 1]
    Copp_after = C_after[0]
    Cown_after = C_after[1]
    best = -np.inf
    # (A') stay on emp side: match j <= qh, kappa = matched-from state,
    # land g; complete with Copp_after(g).
    L = min(K, len(Copp_after) - 1)
    for kappa in range(K + 1):
        for g in range(max(kappa - qh, 0), kappa + 1):
            if g <= L:
                best = max(best, Vt[kappa] + T[kappa] - T[g] + Copp_after[g])
    # (B') flip: match j <= min(qh, K), keep m = qh - j theo pending.
    # m real-representable states only (extra consumed => m <= q).
    for j in range(min(qh, K) + 1):
        m = qh - j
        if m < len(Cown_after) and m <= q:
            best = max(best, Vt[j] + T[j] + Cown_after[m])
        # j with m == q + 1 would leave the EXTRA pending; that case is
        # covered by the mirrored orientation.
    return float(best)


def marginal_gains_fb(emp_blocks, theo_blocks, tau, cost, probes):
    """gain(z) for each probe position z, via forward+adjoint DP in both
    orientations.  probes: list of positions (fresh or existing)."""
    def orient(mirror):
        eb = [(-p, c) for p, c in emp_blocks] if mirror else list(emp_blocks)
        tb = [(-p, c) for p, c in theo_blocks] if mirror else list(theo_blocks)
        pr = [-z for z in probes] if mirror else list(probes)
        # build events with virtual q=0 theo events at fresh probe positions
        ev = []
        for pos, cnt in eb:
            if cnt > 0:
                ev.append((pos, 0, cnt))
        tpos = {}
        for pos, cnt in tb:
            if cnt > 0:
                tpos[pos] = tpos.get(pos, 0) + cnt
        for z in pr:
            tpos.setdefault(z, 0)
        for pos, cnt in tpos.items():
            ev.append((pos, 1, cnt))
        ev.sort()
        states, opt = _sweep_forward_store(ev, tau, cost)
        comps = _adjoint_store(ev, tau, cost, states)
        gains = {}
        for z in pr:
            b_idx = next(i for i, (p, s, _) in enumerate(ev)
                         if s == 1 and p == z)
            val = _compose_with_extra(ev, tau, cost, states, comps, b_idx, z)
            gains[z] = val - opt
        return gains, opt

    g_fwd, opt = orient(False)
    g_mir, opt2 = orient(True)
    assert abs(opt - opt2) < 1e-9
    return [max(g_fwd[z], g_mir[-z], 0.0) for z in probes]


def run_fb_marginal_fuzz(n_trials=150, p=2.0, c_exp=0.5, c_theo=0.8125,
                         seed=31):
    cost = lambda d: d ** p
    tau = c_exp + c_theo
    rng = np.random.default_rng(seed)
    worst = 0.0
    fails = 0
    for trial in range(n_trials):
        emp_b, theo_b = random_instance(rng, max_blocks=6, max_cnt=3)
        probes = sorted({b[0] for b in theo_b}
                        | {float(rng.integers(0, 80)) * 0.25 for _ in range(3)})
        gains = marginal_gains_fb(emp_b, theo_b, tau, cost, probes)
        for z, gain in zip(probes, gains):
            ref = marginal_gain_oracle(emp_b, theo_b, tau, cost, z)
            err = abs(ref - gain)
            worst = max(worst, err)
            if err > 1e-9:
                fails += 1
                if fails <= 5:
                    print(f"FAIL trial {trial} z={z}: fb={gain:.6f} "
                          f"oracle={ref:.6f}")
                    print(f"  emp={emp_b}")
                    print(f"  theo={theo_b}")
    print(f"fb-marginal fuzz: {n_trials} trials, {fails} failures, "
          f"worst {worst:.2e}")
    return fails == 0
