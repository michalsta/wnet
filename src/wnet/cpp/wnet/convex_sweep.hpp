#ifndef WNET_CONVEX_SWEEP_HPP
#define WNET_CONVEX_SWEEP_HPP

// Chain-native exact solver for W_p^p with p > 1 (any strictly convex cost):
// maximum-profit monotone partial matching on a line, swept over merged
// sorted position blocks.  See docs/wp_chain_design.md and
// prototypes/wp_sweep_prototype.py (this is a C++ port of the validated
// prototype).
//
// Kernel: per-pair profit pi(a, b) = tau - c(|x_a - x_b|) in integer
// (quantized) units; unmatched units are free here (their price is the
// trash bracket handled by the caller).  DP state (side, k): k pending
// units of one side; the pending set is always the suffix (most recent
// units) of that side — an optimal solution never matches a unit rightward
// past an unmatched same-side unit.  Matching consumes the OLDEST pendings
// (monotonicity); with T(k) = newest-rank prefix profit sums the transition
// telescopes into windowed maxima.
//
// Value functions over k are piecewise linear with O(blocks-in-window)
// vertices; concavity does NOT hold (refuted in the prototype), so this is
// a general PL engine.  Exactness discipline: every evaluation at integer k
// is exact because any segment with non-integer slope spans only adjacent
// integers (crossing insertions always add the two integers bracketing the
// crossing), and normalize() only merges exactly-collinear vertices.

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <vector>

namespace convex_sweep {

using i64 = int64_t;

#if defined(__SIZEOF_INT128__)
using i128 = __int128;
#else
#error "convex_sweep requires __int128"
#endif

constexpr i64 NEG_INF = std::numeric_limits<i64>::min() / 4;

// ---------------------------------------------------------------------- //
// Piecewise-linear function on integer domain [0, K].
// ---------------------------------------------------------------------- //
struct PLF {
    struct V { i64 k; i64 v; };
    std::vector<V> vs;  // k strictly increasing; front().k == 0

    static PLF constant(i64 K, i64 value) {
        PLF f;
        f.vs.push_back({0, value});
        if (K > 0) f.vs.push_back({K, value});
        return f;
    }

    i64 K() const { return vs.back().k; }

    i64 eval(i64 k) const {
        if (k <= vs.front().k) return vs.front().v;
        if (k >= vs.back().k) return vs.back().v;
        size_t lo = 0, hi = vs.size() - 1;
        while (hi - lo > 1) {
            size_t mid = (lo + hi) / 2;
            if (vs[mid].k <= k) lo = mid; else hi = mid;
        }
        const V& a = vs[lo];
        const V& b = vs[hi];
        i128 num = (i128)(b.v - a.v) * (k - a.k);
        return a.v + (i64)(num / (b.k - a.k));
    }

    void normalize() {
        std::vector<V> out;
        out.reserve(vs.size());
        for (const V& x : vs) {
            if (!out.empty() && out.back().k == x.k) {
                if (x.v > out.back().v) out.back().v = x.v;
                continue;
            }
            while (out.size() >= 2) {
                const V& a = out[out.size() - 2];
                const V& b = out.back();
                i128 lhs = (i128)(b.v - a.v) * (x.k - b.k);
                i128 rhs = (i128)(x.v - b.v) * (b.k - a.k);
                if (lhs == rhs) out.pop_back(); else break;
            }
            out.push_back(x);
        }
        vs = std::move(out);
    }

    // f(k) := max_{k' >= k} f(k')  — free drops of the oldest pendings.
    void suffix_max() {
        std::vector<V> out;
        i64 run = NEG_INF;  // suffix max over processed (right) part
        for (size_t i = vs.size(); i-- > 0;) {
            const V b = vs[i];
            if (i + 1 < vs.size()) {
                const V a = vs[i + 1];  // right neighbour
                if (b.v > run && a.v < run) {
                    // decreasing-to-the-right segment crosses the plateau:
                    // find last integer with line(k) > run, insert pair.
                    i64 lo = b.k, hi = a.k;
                    while (hi - lo > 1) {
                        i64 mid = lo + (hi - lo) / 2;
                        i128 num = (i128)(a.v - b.v) * (mid - b.k);
                        i64 lv = b.v + (i64)(num / (a.k - b.k));
                        if (lv > run) lo = mid; else hi = mid;
                    }
                    out.push_back({hi, run});
                    if (lo != b.k) {
                        i128 num = (i128)(a.v - b.v) * (lo - b.k);
                        out.push_back({lo, b.v + (i64)(num / (a.k - b.k))});
                    }
                }
            }
            run = std::max(run, b.v);
            out.push_back({b.k, run});
        }
        std::reverse(out.begin(), out.end());
        vs = std::move(out);
        normalize();
    }

    // f'(k') = f(clamp(k' - q, 0, K)); domain grows to K + q.  Apply after
    // suffix_max() when free drops are intended.
    void shift_append(i64 q) {
        if (q <= 0) return;
        std::vector<V> out;
        out.reserve(vs.size() + 2);
        out.push_back({0, vs.front().v});
        out.push_back({q, vs.front().v});
        for (const V& x : vs)
            if (x.k > 0) out.push_back({x.k + q, x.v});
        vs = std::move(out);
        normalize();
    }

    // Restrict the domain to [0, K_new].
    void truncate(i64 K_new) {
        if (K_new >= K()) return;
        i64 vK = eval(K_new);
        std::vector<V> out;
        for (const V& x : vs) {
            if (x.k < K_new) out.push_back(x);
            else break;
        }
        out.push_back({K_new, vK});
        if (out.front().k != 0) out.insert(out.begin(), {0, vs.front().v});
        vs = std::move(out);
        normalize();
    }

    // In-place pointwise max with `other` (domains may differ; outside its
    // domain a function does not contribute).
    void pointwise_max(const PLF& other) {
        std::vector<i64> ks;
        ks.reserve(vs.size() + other.vs.size());
        for (auto& x : vs) ks.push_back(x.k);
        for (auto& x : other.vs) ks.push_back(x.k);
        std::sort(ks.begin(), ks.end());
        ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
        i64 Ka = K(), Kb = other.K();
        auto fval = [&](i64 k) { return k <= Ka ? eval(k) : NEG_INF; };
        auto gval = [&](i64 k) { return k <= Kb ? other.eval(k) : NEG_INF; };
        std::vector<V> out;
        for (size_t i = 0; i < ks.size(); ++i) {
            i64 k = ks[i];
            out.push_back({k, std::max(fval(k), gval(k))});
            if (i + 1 < ks.size() && ks[i + 1] - k > 1) {
                i64 k2 = ks[i + 1];
                i64 d1 = fval(k) - gval(k), d2 = fval(k2) - gval(k2);
                if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) {
                    i64 lo = k, hi = k2;
                    while (hi - lo > 1) {
                        i64 mid = lo + (hi - lo) / 2;
                        i64 dm = fval(mid) - gval(mid);
                        if ((d1 > 0) ? (dm > 0) : (dm < 0)) lo = mid;
                        else hi = mid;
                    }
                    out.push_back({lo, std::max(fval(lo), gval(lo))});
                    out.push_back({hi, std::max(fval(hi), gval(hi))});
                }
            }
        }
        std::sort(out.begin(), out.end(),
                  [](const V& a, const V& b) { return a.k < b.k; });
        vs = std::move(out);
        normalize();
    }

    i64 max_value() const {
        i64 m = NEG_INF;
        for (const V& x : vs) m = std::max(m, x.v);
        return m;
    }
};

// Monotone O(1)-amortized evaluator: valid only for non-decreasing query ks.
struct Walker {
    const PLF& f;
    size_t seg = 0;
    explicit Walker(const PLF& f) : f(f) {}
    i64 eval(i64 k) {
        const auto& vs = f.vs;
        if (k <= vs.front().k) return vs.front().v;
        if (k >= vs.back().k) return vs.back().v;
        while (vs[seg + 1].k <= k) ++seg;
        const PLF::V& a = vs[seg];
        const PLF::V& b = vs[seg + 1];
        if (k == a.k) return a.v;
        i128 num = (i128)(b.v - a.v) * (k - a.k);
        return a.v + (i64)(num / (b.k - a.k));
    }
};

template <bool SUB>
inline PLF addsub_pl(const PLF& a, const PLF& b) {
    PLF out;
    out.vs.reserve(a.vs.size() + b.vs.size());
    Walker wa(a), wb(b);
    size_t ia = 0, ib = 0;
    i64 prev = std::numeric_limits<i64>::min();
    while (ia < a.vs.size() || ib < b.vs.size()) {
        i64 k;
        if (ib >= b.vs.size()) k = a.vs[ia++].k;
        else if (ia >= a.vs.size()) k = b.vs[ib++].k;
        else if (a.vs[ia].k < b.vs[ib].k) k = a.vs[ia++].k;
        else if (b.vs[ib].k < a.vs[ia].k) k = b.vs[ib++].k;
        else { k = a.vs[ia].k; ++ia; ++ib; }
        if (k == prev) continue;
        prev = k;
        i64 va = wa.eval(k), vb = wb.eval(k);
        out.vs.push_back({k, SUB ? va - vb : va + vb});
    }
    out.normalize();
    return out;
}

inline PLF add_pl(const PLF& a, const PLF& b) { return addsub_pl<false>(a, b); }
inline PLF sub_pl(const PLF& a, const PLF& b) { return addsub_pl<true>(a, b); }

// result(g) = max_{g <= k <= min(K, g+q)} f(k), domain [0, K].
// Monotone-deque sliding max over candidate breakpoints (vertices and
// their -q images), with adjacent-integer inserts where the window-edge
// lines cross between candidates.  O(n log n).
inline PLF windowed_max(const PLF& f, i64 q) {
    const i64 K = f.K();
    std::vector<i64> ks;
    ks.reserve(2 * f.vs.size() + 2);
    for (auto& x : f.vs) {
        ks.push_back(x.k);
        if (x.k - q >= 0) ks.push_back(x.k - q);
    }
    ks.push_back(0);
    ks.push_back(K);
    std::sort(ks.begin(), ks.end());
    ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
    while (!ks.empty() && ks.back() > K) ks.pop_back();

    // Sliding max of interior vertex values over windows (g, g+q):
    // process candidates in ascending g with a monotone deque of vertex
    // indices whose k lies in the open window.  Edge evaluations use
    // monotone walkers (g and g+q are both ascending).
    std::deque<size_t> dq;      // vertex indices, values decreasing
    size_t next_v = 0;          // next vertex to admit
    Walker w_lo(f), w_hi(f);
    auto win_val = [&](i64 g) -> i64 {
        i64 hi = std::min(K, g + q);
        while (next_v < f.vs.size() && f.vs[next_v].k < hi) {
            // admit only vertices strictly inside (g, hi); ones at <= g are
            // evicted below
            while (!dq.empty() && f.vs[dq.back()].v <= f.vs[next_v].v)
                dq.pop_back();
            dq.push_back(next_v);
            ++next_v;
        }
        while (!dq.empty() && f.vs[dq.front()].k <= g) dq.pop_front();
        i64 m = std::max(w_lo.eval(g), w_hi.eval(hi));
        if (!dq.empty()) m = std::max(m, f.vs[dq.front()].v);
        return m;
    };

    PLF out;
    out.vs.reserve(ks.size() * 2);
    std::vector<PLF::V> extra;
    Walker w_c1(f), w_c2(f), w_c3(f), w_c4(f);
    i64 prev_g = -1, prev_val = 0;
    for (i64 g : ks) {
        i64 val = win_val(g);
        if (prev_g >= 0 && g - prev_g > 1) {
            // Between candidates the result is the max of the two window
            // edge lines (interior vertex max is constant between
            // candidates); insert the integer pair around a crossing.
            i64 dl = 0, dr = 0;
            dl = w_c1.eval(prev_g) - w_c2.eval(std::min(K, prev_g + q));
            dr = w_c3.eval(g) - w_c4.eval(std::min(K, g + q));
            if ((dl > 0 && dr < 0) || (dl < 0 && dr > 0)) {
                i64 lo = prev_g, hi2 = g;
                while (hi2 - lo > 1) {
                    i64 mid = lo + (hi2 - lo) / 2;
                    i64 dm = f.eval(mid) - f.eval(std::min(K, mid + q));
                    if ((dl > 0) ? (dm > 0) : (dm < 0)) lo = mid;
                    else hi2 = mid;
                }
                // evaluate exactly with a fresh scan (rare path)
                auto exact_at = [&](i64 gg) -> i64 {
                    i64 hh = std::min(K, gg + q);
                    i64 m = std::max(f.eval(gg), f.eval(hh));
                    for (const auto& x : f.vs) {
                        if (x.k > gg && x.k < hh) m = std::max(m, x.v);
                        if (x.k >= hh) break;
                    }
                    return m;
                };
                extra.push_back({lo, exact_at(lo)});
                extra.push_back({hi2, exact_at(hi2)});
            }
        }
        out.vs.push_back({g, val});
        prev_g = g;
        prev_val = val;
    }
    (void)prev_val;
    for (const auto& x : extra) out.vs.push_back(x);
    std::sort(out.vs.begin(), out.vs.end(),
              [](const PLF::V& a, const PLF::V& b) { return a.k < b.k; });
    out.normalize();
    return out;
}

// ---------------------------------------------------------------------- //
// Sweep kernel.
// ---------------------------------------------------------------------- //

struct Event {
    double pos;
    int side;    // 0 = empirical, 1 = theoretical
    i64 count;   // integer units
};

struct SideState {
    std::deque<std::pair<double, i64>> blocks;  // oldest first (pos, count)
    PLF V = PLF::constant(0, 0);
};

// T(k): newest-rank prefix profit sums of st's pendings against z, as a
// PL function of k (vertices at cumulative block counts; slopes = per-unit
// profits, integer).  ProfitFn: (pending_pos, z) -> i64.
template <typename ProfitFn>
PLF rank_prefix_profits(const SideState& st, double z, ProfitFn profit) {
    PLF T;
    T.vs.push_back({0, 0});
    i64 k = 0, acc = 0;
    for (size_t i = st.blocks.size(); i-- > 0;) {
        const auto& [pos, cnt] = st.blocks[i];
        i64 pi = profit(pos, z);
        acc += pi * cnt;
        k += cnt;
        T.vs.push_back({k, acc});
    }
    T.normalize();
    return T;
}

template <typename ProfitFn>
void sweep_step(SideState& own, SideState& opp, double z, i64 q,
                ProfitFn profit, double radius) {
    auto prune = [&](SideState& st) {
        i64 drop = 0;
        while (!st.blocks.empty() && (z - st.blocks.front().first) > radius) {
            drop += st.blocks.front().second;
            st.blocks.pop_front();
        }
        if (drop) {
            st.V.suffix_max();
            st.V.truncate(std::max<i64>(st.V.K() - drop, 0));
        }
    };
    prune(own);
    prune(opp);

    // Opposite side: windowed matching.
    opp.V.suffix_max();
    PLF T = rank_prefix_profits(opp, z, profit);
    PLF VT = add_pl(opp.V, T);
    PLF newVopp = sub_pl(windowed_max(VT, q), T);

    // Flip: match exactly j pendings (all remaining), keep m = q - j fresh
    // units pending on own: flip(m) = VT(j), m = q - j, 0 <= j <= min(K,q).
    // flip is VT reflected: vertex (j, v) -> (q - j, v).
    const i64 jmax = std::min(opp.V.K(), q);
    PLF flip;
    {
        // flip(m) = VT(q - m) for m in [q - jmax, q]; unreachable below
        // (j > jmax would need more pendings than exist): NEG_INF with the
        // adjacent-integer drop so interpolation never manufactures values.
        std::vector<PLF::V> rv;
        for (auto it = VT.vs.rbegin(); it != VT.vs.rend(); ++it)
            if (it->k <= jmax) rv.push_back({q - it->k, it->v});
        if (rv.empty() || rv.front().k != q - jmax)
            rv.insert(rv.begin(), {q - jmax, VT.eval(jmax)});
        if (rv.front().k > 0) {
            i64 mmin = rv.front().k;
            if (mmin - 1 >= 0)
                rv.insert(rv.begin(), {mmin - 1, NEG_INF});
            if (rv.front().k > 0)
                rv.insert(rv.begin(), {0, NEG_INF});
        }
        flip.vs = std::move(rv);
        flip.normalize();
    }

    // Own side: append this block, then merge flip contributions.
    own.V.suffix_max();
    own.V.shift_append(q);
    own.blocks.push_back({z, q});
    {
        // Extend flip to own's domain with an adjacent-integer drop to
        // NEG_INF so interpolation never manufactures values.
        PLF fe = flip;
        i64 Ko = own.V.K();
        if (fe.K() < Ko) {
            if (fe.K() + 1 <= Ko) fe.vs.push_back({fe.K() + 1, NEG_INF});
            if (fe.K() < Ko) fe.vs.push_back({Ko, NEG_INF});
            fe.normalize();
        }
        own.V.pointwise_max(fe);
    }

    opp.V = newVopp;

    // Neutral-state sync: V_own(0) and V_opp(0) are the same logical state.
    i64 best0 = std::max(own.V.eval(0), opp.V.eval(0));
    if (own.V.vs.front().v < best0) own.V.vs.front().v = best0;
    if (opp.V.vs.front().v < best0) opp.V.vs.front().v = best0;
}

template <typename ProfitFn>
i64 sweep_solve(const std::vector<Event>& events, ProfitFn profit,
                double radius) {
    SideState sides[2];
    for (const Event& e : events) {
        if (e.count <= 0) continue;
        sweep_step(sides[e.side], sides[1 - e.side], e.pos, e.count,
                   profit, radius);
    }
    i64 best = std::max(sides[0].V.max_value(), sides[1].V.max_value());
    return std::max<i64>(best, 0);
}

// ---------------------------------------------------------------------- //
// Marginals: forward pass with snapshots + adjoint (completion) pass.
// gain(z) = best profit improvement from ONE extra theoretical unit at
// position z, with the extra unit matched to an empirical unit on its LEFT
// (this orientation) or unmatched.  The caller runs both orientations
// (mirror positions) and takes max(gain_fwd, gain_mir, 0).
// Validated against oracle re-solves in prototypes/wp_sweep_prototype.py.
// ---------------------------------------------------------------------- //

inline PLF reflect_pl(const PLF& f) {
    // r(k) = f(K - k)
    PLF r;
    i64 K = f.K();
    r.vs.reserve(f.vs.size());
    for (auto it = f.vs.rbegin(); it != f.vs.rend(); ++it)
        r.vs.push_back({K - it->k, it->v});
    r.normalize();
    return r;
}

inline PLF prefix_max_pl(const PLF& f) {
    // g(k) = max_{k' <= k} f(k')
    PLF r = reflect_pl(f);
    r.suffix_max();
    return reflect_pl(r);
}

// g(k) = max_{max(0, k-q) <= k' <= k} f(k')
inline PLF windowed_max_left(const PLF& f, i64 q) {
    return reflect_pl(windowed_max(reflect_pl(f), q));
}

// g(m) = f(min(m + q, K)), domain [0, K]
inline PLF shift_left_clamp(const PLF& f, i64 q) {
    i64 K = f.K();
    PLF out;
    for (const auto& x : f.vs) {
        i64 m = x.k - q;
        if (m < 0) continue;
        out.vs.push_back({m, x.v});
    }
    if (out.vs.empty() || out.vs.front().k != 0)
        out.vs.insert(out.vs.begin(), {0, f.eval(std::min(q, K))});
    if (out.vs.back().k < K)
        out.vs.push_back({K, f.eval(K)});
    out.normalize();
    return out;
}

// Extend a PLF's domain to K_new by repeating its last value.
inline PLF extend_flat(const PLF& f, i64 K_new) {
    PLF out = f;
    if (K_new > out.K()) out.vs.push_back({K_new, out.vs.back().v});
    out.normalize();
    return out;
}

struct Snapshot {
    // Post-prune, pre-transition state at each event (index i), plus final.
    SideState sides[2];
};

template <typename ProfitFn>
i64 forward_store(const std::vector<Event>& events, ProfitFn profit,
                  double radius, std::vector<Snapshot>& snaps) {
    SideState sides[2];
    snaps.clear();
    snaps.reserve(events.size() + 1);
    auto prune_side = [&](SideState& st, double z) {
        i64 drop = 0;
        while (!st.blocks.empty() && (z - st.blocks.front().first) > radius) {
            drop += st.blocks.front().second;
            st.blocks.pop_front();
        }
        if (drop) {
            st.V.suffix_max();
            st.V.truncate(std::max<i64>(st.V.K() - drop, 0));
        }
    };
    for (const Event& e : events) {
        prune_side(sides[0], e.pos);
        prune_side(sides[1], e.pos);
        snaps.push_back({{sides[0], sides[1]}});
        // transition (sweep_step body minus the prune it normally does; the
        // radius below never triggers because we just pruned at this z)
        sweep_step(sides[e.side], sides[1 - e.side], e.pos, e.count, profit,
                   std::numeric_limits<double>::max());
    }
    snaps.push_back({{sides[0], sides[1]}});
    i64 best = std::max(sides[0].V.max_value(), sides[1].V.max_value());
    return std::max<i64>(best, 0);
}

// Backward completion pass; composes the extra-unit value at every
// theoretical event and records it into `composed` (same length as events,
// NEG_INF elsewhere).  C[side] entering iteration i is the completion
// function on snapshot(i+1) domains; the loop rewinds it to snapshot(i).
template <typename ProfitFn>
void adjoint_compose(const std::vector<Event>& events, ProfitFn profit,
                     const std::vector<Snapshot>& snaps,
                     std::vector<i64>& composed) {
    const size_t n = events.size();
    composed.assign(n, NEG_INF);
    PLF C[2] = {PLF::constant(snaps[n].sides[0].V.K(), 0),
                PLF::constant(snaps[n].sides[1].V.K(), 0)};
    for (size_t i = n; i-- > 0;) {
        const Event& e = events[i];
        const int own_s = e.side, opp_s = 1 - e.side;
        const i64 q = e.count;
        const Snapshot& pre = snaps[i];
        const Snapshot& post = snaps[i + 1];
        // Prune adjoint between post-transition of event i and snapshot(i+1):
        // snapshot(i+1) was taken after the NEXT event's prune, so extend C
        // flat to the post-transition domains.
        const i64 K_own_post = pre.sides[own_s].V.K() + q;
        const i64 K_opp_post = pre.sides[opp_s].V.K();
        PLF C_own_after = extend_flat(C[own_s], K_own_post);
        PLF C_opp_after = extend_flat(C[opp_s], K_opp_post);

        // T over the OPP side's pendings before event i (post-prune).
        PLF T = rank_prefix_profits(pre.sides[opp_s], e.pos, profit);

        // ---- composition: one extra unit of side `1` (theoretical) at this
        // event, consumed here (matched to an opp pending or kept among the
        // fresh block's pendings up to q real slots).
        if (e.side == 1) {
            const i64 qh = q + 1;
            PLF Vt = pre.sides[opp_s].V;
            Vt.suffix_max();
            PLF VT = add_pl(Vt, T);
            // (A') stay on opp side: max_g [win_max_{[g,g+qh]}(VT) - T + C_opp]
            PLF X = windowed_max(VT, qh);
            PLF comp = add_pl(sub_pl(X, T), C_opp_after);
            i64 best = comp.max_value();
            // (B') flip: j matched in [1..min(qh, K)], keep m = qh - j <= q
            // (m <= q means the EXTRA unit was consumed, not left pending):
            // value = VT(j) + C_own_after(qh - j).  Vertex-based max.
            {
                i64 j_hi = std::min<i64>(qh, VT.K());
                i64 j_lo = std::max<i64>(1, qh - std::min<i64>(q, C_own_after.K()));
                if (j_lo <= j_hi) {
                    std::vector<i64> js;
                    for (auto& x : VT.vs)
                        if (x.k >= j_lo && x.k <= j_hi) js.push_back(x.k);
                    for (auto& x : C_own_after.vs) {
                        i64 j = qh - x.k;
                        if (j >= j_lo && j <= j_hi) js.push_back(j);
                    }
                    js.push_back(j_lo);
                    js.push_back(j_hi);
                    for (i64 j : js) {
                        i64 v = VT.eval(j) + C_own_after.eval(qh - j);
                        if (v > best) best = v;
                    }
                }
            }
            composed[i] = best;
        }

        // ---- adjoint transition: C over snapshot(i) domains.
        // Opp side (was matched against):
        //   stay: C_pre(k) = max_{kappa <= k} [ T(kappa)
        //          + max_{g in [kappa-q, kappa]} (C_opp_after(g) - T(g)) ]
        //   flip: C_pre(k) >= max_{j <= min(q,k)} [ T(j) + C_own_after(q-j) ]
        PLF CT = sub_pl(C_opp_after, T);
        PLF H = add_pl(T, windowed_max_left(CT, q));
        PLF C_opp_pre = prefix_max_pl(H);
        {
            // flip term as PLF over k: C_pre(k) >= max_{j <= min(q, k)} D(j),
            // D(j) = T(j) + C_own_after(q - j), reachable j in [j_lo, j_hi].
            i64 j_hi = std::min<i64>(q, T.K());
            i64 j_lo = std::max<i64>(0, q - C_own_after.K());
            if (j_lo <= j_hi) {
                // D on full domain [0, j_hi], NEG_INF below j_lo with the
                // adjacent-integer drop so interpolation stays honest.
                std::vector<i64> ks;
                for (auto& x : T.vs)
                    if (x.k >= j_lo && x.k <= j_hi) ks.push_back(x.k);
                for (auto& x : C_own_after.vs) {
                    i64 j = q - x.k;
                    if (j >= j_lo && j <= j_hi) ks.push_back(j);
                }
                ks.push_back(j_lo);
                ks.push_back(j_hi);
                std::sort(ks.begin(), ks.end());
                ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
                PLF D;
                if (j_lo > 0) {
                    D.vs.push_back({0, NEG_INF});
                    if (j_lo > 1) D.vs.push_back({j_lo - 1, NEG_INF});
                }
                for (i64 j : ks)
                    D.vs.push_back({j, T.eval(j) + C_own_after.eval(q - j)});
                D.normalize();
                // flipk(k) = PM(min(k, j_hi)) = flat extension of prefix max
                PLF flipk = extend_flat(prefix_max_pl(D), C_opp_pre.K());
                C_opp_pre.pointwise_max(flipk);
            }
        }
        C_opp_pre.truncate(pre.sides[opp_s].V.K());
        // Own side (the block was appended): C_pre(m) = max_{m' <= m+q} C(m')
        PLF C_own_pre = shift_left_clamp(prefix_max_pl(C_own_after), q);
        C_own_pre.truncate(pre.sides[own_s].V.K());

        C[own_s] = std::move(C_own_pre);
        C[opp_s] = std::move(C_opp_pre);
    }
}

}  // namespace convex_sweep

#endif  // WNET_CONVEX_SWEEP_HPP
