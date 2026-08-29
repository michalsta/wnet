#ifndef WNET_DECOMPOSITABLE_GRAPH_HPP
#define WNET_DECOMPOSITABLE_GRAPH_HPP

#include <vector>
#include <span>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <optional>
#include <deque>
#include <queue>
#include <variant>
#include <type_traits>


#define LEMON_ONLY_TEMPLATES
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>
#include <lemon/cycle_canceling.h>
#include <lemon/cost_scaling.h>
#include <lemon/capacity_scaling.h>
#include <pylmcf/network_simplex_lct_adapter.h>

// Pivot rule for NetworkSimplex. Values match lemon::NetworkSimplex::PivotRule.
enum class NSPivotRule {
    FIRST_ELIGIBLE, BEST_ELIGIBLE, BLOCK_SEARCH, CANDIDATE_LIST, ALTERING_LIST
};

// Internal method for CostScaling. Values match lemon::CostScaling::Method.
enum class CSMethod { PUSH, AUGMENT, PARTIAL_AUGMENT };

// Internal method for CycleCanceling. Values match lemon::CycleCanceling::Method.
enum class CCMethod {
    SIMPLE_CYCLE_CANCELING, MINIMUM_MEAN_CYCLE_CANCELING, CANCEL_AND_TIGHTEN
};

// Warm-restart strategy for NetworkSimplex across successive solves:
//   None       - always cold init() (no basis reuse)
//   Simple     - reuse the basis via repairTreeFlows(); cold-fallback if it fails
//   Dual       - Simple, plus a bounded dual-simplex repair before cold-fallback
//   Primal     - Simple, plus a bounded primal-pivot repair before cold-fallback
//   DualRatio  - Dual with bound-flipping long step; fewer cold-fallbacks on
//                large/hard graphs; default choice across most dataset families
//   DualGreedy - Like DualRatio but enters the max-capacity arc (not min-|rc|);
//                may fix violations in fewer pivots when capacities vary widely
//   LinkCut    - EXPERIMENTAL alternative backend: pylmcf::NetworkSimplexLCT
//                (link-cut-tree spanning tree, O(log n) pivot tree updates).
//                Currently Simple-strategy warm only (repair-or-cold); selects
//                a different solver implementation, not a repair strategy.
enum class NSWarmMode { None, Simple, Dual, Primal, DualRatio, DualGreedy, LinkCut };

struct NetworkSimplexConfig {
    NSPivotRule pivot = NSPivotRule::BLOCK_SEARCH;
    NSWarmMode warm = NSWarmMode::DualRatio;
    // Warm/cold repair policy, forwarded to
    // NetworkSimplex::setWarmViolationLimit() before each warmRun():
    //   -2 (default) — auto/unset: same as -1 unless a caller that knows the
    //       problem structure resolves it (wnetdeconv sets 0 for shared-grid
    //       profile data);
    //   -1 — always attempt repair;
    //    0 — never repair (any violation after the cheap tree patch goes
    //       straight to a cold start; measured 2-5x faster on shared-grid
    //       profile chains);
    //   >0 — max violated basic arcs before skipping repair (see the
    //       trajectory-instability note in network_simplex.h).
    long warm_violation_limit = -2;
};
struct CostScalingConfig {
    CSMethod method = CSMethod::PARTIAL_AUGMENT;
    int factor = 16;
};
struct CycleCancelingConfig {
    CCMethod method = CCMethod::CANCEL_AND_TIGHTEN;
};
struct CapacityScalingConfig {
    int factor = 4;
};
// Chain-native exact solver: one-pass convex slope DP over the 1-D chain
// (O(K log K) per solve, cold every solve).  Only valid for chain-factory
// subgraphs (no MatchingEdges); bit-exact against NetworkSimplex.  Measured
// ~300x faster than a cold NS solve on dense profile chains (pinene 70k:
// 21 ms vs 6.2 s merged / ~26 s on the full wnet graph).
struct SlopeDPConfig {};

using SolverConfig = std::variant<
    NetworkSimplexConfig, CostScalingConfig,
    CycleCancelingConfig, CapacityScalingConfig, SlopeDPConfig>;

// Convex piecewise-linear function support for the slope-DP chain solver.
// F(s): slope(s) = sum(right masses at keys < s) - sum(left masses at keys > s);
// invariant max(left keys) <= min(right keys); F = minval on the plateau.
namespace slopedp_detail {

// 128-bit signed accumulator for slope-DP cost sums.  Intermediate products
// (mass x distance, with wall masses near 2^53) overflow int64 even though
// every FINAL total fits VALUE_TYPE, so the accumulator must be 128-bit.
// GCC/Clang provide __int128; MSVC does not, so a minimal two's-complement
// fallback covers exactly the operations the DP uses: construction from
// int64, (wide)a * b products of two int64s, +=, +, -, unary -, ==/!=, and
// an explicit narrowing cast.  No ordering, division or shifts.
// WNET_FORCE_SOFT_INT128 forces the fallback for testing it on GCC/Clang.
#if defined(__SIZEOF_INT128__) && !defined(WNET_FORCE_SOFT_INT128)
using wide_t = __int128;
#else
struct wide_t {
    uint64_t lo = 0;
    int64_t  hi = 0;

    constexpr wide_t() = default;
    constexpr wide_t(long long v)                    // implicit: `W x = 0`
        : lo(static_cast<uint64_t>(v)), hi(v < 0 ? -1 : 0) {}

    constexpr wide_t& operator+=(const wide_t& b) {
        lo += b.lo;
        hi += b.hi + (lo < b.lo ? 1 : 0);
        return *this;
    }
    friend constexpr wide_t operator+(wide_t a, const wide_t& b) { return a += b; }
    friend constexpr wide_t operator-(const wide_t& a) {
        wide_t r;
        r.lo = ~a.lo;
        r.hi = ~a.hi;
        r.lo += 1;
        if (r.lo == 0) r.hi += 1;
        return r;
    }
    friend constexpr wide_t operator-(const wide_t& a, const wide_t& b) {
        return a + (-b);
    }
    friend constexpr bool operator==(const wide_t& a, const wide_t& b) {
        return a.lo == b.lo && a.hi == b.hi;
    }
    friend constexpr bool operator!=(const wide_t& a, const wide_t& b) {
        return !(a == b);
    }
    // (wide)a * b with |*this| known to fit in int64 — always true at the
    // DP's call sites, where the left factor is a freshly promoted int64.
    constexpr wide_t operator*(long long b) const {
        const auto a = static_cast<long long>(lo);   // value fits: hi is sign fill
        const bool neg = (a < 0) != (b < 0);
        const uint64_t ua = a < 0 ? 0 - static_cast<uint64_t>(a) : static_cast<uint64_t>(a);
        const uint64_t ub = b < 0 ? 0 - static_cast<uint64_t>(b) : static_cast<uint64_t>(b);
        // schoolbook 64x64 -> 128 on 32-bit limbs
        const uint64_t a0 = ua & 0xffffffffu, a1 = ua >> 32;
        const uint64_t b0 = ub & 0xffffffffu, b1 = ub >> 32;
        const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
        const uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffu) + (p10 & 0xffffffffu);
        wide_t r;
        r.lo = (p00 & 0xffffffffu) | (mid << 32);
        r.hi = static_cast<int64_t>(p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32));
        return neg ? -r : r;
    }
    template <typename T>
    explicit constexpr operator T() const {          // narrowing, like (V)__int128
        return static_cast<T>(static_cast<long long>(lo));
    }
};
#endif

template <typename V>
struct ConvexPL {
    using W = wide_t;
    static constexpr V NEG_BIG = std::numeric_limits<V>::min() / 4;
    static constexpr V POS_BIG = std::numeric_limits<V>::max() / 4;
    // Flat storage (measured ~2.4x faster than std::map here): L inserts land
    // at the back (post-pop-loop or in-plateau), R inserts are found by
    // binary search (the plateau drifts with the lazy offsets, so R's insert
    // point wanders a few dozen entries deep); both structures stay small
    // (~1e3-1e4 entries on 2e5-node chains).
    struct BP { V key, mass; };
    std::vector<BP> L;      // ascending keys; back = max
    std::deque<BP> R;       // ascending keys
    V offL = 0, offR = 0;
    V rTotal = 0;
    W minval = 0;

    V maxLeft()  const { return L.empty() ? NEG_BIG : L.back().key + offL; }
    V minRight() const { return R.empty() ? POS_BIG : R.front().key + offR; }

    void insL(V pos, V m) {
        if (!m) return;
        V k = pos - offL;
        size_t i = L.size();
        while (i > 0 && L[i - 1].key > k) --i;
        if (i > 0 && L[i - 1].key == k) L[i - 1].mass += m;
        else L.insert(L.begin() + i, {k, m});
    }
    void insR(V pos, V m) {
        if (!m) return;
        V k = pos - offR;
        size_t i = std::lower_bound(R.begin(), R.end(), k,
                       [](const BP& e, V kk) { return e.key < kk; }) - R.begin();
        if (i < R.size() && R[i].key == k) R[i].mass += m;
        else R.insert(R.begin() + i, {k, m});
        rTotal += m;
    }

    // F += gamma * |s - b|
    void addAbs(V b, V gamma) {
        if (gamma <= 0) return;
        V ml = maxLeft(), mr = minRight();
        if (b >= ml && b <= mr) { insL(b, gamma); insR(b, gamma); return; }
        if (b < ml) {
            V rem = gamma;
            while (rem > 0 && !L.empty()) {
                BP& e = L.back();
                V key = e.key + offL;
                if (key <= b) break;
                V m = std::min(e.mass, rem);
                minval += (W)m * (key - b);
                e.mass -= m;
                if (e.mass == 0) L.pop_back();
                insR(key, m);
                rem -= m;
            }
            // the |s-b| kink carries slope-delta 2*gamma in total; transferred
            // mass keeps contributing right of its own key
            insL(b, 2 * gamma - rem);
            insR(b, rem);
        } else {
            V rem = gamma;
            while (rem > 0 && !R.empty()) {
                BP& e = R.front();
                V key = e.key + offR;
                if (key >= b) break;
                V m = std::min(e.mass, rem);
                minval += (W)m * (b - key);
                e.mass -= m; rTotal -= m;
                if (e.mass == 0) R.pop_front();
                insL(key, m);
                rem -= m;
            }
            insR(b, 2 * gamma - rem);
            insL(b, rem);
        }
    }

    // F <- conv(F, g) where g has slopes: wall below -t, 0 on [-t, v]
    // (v = e - t), +tau above v (phantom pricing, never profitable).
    // Right-clip at tau; left keys shift by -t, right keys by +v.
    // Returns (lo, hi) clamp interval in PRE-conv coordinates.
    std::pair<V, V> convV(V tau, V v, V t) {
        V hi = POS_BIG;
        // slope == tau ties with the phantom continuation; pin hi at the last
        // real breakpoint in that case too (>=, not >) so recovery cannot
        // wander onto the phantom branch.
        if (rTotal >= tau) {
            V excess = rTotal - tau;
            while (excess > 0) {
                BP& e = R.back();
                if (e.mass <= excess) { excess -= e.mass; rTotal -= e.mass; R.pop_back(); }
                else { e.mass -= excess; rTotal -= excess; excess = 0; }
            }
            hi = R.empty() ? POS_BIG : R.back().key + offR;
        }
        V lo = maxLeft();
        offL += -t; offR += v;
        return {lo, hi};
    }

    W evalAt(V s) const {
        V ml = maxLeft(), mr = minRight();
        W v = minval;
        if (s < ml) {
            for (auto it = L.rbegin(); it != L.rend(); ++it) {
                V key = it->key + offL;
                if (key <= s) break;
                v += (W)it->mass * (key - s);
            }
        } else if (s > mr) {
            for (const auto& e : R) {
                V key = e.key + offR;
                if (key >= s) break;
                v += (W)e.mass * (s - key);
            }
        }
        return v;
    }
};
}  // namespace slopedp_detail

//#include "pylmcf/py_support.h"
#include "graph_elements.hpp"
#include "distribution.hpp"
#include "distances.hpp"

#include <iostream>
#include <cstdint>

// ---------------------------------------------------------------------------
// Cost scaling for order-p (Lp) Wasserstein.
//
// Edges carry their *real* (unscaled, possibly fractional) cost as a double.
// For p != 1 the cost d^p is fractional, so truncating straight to int64 would
// destroy precision for small distances.  Instead we pick an integer scale S
// from the global maximum real cost and store round(S * real) in the solver's
// integer cost map, undoing /S at the public boundary.  p == 1 is special-cased
// to S == 1 and plain truncation, reproducing the legacy integer behaviour
// bit-for-bit.
// ---------------------------------------------------------------------------

// Choose the integer cost scale S subject to two ceilings:
//
//  * per-edge: S * c_max <= 2^52, so every round(S*cost) lands inside the
//    double exact-integer range (round and the /S unscale are both exact) and
//    well below the NetworkSimplex potential-overflow ceiling (ART_COST=2^62).
//
//  * accumulator: the solver sums sum(scaled_cost * flow) in int64, and the
//    total flow is bounded by the total intensity, so S * c_max * total_flow
//    must stay within the int64 accumulator.  Bounding only the per-edge cost
//    (the old behaviour) silently overflowed totalCost() once total_flow grew
//    past ~2^11 — e.g. p=2 over a single peak-pair of mass 1e4 wrapped to a
//    negative total, yielding a complex distance after the ^(1/p) root.
//
// S is the floor of the tighter of the two ceilings (>= 1).  total_flow <= 0
// (unknown / p==1) falls back to the per-edge bound alone.
inline int64_t pick_cost_scale(double c_max, double total_flow, bool p_is_one) {
    if (p_is_one || !(c_max > 0.0)) return 1;
    constexpr double PER_EDGE_TARGET = 4503599627370496.0;   // 2^52
    constexpr double ACCUMULATOR_TARGET = 4611686018427387904.0; // 2^62
    // Ceiling on the scale itself: with a tiny c_max (< ~4.9e-4) the raw
    // quotient exceeds the int64 range and the double->int64 cast below is
    // UB — observed wrapping to INT64_MIN, i.e. a *negative* scale that
    // quantizes every cost negative and silently corrupts total_cost().
    // 2^60 is far above any scale the two budget ceilings can make useful,
    // is exactly representable in double, and keeps the cast well-defined.
    constexpr double SCALE_CEILING = 1152921504606846976.0;  // 2^60
    double s = std::floor(PER_EDGE_TARGET / c_max);
    if (total_flow > 0.0) {
        const double s_acc = std::floor(ACCUMULATOR_TARGET / (c_max * total_flow));
        if (s_acc < s) s = s_acc;
    }
    if (s > SCALE_CEILING) s = SCALE_CEILING;
    return s >= 1.0 ? static_cast<int64_t>(s) : 1;
}

// Quantise a real edge cost to the solver's integer cost type.
// p == 1: truncate the raw value (legacy, exact).  p != 1: round(S * real).
template <typename VALUE_TYPE>
inline VALUE_TYPE quantize_cost(double real_cost, int64_t scale, bool p_is_one) {
    const double scaled = p_is_one ? real_cost : static_cast<double>(scale) * real_cost;
    if (!std::isfinite(scaled) ||
        scaled > static_cast<double>(std::numeric_limits<VALUE_TYPE>::max()))
        throw std::overflow_error(
            "quantize_cost: scaled cost " + std::to_string(scaled) +
            " overflows the solver cost type (max " +
            std::to_string(std::numeric_limits<VALUE_TYPE>::max()) + ")");
    return p_is_one ? static_cast<VALUE_TYPE>(scaled)
                    : static_cast<VALUE_TYPE>(std::llround(scaled));
}


template <typename VALUE_TYPE, typename intensity_type>
class WassersteinNetworkSubgraph {
    std::vector<FlowNode<intensity_type>> nodes;
    std::vector<FlowEdge<intensity_type>> edges;
    lemon::StaticDigraph lemon_graph;
    lemon::StaticDigraph::NodeMap<VALUE_TYPE> node_supply_map;
    lemon::StaticDigraph::ArcMap<VALUE_TYPE> capacities_map;
    lemon::StaticDigraph::ArcMap<VALUE_TYPE> costs_map;
    std::optional<lemon::NetworkSimplex<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>> ns_solver;
    std::optional<pylmcf::NetworkSimplexLCTAdapter<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>> ns_lct_solver;
    std::optional<lemon::CycleCanceling<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>> cc_solver;
    std::optional<lemon::CostScaling<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>> cs_solver;
    std::optional<lemon::CapacityScaling<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>> cap_solver;
    SolverConfig _config = NetworkSimplexConfig{};
    LEMON_INDEX simple_trash_idx;
    bool simple_trash_added = false;
    bool experimental_trash_added = false;
    bool theoretical_trash_added = false;
    // Independent asymmetric trash (dualdeconv4 semantics): implemented by
    // cost shifting, not extra plumbing.  The identity
    //   sum_matched f*d + Xe*C_exp + Xt*C_theo
    //     = sum_matched f*(d - C_exp - C_theo) + E*C_exp + T(w)*C_theo
    // lets the MCF run with zero-cost trash arcs and matching costs shifted
    // by -(C_exp + C_theo); total_cost() adds the bracket back (linear in
    // this subgraph's own E and T(w), so the subgraph decomposition is
    // exact), and the theoretical marginals gain the constant C_theo term.
    // Chain factories: the per-matched-unit shift cannot ride per-hop chain
    // arcs, so only the SlopeDP backend supports independent trash there —
    // it prices trash analytically (trash_of(M) is exactly affine, marginal
    // tau = C_exp + C_theo) and reports the same shifted total; other
    // solvers on the chain are rejected at build().
    bool independent_trash = false;
    double _ind_c_exp = 0.0, _ind_c_theo = 0.0;   // real costs
    VALUE_TYPE _ind_c_exp_q = 0, _ind_c_theo_q = 0;  // quantized in build_impl
    VALUE_TYPE lemon_empirical_intensity;
    VALUE_TYPE lemon_theoretical_intensity;
    const size_t no_target_distributions;
    bool built = false;
    int _cold_starts_via_run = 0;

    // Warm-sequence capture (WNET_DUMP_WARMSEQ): when the env var holds a
    // path prefix, every NetworkSimplex solve of this subgraph appends its
    // inputs (costs on change, supplies, caps) to <prefix>.<seq>.wseq so the
    // exact solver load can be replayed offline (see p2_work replay tooling).
    // Format v2: int32 magic=-2, n, m, np=-1 (read until EOF), src[m], tgt[m]
    // (int32), then per point: uint8 costs_changed, [cost[m]], supply[n],
    // cap[m] (all int64).  First point always carries costs.
    std::unique_ptr<std::ofstream> _dump_stream;

    // Cost scaling (set by the owning network at build): integer cost map holds
    // quantize_cost(real, _scale, _p_is_one).  _p_is_one => legacy S=1 truncation.
    int64_t _scale = 1;
    bool _p_is_one = true;
    // Intensity scaling (set by the owning network at build): real (double) node
    // intensities are mapped to integer LEMON supplies/capacities as
    // round-toward-zero(real * _intensity_scale).  1.0 reproduces the legacy
    // behaviour (intensities consumed verbatim, truncated to the integer type).
    double _intensity_scale = 1.0;

    // Cached residual/derivative context.  The context is a pure function of
    // the post-solve solver state (potentials, flow, capacities, costs),
    // which only changes via _run_solver(); _solution_version is bumped
    // there, so a cached context whose stamp matches is bit-identical to a
    // fresh recompute (value-exact memoization across multiple derivative
    // queries / identical re-solves on the same solution).  Two independent
    // slots: [0] = exact residual, [1] = fast dual-pi approximation, so the
    // same solved solution can be queried in either form without thrash.
    struct DerivContext {
        VALUE_TYPE INF;
        bool supply_fixed;
        bool asymmetric;
        // Independent trash: the supply_fixed one-sided dispatch encodes the
        // annihilating max(E, T) budget and misses valid adjustments for the
        // independent model (e.g. re-matching a trashed empirical unit when
        // T >= E).  The correct shifted marginal is min(dist_src, dist_sink):
        // with E > T a trashed empirical unit always exists (Xe >= E-T > 0),
        // so a zero-cost sink->emp->src relay makes dist_sink <= dist_src and
        // the min never admits an infeasible source-augmenting path; with
        // T >= E both the cycle and the source augmentation are feasible.
        bool independent;
        VALUE_TYPE trash_cost, src_adjust, sink_adjust;
        std::vector<VALUE_TYPE> dist_src, dist_sink;
        std::vector<VALUE_TYPE> theo_sink_slack;
    };
    std::uint64_t _solution_version = 0;
    mutable std::uint64_t _deriv_ctx_version[2] = {
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max()};
    mutable DerivContext _deriv_ctx_cache[2];

    struct MatchingEdgeInfo {
        lemon::StaticDigraph::Arc arc;
        intensity_type emp_intensity;
        intensity_type theo_intensity;
        size_t spectrum_id;
        LEMON_INDEX emp_peak_index;
        LEMON_INDEX theo_peak_index;
    };
    struct TheoSinkEdgeInfo {
        lemon::StaticDigraph::Arc arc;
        intensity_type theo_intensity;
        size_t spectrum_id;
    };
    std::vector<MatchingEdgeInfo> _matching_edge_cache;
    std::vector<TheoSinkEdgeInfo> _theo_sink_edge_cache;
    std::vector<uint8_t> _unlimited_arc;  // true for MatchingEdge and ChainEdge
    std::vector<VALUE_TYPE> _costs_buf;   // reusable scratch for update_positions_and_solve
    mutable std::vector<VALUE_TYPE> _chain_R_buf;    // K-1; reused for R/c_right in chain functions
    mutable std::vector<VALUE_TYPE> _chain_L_buf;    // K-1; reused for L/c_left in chain functions
    mutable std::vector<VALUE_TYPE> _chain_dist_buf; // n nodes; reused by chain_residual_distances
    std::vector<double> _chain_pos_buf;              // K; reused by update_positions_and_solve
    mutable std::vector<uint8_t>    _chain_has_src_fwd,  _chain_has_src_rev;
    mutable std::vector<uint8_t>    _chain_has_sink_fwd, _chain_has_sink_rev;
    mutable std::vector<VALUE_TYPE> _chain_exp_trash_cost, _chain_theo_trash_cost;
    mutable std::vector<uint8_t>    _chain_exp_trash_fwd, _chain_exp_trash_rev;
    mutable std::vector<uint8_t>    _chain_theo_trash_fwd, _chain_theo_trash_rev;

    struct ChainTopology {
        std::vector<LEMON_INDEX> order;
        std::vector<LEMON_INDEX> right_arc_ids;
        std::vector<LEMON_INDEX> left_arc_ids;
        std::vector<VALUE_TYPE>  gap_cost;
        std::vector<size_t> node_to_pos;
    };
    std::optional<ChainTopology> _chain_topo;

    // True when the NetworkSimplex backend is the experimental link-cut-tree
    // implementation (NSWarmMode::LinkCut) rather than LEMON's array solver.
    bool _use_lct() const {
        return std::holds_alternative<NetworkSimplexConfig>(_config) &&
               std::get<NetworkSimplexConfig>(_config).warm == NSWarmMode::LinkCut;
    }
    bool _use_slopedp() const { return std::holds_alternative<SlopeDPConfig>(_config); }

    // ---- slope-DP backend state (SlopeDPConfig) -------------------------- //
    // Per chain position: incident arc ids (-1 when absent) and the fixed
    // empirical cap; per-solve: per-arc flows and the total.
    mutable bool _sdp_cache_built = false;
    mutable std::vector<LEMON_INDEX> _sdp_src_arc, _sdp_sink_arc, _sdp_et_arc, _sdp_tt_arc;
    mutable std::vector<VALUE_TYPE> _sdp_emp_cap;      // per chain pos (0 for theo)
    mutable std::vector<VALUE_TYPE> _sdp_pos;          // per chain pos: prefix of gap_cost
    mutable std::vector<size_t> _sdp_cluster_start;    // C+1 offsets into positions
    mutable VALUE_TYPE _sdp_c_exp = -1, _sdp_c_theo = -1, _sdp_c_s = -1;   // quantized
    mutable std::vector<VALUE_TYPE> _slope_flow;       // per LEMON arc id
    mutable VALUE_TYPE _slope_total = 0;
    mutable bool _slope_solved = false;

    VALUE_TYPE _solver_potential(lemon::StaticDigraph::Node nd) const {
        if (_use_slopedp())
            throw std::runtime_error("SlopeDP backend does not expose node potentials.");
        return _use_lct() ? ns_lct_solver->potential(nd)
                          : ns_solver->potential(nd);
    }
    bool _solver_has_value() const {
        if (_use_slopedp()) return _slope_solved;
        if (std::holds_alternative<NetworkSimplexConfig>(_config))
            return _use_lct() ? ns_lct_solver.has_value() : ns_solver.has_value();
        if (std::holds_alternative<CycleCancelingConfig>(_config)) return cc_solver.has_value();
        if (std::holds_alternative<CostScalingConfig>(_config))    return cs_solver.has_value();
        return cap_solver.has_value();
    }
    VALUE_TYPE _solver_flow(lemon::StaticDigraph::Arc arc) const {
        if (_use_slopedp()) return _slope_flow[lemon_graph.id(arc)];
        if (std::holds_alternative<NetworkSimplexConfig>(_config))
            return _use_lct() ? ns_lct_solver->flow(arc) : ns_solver->flow(arc);
        if (std::holds_alternative<CycleCancelingConfig>(_config)) return cc_solver->flow(arc);
        if (std::holds_alternative<CostScalingConfig>(_config))    return cs_solver->flow(arc);
        return cap_solver->flow(arc);
    }
    VALUE_TYPE _solver_total_cost() const {
        if (_use_slopedp()) return _slope_total;
        if (std::holds_alternative<NetworkSimplexConfig>(_config))
            return _use_lct() ? ns_lct_solver->totalCost() : ns_solver->totalCost();
        if (std::holds_alternative<CycleCancelingConfig>(_config)) return cc_solver->totalCost();
        if (std::holds_alternative<CostScalingConfig>(_config))    return cs_solver->totalCost();
        return cap_solver->totalCost();
    }

    void _build_chain_topology() {
        std::vector<std::vector<std::pair<LEMON_INDEX, LEMON_INDEX>>> chain_adj(nodes.size());
        bool any_chain = false;
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii) {
            if (std::holds_alternative<ChainEdge>(edges[ii].get_type())) {
                const LEMON_INDEX u = edges[ii].get_start_node_id();
                const LEMON_INDEX v = edges[ii].get_end_node_id();
                chain_adj[u].push_back({v, ii});
                any_chain = true;
            }
        }
        if (!any_chain) return;

        LEMON_INDEX start_node = -1;
        for (LEMON_INDEX node_id = 0; node_id < static_cast<LEMON_INT>(chain_adj.size()); ++node_id) {
            if (chain_adj[node_id].size() == 1) { start_node = node_id; break; }
        }
        if (start_node == -1)
            throw std::runtime_error("Chain subgraph has no endpoint — malformed chain.");

        ChainTopology topo;
        topo.order.push_back(start_node);
        LEMON_INDEX prev = -1, curr = start_node;
        while (true) {
            LEMON_INDEX next = -1, out_arc = -1;
            for (const auto& [neighbor, arc_id] : chain_adj[curr]) {
                if (neighbor != prev) { next = neighbor; out_arc = arc_id; break; }
            }
            if (next == -1) break;
            LEMON_INDEX in_arc = -1;
            for (const auto& [neighbor, arc_id] : chain_adj[next]) {
                if (neighbor == curr) { in_arc = arc_id; break; }
            }
            if (in_arc == -1)
                throw std::runtime_error("Chain arc is not bidirectional — malformed chain.");
            topo.order.push_back(next);
            topo.right_arc_ids.push_back(out_arc);
            topo.left_arc_ids.push_back(in_arc);
            topo.gap_cost.push_back(costs_map[lemon_graph.arcFromId(out_arc)]);
            prev = curr; curr = next;
        }
        topo.node_to_pos.assign(nodes.size(), std::numeric_limits<size_t>::max());
        for (size_t i = 0; i < topo.order.size(); ++i)
            topo.node_to_pos[topo.order[i]] = i;
        _chain_topo = std::move(topo);
    }

    // Append this solve's inputs to the capture stream (see _dump_stream).
    void _dump_point(bool costs_changed) {
        static const char* prefix = std::getenv("WNET_DUMP_WARMSEQ");
        if (!prefix || !*prefix) return;
        const int32_t n = lemon_graph.nodeNum(), m = lemon_graph.arcNum();
        auto put32 = [&](int32_t v) {
            _dump_stream->write(reinterpret_cast<const char*>(&v), sizeof v);
        };
        auto put64 = [&](int64_t v) {
            _dump_stream->write(reinterpret_cast<const char*>(&v), sizeof v);
        };
        bool first = false;
        if (!_dump_stream) {
            static std::atomic<int> seq_counter{0};
            const std::string path = std::string(prefix) + "." +
                std::to_string(seq_counter++) + ".wseq";
            _dump_stream = std::make_unique<std::ofstream>(path, std::ios::binary);
            first = true;
            put32(-2); put32(n); put32(m); put32(-1);
            for (int i = 0; i < m; ++i) put32(lemon_graph.id(lemon_graph.source(lemon_graph.arcFromId(i))));
            for (int i = 0; i < m; ++i) put32(lemon_graph.id(lemon_graph.target(lemon_graph.arcFromId(i))));
        }
        const uint8_t cc = (first || costs_changed) ? 1 : 0;
        _dump_stream->write(reinterpret_cast<const char*>(&cc), 1);
        if (cc)
            for (int i = 0; i < m; ++i) put64(int64_t(costs_map[lemon_graph.arcFromId(i)]));
        for (int i = 0; i < n; ++i) put64(int64_t(node_supply_map[lemon_graph.nodeFromId(i)]));
        for (int i = 0; i < m; ++i) put64(int64_t(capacities_map[lemon_graph.arcFromId(i)]));
        _dump_stream->flush();
    }

    // Run (or warm-restart) the configured solver using the current
    // capacities_map, costs_map and node_supply_map.
    // costs_changed=true: costs were updated since the last solve (e.g.
    // after update_positions); for NetworkSimplex the cost map must be
    // re-pushed before warmRun so that the dual variables are consistent.
    void _run_solver(bool costs_changed = false) {
        // LEMON solvers report INFEASIBLE/UNBOUNDED via their return value
        // and leave whatever flow state they last had; discarding the status
        // (the old behaviour) served that state as if it were a valid
        // solution.  Throw instead, naming the status.
        auto require_optimal = [](auto status, const char* solver_name) {
            using PT = decltype(status);
            if (status == PT::INFEASIBLE)
                throw InfeasibleException(std::string(solver_name) +
                    ": flow problem is INFEASIBLE (no flow satisfies the "
                    "current supplies/capacities); the solver state is not a "
                    "valid solution.");
            if (status == PT::UNBOUNDED)
                throw std::runtime_error(std::string(solver_name) +
                    ": flow problem is UNBOUNDED (negative-cost cycle with "
                    "unlimited capacity).");
        };
        std::visit([&](const auto& cfg) {
            using T = std::decay_t<decltype(cfg)>;
            if constexpr (std::is_same_v<T, NetworkSimplexConfig>) {
              _dump_point(costs_changed);
              if (cfg.warm == NSWarmMode::LinkCut) {
                // Experimental link-cut-tree backend (Simple warm strategy:
                // repair-or-cold).  pivot/strategy are not applicable.
                if (ns_lct_solver.has_value()) {
                    ns_lct_solver->upperMap(capacities_map);
                    ns_lct_solver->supplyMap(node_supply_map);
                    if (costs_changed) ns_lct_solver->costMap(costs_map);
                    require_optimal(ns_lct_solver->warmRun(),
                                    "NetworkSimplexLCT::warmRun()");
                } else {
                    ++_cold_starts_via_run;
                    ns_lct_solver.emplace(lemon_graph);
                    ns_lct_solver->upperMap(capacities_map);
                    ns_lct_solver->costMap(costs_map);
                    ns_lct_solver->supplyMap(node_supply_map);
                    require_optimal(ns_lct_solver->run(),
                                    "NetworkSimplexLCT::run()");
                }
              } else {
                using LemonPR = lemon::NetworkSimplex<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>::PivotRule;
                const auto pivot = static_cast<LemonPR>(cfg.pivot);
                if (cfg.warm != NSWarmMode::None && ns_solver.has_value()) {
                    // Warm start: reuse the existing solver and its spanning-tree
                    // basis.  Only capacities and supplies change between calls
                    // (costs are fixed at build() time), so warmRun() can repair
                    // the previous optimal basis and reach the new optimum with
                    // far fewer pivots.  Dual/Primal modes additionally attempt
                    // a dual-simplex / primal-pivot repair before any cold
                    // fallback; all modes fall back to a cold start if the
                    // basis cannot be repaired.
                    using LemonWR = lemon::NetworkSimplex<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>::WarmRepair;
                    const LemonWR strategy =
                        cfg.warm == NSWarmMode::Dual       ? LemonWR::Dual       :
                        cfg.warm == NSWarmMode::Primal     ? LemonWR::Primal     :
                        cfg.warm == NSWarmMode::DualRatio  ? LemonWR::DualRatio  :
                        cfg.warm == NSWarmMode::DualGreedy ? LemonWR::DualGreedy :
                                                             LemonWR::RepairOnly;
                    ns_solver->upperMap(capacities_map);
                    ns_solver->supplyMap(node_supply_map);
                    if (costs_changed) ns_solver->costMap(costs_map);
                    ns_solver->setWarmViolationLimit(cfg.warm_violation_limit);
                    // costs_changed disables the "repair == optimal" fast
                    // path inside warmRun (stale potentials); the basis is
                    // still reused and start() reoptimizes.
                    require_optimal(ns_solver->warmRun(pivot, strategy, costs_changed),
                                    "NetworkSimplex::warmRun()");
                } else {
                    ++_cold_starts_via_run;
                    ns_solver.emplace(lemon_graph);
                    ns_solver->upperMap(capacities_map);
                    ns_solver->costMap(costs_map);
                    ns_solver->supplyMap(node_supply_map);
                    require_optimal(ns_solver->run(pivot),
                                    "NetworkSimplex::run()");
                }
              }
            } else if constexpr (std::is_same_v<T, CycleCancelingConfig>) {
                using LemonM = lemon::CycleCanceling<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>::Method;
                cc_solver.emplace(lemon_graph);
                cc_solver->upperMap(capacities_map);
                cc_solver->costMap(costs_map);
                cc_solver->supplyMap(node_supply_map);
                require_optimal(cc_solver->run(static_cast<LemonM>(cfg.method)),
                                "CycleCanceling::run()");
            } else if constexpr (std::is_same_v<T, CostScalingConfig>) {
                using LemonM = lemon::CostScaling<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>::Method;
                cs_solver.emplace(lemon_graph);
                cs_solver->upperMap(capacities_map);
                cs_solver->costMap(costs_map);
                cs_solver->supplyMap(node_supply_map);
                require_optimal(
                    cs_solver->run(static_cast<LemonM>(cfg.method), cfg.factor),
                    "CostScaling::run()");
            } else if constexpr (std::is_same_v<T, CapacityScalingConfig>) {
                cap_solver.emplace(lemon_graph);
                cap_solver->upperMap(capacities_map);
                cap_solver->costMap(costs_map);
                cap_solver->supplyMap(node_supply_map);
                require_optimal(cap_solver->run(cfg.factor),
                                "CapacityScaling::run()");
            } else {
                static_assert(std::is_same_v<T, SlopeDPConfig>);
                // SlopeDP is dispatched from set_point(); reaching here means
                // a subgraph the chain solver cannot handle.
                throw std::runtime_error(
                    "SlopeDP backend supports only chain-factory subgraphs "
                    "(no MatchingEdges).");
            }
        }, _config);
        // Any solve may have changed potentials/flow -> invalidate the
        // cached derivative context.
        ++_solution_version;
    }

public:
    WassersteinNetworkSubgraph(
        const std::vector<LEMON_INDEX>& subgraph_node_ids,
        const std::vector<FlowNode<intensity_type>>& all_nodes,
        const std::vector<FlowEdge<intensity_type>*>& my_edges,
        size_t no_target_distributions_
    ) :
        lemon_graph(),
        node_supply_map(lemon_graph),
        capacities_map(lemon_graph),
        costs_map(lemon_graph),
        simple_trash_idx(std::numeric_limits<LEMON_INDEX>::max()),
        lemon_empirical_intensity(0),
        lemon_theoretical_intensity(0),
        no_target_distributions(no_target_distributions_)
    {
        nodes.reserve(subgraph_node_ids.size()+2);
        nodes.push_back(FlowNode<intensity_type>(0, SourceNode()));
        nodes.push_back(FlowNode<intensity_type>(1, SinkNode()));
        auto& source_node = nodes[0];
        auto& sink_node = nodes[1];

        std::vector<LEMON_INDEX> node_id_map(all_nodes.size(), -1);

        for (const auto& node_id : subgraph_node_ids)
        {
            node_id_map[node_id] = nodes.size();
            const FlowNodeType<intensity_type>& node_type = all_nodes[node_id].get_type();
            nodes.push_back(FlowNode<intensity_type>(nodes.size(), node_type));
            auto& new_node = nodes.back();
            if(std::holds_alternative<EmpiricalNode<intensity_type>>(node_type))
            {
                edges.emplace_back(
                    edges.size(),
                    source_node,
                    new_node,
                    SrcToEmpiricalEdge()
                );
            }
            else if(std::holds_alternative<TheoreticalNode<intensity_type>>(node_type))
            {
                edges.emplace_back(
                    edges.size(),
                    new_node,
                    sink_node,
                    TheoreticalToSinkEdge()
                );
            }
            else throw std::runtime_error("Invalid FlowNode type. This shouldn't happen.");
        }

        for (const FlowEdge<intensity_type>* edge : my_edges)
        {
            const FlowNode<intensity_type>& start_node = edge->get_start_node();
            const LEMON_INDEX start_local = node_id_map[start_node.get_id()];
            if (start_local == -1) throw std::runtime_error("Start node of edge not found in subgraph nodes.");
            const FlowNode<intensity_type>& end_node = edge->get_end_node();
            const LEMON_INDEX end_local = node_id_map[end_node.get_id()];
            if (end_local == -1) throw std::runtime_error("End node of edge not found in subgraph nodes.");
            edges.emplace_back(
                    edges.size(),
                    nodes[start_local],
                    nodes[end_local],
                    edge->get_type()
            );
        }
    }

    WassersteinNetworkSubgraph(const WassersteinNetworkSubgraph&) = delete;
    WassersteinNetworkSubgraph& operator=(const WassersteinNetworkSubgraph&) = delete;
    WassersteinNetworkSubgraph(WassersteinNetworkSubgraph&&) = delete;
    WassersteinNetworkSubgraph& operator=(WassersteinNetworkSubgraph&&) = delete;

    void add_simple_trash(double cost) {
        if (simple_trash_added)
            throw std::runtime_error("Simple trash edge already added.");
        if (experimental_trash_added || theoretical_trash_added)
            throw std::runtime_error("add_simple_trash() is exclusive with experimental/theoretical trash.");
        if (built)
            throw std::runtime_error("add_simple_trash() must be called before build(), not after.");
        edges.emplace_back(
            edges.size(),
            nodes[0],
            nodes[1],
            SimpleTrashEdge(cost)
        );
        simple_trash_added = true;
    }

    void add_experimental_trash(double cost) {
        if (simple_trash_added)
            throw std::runtime_error("add_experimental_trash() is exclusive with simple trash.");
        if (experimental_trash_added)
            throw std::runtime_error("Experimental trash already added.");
        if (built)
            throw std::runtime_error("add_experimental_trash() must be called before build().");
        // One EmpiricalTrashEdge per empirical node: EmpiricalNode -> Sink.
        // Capacity is set to INF in build(); non-binding (bounded by source supply).
        for (const auto& node : nodes) {
            if (!std::holds_alternative<EmpiricalNode<intensity_type>>(node.get_type())) continue;
            edges.emplace_back(edges.size(), node, nodes[1], EmpiricalTrashEdge(cost));
        }
        experimental_trash_added = true;
    }

    void add_theoretical_trash(double cost) {
        if (simple_trash_added)
            throw std::runtime_error("add_theoretical_trash() is exclusive with simple trash.");
        if (theoretical_trash_added)
            throw std::runtime_error("Theoretical trash already added.");
        if (built)
            throw std::runtime_error("add_theoretical_trash() must be called before build().");
        // One TheoreticalTrashEdge per theoretical node: Source -> TheoreticalNode.
        // Capacity is set to INF in build(); non-binding (bounded by source supply).
        for (const auto& node : nodes) {
            if (!std::holds_alternative<TheoreticalNode<intensity_type>>(node.get_type())) continue;
            edges.emplace_back(edges.size(), nodes[0], node, TheoreticalTrashEdge(cost));
        }
        theoretical_trash_added = true;
    }

    // Independent asymmetric trash (see the member comment for the cost-shift
    // identity).  Creates zero-cost trash arcs on both sides; the real costs
    // are charged via the matching shift and the analytic term.
    void add_independent_asymmetric_trash(double C_exp, double C_theo) {
        if (independent_trash)
            throw std::runtime_error("Independent trash already added.");
        if (built)
            throw std::runtime_error(
                "add_independent_asymmetric_trash() must be called before build().");
        add_experimental_trash(0.0);
        add_theoretical_trash(0.0);
        independent_trash = true;
        _ind_c_exp = C_exp;
        _ind_c_theo = C_theo;
    }

    // Real (unscaled) simple-trash cost.  The scaled value used inside the
    // solver/derivatives is costs_map[arc at simple_trash_idx].
    double simple_trash_cost() const {
        if (simple_trash_idx == std::numeric_limits<LEMON_INDEX>::max())
            throw std::runtime_error("Simple trash edge not added.");
        return std::visit([](const auto& arg) -> double {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, SimpleTrashEdge>) return arg.get_cost();
            else { throw std::runtime_error("Invalid FlowEdgeType at simple_trash_idx"); };
        }, edges[simple_trash_idx].get_type());
    }

    void build_impl() {
        assert_fits_lemon_index(nodes.size(), "subgraph nodes");
        assert_fits_lemon_index(edges.size(), "subgraph edges");
        edges = std::move(sorted_copy(edges, [](const FlowEdge<intensity_type>& a, const FlowEdge<intensity_type>& b) {
            if(a.get_start_node_id() != b.get_start_node_id())
                return a.get_start_node_id() < b.get_start_node_id();
            return a.get_end_node_id() < b.get_end_node_id();
        }));
        std::vector<std::pair<LEMON_INDEX, LEMON_INDEX>> arcs;
        arcs.reserve(edges.size());
        for (const FlowEdge<intensity_type>& edge : edges)
            arcs.emplace_back(edge.get_start_node_id(), edge.get_end_node_id());
        lemon_graph.build(nodes.size(), arcs.begin(), arcs.end());

        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(nodes.size()); ++ii)
            node_supply_map[lemon_graph.nodeFromId(ii)] = 0;

        // Independent trash: quantize the analytic per-unit costs with the
        // network-wide scale, and shift matching costs by their (real) sum
        // below — shifting before quantization costs one rounding, not two.
        if (independent_trash) {
            _ind_c_exp_q = quantize_cost<VALUE_TYPE>(_ind_c_exp, _scale, _p_is_one);
            _ind_c_theo_q = quantize_cost<VALUE_TYPE>(_ind_c_theo, _scale, _p_is_one);
        }
        const double _ind_shift = independent_trash ? (_ind_c_exp + _ind_c_theo) : 0.0;

        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii)
            costs_map[lemon_graph.arcFromId(ii)] = std::visit([&](const auto& arg) -> VALUE_TYPE {
                    using T = std::decay_t<decltype(arg)>;
                    // Cost-bearing edges hold a real (double) cost; quantise it to
                    // the integer solver cost here using the network-wide scale.
                    if constexpr (std::is_same_v<T, MatchingEdge>) return quantize_cost<VALUE_TYPE>(arg.get_cost() - _ind_shift, _scale, _p_is_one);
                    else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) return (VALUE_TYPE) 0;
                    else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) return (VALUE_TYPE) 0;
                    else if constexpr (std::is_same_v<T, SimpleTrashEdge>) { simple_trash_idx = ii; return quantize_cost<VALUE_TYPE>(arg.get_cost(), _scale, _p_is_one); }
                    else if constexpr (std::is_same_v<T, ChainEdge>) {
                        // On the chain the per-match shift cannot ride hop arcs;
                        // only the SlopeDP backend prices independent trash
                        // analytically (tau = C_exp + C_theo), so other solvers
                        // must use the dense factory.
                        if (independent_trash && !_use_slopedp())
                            throw std::runtime_error(
                                "Independent asymmetric trash on the 1-D chain requires "
                                "the SlopeDP solver (chain hop arcs cannot carry the "
                                "per-match cost shift); use SlopeDP, or build the "
                                "network with force_dense_1d=True.");
                        return quantize_cost<VALUE_TYPE>(arg.get_cost(), _scale, _p_is_one);
                    }
                    // Independent trash + SlopeDP (chain): the solver prices trash
                    // analytically and ignores these costs, but the residual
                    // derivative search reads them.  It must see the residual of
                    // the SHIFTED problem (matched unit worth s = C_exp + C_theo,
                    // trash free, analytic bracket added outside): with no
                    // matching arcs to carry -s, the shift moves to the matched-
                    // mass carriers — emp-trash arcs cost +s here and src arcs
                    // carry -/+s in the chain relay (_chain_src_shift).  Phantom
                    // (theoretical trash) arcs stay 0 like the dense shifted net.
                    else if constexpr (std::is_same_v<T, EmpiricalTrashEdge>) {
                        if (independent_trash)
                            return _use_slopedp()
                                ? (VALUE_TYPE)(_ind_c_exp_q + _ind_c_theo_q)
                                : (VALUE_TYPE) quantize_cost<VALUE_TYPE>(arg.get_cost(), _scale, _p_is_one);
                        return quantize_cost<VALUE_TYPE>(arg.get_cost(), _scale, _p_is_one);
                    }
                    else if constexpr (std::is_same_v<T, TheoreticalTrashEdge>) return quantize_cost<VALUE_TYPE>(arg.get_cost(), _scale, _p_is_one);
                    else { throw std::runtime_error("Invalid FlowEdgeType"); };
                }, edges[ii].get_type());

        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii)
        {
            capacities_map[lemon_graph.arcFromId(ii)] = std::visit([&](const auto& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, MatchingEdge>) return (VALUE_TYPE) 0;
                    else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) {
                        VALUE_TYPE lemon_intensity = static_cast<VALUE_TYPE>(
                            std::get<EmpiricalNode<intensity_type>>(edges[ii].get_end_node().get_type()).get_intensity() * _intensity_scale);
                        lemon_empirical_intensity += lemon_intensity;
                        return lemon_intensity;
                    }
                    else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) return (VALUE_TYPE) 0;
                    // Simple trash carries flow up to lemon_total_flow (the supply set on
                    // source/sink). A tight cap = lemon_total_flow is redundant — flow is
                    // already bounded by supply — and causes warm-restart to fail whenever
                    // lemon_total_flow shrinks below the previous trash flow. Use INF so the
                    // cap is non-binding and never needs updating.
                    else if constexpr (std::is_same_v<T, SimpleTrashEdge>) return std::numeric_limits<VALUE_TYPE>::max();
                    // Chain, empirical-trash, and theoretical-trash edges carry unlimited
                    // flow. Use INF (= numeric_limits::max() for int64, which equals LEMON's
                    // INF sentinel). LEMON's findLeavingArc guards c >= MAX → INF so residual
                    // capacity is correctly treated as infinite; LEMON uses this same value
                    // for its own artificial arcs. max/2 was wrong: it bypassed the guard,
                    // returning max/2 - flow (finite) instead of INF.
                    else if constexpr (std::is_same_v<T, ChainEdge>) return std::numeric_limits<VALUE_TYPE>::max();
                    // Asymmetric trash edges: the adjacent anchor edge is always the
                    // binding constraint (SrcToEmpiricalEdge caps empirical inflow;
                    // TheoreticalToSinkEdge caps theoretical outflow), so a redundant
                    // tight cap here adds pivot candidates without shrinking the feasible
                    // region. Use INF like ChainEdge and skip set_point updates.
                    else if constexpr (std::is_same_v<T, EmpiricalTrashEdge>) return std::numeric_limits<VALUE_TYPE>::max();
                    else if constexpr (std::is_same_v<T, TheoreticalTrashEdge>) return std::numeric_limits<VALUE_TYPE>::max();
                    else { throw std::runtime_error("Invalid FlowEdgeType"); };
                }, edges[ii].get_type());
        }
        ns_solver.reset();
        ns_lct_solver.reset();
        cc_solver.reset();
        _slope_solved = false;
        _sdp_cache_built = false;
        _chain_arc_cache_built = false;
        _build_chain_topology();
        _matching_edge_cache.clear();
        _theo_sink_edge_cache.clear();
        _unlimited_arc.assign(edges.size(), false);
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii) {
            std::visit([&](const auto& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, MatchingEdge>) {
                    _unlimited_arc[ii] = true;
                    const auto& theo = std::get<TheoreticalNode<intensity_type>>(edges[ii].get_end_node().get_type());
                    const auto& emp  = std::get<EmpiricalNode<intensity_type>>(edges[ii].get_start_node().get_type());
                    _matching_edge_cache.push_back({lemon_graph.arcFromId(ii), emp.get_intensity(), theo.get_intensity(), theo.get_spectrum_id(), emp.get_peak_index(), theo.get_peak_index()});
                } else if constexpr (std::is_same_v<T, ChainEdge>) {
                    _unlimited_arc[ii] = true;
                } else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) {
                    const auto& theo = std::get<TheoreticalNode<intensity_type>>(edges[ii].get_start_node().get_type());
                    _theo_sink_edge_cache.push_back({lemon_graph.arcFromId(ii), theo.get_intensity(), theo.get_spectrum_id()});
                }
            }, edges[ii].get_type());
        }
        _costs_buf.assign(edges.size(), VALUE_TYPE(0));
        if (_chain_topo.has_value()) {
            const size_t K = _chain_topo->order.size();
            const size_t Km1 = K > 1 ? K - 1 : 0;
            _chain_R_buf.resize(Km1);
            _chain_L_buf.resize(Km1);
            _chain_pos_buf.resize(K);
            _chain_dist_buf.resize(nodes.size());
            _chain_has_src_fwd.resize(K);   _chain_has_src_rev.resize(K);
            _chain_has_sink_fwd.resize(K);  _chain_has_sink_rev.resize(K);
            _chain_exp_trash_cost.resize(K); _chain_theo_trash_cost.resize(K);
            _chain_exp_trash_fwd.resize(K); _chain_exp_trash_rev.resize(K);
            _chain_theo_trash_fwd.resize(K); _chain_theo_trash_rev.resize(K);
        }
        built = true;
    }

    // Set by the owning network before build() so build_impl() quantises costs
    // with the network-wide scale.  Kept separate from build() to leave the
    // Python-facing build(config) signature unchanged.
    void set_cost_scaling(int64_t scale, bool p_is_one, double intensity_scale = 1.0) {
        _scale = scale;
        _p_is_one = p_is_one;
        _intensity_scale = intensity_scale;
    }

    void build(SolverConfig config = NetworkSimplexConfig{}) {
        _config = config;
        build_impl();
    }

    void set_point(const std::vector<double>& point) {
        if(point.size() != no_target_distributions)
            throw std::runtime_error("Point dimension: " + std::to_string(point.size()) + " does not match number of target distributions: " + std::to_string(no_target_distributions));
        lemon_theoretical_intensity = 0;
        for (const auto& e : _matching_edge_cache) {
            capacities_map[e.arc] = (VALUE_TYPE) std::min<double>(
                e.theo_intensity * point[e.spectrum_id] * _intensity_scale,
                e.emp_intensity * _intensity_scale);
        }
        for (const auto& e : _theo_sink_edge_cache) {
            VALUE_TYPE lemon_intensity = (VALUE_TYPE) (e.theo_intensity * point[e.spectrum_id] * _intensity_scale);
            capacities_map[e.arc] = lemon_intensity;
            lemon_theoretical_intensity += lemon_intensity;
        }
        // Determine how many units to push from source to sink.
        // When both sides can absorb excess (simple trash, or both asymmetric
        // trash types), use max so every peak participates.  When only one
        // asymmetric trash direction is present, cap supply to the side that
        // has a valid escape route so the MCF is feasible.  No-trash always
        // throws — see the comment inside the else branch.
        VALUE_TYPE lemon_total_flow = 0;
        if (simple_trash_added || (experimental_trash_added && theoretical_trash_added)) {
            lemon_total_flow = std::max<VALUE_TYPE>(lemon_empirical_intensity, lemon_theoretical_intensity);
        } else if (experimental_trash_added) {
            lemon_total_flow = lemon_empirical_intensity;
        } else if (theoretical_trash_added) {
            lemon_total_flow = lemon_theoretical_intensity;
        } else {
            // No trash edges present.  All MCF solvers are unsafe or produce
            // incorrect results without a trash escape route:
            //
            // NetworkSimplex: signed-integer overflow UB in updatePotential().
            //   ART_COST = 2^62; potential accumulation during init pivots can
            //   reach 2^63 — UB.  GCC wraps and terminates; Clang loops forever.
            //
            // CostScaling / CapacityScaling: no ART_COST issue, but
            //   lemon_total_flow = min(emp, theo) is infeasible on sparse graphs
            //   (some units have no matching path), causing these solvers to
            //   return INFEASIBLE with totalCost() = 0 — silently wrong.
            //
            // The correct fix is to add trash edges before calling solve():
            //   use add_simple_trash(cost) to give every unit an escape route.
            // No trash: safe only when supply == demand (empirical == theoretical
            // intensity after integer quantisation).  A balanced, dense-matching
            // network is always feasible; LEMON drives out its artificial arcs
            // immediately so the ART_COST = 2^62 potential-accumulation UB
            // cannot occur.  Verified empirically under ASan+UBSan: 298 tests,
            // zero reports.  Unbalanced or sparse cases must use add_simple_trash().
            if (lemon_empirical_intensity != lemon_theoretical_intensity) {
                std::string msg =
                    "wnet: solve() without trash edges requires equal empirical and "
                    "theoretical intensities (balanced supply/demand); got empirical = " +
                    std::to_string(lemon_empirical_intensity) + " vs theoretical = " +
                    std::to_string(lemon_theoretical_intensity) +
                    " in scaled integer units.";
                if (_intensity_scale != 1.0)
                    msg += " Intensities are quantised as trunc(real * " +
                        std::to_string(_intensity_scale) +
                        ") per peak, so equal real-valued totals can still land on "
                        "unequal integers.";
                msg += " Call add_simple_trash() before build(), or pass integer "
                       "intensities that balance exactly.";
                throw InfeasibleException(msg);
            }
            lemon_total_flow = lemon_empirical_intensity;
        }
        // Trash cap/cost are fixed at build time (cap = INF, cost = SimpleTrashEdge.cost);
        // touching them here would force a warm-restart cold fallback whenever lemon_total_flow
        // changes between solves. Flow on the trash arc is already bounded by source supply.
        node_supply_map[lemon_graph.nodeFromId(0)] = lemon_total_flow;
        node_supply_map[lemon_graph.nodeFromId(1)] = -lemon_total_flow;
        if (_use_slopedp()) {
            _slope_dp_solve(lemon_total_flow);
            ++_solution_version;
            return;
        }
        _run_solver();
    }

  private:
    // ---- slope-DP chain solver ------------------------------------------- //
    void _sdp_build_cache() const {
        const size_t K = _chain_topo.has_value() ? _chain_topo->order.size() : 0;
        // Node-id -> chain position (or, without chain topo, each node is its
        // own singleton "position" in node-id order).
        std::vector<size_t> node_pos(nodes.size(), std::numeric_limits<size_t>::max());
        size_t P;
        if (_chain_topo.has_value()) {
            P = K;
            for (size_t i = 0; i < K; ++i) node_pos[_chain_topo->order[i]] = i;
        } else {
            // isolated peaks (no chain edges): source/sink are ids 0/1
            P = 0;
            for (size_t nid = 2; nid < nodes.size(); ++nid) node_pos[nid] = P++;
        }
        _sdp_src_arc.assign(P, -1);
        _sdp_sink_arc.assign(P, -1);
        _sdp_et_arc.assign(P, -1);
        _sdp_tt_arc.assign(P, -1);
        _sdp_emp_cap.assign(P, 0);
        _sdp_c_exp = _sdp_c_theo = _sdp_c_s = -1;
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii) {
            const auto& et = edges[ii].get_type();
            std::visit([&](const auto& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) {
                    size_t p = node_pos[edges[ii].get_end_node_id()];
                    _sdp_src_arc[p] = ii;
                    _sdp_emp_cap[p] = capacities_map[lemon_graph.arcFromId(ii)];
                } else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) {
                    _sdp_sink_arc[node_pos[edges[ii].get_start_node_id()]] = ii;
                } else if constexpr (std::is_same_v<T, EmpiricalTrashEdge>) {
                    _sdp_et_arc[node_pos[edges[ii].get_start_node_id()]] = ii;
                    _sdp_c_exp = costs_map[lemon_graph.arcFromId(ii)];
                } else if constexpr (std::is_same_v<T, TheoreticalTrashEdge>) {
                    _sdp_tt_arc[node_pos[edges[ii].get_end_node_id()]] = ii;
                    _sdp_c_theo = costs_map[lemon_graph.arcFromId(ii)];
                } else if constexpr (std::is_same_v<T, SimpleTrashEdge>) {
                    _sdp_c_s = costs_map[lemon_graph.arcFromId(ii)];
                } else if constexpr (std::is_same_v<T, MatchingEdge>) {
                    throw std::runtime_error(
                        "SlopeDP backend supports only chain-factory subgraphs "
                        "(found a MatchingEdge; use the NetworkSimplex solver).");
                }
            }, et);
        }
        _sdp_pos.assign(P, 0);
        if (_chain_topo.has_value())
            for (size_t g = 0; g + 1 < P; ++g)
                _sdp_pos[g + 1] = _sdp_pos[g] + _chain_topo->gap_cost[g];
        // Cluster segmentation: co-located positions (zero-cost gaps) merge
        // into one DP node — ~3x fewer conv steps on profile grids, and all
        // intra-cluster gaps (the zero-cost tie surface) leave the DP.
        _sdp_cluster_start.clear();
        _sdp_cluster_start.push_back(0);
        for (size_t p = 1; p < P; ++p)
            if (_sdp_pos[p] != _sdp_pos[p - 1]) _sdp_cluster_start.push_back(p);
        _sdp_cluster_start.push_back(P);
        _sdp_cache_built = true;
    }

    void _slope_dp_solve(VALUE_TYPE F) const {
        using W = slopedp_detail::wide_t;
        if (!_sdp_cache_built) _sdp_build_cache();
        const size_t P = _sdp_emp_cap.size();
        const bool chained = _chain_topo.has_value();

        // per-position masses for this solve
        std::vector<VALUE_TYPE> e(P, 0), t(P, 0);
        VALUE_TYPE E = 0, T = 0;
        for (size_t p = 0; p < P; ++p) {
            e[p] = _sdp_emp_cap[p];
            E += e[p];
            if (_sdp_sink_arc[p] >= 0) {
                t[p] = capacities_map[lemon_graph.arcFromId(_sdp_sink_arc[p])];
                T += t[p];
            }
        }

        // trash channels (quantized prices; cap at M=0)
        struct Ch { VALUE_TYPE price; VALUE_TYPE cap0; int kind; };  // 0=exp,1=theo,2=simple
        std::vector<Ch> ch;
        if (!independent_trash) {
            if (_sdp_c_exp >= 0)  ch.push_back({_sdp_c_exp, E, 0});
            if (_sdp_c_theo >= 0) ch.push_back({_sdp_c_theo, T, 1});
            if (_sdp_c_s >= 0)    ch.push_back({_sdp_c_s, std::numeric_limits<VALUE_TYPE>::max(), 2});
            std::sort(ch.begin(), ch.end(), [](const Ch& a, const Ch& b) { return a.price < b.price; });
        }
        // No trash channel at all: the DP prices matching against the trash
        // alternative (finite tau), so it cannot represent the forced
        // full-matching of a trash-less balanced chain — a configuration
        // NetworkSimplex handles fine.  Say so, instead of the generic
        // "infeasible trash configuration" the trash_of() bookkeeping would
        // otherwise report for a perfectly feasible problem.
        if (!independent_trash && ch.empty() && F > 0)
            throw std::runtime_error(
                "SlopeDP backend requires trash edges "
                "(add_simple_trash/add_experimental_trash/"
                "add_theoretical_trash before build()); "
                "use the NetworkSimplex solver for trash-less chain networks.");
        // Independent trash: every unmatched empirical unit pays C_exp and
        // every unfilled theoretical unit pays C_theo, charged independently —
        // trash_of(M) = C_exp*(E-M) + C_theo*(T-M), exactly affine in M, so
        // the DP's constant marginal tau = C_exp + C_theo is exact (the
        // annihilating channel greedy below is affine too, but only piecewise
        // per price ordering; here there is no channel coupling at all).
        auto trash_of = [&](VALUE_TYPE M) -> W {
            if (independent_trash) {
                if (M > E || M > T)
                    throw std::runtime_error("SlopeDP: matched mass exceeds supply.");
                return (W)_ind_c_exp_q * (E - M) + (W)_ind_c_theo_q * (T - M);
            }
            VALUE_TYPE need = F - M;
            if (need < 0) throw std::runtime_error("SlopeDP: matched mass exceeds supply.");
            W c = 0;
            for (const auto& x : ch) {
                if (need <= 0) break;
                VALUE_TYPE cap = x.kind == 2 ? need : x.cap0 - M;
                VALUE_TYPE take = std::min(need, std::max<VALUE_TYPE>(cap, 0));
                c += (W)x.price * take; need -= take;
            }
            if (need > 0) throw std::runtime_error("SlopeDP: infeasible trash configuration.");
            return c;
        };
        const W trash0 = trash_of(0);
        const VALUE_TYPE tau = std::min(E, T) >= 1 ? (VALUE_TYPE)(trash0 - trash_of(1)) : 0;

        std::vector<VALUE_TYPE> u(P, 0), y(P, 0), s(P > 0 ? P - 1 : 0, 0);
        W var_total = 0;
        if (tau > 0 && chained) {
            // DP over co-located CLUSTERS (zero-cost gaps merged): ~3x fewer
            // conv steps on shared-grid profile data, and no zero-gap ties.
            const auto& cs = _sdp_cluster_start;
            const size_t C = cs.size() - 1;
            std::vector<VALUE_TYPE> ce(C, 0), ct(C, 0);
            for (size_t c = 0; c < C; ++c)
                for (size_t p = cs[c]; p < cs[c + 1]; ++p) { ce[c] += e[p]; ct[c] += t[p]; }

            slopedp_detail::ConvexPL<VALUE_TYPE> DP;
            const VALUE_TYPE WALLM = std::numeric_limits<VALUE_TYPE>::max() / 1024;
            DP.insL(0, WALLM);
            DP.insR(0, WALLM);
            std::vector<VALUE_TYPE> lo(C), hi(C), vv(C);
            for (size_t c = 0; c < C; ++c) {
                vv[c] = ce[c] - ct[c];
                auto [l, h] = DP.convV(tau, vv[c], ct[c]);
                lo[c] = l; hi[c] = h;
                DP.minval += -(W)tau * ct[c];
                if (c + 1 < C) DP.addAbs(0, _sdp_pos[cs[c + 1]] - _sdp_pos[cs[c]]);
            }
            var_total = DP.evalAt(0);
            // backward recovery of inter-cluster flows and cluster totals
            std::vector<VALUE_TYPE> uc(C), yc(C), sc(C > 0 ? C - 1 : 0, 0);
            VALUE_TYPE s_cur = 0;
            for (size_t cr = C; cr-- > 0; ) {
                VALUE_TYPE want = s_cur - vv[cr];
                VALUE_TYPE s_prev = std::min(std::max(want, lo[cr]), hi[cr]);
                if (s_cur - s_prev < -ct[cr]) s_prev = s_cur + ct[cr];  // absorb wall
                if (s_cur - s_prev > ce[cr])  s_prev = s_cur - ce[cr];  // phantom tie-break
                if (cr == 0) s_prev = 0;
                VALUE_TYPE a_c = s_cur - s_prev;
                VALUE_TYPE yy = std::max<VALUE_TYPE>(std::min(ct[cr], ce[cr] - a_c), 0);
                VALUE_TYPE uu = std::min(std::max<VALUE_TYPE>(a_c + yy, 0), ce[cr]);
                yc[cr] = yy; uc[cr] = uu;
                if (cr > 0) sc[cr - 1] = s_prev;
                s_cur = s_prev;
            }
            // expand: distribute cluster totals to members (greedy in chain
            // order; any split is optimal — intra-cluster arcs cost 0) and
            // derive intra-cluster gap flows by prefix balance.
            for (size_t c = 0; c < C; ++c) {
                VALUE_TYPE uleft = uc[c], yleft = yc[c];
                VALUE_TYPE run = c > 0 ? sc[c - 1] : 0;   // flow entering from the left
                for (size_t p = cs[c]; p < cs[c + 1]; ++p) {
                    u[p] = std::min(e[p], uleft); uleft -= u[p];
                    y[p] = std::min(t[p], yleft); yleft -= y[p];
                    if (p + 1 < cs[c + 1]) {              // intra-cluster gap
                        run += u[p] - y[p];
                        s[p] = run;
                    }
                }
                if (c + 1 < C) s[cs[c + 1] - 1] = sc[c];  // inter-cluster gap
            }
        }
        // matched mass and transport
        VALUE_TYPE M = 0;
        W transport = 0;
        for (size_t p = 0; p < P; ++p) M += y[p];
        for (size_t g = 0; g + 1 < P; ++g) {
            VALUE_TYPE gap = _sdp_pos[g + 1] - _sdp_pos[g];
            transport += (W)(s[g] >= 0 ? s[g] : -s[g]) * gap;
        }
        const W total = transport + trash_of(M);
        if (tau > 0 && chained) {
            const W total_dp = var_total + trash0;
            if (total != total_dp)
                throw std::runtime_error("SlopeDP: flow recovery inconsistent with DP cost.");
        }

        // channel amounts for M, then per-node distribution (greedy in order)
        VALUE_TYPE D = 0, PH = 0, SI = 0;
        if (independent_trash) {
            // All excess is disposed on its own side, no budget sharing.
            D = E - M;
            PH = T - M;
        } else {
            VALUE_TYPE need = F - M;
            for (const auto& x : ch) {
                if (need <= 0) break;
                VALUE_TYPE cap = x.kind == 2 ? need : x.cap0 - M;
                VALUE_TYPE take = std::min(need, std::max<VALUE_TYPE>(cap, 0));
                if (x.kind == 0) D = take; else if (x.kind == 1) PH = take; else SI = take;
                need -= take;
            }
        }

        // per-arc flows
        _slope_flow.assign(edges.size(), 0);
        VALUE_TYPE Dleft = D, Pleft = PH;
        for (size_t p = 0; p < P; ++p) {
            if (_sdp_src_arc[p] >= 0) {
                VALUE_TYPE d = std::min(Dleft, e[p] - u[p]);
                if (_sdp_et_arc[p] < 0) d = 0;             // no trash arc here
                Dleft -= d;
                _slope_flow[_sdp_src_arc[p]] = u[p] + d;
                if (_sdp_et_arc[p] >= 0) _slope_flow[_sdp_et_arc[p]] = d;
            }
            if (_sdp_sink_arc[p] >= 0) {
                VALUE_TYPE ph = std::min(Pleft, t[p] - y[p]);
                if (_sdp_tt_arc[p] < 0) ph = 0;
                Pleft -= ph;
                _slope_flow[_sdp_sink_arc[p]] = y[p] + ph;
                if (_sdp_tt_arc[p] >= 0) _slope_flow[_sdp_tt_arc[p]] = ph;
            }
        }
        if (Dleft != 0 || Pleft != 0)
            throw std::runtime_error("SlopeDP: trash distribution failed.");
        if (simple_trash_added && simple_trash_idx != std::numeric_limits<LEMON_INDEX>::max())
            _slope_flow[simple_trash_idx] = SI;
        if (chained) {
            const auto& topo = *_chain_topo;
            for (size_t g = 0; g + 1 < P; ++g) {
                _slope_flow[topo.right_arc_ids[g]] = s[g] > 0 ?  s[g] : 0;
                _slope_flow[topo.left_arc_ids[g]]  = s[g] < 0 ? -s[g] : 0;
            }
        }
        // Independent trash reports the SHIFTED total (transport - tau*M), the
        // same convention as the dense factory's shifted matching costs:
        // total_cost() then adds the analytic bracket E*C_exp + T*C_theo.
        _slope_total = independent_trash
            ? (VALUE_TYPE)(transport - (W)tau * M)
            : (VALUE_TYPE)total;
        _slope_solved = true;
    }

  public:

    VALUE_TYPE total_cost() const {
        if(!_solver_has_value()) throw std::runtime_error("You must call build() and set_point() before calling total_cost().");
        VALUE_TYPE cost = _solver_total_cost();
        // Independent trash: the solver ran on shifted matching costs with
        // free trash arcs; add back the analytic bracket E*C_exp + T(w)*C_theo
        // (both totals are this subgraph's own, so decomposition stays exact).
        if (independent_trash)
            cost += _ind_c_exp_q * lemon_empirical_intensity
                    + _ind_c_theo_q * lemon_theoretical_intensity;
        return cost;
    };

    int warm_start_count() const {
        if (_use_lct())
            return ns_lct_solver.has_value() ? ns_lct_solver->warmStartCount() : 0;
        return ns_solver.has_value() ? ns_solver->warmStartCount() : 0;
    }
    int cold_start_count() const {
        if (_use_lct())
            return _cold_starts_via_run +
                   (ns_lct_solver.has_value() ? ns_lct_solver->coldStartCount() : 0);
        return _cold_starts_via_run +
               (ns_solver.has_value() ? ns_solver->coldStartCount() : 0);
    }
    int dual_repair_count() const {
        if (_use_lct())
            return ns_lct_solver.has_value() ? ns_lct_solver->dualRepairCount() : 0;
        return ns_solver.has_value() ? ns_solver->dualRepairCount() : 0;
    }
    int primal_repair_count() const {
        if (_use_lct())
            return ns_lct_solver.has_value() ? ns_lct_solver->primalRepairCount() : 0;
        return ns_solver.has_value() ? ns_solver->primalRepairCount() : 0;
    }


    std::string to_string() const {
        std::string result;
        result += "FlowSubgraph:\n";
        result += "Nodes:\n";
        for (const auto& node : nodes) {
            result += node.to_string() + "\n";
        }
        result += "Edges:\n";
        for (int ii = 0; ii < lemon_graph.arcNum(); ++ii) {
            result += "Edge " + std::to_string(lemon_graph.id(lemon_graph.arcFromId(ii))) + ": " +
                      std::to_string(lemon_graph.id(lemon_graph.source(lemon_graph.arcFromId(ii)))) + " -> " +
                      std::to_string(lemon_graph.id(lemon_graph.target(lemon_graph.arcFromId(ii)))) + " cost: " +
                      std::to_string(costs_map[lemon_graph.arcFromId(ii)]) + " capacity: " +
                      std::to_string(capacities_map[lemon_graph.arcFromId(ii)]) + " flow: " +
                      (_solver_has_value() ?
                      std::to_string(_solver_flow(lemon_graph.arcFromId(ii))) + "\n" :  "not yet computed\n");
        }
        return result;
    };

    std::string lemon_to_string() const {
        std::string result;
        result += "Lemon graph:\n";
        result += "Nodes:\n";
        for (int ii = 0; ii < lemon_graph.nodeNum(); ++ii) {
            result += "Node " + std::to_string(lemon_graph.id(lemon_graph.nodeFromId(ii))) + " supply: " +
                      std::to_string(node_supply_map[lemon_graph.nodeFromId(ii)]) + "\n";
        }
        result += "Edges:\n";
        for (int ii = 0; ii < lemon_graph.arcNum(); ++ii) {
            result += "Edge " + std::to_string(lemon_graph.id(lemon_graph.arcFromId(ii))) + ": " +
                      std::to_string(lemon_graph.id(lemon_graph.source(lemon_graph.arcFromId(ii)))) + " -> " +
                      std::to_string(lemon_graph.id(lemon_graph.target(lemon_graph.arcFromId(ii)))) + " cost: " +
                      std::to_string(costs_map[lemon_graph.arcFromId(ii)]) + " capacity: " +
                      std::to_string(capacities_map[lemon_graph.arcFromId(ii)]) + " flow: " +
                      (_solver_has_value() ?
                      std::to_string(_solver_flow(lemon_graph.arcFromId(ii))) + "\n" :  "not yet computed\n");
        }
        return result;
    };

    size_t no_nodes() const {
        return nodes.size();
    };

    size_t no_edges() const {
        return edges.size();
    };

    const std::vector<FlowNode<intensity_type>>& get_nodes() const {
        return nodes;
    };

    const std::vector<FlowEdge<intensity_type>>& get_edges() const {
        return edges;
    };

    std::vector<VALUE_TYPE>& costs_scratch() { return _costs_buf; }
    std::vector<double>& chain_pos_scratch() { return _chain_pos_buf; }

    void flows_for_target(size_t spectrum_id,
                            std::vector<LEMON_INDEX>& empirical_peak_indices,
                            std::vector<LEMON_INDEX>& theoretical_peak_indices,
                            std::vector<VALUE_TYPE>& flows) const
    {
        for (const auto& e : _matching_edge_cache) {
            if (e.spectrum_id != spectrum_id) continue;
            const VALUE_TYPE flow = _solver_flow(e.arc);
            if (flow == 0) continue;
            empirical_peak_indices.push_back(e.emp_peak_index);
            theoretical_peak_indices.push_back(e.theo_peak_index);
            flows.push_back(flow);
        }
        if (has_chain_edges())
            flows_for_target_chain(
                spectrum_id, empirical_peak_indices,
                theoretical_peak_indices, flows);
    };

    // Sweep-line reconstruction of per-(empirical, theoretical) flows from
    // chain-arc flows. Each chain-edge subgraph holds one linear chain of
    // empirical and theoretical nodes connected by bidirectional ChainEdges.
    // Flow on those arcs encodes transport between pairs without recording
    // which empirical unit ended up at which theoretical. We recover one
    // valid FIFO decomposition in two passes: left-to-right for rightward
    // net flow, right-to-left for leftward net flow. In canonical min-cost
    // solutions, at most one direction carries flow on any positive-cost
    // gap; on zero-cost gaps both directions may be non-zero, which the
    // two-pass split still handles correctly.
    void flows_for_target_chain(
        size_t spectrum_id,
        std::vector<LEMON_INDEX>& empirical_peak_indices,
        std::vector<LEMON_INDEX>& theoretical_peak_indices,
        std::vector<VALUE_TYPE>& flows) const
    {
        if (!_chain_topo.has_value()) return;
        const auto& topo = *_chain_topo;
        const size_t K = topo.order.size();
        if (K < 2) return;  // Isolated node — no gap flow to decompose.

        // Read per-gap forward/reverse flows from the solver.
        std::vector<VALUE_TYPE> R(K - 1), L(K - 1);
        for (size_t g = 0; g < K - 1; ++g) {
            R[g] = _solver_flow(lemon_graph.arcFromId(topo.right_arc_ids[g]));
            L[g] = _solver_flow(lemon_graph.arcFromId(topo.left_arc_ids[g]));
        }

        // Pass 1 — rightward decomposition.
        // delta = R[i] - R[i-1] is the change in the rightward conveyor
        // across node i. delta > 0 means this (empirical) node injects flow
        // onto the conveyor; delta < 0 means this (theoretical) node drains
        // it. In a canonical min-cost flow the sign of delta at a node
        // matches that node's role.
        {
            std::deque<std::pair<LEMON_INDEX, VALUE_TYPE>> queue;
            for (size_t i = 0; i < K; ++i) {
                const VALUE_TYPE r_in = (i == 0) ? 0 : R[i - 1];
                const VALUE_TYPE r_out = (i == K - 1) ? 0 : R[i];
                const VALUE_TYPE delta = r_out - r_in;
                const auto& node_type = nodes[topo.order[i]].get_type();
                if (delta > 0) {
                    const auto* emp = std::get_if<EmpiricalNode<intensity_type>>(&node_type);
                    if (emp == nullptr) continue;
                    queue.push_back({emp->get_peak_index(), delta});
                } else if (delta < 0) {
                    const auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&node_type);
                    if (theo == nullptr) continue;
                    const bool is_target = (theo->get_spectrum_id() == spectrum_id);
                    VALUE_TYPE remaining = -delta;
                    while (remaining > 0 && !queue.empty()) {
                        auto& front = queue.front();
                        const VALUE_TYPE take = std::min(remaining, front.second);
                        if (is_target) {
                            empirical_peak_indices.push_back(front.first);
                            theoretical_peak_indices.push_back(theo->get_peak_index());
                            flows.push_back(take);
                        }
                        remaining -= take;
                        front.second -= take;
                        if (front.second == 0) queue.pop_front();
                    }
                }
            }
        }

        // Pass 2 — leftward decomposition (mirror of pass 1, walking R→L).
        {
            std::deque<std::pair<LEMON_INDEX, VALUE_TYPE>> queue;
            for (size_t ii = 0; ii < K; ++ii) {
                const size_t i = K - 1 - ii;
                const VALUE_TYPE l_in = (i == K - 1) ? 0 : L[i];
                const VALUE_TYPE l_out = (i == 0) ? 0 : L[i - 1];
                const VALUE_TYPE delta = l_out - l_in;
                const auto& node_type = nodes[topo.order[i]].get_type();
                if (delta > 0) {
                    const auto* emp = std::get_if<EmpiricalNode<intensity_type>>(&node_type);
                    if (emp == nullptr) continue;
                    queue.push_back({emp->get_peak_index(), delta});
                } else if (delta < 0) {
                    const auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&node_type);
                    if (theo == nullptr) continue;
                    const bool is_target = (theo->get_spectrum_id() == spectrum_id);
                    VALUE_TYPE remaining = -delta;
                    while (remaining > 0 && !queue.empty()) {
                        auto& front = queue.front();
                        const VALUE_TYPE take = std::min(remaining, front.second);
                        if (is_target) {
                            empirical_peak_indices.push_back(front.first);
                            theoretical_peak_indices.push_back(theo->get_peak_index());
                            flows.push_back(take);
                        }
                        remaining -= take;
                        front.second -= take;
                        if (front.second == 0) queue.pop_front();
                    }
                }
            }
        }
    };

    std::unordered_map<LEMON_INDEX, VALUE_TYPE> get_flow_map() const {
        std::unordered_map<LEMON_INDEX, VALUE_TYPE> result;
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii)
        {
            const FlowEdge<intensity_type>& edge = edges[ii];
            const VALUE_TYPE flow = _solver_flow(lemon_graph.arcFromId(ii));
            if (flow == 0) continue;
            result[edge.get_id()] = flow;
        }
        return result;
    }

    template<typename T>
    size_t count_nodes_of_type() const {
        size_t result = 0;
        for (const auto& node : nodes)
            if(std::holds_alternative<T>(node.get_type()))
                result++;
        return result;
    }

    template<typename T>
    size_t count_edges_of_type() const {
        size_t result = 0;
        for (const auto& edge : edges)
            if(std::holds_alternative<T>(edge.get_type()))
                result++;
        return result;
    }

    double matching_density() const {
        const double nominator = count_edges_of_type<MatchingEdge>() + count_edges_of_type<ChainEdge>() / 2.0;
        const double denominator = count_nodes_of_type<EmpiricalNode<intensity_type>>() * count_nodes_of_type<TheoreticalNode<intensity_type>>();
        if (denominator == 0) return std::numeric_limits<double>::quiet_NaN();
        return nominator / denominator;
    }

    std::vector<size_t> theoretical_spectra_involved() const {
        std::unique_ptr<bool[]> involved = std::make_unique<bool[]>(no_target_distributions);
        std::fill(involved.get(), involved.get() + no_target_distributions, false);
        for (const auto& node : nodes)
        {
            if (auto node_type = std::get_if<TheoreticalNode<intensity_type>>(&node.get_type()))
            {
                const auto& theoretical_node = *node_type;
                involved[theoretical_node.get_spectrum_id()] = true;
            }
        }
        std::vector<size_t> result;
        for (size_t ii = 0; ii < no_target_distributions; ++ii)
            if(involved[ii])
                result.push_back(ii);
        return result;
    }

    bool is_solved() const {
        return _solver_has_value();
    }

    // Chain-specialized residual shortest distances. Linear sweep variant of
    // bellman_ford_residual for chain subgraphs: rather than relaxing every
    // arc O(n) times, we propagate along the chain (L→R and R→L sweeps)
    // interleaved with src/sink relays. Each round is O(K); the loop exits
    // once the distance vector stops changing, typically after 2–3 rounds.
    // The residual graph of an optimal MCF has no negative cycles, so the
    // fixpoint is well-defined; the loop is capped at K+2 rounds as a safety
    // net (matching the Bellman-Ford bound) but real inputs exit much sooner.
    //
    // Requires: at least one ChainEdge present (the caller is responsible).
    // Fills _chain_R_buf, _chain_L_buf, and all flag/cost scratch arrays from
    // the current solved flow. Must be called before _chain_run_search().
    // Static per-position arc cache for the chain search state: which arcs
    // touch each chain position, their fixed caps and (scaled) trash costs.
    // Everything here is invariant across set_point() solves — only flows and
    // the theo->sink caps change — so it is built once per build().
    // apply_new_costs() only touches Matching/Chain edge costs, so the cached
    // trash costs stay valid.
    mutable bool _chain_arc_cache_built = false;
    mutable std::vector<LEMON_INDEX> _chain_src_arc, _chain_sink_arc,
                                     _chain_et_arc, _chain_tt_arc;   // -1 = absent
    mutable std::vector<VALUE_TYPE> _chain_src_cap;                  // fixed emp caps
    // Independent trash (SlopeDP chain): the residual search runs on the
    // shifted problem, where src arcs are matched-mass carriers and cost
    // -s forward / +s reverse (s = C_exp + C_theo).  0 otherwise.
    mutable VALUE_TYPE _chain_src_shift = 0;

    void _chain_build_arc_cache() const {
        const auto& topo = *_chain_topo;
        const size_t K = topo.order.size();
        const VALUE_TYPE INF = std::numeric_limits<VALUE_TYPE>::max();
        _chain_src_arc.assign(K, -1);
        _chain_sink_arc.assign(K, -1);
        _chain_et_arc.assign(K, -1);
        _chain_tt_arc.assign(K, -1);
        _chain_src_cap.assign(K, 0);
        _chain_src_shift = independent_trash
            ? (VALUE_TYPE)(_ind_c_exp_q + _ind_c_theo_q) : (VALUE_TYPE)0;
        // Static parts of the search state: trash costs (scaled — see below)
        // and the trash forward flags (trash arc caps are INF, so residual
        // forward capacity is a fixed property of arc existence).
        std::fill(_chain_exp_trash_cost.begin(),  _chain_exp_trash_cost.end(),  INF);
        std::fill(_chain_theo_trash_cost.begin(), _chain_theo_trash_cost.end(), INF);
        std::fill(_chain_exp_trash_fwd.begin(),   _chain_exp_trash_fwd.end(),   uint8_t(0));
        std::fill(_chain_theo_trash_fwd.begin(),  _chain_theo_trash_fwd.end(),  uint8_t(0));
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii) {
            const auto& et = edges[ii].get_type();
            if (std::holds_alternative<SrcToEmpiricalEdge>(et)) {
                const size_t pos = topo.node_to_pos[edges[ii].get_end_node_id()];
                if (pos == std::numeric_limits<size_t>::max()) continue;
                _chain_src_arc[pos] = ii;
                _chain_src_cap[pos] = capacities_map[lemon_graph.arcFromId(ii)];
            } else if (std::holds_alternative<TheoreticalToSinkEdge>(et)) {
                const size_t pos = topo.node_to_pos[edges[ii].get_start_node_id()];
                if (pos == std::numeric_limits<size_t>::max()) continue;
                _chain_sink_arc[pos] = ii;
            } else if (std::holds_alternative<EmpiricalTrashEdge>(et)) {
                const size_t pos = topo.node_to_pos[edges[ii].get_start_node_id()];
                if (pos == std::numeric_limits<size_t>::max()) continue;
                _chain_et_arc[pos] = ii;
                // Use the SCALED cost (costs_map), not the raw get_cost(): the
                // chain gap costs (topo.gap_cost) come from costs_map, so the
                // trash costs must share those units.  With cost scaling on they
                // differ by the cost scale; reading get_cost() here made the
                // trash term negligible against the scaled matching costs and
                // dropped the trash-reclaim from the residual marginal.
                _chain_exp_trash_cost[pos] = costs_map[lemon_graph.arcFromId(ii)];
                _chain_exp_trash_fwd[pos] = true;   // cap INF: always residual fwd
            } else if (std::holds_alternative<TheoreticalTrashEdge>(et)) {
                const size_t pos = topo.node_to_pos[edges[ii].get_end_node_id()];
                if (pos == std::numeric_limits<size_t>::max()) continue;
                _chain_tt_arc[pos] = ii;
                // Scaled cost (see EmpiricalTrashEdge branch above).
                _chain_theo_trash_cost[pos] = costs_map[lemon_graph.arcFromId(ii)];
                _chain_theo_trash_fwd[pos] = true;
            }
        }
        _chain_arc_cache_built = true;
    }

    void _chain_build_search_state() const {
        const auto& topo = *_chain_topo;
        const size_t K = topo.order.size();
        if (!_chain_arc_cache_built) _chain_build_arc_cache();

        // Hoist the backend dispatch out of the per-arc loops: _solver_flow()
        // re-checks the config variant on every call, which dominates when
        // called ~4K times per gradient evaluation.
        const std::vector<VALUE_TYPE>* sdp_flow =
            _use_slopedp() ? &_slope_flow : nullptr;
        auto flow_of = [&](LEMON_INDEX ii) -> VALUE_TYPE {
            return sdp_flow ? (*sdp_flow)[ii]
                            : _solver_flow(lemon_graph.arcFromId(ii));
        };

        auto& c_right = _chain_R_buf;
        auto& c_left  = _chain_L_buf;
        for (size_t g = 0; g + 1 < K; ++g) {
            const VALUE_TYPE R = flow_of(topo.right_arc_ids[g]);
            const VALUE_TYPE L = flow_of(topo.left_arc_ids[g]);
            c_right[g] = (L > 0) ? -topo.gap_cost[g] : topo.gap_cost[g];
            c_left[g]  = (R > 0) ? -topo.gap_cost[g] : topo.gap_cost[g];
        }

        // Dynamic flags only (flows and theo->sink caps change per solve);
        // trash costs and trash fwd flags are static, filled by the cache.
        for (size_t pos = 0; pos < K; ++pos) {
            LEMON_INDEX ii;
            if ((ii = _chain_src_arc[pos]) >= 0) {
                const VALUE_TYPE flow = flow_of(ii);
                _chain_has_src_fwd[pos] = flow < _chain_src_cap[pos];
                _chain_has_src_rev[pos] = flow > 0;
            } else {
                _chain_has_src_fwd[pos] = _chain_has_src_rev[pos] = false;
            }
            if ((ii = _chain_sink_arc[pos]) >= 0) {
                const VALUE_TYPE flow = flow_of(ii);
                _chain_has_sink_fwd[pos] = flow < capacities_map[lemon_graph.arcFromId(ii)];
                _chain_has_sink_rev[pos] = flow > 0;
            } else {
                _chain_has_sink_fwd[pos] = _chain_has_sink_rev[pos] = false;
            }
            if ((ii = _chain_et_arc[pos]) >= 0)
                _chain_exp_trash_rev[pos] = flow_of(ii) > 0;
            else
                _chain_exp_trash_rev[pos] = false;
            if ((ii = _chain_tt_arc[pos]) >= 0)
                _chain_theo_trash_rev[pos] = flow_of(ii) > 0;
            else
                _chain_theo_trash_rev[pos] = false;
        }
    }

    // Runs the relay+sweep search from source_id using pre-filled scratch state.
    // _chain_build_search_state() must have been called first.
    std::vector<VALUE_TYPE> _chain_run_search(LEMON_INDEX source_id) const {
        const auto& topo = *_chain_topo;
        const size_t K = topo.order.size();
        const VALUE_TYPE INF = std::numeric_limits<VALUE_TYPE>::max();
        const LEMON_INDEX src_id = 0;
        const LEMON_INDEX sink_id = 1;

        auto& dist         = _chain_dist_buf;
        auto& c_right      = _chain_R_buf;
        auto& c_left       = _chain_L_buf;
        auto& has_src_fwd  = _chain_has_src_fwd;   auto& has_src_rev  = _chain_has_src_rev;
        auto& has_sink_fwd = _chain_has_sink_fwd;  auto& has_sink_rev = _chain_has_sink_rev;
        auto& exp_trash_cost  = _chain_exp_trash_cost; auto& theo_trash_cost = _chain_theo_trash_cost;
        auto& exp_trash_fwd   = _chain_exp_trash_fwd;  auto& exp_trash_rev  = _chain_exp_trash_rev;
        auto& theo_trash_fwd  = _chain_theo_trash_fwd; auto& theo_trash_rev = _chain_theo_trash_rev;

        std::fill(dist.begin(), dist.end(), INF);
        dist[source_id] = 0;

        bool changed = false;
        auto update_min = [&](VALUE_TYPE& a, VALUE_TYPE b) { if (b < a) { a = b; changed = true; } };

        auto relay = [&]() {
            for (size_t i = 0; i < K; ++i) {
                const VALUE_TYPE d = dist[topo.order[i]];
                if (d == INF) continue;
                // Reverse SrcToEmpiricalEdge (+shift under independent trash,
                // else cost 0); forward TheoreticalToSinkEdge (cost 0).
                if (has_src_rev[i])  update_min(dist[src_id],  d + _chain_src_shift);
                if (has_sink_fwd[i]) update_min(dist[sink_id], d);
                // Forward EmpiricalTrashEdge (Emp→Sink, cost +C_exp).
                if (exp_trash_fwd[i]) update_min(dist[sink_id], d + exp_trash_cost[i]);
                // Reverse TheoreticalTrashEdge (Theo→Source, cost -C_theo).
                if (theo_trash_rev[i]) update_min(dist[src_id], d - theo_trash_cost[i]);
            }
            const VALUE_TYPE ds = dist[src_id], dk = dist[sink_id];
            for (size_t i = 0; i < K; ++i) {
                // Forward SrcToEmpiricalEdge (-shift under independent trash,
                // else cost 0); reverse TheoreticalToSinkEdge (cost 0).
                if (ds != INF && has_src_fwd[i])  update_min(dist[topo.order[i]], ds - _chain_src_shift);
                if (dk != INF && has_sink_rev[i]) update_min(dist[topo.order[i]], dk);
                // Forward TheoreticalTrashEdge (Source→Theo, cost +C_theo).
                if (ds != INF && theo_trash_fwd[i]) update_min(dist[topo.order[i]], ds + theo_trash_cost[i]);
                // Reverse EmpiricalTrashEdge (Sink→Emp, cost -C_exp).
                if (dk != INF && exp_trash_rev[i]) update_min(dist[topo.order[i]], dk - exp_trash_cost[i]);
            }
        };
        auto chain_sweep = [&]() {
            for (size_t i = 1; i < K; ++i) {
                const VALUE_TYPE d = dist[topo.order[i-1]];
                if (d != INF) update_min(dist[topo.order[i]], d + c_right[i-1]);
            }
            for (size_t ii = 1; ii < K; ++ii) {
                const size_t i = K - 1 - ii;
                const VALUE_TYPE d = dist[topo.order[i+1]];
                if (d != INF) update_min(dist[topo.order[i]], d + c_left[i]);
            }
        };

        const size_t MAX_ROUNDS = K + 2;
        for (size_t round = 0; round < MAX_ROUNDS; ++round) {
            changed = false;
            relay();
            chain_sweep();
            if (!changed) break;
        }
        return dist;
    }

    std::vector<VALUE_TYPE> chain_residual_distances(LEMON_INDEX source_id) const {
        if (!_chain_topo.has_value())
            throw std::runtime_error(
                "chain_residual_distances() called on non-chain subgraph.");
        _chain_build_search_state();
        return _chain_run_search(source_id);
    }

    bool has_chain_edges() const {
        return _chain_topo.has_value();
    }

    // Returns the node IDs of the chain in sorted position order, or an empty
    // vector if this subgraph has no chain topology.  Used by update_positions_and_solve
    // to validate that new positions don't change the chain's sorted order.
    const std::vector<LEMON_INDEX>& get_chain_order() const {
        static const std::vector<LEMON_INDEX> empty;
        return _chain_topo.has_value() ? _chain_topo->order : empty;
    }

    // Accumulate position gradients from this subgraph into caller-owned spans.
    // emp_grad is flat [N_emp * DIM] row-major; theo_grads[s] is [N_s * DIM] row-major.
    // Caller must zero both before the first call (multiple subgraphs accumulate additively).
    // Only MatchingEdge arcs with nonzero flow contribute.  Allocates nothing.
    template<typename Distribution_t, typename DistMetric>
    void accumulate_position_gradients(
        const Distribution_t* new_empirical,
        const std::vector<Distribution_t*>& new_theoretical,
        std::span<double> emp_grad,
        std::vector<std::span<double>>& theo_grads,
        double p = 1.0
    ) const {
        static constexpr size_t DIM = std::tuple_size_v<typename Distribution_t::Point_t>;
        for (const auto& e : _matching_edge_cache) {
            const VALUE_TYPE flow = _solver_flow(e.arc);
            if (flow == 0) continue;
            const auto emp_pt  = new_empirical->get_point(e.emp_peak_index);
            const auto theo_pt = new_theoretical[e.spectrum_id]->get_point(e.theo_peak_index);
            const auto g = DistMetric::grad_x(emp_pt, theo_pt);
            // Cost is d^p, so d(cost)/dx = p * d^(p-1) * grad_x(d).  p == 1 gives
            // factor 1 (bit-identical to the legacy W_1 gradient).  The gradient is
            // in REAL units (independent of the cost scale).
            double factor = 1.0;
            if (p != 1.0) {
                const double d = DistMetric::dist(emp_pt, theo_pt);
                factor = p * std::pow(d, p - 1.0);
            }
            for (size_t d = 0; d < DIM; ++d) {
                emp_grad[e.emp_peak_index * DIM + d]                   += static_cast<double>(flow) * factor * g[d];
                theo_grads[e.spectrum_id][e.theo_peak_index * DIM + d] -= static_cast<double>(flow) * factor * g[d];
            }
        }
    }

    // Accumulate position gradients for a chain (1D) subgraph.
    // Total cost = sum_g (R[g]+L[g])*gap_g.  Moving a peak at chain position k
    // changes gap_{k-1} by +delta and gap_k by -delta, so the gradient is
    // dir_sign*(left_total - right_total).  dir_sign is +1 for an ascending
    // chain (pos[0] < pos[1]) and -1 for descending.
    // DistMetric is accepted for API symmetry but unused (all 1D metrics agree).
    template<typename Distribution_t, typename DistMetric>
    void accumulate_position_gradients_chain(
        const Distribution_t* new_empirical,
        const std::vector<Distribution_t*>& new_theoretical,
        std::span<double> emp_grad,
        std::vector<std::span<double>>& theo_grads
    ) const {
        static constexpr size_t DIM = std::tuple_size_v<typename Distribution_t::Point_t>;
        static_assert(DIM == 1,
            "accumulate_position_gradients_chain requires 1D distributions");

        if (!_chain_topo.has_value()) return;
        const auto& topo = *_chain_topo;
        const size_t K = topo.order.size();
        if (K < 2) return;

        auto& R = _chain_R_buf;
        auto& L = _chain_L_buf;
        for (size_t g = 0; g < K - 1; ++g) {
            R[g] = _solver_flow(lemon_graph.arcFromId(topo.right_arc_ids[g]));
            L[g] = _solver_flow(lemon_graph.arcFromId(topo.left_arc_ids[g]));
        }

        auto get_node_pos = [&](LEMON_INDEX nid) -> double {
            const auto& ntype = nodes[nid].get_type();
            if (const auto* emp = std::get_if<EmpiricalNode<intensity_type>>(&ntype))
                return new_empirical->get_point(emp->get_peak_index())[0];
            if (const auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&ntype))
                return new_theoretical[theo->get_spectrum_id()]->get_point(theo->get_peak_index())[0];
            return 0.0;
        };
        // Orientation from the first strictly different position pair: on
        // shared-grid data the head pair is routinely co-located, so a bare
        // pos[1] > pos[0] test ties to descending and negates every gradient
        // in an ascending chain.  A fully co-located chain keeps +1 (all gaps
        // are 0, so either sign is a valid subgradient of |.| there).
        double dir_sign = 1.0;
        for (size_t k = 1; k < K; ++k) {
            const double diff =
                get_node_pos(topo.order[k]) - get_node_pos(topo.order[k - 1]);
            if (diff != 0.0) {
                dir_sign = (diff > 0.0) ? 1.0 : -1.0;
                break;
            }
        }

        for (size_t k = 0; k < K; ++k) {
            const VALUE_TYPE left_total  = (k > 0)     ? R[k-1] + L[k-1] : 0;
            const VALUE_TYPE right_total = (k < K - 1) ? R[k]   + L[k]   : 0;
            const double grad_val =
                dir_sign * static_cast<double>(left_total - right_total);

            const LEMON_INDEX nid = topo.order[k];
            const auto& ntype = nodes[nid].get_type();
            if (const auto* emp = std::get_if<EmpiricalNode<intensity_type>>(&ntype)) {
                emp_grad[emp->get_peak_index()] += grad_val;
            } else if (const auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&ntype)) {
                theo_grads[theo->get_spectrum_id()][theo->get_peak_index()] += grad_val;
            }
        }
    }

    // Update MatchingEdge and ChainEdge costs in the already-built LEMON graph,
    // then immediately re-run the solver (warm-restarting for NetworkSimplex).
    // new_costs_per_edge_idx[i] is the new cost for edge i; entries for other
    // edge types (SrcToEmpirical, TheoreticalToSink, trash, ...) are ignored.
    // Precondition: build() and at least one solve() must have been called.
    void apply_new_costs(const std::vector<VALUE_TYPE>& new_costs_per_edge_idx) {
        if (!built)
            throw std::runtime_error("apply_new_costs() must be called after build().");
        if (new_costs_per_edge_idx.size() != edges.size())
            throw std::runtime_error("apply_new_costs(): cost vector size mismatch.");
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii) {
            const auto& etype = edges[ii].get_type();
            if (std::holds_alternative<MatchingEdge>(etype) || std::holds_alternative<ChainEdge>(etype))
                costs_map[lemon_graph.arcFromId(ii)] = new_costs_per_edge_idx[ii];
        }
        // Re-sync gap_cost from costs_map so chain_residual_distances stays correct.
        if (_chain_topo.has_value()) {
            for (size_t g = 0; g < _chain_topo->right_arc_ids.size(); ++g)
                _chain_topo->gap_cost[g] = costs_map[lemon_graph.arcFromId(_chain_topo->right_arc_ids[g])];
        }
        if (_use_slopedp()) {
            // positions cache derives from gap_cost — rebuild it, then re-solve
            // with the supplies of the last set_point()
            _sdp_cache_built = false;
            _slope_dp_solve(node_supply_map[lemon_graph.nodeFromId(0)]);
            ++_solution_version;
            return;
        }
        _run_solver(/*costs_changed=*/true);
    }

    // Dijkstra variant of residual shortest-path using NetworkSimplex potentials.
    // After an optimal NS solve, reduced costs c_r(u,v) = c(u,v) + pi[u] - pi[v]
    // are >= 0 on every residual arc (complementary slackness), so Dijkstra applies.
    // O((V+E) log V) vs Bellman-Ford's O(V*E).
    // true_dist[v] = dijkstra_reduced_dist[v] + pi[v] - pi[source]
    //
    // Caveat — the "unlimited" matching arcs break that premise in one case:
    // the residual graph deliberately treats MatchingEdge arcs as traversable
    // even when saturated (their LEMON capacity, min of the endpoint
    // intensities, is an optimization, not a real bound — the derivative
    // semantics ask about *intensity* perturbations).  For the capped LP that
    // LEMON solved, a saturated arc may legally carry NEGATIVE reduced cost,
    // and which dual the solver lands on is basis-dependent: a cold solve and
    // a warm-restarted solve of the same optimum can disagree.  Clamping such
    // an arc's rcost to 0 silently understates residual distances (observed:
    // exact marginal -90 fresh vs -60 after warm point cycling on the same
    // flows).  Dijkstra is therefore only run after verifying every
    // saturated unlimited arc has nonnegative reduced cost; otherwise fall
    // back to Bellman-Ford, which uses true costs and no potentials.
    std::vector<VALUE_TYPE> dijkstra_residual(LEMON_INDEX source_id) const {
        const LEMON_INDEX n = lemon_graph.nodeNum();
        const VALUE_TYPE INF = std::numeric_limits<VALUE_TYPE>::max();

        std::vector<VALUE_TYPE> pi(n);
        for (LEMON_INDEX i = 0; i < n; ++i)
            pi[i] = _solver_potential(lemon_graph.nodeFromId(i));

        // Premise check (see the caveat above).  Only saturated unlimited
        // arcs can violate rc >= 0 on a traversable residual arc: reverse
        // arcs (flow > 0) have rc_rev = -rc >= 0 and unsaturated forward
        // arcs have rc >= 0 by optimality of the capped LP.
        for (LEMON_INDEX ii = 0; ii < lemon_graph.arcNum(); ++ii) {
            if (!_unlimited_arc[ii]) continue;
            const auto arc = lemon_graph.arcFromId(ii);
            if (_solver_flow(arc) < capacities_map[arc]) continue;
            const LEMON_INDEX u = lemon_graph.id(lemon_graph.source(arc));
            const LEMON_INDEX v = lemon_graph.id(lemon_graph.target(arc));
            if (costs_map[arc] + pi[u] - pi[v] < 0)
                return bellman_ford_residual(source_id);
        }

        std::vector<VALUE_TYPE> rdist(n, INF);
        rdist[source_id] = 0;
        using Entry = std::pair<VALUE_TYPE, LEMON_INDEX>;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;
        pq.emplace(0, source_id);

        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > rdist[u]) continue;
            const auto u_node = lemon_graph.nodeFromId(u);

            // Forward residual: outgoing arcs of u
            for (lemon::StaticDigraph::OutArcIt a(lemon_graph, u_node); a != lemon::INVALID; ++a) {
                const LEMON_INDEX ii = lemon_graph.id(a);
                if (ii == simple_trash_idx) continue;
                if (!_unlimited_arc[ii] && _solver_flow(a) >= capacities_map[a]) continue;
                const LEMON_INDEX v = lemon_graph.id(lemon_graph.target(a));
                const VALUE_TYPE rcost = std::max<VALUE_TYPE>(0, costs_map[a] + pi[u] - pi[v]);
                const VALUE_TYPE nd = d + rcost;
                if (nd < rdist[v]) { rdist[v] = nd; pq.emplace(nd, v); }
            }

            // Reverse residual: incoming arcs of u (arc goes v→u in original, u→v in residual)
            for (lemon::StaticDigraph::InArcIt a(lemon_graph, u_node); a != lemon::INVALID; ++a) {
                const LEMON_INDEX ii = lemon_graph.id(a);
                if (ii == simple_trash_idx) continue;
                if (_solver_flow(a) <= 0) continue;
                const LEMON_INDEX v = lemon_graph.id(lemon_graph.source(a));
                const VALUE_TYPE rcost = std::max<VALUE_TYPE>(0, -costs_map[a] + pi[u] - pi[v]);
                const VALUE_TYPE nd = d + rcost;
                if (nd < rdist[v]) { rdist[v] = nd; pq.emplace(nd, v); }
            }
        }

        // Convert reduced distances back to true distances
        std::vector<VALUE_TYPE> dist(n, INF);
        const VALUE_TYPE pi_src = pi[source_id];
        for (LEMON_INDEX i = 0; i < n; ++i)
            if (rdist[i] != INF)
                dist[i] = rdist[i] + pi[i] - pi_src;
        return dist;
    }

    // Compute single-source shortest distances on the implicit residual graph
    // (excluding the trash edge). For each arc:
    //   forward residual if flow < capacity (cost = original)
    //   reverse residual if flow > 0 (cost = -original)
    std::vector<VALUE_TYPE> bellman_ford_residual(LEMON_INDEX source_id) const {
        const LEMON_INDEX n = lemon_graph.nodeNum();
        const LEMON_INDEX m = lemon_graph.arcNum();
        const VALUE_TYPE INF = std::numeric_limits<VALUE_TYPE>::max();
        std::vector<VALUE_TYPE> dist(n, INF);
        dist[source_id] = 0;

        for (LEMON_INDEX iter = 0; iter < n - 1; ++iter) {
            bool changed = false;
            for (LEMON_INDEX ii = 0; ii < m; ++ii) {
                if (ii == simple_trash_idx) continue;
                auto arc = lemon_graph.arcFromId(ii);
                LEMON_INDEX u = lemon_graph.id(lemon_graph.source(arc));
                LEMON_INDEX v = lemon_graph.id(lemon_graph.target(arc));
                VALUE_TYPE cost = costs_map[arc];
                VALUE_TYPE cap = capacities_map[arc];
                VALUE_TYPE flow = _solver_flow(arc);
                // Matching edges have unlimited base capacity; the LEMON
                // capacity (min of endpoint intensities) is an optimization
                // that should not limit the residual graph. Chain edges
                // also represent unlimited-capacity residual transitions.
                bool unlimited = std::holds_alternative<MatchingEdge>(edges[ii].get_type())
                              || std::holds_alternative<ChainEdge>(edges[ii].get_type());

                if ((unlimited || flow < cap) && dist[u] != INF && dist[u] + cost < dist[v]) {
                    dist[v] = dist[u] + cost;
                    changed = true;
                }
                if (flow > 0 && dist[v] != INF && dist[v] - cost < dist[u]) {
                    dist[u] = dist[v] - cost;
                    changed = true;
                }
            }
            if (!changed) break;
        }
        return dist;
    }

    // Opt-in (GradientMode::DualPi) fast, approximate replacement for
    // dijkstra_residual(): the pure dual-potential difference
    //   dist[i] = pi[i] - pi[source_id].
    // This is exactly the residual-shortest-path distance with the (>= 0)
    // reduced-cost detour term dropped, so it is correct only for nodes on
    // the optimal flow support (reduced-cost-0 reachable) and a lower bound
    // elsewhere; it is also basis-dependent at degenerate optima.  O(n), no
    // search.  NOT value-equivalent to the residual marginal — only reachable
    // via the *_fast_approx() entry points.
    std::vector<VALUE_TYPE> pi_distances(LEMON_INDEX source_id) const {
        const LEMON_INDEX n = lemon_graph.nodeNum();
        std::vector<VALUE_TYPE> dist(n);
        const VALUE_TYPE pi_src =
            _solver_potential(lemon_graph.nodeFromId(source_id));
        for (LEMON_INDEX i = 0; i < n; ++i)
            dist[i] = _solver_potential(lemon_graph.nodeFromId(i)) - pi_src;
        return dist;
    }

    // Per-peak marginal cost of increasing each theoretical signal by 1.
    // Returns vector of (spectrum_id, peak_index, derivative).
    //
    // Simple trash: Bellman-Ford excludes the trash edge; src_adjust/sink_adjust
    // manually account for the Source↔Sink shortcut it provides.
    //
    // Asymmetric trash: full residual already contains the shortcuts (reverse
    // EmpiricalTrashEdge: Sink→Emp at -C_exp; forward TheoreticalTrashEdge:
    // Source→Theo at C_theo), so no adjustments are needed.
    //   E > T (supply fixed): augmenting cycle through T_node uses dist_sink.
    //   T >= E (supply +1):   extra Source unit routed to T_node uses dist_src.
    // (DerivContext is declared near the top of the class so it can be cached.)

    // want_pi=false: exact residual marginals.  want_pi=true: fast dual-pi
    // approximation (only when the NetworkSimplex solver is in use; chain /
    // non-NS subgraphs have no potentials so they ignore want_pi and stay
    // exact).
    DerivContext _make_deriv_context(bool want_pi) const {
        if (!_solver_has_value())
            throw std::runtime_error("Must call solve() before signal_part_derivatives().");
        if (simple_trash_idx == std::numeric_limits<LEMON_INDEX>::max()
                && !experimental_trash_added && !theoretical_trash_added)
            throw std::runtime_error("signal_part_derivatives() requires trash edges.");

        DerivContext ctx;
        ctx.INF = std::numeric_limits<VALUE_TYPE>::max();
        ctx.supply_fixed = (lemon_empirical_intensity > lemon_theoretical_intensity);
        ctx.asymmetric = experimental_trash_added || theoretical_trash_added;
        ctx.independent = independent_trash;
        const LEMON_INDEX sink_id = 1;
        const bool use_chain = has_chain_edges();
        const bool use_dijkstra = !use_chain &&
            (_use_lct() ? ns_lct_solver.has_value() : ns_solver.has_value());
        const bool need_src  = !ctx.asymmetric || !ctx.supply_fixed || ctx.independent;
        const bool need_sink = !ctx.asymmetric ||  ctx.supply_fixed || ctx.independent;
        if (use_chain) {
            _chain_build_search_state();
            if (need_src)  ctx.dist_src  = _chain_run_search(0);
            if (need_sink) ctx.dist_sink = _chain_run_search(sink_id);
        } else {
            const bool use_pi = want_pi && use_dijkstra;
            auto compute_dist = [&](LEMON_INDEX id) -> std::vector<VALUE_TYPE> {
                if (use_pi)       return pi_distances(id);
                return use_dijkstra ? dijkstra_residual(id)
                                    : bellman_ford_residual(id);
            };
            if (need_src)  ctx.dist_src  = compute_dist(0);
            if (need_sink) ctx.dist_sink = compute_dist(sink_id);
        }

        ctx.trash_cost = 0; ctx.src_adjust = 0; ctx.sink_adjust = 0;
        if (!ctx.asymmetric) {
            // Scaled trash cost (same units as the residual distances, which
            // come from the scaled costs_map).
            ctx.trash_cost  = costs_map[lemon_graph.arcFromId(simple_trash_idx)];
            ctx.src_adjust  = ctx.supply_fixed ? -ctx.trash_cost : 0;
            ctx.sink_adjust = ctx.supply_fixed ?  0 : ctx.trash_cost;
        }

        ctx.theo_sink_slack.assign(nodes.size(), VALUE_TYPE(-1));
        for (const auto& e : _theo_sink_edge_cache) {
            const LEMON_INDEX node_id = lemon_graph.id(lemon_graph.source(e.arc));
            ctx.theo_sink_slack[node_id] = capacities_map[e.arc] - _solver_flow(e.arc);
        }
        return ctx;
    }

    // Value-exact memoized accessor: rebuilds the context only when the
    // solution changed since the last build (_solution_version), otherwise
    // returns the cached one.  Eliminates the redundant residual recompute
    // when several derivative queries (spectrum + signal + repeated/identical
    // re-solves) hit the same solved solution.  Bit-identical to
    // _make_deriv_context() because the context is a pure function of the
    // post-solve solver state.
    const DerivContext& _get_deriv_context(bool use_pi) const {
        const int k = use_pi ? 1 : 0;
        if (_deriv_ctx_version[k] != _solution_version) {
            _deriv_ctx_cache[k] = _make_deriv_context(use_pi);
            _deriv_ctx_version[k] = _solution_version;
        }
        return _deriv_ctx_cache[k];
    }

    VALUE_TYPE _node_deriv(LEMON_INDEX node_id, const DerivContext& ctx) const {
        const VALUE_TYPE slack = ctx.theo_sink_slack[node_id];
        if (slack != VALUE_TYPE(-1) && slack > 0 && ctx.supply_fixed && !ctx.independent)
            return VALUE_TYPE(0);
        VALUE_TYPE deriv;
        if (ctx.independent) {
            // Shifted marginal: min over the source augmentation and the
            // residual cycle through the sink (see the DerivContext comment).
            deriv = std::min(ctx.dist_src[node_id], ctx.dist_sink[node_id]);
            if (deriv == ctx.INF) deriv = 0;
        } else if (ctx.asymmetric) {
            deriv = ctx.supply_fixed ? ctx.dist_sink[node_id] : ctx.dist_src[node_id];
            if (deriv == ctx.INF) deriv = 0;
        } else {
            deriv = ctx.trash_cost;
            if (!ctx.dist_src.empty() && ctx.dist_src[node_id] != ctx.INF)
                deriv = std::min(deriv, ctx.dist_src[node_id] + ctx.src_adjust);
            if (!ctx.dist_sink.empty() && ctx.dist_sink[node_id] != ctx.INF)
                deriv = std::min(deriv, ctx.dist_sink[node_id] + ctx.sink_adjust);
        }
        return deriv;
    }

    std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>>
    _signal_part_derivatives_impl(const DerivContext& ctx) const {
        // Independent trash: each extra unit of theoretical supply also pays
        // the analytic C_theo term on top of the (shifted-cost) MCF marginal.
        const VALUE_TYPE ind_theo = independent_trash ? _ind_c_theo_q : 0;
        std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>> result;
        for (const auto& node : nodes) {
            auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&node.get_type());
            if (!theo) continue;
            result.emplace_back(theo->get_spectrum_id(), theo->get_peak_index(),
                                _node_deriv(node.get_id(), ctx) + ind_theo);
        }
        return result;
    }

    std::vector<std::pair<size_t, double>>
    _spectrum_proportion_derivatives_impl(const DerivContext& ctx) const {
        // Weight each peak's integer per-supply marginal by its REAL (double)
        // intensity — d(cost)/dw = sum_i marginal_i * real_intensity_i.  The
        // intensity scale is deliberately absent (it cancels: it enters both
        // dS/dw and the total_cost unscale), and the real intensity may be
        // fractional, so this must accumulate in double, not the integer
        // VALUE_TYPE (which would floor sub-unit intensities to zero).
        std::vector<double> accum(no_target_distributions, 0.0);
        const VALUE_TYPE ind_theo = independent_trash ? _ind_c_theo_q : 0;
        for (const auto& node : nodes) {
            auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&node.get_type());
            if (!theo) continue;
            accum[theo->get_spectrum_id()] +=
                static_cast<double>(_node_deriv(node.get_id(), ctx) + ind_theo)
                * static_cast<double>(theo->get_intensity());
        }
        std::vector<std::pair<size_t, double>> result;
        result.reserve(no_target_distributions);
        for (size_t s = 0; s < no_target_distributions; ++s)
            result.emplace_back(s, accum[s]);
        return result;
    }

public:
    // Per-peak marginal cost of increasing each theoretical signal by 1
    // (spectrum_id, peak_index, derivative).  Exact: cheapest residual
    // augmenting route (Dijkstra on reduced costs).
    std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>> signal_part_derivatives() const {
        return _signal_part_derivatives_impl(_get_deriv_context(/*use_pi=*/false));
    }

    // Fast, APPROXIMATE per-peak marginals: the pure dual-potential
    // difference (pi[v] - pi[src]), skipping the residual search.  A lower
    // bound on the true marginal — exact only for peaks on the optimal flow
    // support, basis-dependent at degenerate optima.  Different VALUES from
    // signal_part_derivatives(); opt-in only.
    std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>>
    signal_part_derivatives_fast_approx() const {
        return _signal_part_derivatives_impl(_get_deriv_context(/*use_pi=*/true));
    }

    // Gradient of total cost w.r.t. scaling each spectrum's proportion
    // (spectrum_id, derivative) = sum_i(peak_derivative_i * intensity_i).
    // Exact residual marginals.
    std::vector<std::pair<size_t, double>> spectrum_proportion_derivatives() const {
        return _spectrum_proportion_derivatives_impl(_get_deriv_context(/*use_pi=*/false));
    }

    // Fast, APPROXIMATE spectrum-proportion gradient (dual-potential
    // difference; see signal_part_derivatives_fast_approx for the accuracy
    // caveat).  Different VALUES from spectrum_proportion_derivatives().
    std::vector<std::pair<size_t, double>>
    spectrum_proportion_derivatives_fast_approx() const {
        return _spectrum_proportion_derivatives_impl(_get_deriv_context(/*use_pi=*/true));
    }
};

template <typename VALUE_TYPE, typename intensity_type>
class WassersteinNetwork {
    std::vector<FlowNode<intensity_type>> nodes;
    std::vector<FlowEdge<intensity_type>> edges;

    const size_t _no_theoretical_spectra;
    const std::vector<size_t> _theoretical_spectra_sizes;

    std::vector<LEMON_INDEX> dead_end_node_ids;
    std::vector<std::unique_ptr<WassersteinNetworkSubgraph<VALUE_TYPE, intensity_type>>> flow_subgraphs;

    intensity_type _isolated_empirical_intensity = 0;
    std::vector<intensity_type> _isolated_theoretical_intensity;
    // Real (unscaled) per-unit trash costs for isolated/dead-end nodes.
    double _isolated_exp_trash_cost = 0;
    double _isolated_theo_trash_cost = 0;
    std::vector<double> _last_point;

    bool built = false;

    // Wasserstein transport order p.  Edge cost = ground_distance^p, so the
    // network optimises/reports the W_p^p objective; the ^(1/p) root is applied
    // by the high-level Python wrappers.  p == 1 reproduces the legacy behaviour
    // bit-for-bit.  p != 1 requires the dense factory (chain cost is not additive
    // under exponentiation).
    double _p_order = 1.0;

    // Cost scaling: integer edge costs are quantize_cost(real, _scale, p==1).
    // _scale is chosen at build() from the largest real cost (matching + trash).
    // p == 1 forces _scale == 1 (legacy truncation).  _max_real_cost tracks the
    // running maximum real edge/trash cost so build() can size _scale.
    int64_t _scale = 1;
    double _max_real_cost = 0.0;

    // Opt-in cost scaling (see set_cost_scaling): when requested, p == 1 costs
    // are scaled+rounded like p != 1 instead of truncated.  _explicit_cost_scale
    // <= 0 means auto (pick_cost_scale); > 0 is used verbatim.
    bool _cost_scaling_requested = false;
    int64_t _explicit_cost_scale = 0;

    // Single source of truth for the quantize_cost "p_is_one" (truncate) flag.
    // Costs are scaled+rounded whenever p != 1 OR the caller opted into cost
    // scaling; only legacy p == 1 without cost scaling truncates.  Every
    // quantize_cost() call site must use this, not a bare `_p_order == 1.0`,
    // so the solver's cost map and any re-quantised (isolated-trash, position
    // update) costs stay in the same units.
    bool _costs_truncated() const {
        return _p_order == 1.0 && !_cost_scaling_requested;
    }

    // Intensity scaling: real (double) node intensities map to integer LEMON
    // supplies as round-toward-zero(real * _intensity_scale).  Set via
    // set_intensity_scale() before build(); propagated to every subgraph in
    // build().  1.0 (the default) reproduces the legacy verbatim-intensity
    // behaviour.  total_cost() and the proportion derivatives are in scaled
    // units (cost_scale * intensity_scale); the Python wrapper unscales by
    // scale_factor() * intensity_scale_factor().
    double _intensity_scale = 1.0;

    // Cost-accumulator overflow bookkeeping (see set_flow_budget / solve):
    // _flow_budget is the caller-declared bound on point-scaled total flow in
    // real (pre-intensity-scale) units (0 = none declared; build() then sizes
    // the cost scale only for the build-time supplies at point == 1).
    // _emp_flow_total / _theo_flow_totals are per-spectrum real intensity
    // totals precomputed in build() so solve() can bound the accumulated cost
    // of an arbitrary point in O(#spectra).
    double _flow_budget = 0.0;
    double _emp_flow_total = 0.0;
    std::vector<double> _theo_flow_totals;

    // See add_independent_asymmetric_trash().
    bool _independent_trash_added = false;

public:
    WassersteinNetwork(std::vector<FlowNode<intensity_type>>&& nodes_,
                       std::vector<FlowEdge<intensity_type>>&& edges_,
                       size_t no_theoretical_spectra_,
                       std::vector<size_t>&& theoretical_spectra_sizes_,
                       std::vector<LEMON_INDEX>&& dead_end_node_ids_,
                       double p_order = 1.0,
                       double max_real_cost = 0.0
    ) :
    nodes(std::move(nodes_)),
    edges(std::move(edges_)),
    _no_theoretical_spectra(no_theoretical_spectra_),
    _theoretical_spectra_sizes(std::move(theoretical_spectra_sizes_)),
    dead_end_node_ids(std::move(dead_end_node_ids_)),
    _p_order(p_order),
    _max_real_cost(max_real_cost)
    {
        build_subgraphs();
    };

    double p_order() const { return _p_order; }
    int64_t scale_factor() const { return _scale; }
    double intensity_scale_factor() const { return _intensity_scale; }
    // Must be called before build() to take effect (build() propagates it to
    // the subgraphs and folds it into the cost-scale overflow budget).
    // The integer-intensity backend does no scaling at all: intensities are
    // already exact integers, so the only legal scale is 1.
    void set_intensity_scale(double s) {
        if constexpr (std::is_integral_v<intensity_type>) {
            if (s != 1.0)
                throw std::invalid_argument(
                    "Integer intensity backend does not support intensity scaling "
                    "(set_intensity_scale requires 1; use the double-intensity backend).");
            _intensity_scale = 1.0;
        } else {
            _intensity_scale = s;
        }
    }

    // Opt into scaling the (real, possibly fractional) edge costs to integers,
    // even at p == 1.  By default p == 1 truncates the cost (legacy, bit-exact)
    // and only p != 1 scales; calling this makes the network quantize p == 1
    // costs too (round(scale * real)), which lets a caller pass real distances
    // instead of pre-scaling positions.  `scale <= 0` => auto (pick_cost_scale,
    // coupled with the intensity scale against the int64 budget); `scale > 0`
    // => use it verbatim.  Must be called before build().
    //
    // The integer-intensity backend works in exact integer costs and does no
    // scaling at all, so this is rejected there.
    void set_cost_scaling(int64_t scale = 0) {
        if constexpr (std::is_integral_v<intensity_type>) {
            (void)scale;
            throw std::invalid_argument(
                "Integer intensity backend does not support cost scaling "
                "(it works in exact integer costs; use the double-intensity backend).");
        } else {
            _cost_scaling_requested = true;
            _explicit_cost_scale = scale;
        }
    }

    // Declare an upper bound on the point-scaled total flow this network will
    // be solved with: empirical total plus sum_i(point[i] * theoretical_i
    // total), in real (pre-intensity-scale) units.  build() sizes the auto
    // cost scale against max(build-time flow, budget), so every point within
    // the budget stays inside the int64 cost-accumulator ceiling; solve()
    // rejects points whose worst-case accumulated cost exceeds the ceiling
    // regardless of how the scale was chosen.  Must be called before build().
    // Default 0 keeps the legacy sizing (build-time supplies at point == 1).
    void set_flow_budget(double flow) {
        if (built)
            throw std::runtime_error(
                "set_flow_budget() must be called before build().");
        if (!(flow >= 0.0) || !std::isfinite(flow))
            throw std::invalid_argument(
                "set_flow_budget: flow must be finite and >= 0.");
        _flow_budget = flow;
    }

    WassersteinNetwork(const WassersteinNetwork&) = delete;
    WassersteinNetwork& operator=(const WassersteinNetwork&) = delete;
    WassersteinNetwork(WassersteinNetwork&& other) :
        nodes(std::move(other.nodes)),
        edges(std::move(other.edges)),
        _no_theoretical_spectra(other._no_theoretical_spectra),
        _theoretical_spectra_sizes(std::move(other._theoretical_spectra_sizes)),
        dead_end_node_ids(std::move(other.dead_end_node_ids)),
        flow_subgraphs(std::move(other.flow_subgraphs)),
        _isolated_empirical_intensity(other._isolated_empirical_intensity),
        _isolated_theoretical_intensity(std::move(other._isolated_theoretical_intensity)),
        _isolated_exp_trash_cost(other._isolated_exp_trash_cost),
        _isolated_theo_trash_cost(other._isolated_theo_trash_cost),
        _last_point(std::move(other._last_point)),
        built(other.built),
        _p_order(other._p_order),
        _scale(other._scale),
        _max_real_cost(other._max_real_cost),
        _cost_scaling_requested(other._cost_scaling_requested),
        _explicit_cost_scale(other._explicit_cost_scale),
        _intensity_scale(other._intensity_scale),
        _flow_budget(other._flow_budget),
        _emp_flow_total(other._emp_flow_total),
        _theo_flow_totals(std::move(other._theo_flow_totals)),
        _independent_trash_added(other._independent_trash_added)
    {
        other.built = false;
    }
    WassersteinNetwork& operator=(WassersteinNetwork&& other) = delete;
    size_t no_nodes() const {
        return nodes.size();
    };
    size_t no_edges() const {
        return edges.size();
    };
    size_t no_theoretical_spectra() const {
        return _no_theoretical_spectra;
    };

    const std::vector<size_t>& theoretical_spectra_sizes() const {
        return _theoretical_spectra_sizes;
    };

    const std::vector<FlowNode<intensity_type>>& get_nodes() const {
        return nodes;
    };
    const std::vector<FlowEdge<intensity_type>>& get_edges() const {
        return edges;
    };

    std::vector<std::vector<LEMON_INDEX>> neighbourhood_lists() const {
        std::vector<std::vector<LEMON_INDEX>> neighbourhood_lists;
        neighbourhood_lists.resize(nodes.size());
        for (const auto& edge : edges) {
            const LEMON_INDEX start_node_id = edge.get_start_node_id();
            const LEMON_INDEX end_node_id = edge.get_end_node_id();
            neighbourhood_lists[start_node_id].push_back(end_node_id);
            neighbourhood_lists[end_node_id].push_back(start_node_id);
        }
        return neighbourhood_lists;
    };

    std::pair<std::vector<std::vector<LEMON_INDEX>>, std::vector<LEMON_INDEX>> split_into_subgraphs() const {
        std::vector<std::vector<LEMON_INDEX>> subgraphs;
        std::vector<LEMON_INDEX> dead_end_nodes;

        assert_fits_lemon_index(nodes.size(), "network nodes");
        std::vector<bool> visited(nodes.size(), false);
        visited[0] = true; // Mark the source node as visited
        visited[1] = true; // Mark the sink node as visited
        std::vector<LEMON_INDEX> stack;
        std::vector<std::vector<LEMON_INDEX>> neighbourhood_lists = this->neighbourhood_lists();

        for (LEMON_INDEX node_id = 0; node_id < static_cast<LEMON_INT>(nodes.size()); ++node_id) {
            if (!visited[node_id]) {
                std::vector<LEMON_INDEX>& neighbours = neighbourhood_lists[node_id];
                if(neighbours.size() == 0) {
                    dead_end_nodes.push_back(node_id);
                } else {
                    std::vector<LEMON_INDEX> subgraph;
                    stack.push_back(node_id);
                    while (!stack.empty()) {
                        LEMON_INDEX current_node = stack.back();
                        stack.pop_back();
                        if (!visited[current_node]) {
                            visited[current_node] = true;
                            subgraph.push_back(current_node);
                            for (LEMON_INDEX neighbour : neighbourhood_lists[current_node]) {
                                if (!visited[neighbour]) {
                                    stack.push_back(neighbour);
                                }
                            }
                        }
                    }
                    // TODO: potentially remove this
                    std::sort(subgraph.begin(), subgraph.end());
                    subgraphs.push_back(subgraph);
                }
            }
        }
        return {subgraphs, dead_end_nodes};
    }

    void build_subgraphs() {
        auto [_subgraphs, _dead_end_nodes] = this->split_into_subgraphs();

        dead_end_node_ids = std::move(_dead_end_nodes);

        std::unique_ptr<LEMON_INDEX[]> node_in_subgraph = std::make_unique<LEMON_INDEX[]>(nodes.size());

        #ifdef LEMON_DO_ASSERTS
        for (size_t ii = 0; ii < nodes.size(); ++ii)
            node_in_subgraph[ii] = -10;
        #endif

        for (LEMON_INDEX subgraph_idx = 0; subgraph_idx < static_cast<LEMON_INT>(_subgraphs.size()); ++subgraph_idx)
            for (const auto& node_id : _subgraphs[subgraph_idx])
                node_in_subgraph[node_id] = subgraph_idx;

        #ifdef WNET_DO_ASSERTS
        for(auto dead_end_node_id : dead_end_node_ids)
            node_in_subgraph[dead_end_node_id] = -1;
        for(size_t node_id = 0; node_id < nodes.size(); ++node_id)
            if(node_in_subgraph[node_id] == -10)
                throw std::runtime_error("Node not assigned to any subgraph");
        #endif

        std::vector<std::vector<FlowEdge<intensity_type>*>> subgraph_edges(_subgraphs.size());
        for (auto& edge : edges)
        {
            const LEMON_INDEX start_node_id = edge.get_start_node_id();
            const LEMON_INDEX start_subgraph_idx = node_in_subgraph[start_node_id];
            subgraph_edges[start_subgraph_idx].push_back(&edge);

            #ifdef WNET_DO_ASSERTS
            const LEMON_INDEX end_node_id = edge.get_end_node_id();
            const LEMON_INDEX end_subgraph_idx = node_in_subgraph[end_node_id];
            if(start_subgraph_idx != end_subgraph_idx || start_subgraph_idx == -1)
                throw std::runtime_error("Edge connects nodes from different subgraphs or dead end nodes.");
            #endif
        }


        // TODO: optimize, right now this is needlessly O(subgraphs.size() * edges.size()),
        // can be O(subgraphs.size() + edges.size())
        flow_subgraphs.reserve(_subgraphs.size());
        for (size_t subgraph_idx = 0; subgraph_idx < _subgraphs.size(); ++subgraph_idx)
        {
            #ifdef DO_TONS_OF_PRINTS
            std::cout << "Subgraph" << std::endl;
            #endif
            flow_subgraphs.emplace_back(std::make_unique<WassersteinNetworkSubgraph<VALUE_TYPE, intensity_type>>(
                    _subgraphs[subgraph_idx],
                    nodes,
                    subgraph_edges[subgraph_idx],
                    _no_theoretical_spectra
            ));
        }
        _isolated_theoretical_intensity.assign(_no_theoretical_spectra, 0);
        for (LEMON_INDEX dead_end_id : dead_end_node_ids) {
            std::visit([&](const auto& t) {
                using T = std::decay_t<decltype(t)>;
                if constexpr (std::is_same_v<T, EmpiricalNode<intensity_type>>)
                    _isolated_empirical_intensity += t.get_intensity();
                else if constexpr (std::is_same_v<T, TheoreticalNode<intensity_type>>)
                    _isolated_theoretical_intensity[t.get_spectrum_id()] += t.get_intensity();
            }, nodes[dead_end_id].get_type());
        }
    }

    void add_simple_trash(double cost) {
        if (built)
            throw std::runtime_error("add_simple_trash() must be called before build(), not after.");
        _isolated_exp_trash_cost = cost;
        _isolated_theo_trash_cost = cost;
        _max_real_cost = std::max(_max_real_cost, cost);
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->add_simple_trash(cost);
    };

    void add_experimental_trash(double cost) {
        if (built)
            throw std::runtime_error("add_experimental_trash() must be called before build().");
        _isolated_exp_trash_cost = cost;
        _max_real_cost = std::max(_max_real_cost, cost);
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->add_experimental_trash(cost);
    };

    void add_theoretical_trash(double cost) {
        if (built)
            throw std::runtime_error("add_theoretical_trash() must be called before build().");
        _isolated_theo_trash_cost = cost;
        _max_real_cost = std::max(_max_real_cost, cost);
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->add_theoretical_trash(cost);
    };

    // Independent asymmetric trash (dualdeconv4 semantics): every discarded
    // empirical unit costs C_exp and every phantom-filled theoretical unit
    // costs C_theo, charged independently — an (empirical, theoretical)
    // excess pair costs C_exp + C_theo, never the annihilating model's
    // min(C_exp, C_theo), and the match-vs-dump threshold is C_exp + C_theo.
    // Implemented per subgraph by cost shifting (see the subgraph member
    // comment); the excess pricing is linear per node, so the subgraph
    // decomposition and the isolated-node terms are exact.  On the dense
    // factory any solver works; on the 1-D chain factory only the SlopeDP
    // backend supports it (it prices trash analytically — the per-match cost
    // shift cannot ride chain hop arcs), enforced at build().  Mutually
    // exclusive with the other trash models.
    void add_independent_asymmetric_trash(double C_exp, double C_theo) {
        if (built)
            throw std::runtime_error(
                "add_independent_asymmetric_trash() must be called before build().");
        if (_independent_trash_added)
            throw std::runtime_error("Independent trash already added.");
        if (!(C_exp >= 0.0) || !(C_theo >= 0.0) || !std::isfinite(C_exp + C_theo))
            throw std::invalid_argument(
                "add_independent_asymmetric_trash: costs must be finite and >= 0.");
        // Isolated (dead-end) nodes are charged the full per-unit costs by the
        // existing isolated-trash terms in total_cost()/derivatives.
        _isolated_exp_trash_cost = C_exp;
        _isolated_theo_trash_cost = C_theo;
        _max_real_cost = std::max(_max_real_cost, C_exp + C_theo);
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->add_independent_asymmetric_trash(C_exp, C_theo);
        _independent_trash_added = true;
    };

    void build(SolverConfig config = NetworkSimplexConfig{}) {
        // Total flow upper bound for the cost-scale accumulator ceiling: the
        // per-subgraph flow is max(emp, theo) intensity, so summing all node
        // intensities over-estimates the network-wide flow (safe — it only
        // shrinks the scale).  Per-spectrum totals are kept so solve() can
        // bound the accumulated cost of an arbitrary point; a caller expecting
        // points far above 1 declares that via set_flow_budget(), which widens
        // the sizing here, and solve() rejects points past the ceiling either
        // way.
        _emp_flow_total = 0.0;
        _theo_flow_totals.assign(_no_theoretical_spectra, 0.0);
        for (const auto& node : nodes)
            std::visit([&](const auto& n) {
                using T = std::decay_t<decltype(n)>;
                if constexpr (std::is_same_v<T, EmpiricalNode<intensity_type>>)
                    _emp_flow_total += static_cast<double>(n.get_intensity());
                else if constexpr (std::is_same_v<T, TheoreticalNode<intensity_type>>)
                    _theo_flow_totals[n.get_spectrum_id()] += static_cast<double>(n.get_intensity());
            }, node.get_type());
        double total_flow = _emp_flow_total;
        for (const double t : _theo_flow_totals)
            total_flow += t;
        if (_flow_budget > 0.0) {
            if (_flow_budget > total_flow)
                total_flow = _flow_budget;
        } else {
            // No explicit budget: size the accumulator ceiling for 4x the
            // build-time supplies (2 bits of headroom) instead of exactly 1x,
            // so nearby points — an optimizer probing slightly past 1.0, a
            // moderate proportion sweep — don't trip the solve()-time guard
            // on a network whose scale was sized with zero slack.  Costs the
            // auto scale 2 bits of precision; an explicit set_flow_budget()
            // overrides.  p == 1 legacy truncation (scale 1) is unaffected.
            total_flow *= 4.0;
        }
        // Choose one global cost scale from the largest real cost across the whole
        // network (matching + trash), so every subgraph's integer costs — and the
        // summed total_cost — share the same units.  p == 1 keeps _scale == 1.
        // The integer flow the accumulator actually sees is the real flow times
        // the intensity scale, so size the cost-scale ceiling against that.
        // Costs are scaled (round) whenever p != 1, OR when the caller opted in
        // via set_cost_scaling() (which lets p == 1 carry real fractional costs
        // instead of truncating).  Otherwise p == 1 keeps the legacy S == 1
        // truncation.  `_truncate` drives both the scale choice and quantize_cost.
        const bool _truncate = _costs_truncated();
        if (_cost_scaling_requested && _explicit_cost_scale > 0)
            _scale = _explicit_cost_scale;
        else
            _scale = pick_cost_scale(_max_real_cost, total_flow * _intensity_scale, _truncate);
        for (auto& flow_subgraph : flow_subgraphs) {
            flow_subgraph->set_cost_scaling(_scale, _truncate, _intensity_scale);
            flow_subgraph->build(config);
        }
        built = true;
    };

    // Quantised (scaled) isolated trash costs, matching the subgraph cost map.
    VALUE_TYPE _isolated_exp_trash_cost_scaled() const {
        return quantize_cost<VALUE_TYPE>(_isolated_exp_trash_cost, _scale, _costs_truncated());
    }
    VALUE_TYPE _isolated_theo_trash_cost_scaled() const {
        return quantize_cost<VALUE_TYPE>(_isolated_theo_trash_cost, _scale, _costs_truncated());
    }

    void solve()
    {
        std::vector<double> point(_no_theoretical_spectra, 1.0);
        solve(point);
    };

    // int64 cost-accumulator overflow guard.  A unit of flow traverses at most
    // two cost-bearing edges (an annihilation path crosses both asymmetric
    // trash edges), so the accumulated scaled cost is bounded by
    // 2 * c_max * scale * flow * intensity_scale.  pick_cost_scale targets
    // 2^62, which leaves exactly that factor of 2 below the int64 edge; the
    // guard therefore checks the single-traversal bound against 2^62.  Past
    // it, the accumulator can wrap and return plausible-looking garbage, so
    // reject the point loudly instead.
    void check_accumulator_budget(const std::vector<double>& point) const {
        if (!(_max_real_cost > 0.0)) return;
        double flow = _emp_flow_total;
        const size_t n = std::min(point.size(), _theo_flow_totals.size());
        for (size_t i = 0; i < n; ++i)
            flow += point[i] * _theo_flow_totals[i];
        constexpr double ACCUMULATOR_TARGET = 4611686018427387904.0; // 2^62
        const double worst_cost = _max_real_cost * static_cast<double>(_scale)
                                  * flow * _intensity_scale;
        if (!(worst_cost <= ACCUMULATOR_TARGET))
            throw std::overflow_error(
                "solve(): this point scales the total flow to " + std::to_string(flow) +
                " real intensity units, bounding the accumulated integer cost by " +
                std::to_string(worst_cost) + ", which exceeds the int64 budget of 2^62 "
                "and could overflow silently. Declare the expected point range via "
                "set_flow_budget() before build(), pass a smaller explicit cost scale "
                "(set_cost_scaling), or rescale the theoretical intensities.");
    }

    void solve(const std::vector<double>& point) {
        if(!built)
            throw std::runtime_error("You must call build() before calling solve().");
        // A negative proportion produces negative arc capacities (an
        // infeasible LP the solver cannot represent), and a non-finite one
        // corrupts the double->int64 capacity casts.  Reject both loudly
        // instead of returning a meaningless flow.
        for (size_t i = 0; i < point.size(); ++i)
            if (!std::isfinite(point[i]) || point[i] < 0.0)
                throw std::invalid_argument(
                    "solve(): point[" + std::to_string(i) + "] = " +
                    std::to_string(point[i]) +
                    " is invalid; every spectrum proportion must be finite "
                    "and >= 0.");
        check_accumulator_budget(point);

        _last_point = point;
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->set_point(point);
    };

    // Total cost in SCALED units (sum of scaled per-subgraph costs plus scaled
    // isolated-trash contributions).  The Python wrapper divides by scale_factor()
    // to recover the real W_p**p value.
    VALUE_TYPE total_cost() const {
        // Normally a subgraph throws first when queried before solve(), but a
        // network whose peaks are all dead-ends has no subgraph to object and
        // would read _last_point out of bounds below.
        if (_last_point.size() != _no_theoretical_spectra)
            throw std::runtime_error(
                "You must call solve() before calling total_cost().");
        VALUE_TYPE cost = 0;
        for (const auto& flow_subgraph : flow_subgraphs)
            cost += flow_subgraph->total_cost();
        cost += _isolated_exp_trash_cost_scaled() * _isolated_empirical_intensity * _intensity_scale;
        const VALUE_TYPE theo_trash_scaled = _isolated_theo_trash_cost_scaled();
        for (size_t s = 0; s < _no_theoretical_spectra; ++s)
            cost += static_cast<VALUE_TYPE>(theo_trash_scaled * _isolated_theoretical_intensity[s] * _last_point[s] * _intensity_scale);
        // Nonnegative edge costs cannot sum to a negative total: a negative
        // value here means the integer accumulator overflowed somewhere the
        // solve()-time guard did not anticipate.
        if (cost < 0)
            throw std::overflow_error(
                "total_cost(): accumulated scaled cost is negative, which means the "
                "int64 cost accumulator overflowed. Rebuild with a smaller cost scale "
                "(set_cost_scaling) or declare the expected point range via "
                "set_flow_budget() before build().");
        return cost;
    };

    int warm_start_count() const {
        int total = 0;
        for (const auto& sg : flow_subgraphs)
            total += sg->warm_start_count();
        return total;
    }
    int cold_start_count() const {
        int total = 0;
        for (const auto& sg : flow_subgraphs)
            total += sg->cold_start_count();
        return total;
    }
    int dual_repair_count() const {
        int total = 0;
        for (const auto& sg : flow_subgraphs)
            total += sg->dual_repair_count();
        return total;
    }
    int primal_repair_count() const {
        int total = 0;
        for (const auto& sg : flow_subgraphs)
            total += sg->primal_repair_count();
        return total;
    }

    size_t no_subgraphs() const {
        return flow_subgraphs.size();
    };

    const WassersteinNetworkSubgraph<VALUE_TYPE, intensity_type>& get_subgraph(size_t idx) const {
        if (idx >= flow_subgraphs.size())
            throw std::out_of_range("Subgraph index out of range");
        return *flow_subgraphs[idx];
    };

    std::string to_string() const {
        std::string result;
        for (const auto& flow_subgraph : flow_subgraphs)
            result += flow_subgraph->to_string();
        return result;
    };

    std::string lemon_to_string() const {
        std::string result;
        for (const auto& flow_subgraph : flow_subgraphs)
            result += flow_subgraph->lemon_to_string();
        return result;
    };

    std::tuple<std::vector<LEMON_INDEX>, std::vector<LEMON_INDEX>, std::vector<VALUE_TYPE>> flows_for_target(size_t target_id) const {
        std::vector<LEMON_INDEX> empirical_peak_indices;
        std::vector<LEMON_INDEX> theoretical_peak_indices;
        std::vector<VALUE_TYPE> flows;
        for (const auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->flows_for_target(target_id, empirical_peak_indices, theoretical_peak_indices, flows);
        return {empirical_peak_indices, theoretical_peak_indices, flows};
    };

    size_t count_matching_edges() const {
        size_t result = 0;
        for (const auto& edge : edges)
            std::visit([&](const auto& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, MatchingEdge>) result++;
            },
            edge.get_type());
        return result;
    }

    template<typename T>
    size_t count_nodes_of_type() const {
        size_t result = 0;
        for (const auto& node : nodes)
            if(std::holds_alternative<T>(node.get_type()))
                result++;
        return result;
    }

    template<typename T>
    size_t count_edges_of_type() const {
        size_t result = 0;
        for (const auto& edge : edges)
            if(std::holds_alternative<T>(edge.get_type()))
                result++;
        return result;
    }

    double matching_density() const {
        const double nominator = count_edges_of_type<MatchingEdge>() + count_edges_of_type<ChainEdge>() / 2.0;
        double denominator = 0;
        for (const auto& flow_subgraph : flow_subgraphs)
            denominator += flow_subgraph->template count_nodes_of_type<EmpiricalNode<intensity_type>>() * flow_subgraph->template count_nodes_of_type<TheoreticalNode<intensity_type>>();
        if (denominator == 0) return std::numeric_limits<double>::quiet_NaN();
        return nominator / denominator;
    }

    std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>>
    _signal_part_derivatives(bool fast) const {
        std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>> result;
        for (const auto& sg : flow_subgraphs) {
            auto sg_derivs = fast ? sg->signal_part_derivatives_fast_approx()
                                  : sg->signal_part_derivatives();
            result.insert(result.end(), sg_derivs.begin(), sg_derivs.end());
        }
        const VALUE_TYPE theo_trash_scaled = _isolated_theo_trash_cost_scaled();
        for (LEMON_INDEX dead_end_id : dead_end_node_ids) {
            if (auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&nodes[dead_end_id].get_type()))
                result.emplace_back(theo->get_spectrum_id(), theo->get_peak_index(), theo_trash_scaled);
        }
        return result;
    }

    std::vector<std::pair<size_t, double>>
    _spectrum_proportion_derivatives(bool fast) const {
        std::vector<double> accum(_no_theoretical_spectra, 0.0);
        for (const auto& sg : flow_subgraphs) {
            auto sg_derivs = fast ? sg->spectrum_proportion_derivatives_fast_approx()
                                  : sg->spectrum_proportion_derivatives();
            for (auto& [spec_id, deriv] : sg_derivs)
                accum[spec_id] += deriv;
        }
        const VALUE_TYPE theo_trash_scaled = _isolated_theo_trash_cost_scaled();
        for (size_t s = 0; s < _no_theoretical_spectra; ++s) {
            if (_isolated_theoretical_intensity[s] != 0)
                accum[s] += static_cast<double>(theo_trash_scaled) * static_cast<double>(_isolated_theoretical_intensity[s]);
        }
        std::vector<std::pair<size_t, double>> result;
        result.reserve(_no_theoretical_spectra);
        for (size_t s = 0; s < _no_theoretical_spectra; ++s)
            result.emplace_back(s, accum[s]);
        return result;
    }

public:
    // Exact per-peak / per-spectrum marginals (cheapest residual augmenting
    // route).  This is the default everything uses.
    std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>> signal_part_derivatives() const {
        return _signal_part_derivatives(/*fast=*/false);
    }
    std::vector<std::pair<size_t, double>> spectrum_proportion_derivatives() const {
        return _spectrum_proportion_derivatives(/*fast=*/false);
    }

    // Fast, APPROXIMATE variants: pure dual-potential difference, no residual
    // search.  ~O(n) vs a Dijkstra per subgraph, but the returned gradient
    // VALUES differ (lower bound on the true marginal; exact only on the
    // optimal flow support, basis-dependent at degenerate optima).  Opt-in.
    std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>>
    signal_part_derivatives_fast_approx() const {
        return _signal_part_derivatives(/*fast=*/true);
    }
    std::vector<std::pair<size_t, double>>
    spectrum_proportion_derivatives_fast_approx() const {
        return _spectrum_proportion_derivatives(/*fast=*/true);
    }

    static constexpr size_t value_type_size() {
        return sizeof(VALUE_TYPE);
    }

    static constexpr size_t index_type_size() {
        return sizeof(LEMON_INDEX);
    }

    static constexpr size_t max_value() {
        return std::numeric_limits<VALUE_TYPE>::max();
    }

    static constexpr size_t max_index() {
        return std::numeric_limits<LEMON_INDEX>::max();
    }

    // Update positions of empirical and theoretical peaks and immediately
    // re-solve each subgraph (warm-restarting NetworkSimplex if possible).
    //
    // Graph topology is fixed: only edge costs change.  Intensities are not
    // touched.  The new distributions must have the same number of peaks as
    // the originals (same peak_index range).
    //
    // For chain subgraphs (1D) the sorted position order of peaks in the
    // chain must be preserved; otherwise an exception is thrown.  If peaks
    // have genuinely crossed, rebuild the network from scratch instead.
    template<typename Distribution_t, typename DistMetric>
    void update_positions_and_solve(
        const Distribution_t* new_empirical,
        const std::vector<Distribution_t*>& new_theoretical
    ) {
        if (!built)
            throw std::runtime_error("update_positions_and_solve() must be called after build().");

        for (auto& sg_ptr : flow_subgraphs) {
            auto& sg = *sg_ptr;
            const auto& sg_nodes = sg.get_nodes();
            const auto& sg_edges = sg.get_edges();

            // Option B: reject position updates that would reorder chain nodes.
            // _build_chain_topology() walks from the lowest-ID endpoint, so the
            // chain order is deterministic.  Valid updates keep the sequence
            // monotone; a non-monotone result means peaks have genuinely crossed
            // and the topology is no longer valid.
            // nodes[i].get_id() == i by construction, so sg_nodes[nid] is a direct lookup.
            const auto& chain_order = sg.get_chain_order();
            if (chain_order.size() >= 2) {
                auto& chain_pos = sg.chain_pos_scratch();
                chain_pos.clear();
                for (LEMON_INDEX nid : chain_order) {
                    const auto& ntype = sg_nodes[nid].get_type();
                    if (const auto* emp = std::get_if<EmpiricalNode<intensity_type>>(&ntype))
                        chain_pos.push_back(new_empirical->get_point(emp->get_peak_index())[0]);
                    else if (const auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&ntype))
                        chain_pos.push_back(new_theoretical[theo->get_spectrum_id()]->get_point(theo->get_peak_index())[0]);
                    // source/sink never appear in chain_order; skip anything else
                }
                bool all_nondec = true, all_noninc = true;
                for (size_t k = 1; k < chain_pos.size(); ++k) {
                    if (chain_pos[k] < chain_pos[k - 1]) all_nondec = false;
                    if (chain_pos[k] > chain_pos[k - 1]) all_noninc = false;
                }
                if (!all_nondec && !all_noninc)
                    throw std::invalid_argument(
                        "update_positions_and_solve(): new positions violate the chain's sorted "
                        "order (peaks have crossed). Rebuild the network for the new positions.");
            }

            // Compute new edge costs using the pre-allocated scratch buffer.
            auto& new_costs = sg.costs_scratch();
            std::fill(new_costs.begin(), new_costs.end(), VALUE_TYPE(0));
            for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(sg_edges.size()); ++ii) {
                const auto& edge = sg_edges[ii];
                if (std::holds_alternative<MatchingEdge>(edge.get_type())) {
                    const auto& emp_t  = std::get<EmpiricalNode<intensity_type>>(edge.get_start_node().get_type());
                    const auto& theo_t = std::get<TheoreticalNode<intensity_type>>(edge.get_end_node().get_type());
                    const double d = DistMetric::dist(
                        new_empirical->get_point(emp_t.get_peak_index()),
                        new_theoretical[theo_t.get_spectrum_id()]->get_point(theo_t.get_peak_index()));
                    const double real_cost = (_p_order == 1.0) ? d : std::pow(d, _p_order);
                    // Reuse the fixed build-time scale so the warm-restarted basis
                    // stays in the same cost units.
                    new_costs[ii] = quantize_cost<VALUE_TYPE>(real_cost, _scale, _costs_truncated());
                } else if (std::holds_alternative<ChainEdge>(edge.get_type())) {
                    auto get_pos_1d = [&](const FlowNode<intensity_type>& n) -> double {
                        const auto& nt = n.get_type();
                        if (const auto* emp = std::get_if<EmpiricalNode<intensity_type>>(&nt))
                            return new_empirical->get_point(emp->get_peak_index())[0];
                        if (const auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&nt))
                            return new_theoretical[theo->get_spectrum_id()]->get_point(theo->get_peak_index())[0];
                        throw std::runtime_error("update_positions_and_solve(): chain edge connects non-peak node.");
                    };
                    const double gap = std::abs(get_pos_1d(edge.get_start_node()) - get_pos_1d(edge.get_end_node()));
                    // Match the build-time quantisation (truncate only for legacy
                    // p == 1 without cost scaling; scaled+rounded otherwise).
                    new_costs[ii] = quantize_cost<VALUE_TYPE>(gap, _scale, _costs_truncated());
                }
                // All other edge types (SrcToEmpirical, TheoreticalToSink, trash, …)
                // have position-independent costs; apply_new_costs ignores them (new_costs[ii] = 0).
            }

            sg.apply_new_costs(new_costs);
        }
        // _last_point (spectrum proportions) is unchanged — intensities are fixed.
    }

    // Runtime-dispatch variant: selects the metric policy at run time.
    template<typename Distribution_t>
    void update_positions_and_solve(
        const Distribution_t* new_empirical,
        const std::vector<Distribution_t*>& new_theoretical,
        DistanceMetric metric
    ) {
        if (metric == DistanceMetric::L1)
            update_positions_and_solve<Distribution_t, L1Metric>(new_empirical, new_theoretical);
        else if (metric == DistanceMetric::L2)
            update_positions_and_solve<Distribution_t, L2Metric>(new_empirical, new_theoretical);
        else if (metric == DistanceMetric::LINF)
            update_positions_and_solve<Distribution_t, LinfMetric>(new_empirical, new_theoretical);
        else
            throw std::runtime_error("update_positions_and_solve(): unsupported distance metric.");
    }

    // Layer 1 (span sink): update positions, re-solve, accumulate gradients into
    // caller-owned zero-initialised spans.  emp_grad is [N_emp * DIM] row-major;
    // theo_grads[s] is [N_s * DIM] row-major.  Chain (1D) subgraphs are handled
    // via accumulate_position_gradients_chain(); dense subgraphs via accumulate_position_gradients().
    template<typename Distribution_t, typename DistMetric>
    void update_positions_and_get_gradient(
        const Distribution_t* new_empirical,
        const std::vector<Distribution_t*>& new_theoretical,
        std::span<double> emp_grad,
        std::vector<std::span<double>> theo_grads
    ) {
        static constexpr size_t DIM = std::tuple_size_v<typename Distribution_t::Point_t>;
        update_positions_and_solve<Distribution_t, DistMetric>(new_empirical, new_theoretical);
        for (auto& sg_ptr : flow_subgraphs) {
            if (sg_ptr->has_chain_edges()) {
                if constexpr (DIM == 1)
                    sg_ptr->template accumulate_position_gradients_chain<Distribution_t, DistMetric>(
                        new_empirical, new_theoretical, emp_grad, theo_grads);
                else
                    throw std::logic_error(
                        "update_positions_and_get_gradient: chain edges require DIM == 1");
            } else {
                sg_ptr->template accumulate_position_gradients<Distribution_t, DistMetric>(
                    new_empirical, new_theoretical, emp_grad, theo_grads, _p_order);
            }
        }
    }

    // Runtime-dispatch variant.
    template<typename Distribution_t>
    void update_positions_and_get_gradient(
        const Distribution_t* new_empirical,
        const std::vector<Distribution_t*>& new_theoretical,
        std::span<double> emp_grad,
        std::vector<std::span<double>> theo_grads,
        DistanceMetric metric
    ) {
        if (metric == DistanceMetric::L1)
            update_positions_and_get_gradient<Distribution_t, L1Metric>(
                new_empirical, new_theoretical, emp_grad, theo_grads);
        else if (metric == DistanceMetric::L2)
            update_positions_and_get_gradient<Distribution_t, L2Metric>(
                new_empirical, new_theoretical, emp_grad, theo_grads);
        else if (metric == DistanceMetric::LINF)
            update_positions_and_get_gradient<Distribution_t, LinfMetric>(
                new_empirical, new_theoretical, emp_grad, theo_grads);
        else
            throw std::runtime_error(
                "update_positions_and_get_gradient(): unsupported distance metric.");
    }

    // Layer 2 (vector wrapper): allocates, calls Layer 1, returns by move.
    template<typename Distribution_t, typename DistMetric>
    std::pair<std::vector<double>, std::vector<std::vector<double>>>
    update_positions_and_get_gradient(
        const Distribution_t* new_empirical,
        const std::vector<Distribution_t*>& new_theoretical
    ) {
        static constexpr size_t DIM = std::tuple_size_v<typename Distribution_t::Point_t>;
        std::vector<double> emp_grad(new_empirical->size() * DIM, 0.0);
        std::vector<std::vector<double>> theo_grads;
        theo_grads.reserve(new_theoretical.size());
        for (const auto* t : new_theoretical)
            theo_grads.emplace_back(t->size() * DIM, 0.0);
        std::vector<std::span<double>> theo_spans;
        theo_spans.reserve(new_theoretical.size());
        for (auto& v : theo_grads)
            theo_spans.emplace_back(v.data(), v.size());
        update_positions_and_get_gradient<Distribution_t, DistMetric>(
            new_empirical, new_theoretical,
            std::span<double>(emp_grad.data(), emp_grad.size()),
            theo_spans);
        return {std::move(emp_grad), std::move(theo_grads)};
    }
};



template <typename VALUE_TYPE>
class WassersteinNetworkFactory {
public:
    template<typename Distribution_t, typename DistMetric>
    static WassersteinNetwork<VALUE_TYPE, typename Distribution_t::intensity_type> create(
        const Distribution_t* empirical_spectrum,
        const std::vector<Distribution_t*>& theoretical_spectra,
        double max_dist = std::numeric_limits<double>::max(),
        double p = 1.0
    )
    {
        if (!(std::isfinite(p) && p >= 1.0))
            throw std::invalid_argument("Wasserstein order p must be a finite number >= 1.");
        using intensity_type = typename Distribution_t::intensity_type;
        // The integer-intensity backend is the bit-exact p == 1 legacy: it does
        // no cost/intensity scaling, so fractional-cost orders are unsupported.
        if constexpr (std::is_integral_v<intensity_type>)
            if (p != 1.0)
                throw std::invalid_argument(
                    "Integer intensity backend supports only p == 1; use the "
                    "double-intensity backend for p != 1.");
        double max_real_cost = 0.0;  // largest matching cost; sizes the build-time scale
        std::vector<FlowNode<intensity_type>> nodes;
        std::vector<FlowEdge<intensity_type>> edges;
        std::vector<LEMON_INDEX> dead_end_node_ids;

        static_assert(std::is_same_v<typename Distribution_t::intensity_type, intensity_type>,
                      "intensity_type does not match the intensity_type of the provided Distribution_t");
        // Empty distributions previously triggered UB inside CloserThanIter
        // (out-of-bounds access on empty sorted_indices for empty empirical,
        // and an infinite loop for empty theoretical). Reject explicitly.
        if (empirical_spectrum->size() == 0)
            throw std::invalid_argument("Empirical distribution is empty.");
        for (size_t i = 0; i < theoretical_spectra.size(); ++i)
            if (theoretical_spectra[i]->size() == 0)
                throw std::invalid_argument(
                    "Theoretical distribution at index " + std::to_string(i) + " is empty.");
        {
            size_t no_nodes = 2 + empirical_spectrum->size();
            for (auto& ts : theoretical_spectra)
                no_nodes += ts->size();
            assert_fits_lemon_index(no_nodes, "nodes");
            assert_fits_lemon_index(empirical_spectrum->size(), "empirical peaks");
            for (size_t i = 0; i < theoretical_spectra.size(); ++i)
                assert_fits_lemon_index(theoretical_spectra[i]->size(), "theoretical peaks");
            nodes.reserve(no_nodes);
        }

        // Create placeholder source and sink nodes
        nodes.emplace_back(FlowNode<intensity_type>(0, SourceNode()));
        nodes.emplace_back(FlowNode<intensity_type>(1, SinkNode()));

        for (LEMON_INDEX empirical_idx = 0; empirical_idx < static_cast<LEMON_INT>(empirical_spectrum->size()); ++empirical_idx) {
            nodes.emplace_back(FlowNode<intensity_type>(
                                    nodes.size(),
                                    EmpiricalNode(
                                        empirical_idx,
                                        empirical_spectrum->intensities()[empirical_idx])));
        }

        for (size_t theoretical_spectrum_idx = 0; theoretical_spectrum_idx < theoretical_spectra.size(); ++theoretical_spectrum_idx)
        {
            #ifdef DO_TONS_OF_PRINTS
            size_t no_processed = 0;
            size_t no_included = 0;
            std::cout << "Processing theoretical spectrum " << theoretical_spectrum_idx << " / " << theoretical_spectra.size() << std::endl;
            #endif
            const auto& theoretical_spectrum = theoretical_spectra[theoretical_spectrum_idx];

            const size_t first_theoretical_node_idx = nodes.size();
            for (LEMON_INDEX theoretical_peak_idx = 0; theoretical_peak_idx < static_cast<LEMON_INT>(theoretical_spectrum->size()); ++theoretical_peak_idx) {
                nodes.emplace_back(FlowNode<intensity_type>(
                                        nodes.size(),
                                            TheoreticalNode(
                                                theoretical_spectrum_idx,
                                                theoretical_peak_idx,
                                                theoretical_spectrum->intensities()[theoretical_peak_idx])));
            }

            // Calculate the distances between the empirical and theoretical peaks
            auto it = empirical_spectrum->template closer_than_iter<DistMetric>(*theoretical_spectrum, max_dist);
            while(it.advance())
            {
                auto [empirical_idx, theoretical_peak_idx] = it.get_indices();
                double dist = it.get_distance();
                // Order-p cost: d^p (real, unscaled).  p == 1 leaves dist unchanged
                // (pow(d,1) == d).  Quantisation to the integer solver cost happens
                // at build(), once the global scale is known.
                double real_cost = (p == 1.0) ? dist : std::pow(dist, p);
                if (real_cost > max_real_cost) max_real_cost = real_cost;
                edges.emplace_back(FlowEdge<intensity_type>(
                    edges.size(),
                    nodes[empirical_idx + 2], // +2 to skip the source and sink nodes
                    nodes[first_theoretical_node_idx + theoretical_peak_idx],
                    MatchingEdge(real_cost)
                ));
            }
        }

        std::vector<size_t> theoretical_spectra_sizes;
        theoretical_spectra_sizes.reserve(theoretical_spectra.size());
        for (const auto& theoretical_spectrum : theoretical_spectra)
            theoretical_spectra_sizes.push_back(theoretical_spectrum->size());

        assert_fits_lemon_index(edges.size(), "edges");

        return WassersteinNetwork<VALUE_TYPE, intensity_type>(
            std::move(nodes),
            std::move(edges),
            theoretical_spectra.size(),
            std::move(theoretical_spectra_sizes),
            std::move(dead_end_node_ids),
            p,
            max_real_cost
        );
    };

    template<typename Distribution_t>
    static WassersteinNetwork<VALUE_TYPE, typename Distribution_t::intensity_type> create(
        const Distribution_t* empirical_spectrum,
        const std::vector<Distribution_t*>& theoretical_spectra,
        DistanceMetric distance_metric,
        double max_dist = std::numeric_limits<double>::max(),
        double p = 1.0
    ) {
        if (distance_metric == DistanceMetric::L1) {
            return create<Distribution_t, L1Metric>(empirical_spectrum, theoretical_spectra, max_dist, p);
        } else if (distance_metric == DistanceMetric::L2) {
            return create<Distribution_t, L2Metric>(empirical_spectrum, theoretical_spectra, max_dist, p);
        } else if (distance_metric == DistanceMetric::LINF) {
            return create<Distribution_t, LinfMetric>(empirical_spectrum, theoretical_spectra, max_dist, p);
        } else {
            throw std::runtime_error("Unsupported distance metric.");
        }
    };

    // 1D chain-optimized factory. Instead of O(m·n) matching edges,
    // merges empirical and theoretical peaks into one sorted sequence and
    // emits only O(m+n) chain edges (gap-cost) between adjacent peaks.
    // In 1D, L1 = L2 = L_inf = |position difference|, so the distance
    // metric argument is accepted for API symmetry but has no effect.
    //
    // Parity with `create`: single-side fragments (runs of peaks where all
    // are empirical or all are theoretical, with no cross-side peak within
    // `max_dist`) emit no chain edges, so their nodes get zero neighbours
    // and are classified as dead-end by `split_into_subgraphs`. This
    // matches today's dense behavior of dropping unmatched mass silently.
    template<typename intensity_type_>
    static WassersteinNetwork<VALUE_TYPE, intensity_type_> create_1d(
        const VectorDistribution<1, double, intensity_type_>* empirical_spectrum,
        const std::vector<VectorDistribution<1, double, intensity_type_>*>& theoretical_spectra,
        DistanceMetric /* distance_metric */,
        double max_dist = std::numeric_limits<double>::max(),
        double p = 1.0
    ) {
        // The chain factory's gap costs are additive along the line, which only
        // equals the transport cost for p == 1 (|a-c|^p != |a-b|^p + |b-c|^p).
        if (p != 1.0)
            throw std::invalid_argument(
                "create_1d (chain factory) only supports p=1; use the dense factory for p!=1.");
        using intensity_type = intensity_type_;
        // Largest real per-unit path cost a unit of flow can accumulate: the
        // span (sum of gap costs) of the widest emitted run.  A chain ride may
        // traverse many gap edges, so the max single gap would under-state the
        // accumulator bound; the run span is the true per-unit ceiling.  This
        // sizes the build-time auto cost scale (pick_cost_scale) exactly like
        // the dense factory's max matching cost — without it a trash-less
        // chain network kept _max_real_cost == 0 and scale_factor() == 1, so
        // under set_cost_scaling() fractional gap costs were llround()ed at
        // scale 1 (a 0.6 gap priced as 1).
        double max_real_cost = 0.0;
        std::vector<FlowNode<intensity_type>> nodes;
        std::vector<FlowEdge<intensity_type>> edges;
        std::vector<LEMON_INDEX> dead_end_node_ids;  // recomputed in build_subgraphs()

        // Reject empty inputs for API parity with the dense `create` factory.
        if (empirical_spectrum->size() == 0)
            throw std::invalid_argument("Empirical distribution is empty.");
        for (size_t i = 0; i < theoretical_spectra.size(); ++i)
            if (theoretical_spectra[i]->size() == 0)
                throw std::invalid_argument(
                    "Theoretical distribution at index " + std::to_string(i) + " is empty.");

        // Reserve node storage (source + sink + all empirical + all theoretical).
        {
            size_t no_nodes = 2 + empirical_spectrum->size();
            for (auto& ts : theoretical_spectra)
                no_nodes += ts->size();
            assert_fits_lemon_index(no_nodes, "nodes");
            assert_fits_lemon_index(empirical_spectrum->size(), "empirical peaks");
            for (const auto& ts : theoretical_spectra)
                assert_fits_lemon_index(ts->size(), "theoretical peaks");
            nodes.reserve(no_nodes);
        }

        // Source and sink placeholders (matches dense factory's convention).
        nodes.emplace_back(FlowNode<intensity_type>(0, SourceNode()));
        nodes.emplace_back(FlowNode<intensity_type>(1, SinkNode()));

        // Position entry for the sorted merge. is_empirical tags the side.
        struct PosEntry {
            double position;
            LEMON_INDEX node_id;
            bool is_empirical;
        };
        std::vector<PosEntry> entries;
        {
            size_t no_entries = empirical_spectrum->size();
            for (const auto& ts : theoretical_spectra)
                no_entries += ts->size();
            entries.reserve(no_entries);
        }

        for (LEMON_INDEX empirical_idx = 0;
             empirical_idx < static_cast<LEMON_INT>(empirical_spectrum->size());
             ++empirical_idx) {
            nodes.emplace_back(FlowNode<intensity_type>(
                nodes.size(),
                EmpiricalNode<intensity_type>(
                    empirical_idx,
                    empirical_spectrum->intensities()[empirical_idx])));
            entries.push_back(PosEntry{
                empirical_spectrum->get_point(empirical_idx)[0],
                static_cast<LEMON_INDEX>(nodes.size() - 1),
                true});
        }

        for (size_t theoretical_spectrum_idx = 0;
             theoretical_spectrum_idx < theoretical_spectra.size();
             ++theoretical_spectrum_idx) {
            const auto& ts = theoretical_spectra[theoretical_spectrum_idx];
            for (LEMON_INDEX peak_idx = 0;
                 peak_idx < static_cast<LEMON_INT>(ts->size());
                 ++peak_idx) {
                nodes.emplace_back(FlowNode<intensity_type>(
                    nodes.size(),
                    TheoreticalNode<intensity_type>(
                        theoretical_spectrum_idx,
                        peak_idx,
                        ts->intensities()[peak_idx])));
                entries.push_back(PosEntry{
                    ts->get_point(peak_idx)[0],
                    static_cast<LEMON_INDEX>(nodes.size() - 1),
                    false});
            }
        }

        std::sort(entries.begin(), entries.end(),
                  [](const PosEntry& a, const PosEntry& b) {
                      return a.position < b.position;
                  });

        // Walk sorted entries, splitting on gaps > max_dist.
        // For each maximal run, only emit chain edges if the run contains
        // at least one empirical AND one theoretical node; otherwise, all
        // nodes in the run stay isolated and get dropped as dead-ends.
        auto flush_run = [&](size_t run_start, size_t run_end) {
            bool has_emp = false;
            bool has_theo = false;
            for (size_t i = run_start; i < run_end; ++i) {
                if (entries[i].is_empirical) has_emp = true;
                else has_theo = true;
                if (has_emp && has_theo) break;
            }
            if (!(has_emp && has_theo)) return;  // single-side run → drop

            // Entries are sorted, so the run span equals the sum of its gap
            // costs — the largest real cost one unit can accumulate here.
            const double run_span =
                entries[run_end - 1].position - entries[run_start].position;
            if (run_span > max_real_cost) max_real_cost = run_span;

            for (size_t i = run_start + 1; i < run_end; ++i) {
                const double gap_d = entries[i].position - entries[i-1].position;
                if (gap_d > static_cast<double>(std::numeric_limits<VALUE_TYPE>::max()))
                    throw std::overflow_error(
                        "Chain gap " + std::to_string(gap_d) +
                        " overflows VALUE_TYPE (max " +
                        std::to_string(std::numeric_limits<VALUE_TYPE>::max()) + ")");
                // Store the real gap; build() quantises (p==1 => truncation, the
                // legacy behaviour).  Bidirectional: LEMON needs two arcs for flow
                // in either direction. Both carry cost = gap.
                edges.emplace_back(FlowEdge<intensity_type>(
                    edges.size(),
                    nodes[entries[i-1].node_id],
                    nodes[entries[i].node_id],
                    ChainEdge(gap_d)));
                edges.emplace_back(FlowEdge<intensity_type>(
                    edges.size(),
                    nodes[entries[i].node_id],
                    nodes[entries[i-1].node_id],
                    ChainEdge(gap_d)));
            }
        };

        if (!entries.empty()) {
            size_t run_start = 0;
            for (size_t i = 1; i < entries.size(); ++i) {
                const double gap = entries[i].position - entries[i-1].position;
                if (gap > static_cast<double>(max_dist)) {
                    flush_run(run_start, i);
                    run_start = i;
                }
            }
            flush_run(run_start, entries.size());
        }

        std::vector<size_t> theoretical_spectra_sizes;
        theoretical_spectra_sizes.reserve(theoretical_spectra.size());
        for (const auto& ts : theoretical_spectra)
            theoretical_spectra_sizes.push_back(ts->size());

        assert_fits_lemon_index(edges.size(), "edges");

        return WassersteinNetwork<VALUE_TYPE, intensity_type>(
            std::move(nodes),
            std::move(edges),
            theoretical_spectra.size(),
            std::move(theoretical_spectra_sizes),
            std::move(dead_end_node_ids),
            p,  // validated == 1.0 above
            max_real_cost
        );
    };
};
#endif // WNET_DECOMPOSITABLE_GRAPH_HPP