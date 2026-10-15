// Prototype fork of production convex_sweep.hpp at e645fad. Production dispatch
// is unchanged.
#ifndef WNET_TRIAL_SWEEP_KERNEL_HPP
#define WNET_TRIAL_SWEEP_KERNEL_HPP

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
#include <string>
#include <type_traits>
#include <vector>

#include "augmented_profiles.hpp"
#include "moments.hpp"
#include "ordered_breakpoints.hpp"
#include "wide_int.hpp"

namespace sweep_trials_kernel {

using i64 = int64_t;

// Interpolation and collinearity tests need the exact 128-bit product of two
// i64s; wide_int supplies it natively or in software (MSVC has no __int128).
using wide_int::mul_div_trunc;
using wide_int::products_equal;

constexpr i64 NEG_INF = std::numeric_limits<i64>::min() / 4;

// ---------------------------------------------------------------------- //
// Piecewise-linear function on integer domain [0, K].
// ---------------------------------------------------------------------- //
template <class Storage = breakpoint_storage::FlatProfileStorage,
          bool Linear = false>
struct BasicPLF {
  using PLF = BasicPLF;
  static constexpr bool linear = Linear;
  struct V {
    i64 k;
    i64 v;
  };
  typename Storage::template Vertices<V>
      vs; // k strictly increasing; front().k == 0

  static PLF constant(i64 K, i64 value) {
    PLF f;
    f.vs.push_back({0, value});
    if (K > 0)
      f.vs.push_back({K, value});
    return f;
  }

  i64 K() const { return vs.back().k; }

  i64 eval(i64 k) const {
    if constexpr (requires { vs.eval(k); })
      return vs.eval(k);
    if (k <= vs.front().k)
      return vs.front().v;
    if (k >= vs.back().k)
      return vs.back().v;
    const auto right = vs.lower_bound(k);
    const V &b = *right;
    const V &a = *std::prev(right);
    return a.v + mul_div_trunc(b.v - a.v, k - a.k, b.k - a.k);
  }

  void normalize() {
    TRIAL_COUNT(normalizations, 1);
    TRIAL_COUNT(normalized_vertices, vs.size());
    TRIAL_MAX(largest_profile, vs.size());
    if constexpr (Linear) {
      size_t n = 0;
      for (size_t i = 0; i < vs.size(); ++i) {
        auto x = vs[i];
        if (n && vs[n - 1].k == x.k) {
          vs[n - 1].v = std::max(vs[n - 1].v, x.v);
          continue;
        }
        while (n >= 2 &&
               products_equal(vs[n - 1].v - vs[n - 2].v, x.k - vs[n - 1].k,
                              x.v - vs[n - 1].v, vs[n - 1].k - vs[n - 2].k))
          --n;
        vs[n++] = x;
      }
      vs.resize(n);
      return;
    }
    std::vector<V> out;
    out.reserve(vs.size());
    for (const V &x : vs) {
      if (!out.empty() && out.back().k == x.k) {
        if (x.v > out.back().v)
          out.back().v = x.v;
        continue;
      }
      while (out.size() >= 2) {
        const V &a = out[out.size() - 2];
        const V &b = out.back();
        if (products_equal(b.v - a.v, x.k - b.k, x.v - b.v, b.k - a.k))
          out.pop_back();
        else
          break;
      }
      out.push_back(x);
    }
    vs = std::move(out);
  }

  // f(k) := max_{k' >= k} f(k')  — free drops of the oldest pendings.
  void suffix_max() {
    std::vector<V> out;
    i64 run = NEG_INF; // suffix max over processed (right) part
    for (auto it = vs.rbegin(); it != vs.rend(); ++it) {
      const V b = *it;
      if (it != vs.rbegin()) {
        const V a = *std::prev(it); // right neighbour
        if (b.v > run && a.v < run) {
          // decreasing-to-the-right segment crosses the plateau:
          // find last integer with line(k) > run, insert pair.
          i64 lo = b.k, hi = a.k;
          while (hi - lo > 1) {
            i64 mid = lo + (hi - lo) / 2;
            i64 lv = b.v + mul_div_trunc(a.v - b.v, mid - b.k, a.k - b.k);
            if (lv > run)
              lo = mid;
            else
              hi = mid;
          }
          out.push_back({hi, run});
          if (lo != b.k) {
            out.push_back(
                {lo, b.v + mul_div_trunc(a.v - b.v, lo - b.k, a.k - b.k)});
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
    if (q <= 0)
      return;
    if constexpr (Linear) {
      const i64 value = vs.front().v;
      for (auto &x : vs)
        x.k += q;
      vs.insert(vs.begin(), {0, value});
      normalize();
      return;
    }
    if constexpr (requires { vs.shift_append(q); }) {
      vs.shift_append(q);
      return;
    }
    std::vector<V> out;
    out.reserve(vs.size() + 2);
    out.push_back({0, vs.front().v});
    out.push_back({q, vs.front().v});
    for (const V &x : vs)
      if (x.k > 0)
        out.push_back({x.k + q, x.v});
    vs = std::move(out);
    normalize();
  }

  // Restrict the domain to [0, K_new].
  void truncate(i64 K_new) {
    if (K_new >= K())
      return;
    if constexpr (Linear) {
      const i64 value = eval(K_new);
      auto it = vs.lower_bound(K_new);
      vs.resize(static_cast<size_t>(it - vs.begin()));
      vs.push_back({K_new, value});
      normalize();
      return;
    }
    if constexpr (requires { vs.truncate(K_new); }) {
      vs.truncate(K_new);
      return;
    }
    i64 vK = eval(K_new);
    std::vector<V> out;
    for (const V &x : vs) {
      if (x.k < K_new)
        out.push_back(x);
      else
        break;
    }
    out.push_back({K_new, vK});
    if (out.front().k != 0)
      out.insert(out.begin(), {0, vs.front().v});
    vs = std::move(out);
    normalize();
  }

  // In-place pointwise max with `other` (domains may differ; outside its
  // domain a function does not contribute).
  void pointwise_max(const PLF &other) {
    TRIAL_COUNT(pointwise_merges, 1);
    if constexpr (Linear) {
      static thread_local std::vector<i64> keys;
      static thread_local std::vector<V> out;
      keys.clear();
      out.clear();
      auto a = vs.begin(), b = other.vs.begin();
      while (a != vs.end() || b != other.vs.end()) {
        i64 k;
        if (b == other.vs.end() || (a != vs.end() && a->k < b->k))
          k = (a++)->k;
        else if (a == vs.end() || b->k < a->k)
          k = (b++)->k;
        else {
          k = a->k;
          ++a;
          ++b;
        }
        keys.push_back(k);
      }
      size_t ia = 0, ib = 0;
      auto value = [](const auto &data, size_t cursor, i64 k) {
        if (k > data.back().k)
          return NEG_INF;
        if (k == data.back().k)
          return data.back().v;
        const auto &x = data[cursor];
        const auto &y = data[cursor + 1];
        return x.v + mul_div_trunc(y.v - x.v, k - x.k, y.k - x.k);
      };
      for (size_t i = 0; i < keys.size(); ++i) {
        const i64 k = keys[i];
        while (ia + 1 < vs.size() && vs[ia + 1].k <= k)
          ++ia;
        while (ib + 1 < other.vs.size() && other.vs[ib + 1].k <= k)
          ++ib;
        auto fa = [&](i64 q) { return value(vs, ia, q); };
        auto fb = [&](i64 q) { return value(other.vs, ib, q); };
        out.push_back({k, std::max(fa(k), fb(k))});
        if (i + 1 == keys.size() || keys[i + 1] - k <= 1)
          continue;
        const i64 end = keys[i + 1], d1 = fa(k) - fb(k), d2 = fa(end) - fb(end);
        if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) {
          i64 lo = k, hi = end;
          while (hi - lo > 1) {
            const i64 mid = lo + (hi - lo) / 2, d = fa(mid) - fb(mid);
            if (d1 > 0 ? d > 0 : d < 0)
              lo = mid;
            else
              hi = mid;
          }
          out.push_back({lo, std::max(fa(lo), fb(lo))});
          out.push_back({hi, std::max(fa(hi), fb(hi))});
        }
      }
      vs.swap(out);
      normalize();
      return;
    }
    if constexpr (requires { vs.pointwise_max(other.vs); }) {
      vs.pointwise_max(other.vs);
      return;
    }
    std::vector<i64> ks;
    ks.reserve(vs.size() + other.vs.size());
    for (auto &x : vs)
      ks.push_back(x.k);
    for (auto &x : other.vs)
      ks.push_back(x.k);
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
            if ((d1 > 0) ? (dm > 0) : (dm < 0))
              lo = mid;
            else
              hi = mid;
          }
          out.push_back({lo, std::max(fval(lo), gval(lo))});
          out.push_back({hi, std::max(fval(hi), gval(hi))});
        }
      }
    }
    std::sort(out.begin(), out.end(),
              [](const V &a, const V &b) { return a.k < b.k; });
    vs = std::move(out);
    normalize();
  }

  void set_front_value(i64 value) {
    if constexpr (requires { vs.set_front_value(value); })
      vs.set_front_value(value);
    else
      vs.front().v = value;
  }

  i64 max_value() const {
    if constexpr (requires { vs.vertex_argmax(0, K()); })
      return vs.vertex_argmax(0, K()).second;
    i64 m = NEG_INF;
    for (const V &x : vs)
      m = std::max(m, x.v);
    return m;
  }
};

using PLF = BasicPLF<>;
using LinearPLF = BasicPLF<breakpoint_storage::FlatProfileStorage, true>;
using TreePLF = BasicPLF<breakpoint_storage::MapStorage>;
using AugmentedPLF = BasicPLF<breakpoint_storage::AugmentedStorage<>>;

// Monotone O(1)-amortized evaluator: valid only for non-decreasing query ks.
template <class F> struct Walker {
  const F &f;
  decltype(std::declval<const F &>().vs.begin()) seg;
  explicit Walker(const F &f) : f(f), seg(f.vs.begin()) {}
  i64 eval(i64 k) {
    const auto &vs = f.vs;
    if (k <= vs.front().k)
      return vs.front().v;
    if (k >= vs.back().k)
      return vs.back().v;
    while (std::next(seg)->k <= k)
      ++seg;
    const typename F::V &a = *seg;
    const typename F::V &b = *std::next(seg);
    if (k == a.k)
      return a.v;
    return a.v + mul_div_trunc(b.v - a.v, k - a.k, b.k - a.k);
  }
};

template <bool SUB, class F> inline F addsub_pl(const F &a, const F &b) {
  F out;
  out.vs.reserve(a.vs.size() + b.vs.size());
  Walker wa(a), wb(b);
  auto ia = a.vs.begin(), ib = b.vs.begin();
  i64 prev = std::numeric_limits<i64>::min();
  while (ia != a.vs.end() || ib != b.vs.end()) {
    i64 k;
    if (ib == b.vs.end())
      k = (ia++)->k;
    else if (ia == a.vs.end())
      k = (ib++)->k;
    else if (ia->k < ib->k)
      k = (ia++)->k;
    else if (ib->k < ia->k)
      k = (ib++)->k;
    else {
      k = ia->k;
      ++ia;
      ++ib;
    }
    if (k == prev)
      continue;
    prev = k;
    i64 va = wa.eval(k), vb = wb.eval(k);
    out.vs.push_back({k, SUB ? va - vb : va + vb});
  }
  out.normalize();
  return out;
}

template <class F> inline F add_pl(const F &a, const F &b) {
  return addsub_pl<false>(a, b);
}
template <class F> inline F sub_pl(const F &a, const F &b) {
  return addsub_pl<true>(a, b);
}

// result(g) = max_{g <= k <= min(K, g+q)} f(k), domain [0, K].
// Monotone-deque sliding max over candidate breakpoints (vertices and
// their -q images), with adjacent-integer inserts where the window-edge
// lines cross between candidates.  O(n log n).
template <class F> inline F windowed_max(const F &f, i64 q) {
  TRIAL_COUNT(window_passes, 1);
  const i64 K = f.K();
  std::vector<i64> ks;
  ks.reserve(2 * f.vs.size() + 2);
  for (auto &x : f.vs) {
    ks.push_back(x.k);
    if (x.k - q >= 0)
      ks.push_back(x.k - q);
  }
  ks.push_back(0);
  ks.push_back(K);
  if constexpr (F::linear) {
    ks.clear();
    auto a = f.vs.begin(), b = f.vs.begin();
    while (b != f.vs.end() && b->k < q)
      ++b;
    while (a != f.vs.end() || b != f.vs.end()) {
      i64 k;
      if (b == f.vs.end() || (a != f.vs.end() && a->k < b->k - q))
        k = (a++)->k;
      else if (a == f.vs.end() || b->k - q < a->k)
        k = (b++)->k - q;
      else {
        k = a->k;
        ++a;
        ++b;
      }
      if (ks.empty() || ks.back() != k)
        ks.push_back(k);
    }
  } else {
    std::sort(ks.begin(), ks.end());
    ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
  }
  while (!ks.empty() && ks.back() > K)
    ks.pop_back();

  // Sliding max of interior vertex values over windows (g, g+q):
  // process candidates in ascending g with a monotone deque of vertex
  // iterators whose k lies in the open window.  Edge evaluations use
  // monotone walkers (g and g+q are both ascending).
  std::deque<decltype(f.vs.begin())> dq; // vertex iterators, values decreasing
  auto next_v = f.vs.begin();            // next vertex to admit
  Walker w_lo(f), w_hi(f);
  auto win_val = [&](i64 g) -> i64 {
    i64 hi = std::min(K, g + q);
    while (next_v != f.vs.end() && next_v->k < hi) {
      // admit only vertices strictly inside (g, hi); ones at <= g are
      // evicted below
      while (!dq.empty() && dq.back()->v <= next_v->v)
        dq.pop_back();
      dq.push_back(next_v);
      ++next_v;
    }
    while (!dq.empty() && dq.front()->k <= g)
      dq.pop_front();
    i64 m = std::max(w_lo.eval(g), w_hi.eval(hi));
    if (!dq.empty())
      m = std::max(m, dq.front()->v);
    return m;
  };

  F out;
  out.vs.reserve(ks.size() * 2);
  std::vector<typename F::V> extra;
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
          if ((dl > 0) ? (dm > 0) : (dm < 0))
            lo = mid;
          else
            hi2 = mid;
        }
        // evaluate exactly with a fresh scan (rare path)
        auto exact_at = [&](i64 gg) -> i64 {
          i64 hh = std::min(K, gg + q);
          if constexpr (requires { f.vs.vertex_argmax(gg, hh); })
            return std::max(
                {f.eval(gg), f.eval(hh), f.vs.vertex_argmax(gg, hh).second});
          i64 m = std::max(f.eval(gg), f.eval(hh));
          for (const auto &x : f.vs) {
            if (x.k > gg && x.k < hh)
              m = std::max(m, x.v);
            if (x.k >= hh)
              break;
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
  for (const auto &x : extra)
    out.vs.push_back(x);
  out.vs.sort();
  out.normalize();
  return out;
}

#include "policies.inc"

// ---------------------------------------------------------------------- //
// Sweep kernel.
// ---------------------------------------------------------------------- //

struct Event {
  double pos;
  int side;                // 0 = empirical, 1 = theoretical
  i64 count;               // integer units
  size_t index = SIZE_MAX; // caller's position index (optional for cost-only)
};

struct Match {
  size_t empirical, theoretical;
  i64 count;
};
struct TraceResult {
  i64 profit = 0;
  std::vector<Match> matches;
  size_t records = 0, vertices = 0;
};

template <class F> struct BasicTraceRecord {
  F before[2], profits;
  i64 own_limit = 0, end[2] = {0, 0};
};
template <class F> struct BasicTraceStorage {
  struct Block {
    size_t index;
    i64 end;
  };
  std::vector<BasicTraceRecord<F>> records;
  std::vector<Block> blocks[2];
  i64 end[2] = {0, 0};
};
struct NoTraceStorage {};

// Maximum of an exact PL function on an integer interval; endpoints and
// stored vertices suffice. Select one consistent predecessor at ties.
template <class F>
inline std::pair<i64, i64> argmax(const F &f, i64 lo, i64 hi) {
  if (lo > hi)
    return {0, NEG_INF};
  std::pair<i64, i64> best{lo, f.eval(lo)};
  auto consider = [&](i64 k, i64 value) {
    if (value > best.second)
      best = {k, value};
  };
  consider(hi, f.eval(hi));
  if constexpr (requires { f.vs.vertex_argmax(lo, hi); }) {
    auto candidate = f.vs.vertex_argmax(lo, hi);
    consider(candidate.first, candidate.second);
    return best;
  }
  for (const auto &v : f.vs)
    if (v.k >= lo && v.k <= hi)
      consider(v.k, v.v);
  return best;
}

template <class F> struct BasicSideState {
  std::deque<std::pair<double, i64>> blocks; // oldest first (pos, count)
  F V = F::constant(0, 0);
};

// T(k): newest-rank prefix profit sums of st's pendings against z, as a
// PL function of k (vertices at cumulative block counts; slopes = per-unit
// profits, integer).  ProfitFn: (pending_pos, z) -> i64.
template <class F, typename ProfitFn>
F rank_prefix_profits(const BasicSideState<F> &st, double z, ProfitFn profit) {
  F T;
  T.vs.push_back({0, 0});
  i64 k = 0, acc = 0;
  for (size_t i = st.blocks.size(); i-- > 0;) {
    const auto &[pos, cnt] = st.blocks[i];
    i64 pi = profit(pos, z);
    acc += pi * cnt;
    k += cnt;
    T.vs.push_back({k, acc});
  }
  T.normalize();
  return T;
}

template <bool Traceback = false, class Policy = ReferencePolicy, class F,
          typename ProfitFn>
void sweep_step(BasicSideState<F> &own, BasicSideState<F> &opp, double z, i64 q,
                ProfitFn profit, double radius,
                BasicTraceRecord<F> *trace = nullptr, int own_side = 0) {
  if constexpr (Traceback) {
    trace->before[own_side] = own.V;
    trace->before[1 - own_side] = opp.V;
  }
  auto prune = [&](BasicSideState<F> &st) {
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
  auto T = Policy::template curve<F>(opp, z, profit);
  if constexpr (Traceback) {
    trace->profits = Policy::template materialize<F>(T);
    trace->own_limit = own.V.K();
  }
  F VT = Policy::template combine<F>(opp.V, T);
  F newVopp = Policy::template window<F>(VT, q, T);

  // Flip: match exactly j pendings (all remaining), keep m = q - j fresh
  // units pending on own: flip(m) = VT(j), m = q - j, 0 <= j <= min(K,q).
  // flip is VT reflected: vertex (j, v) -> (q - j, v).
  const i64 jmax = std::min(opp.V.K(), q);
  F flip;
  {
    // flip(m) = VT(q - m) for m in [q - jmax, q]; unreachable below
    // (j > jmax would need more pendings than exist): NEG_INF with the
    // adjacent-integer drop so interpolation never manufactures values.
    std::vector<typename F::V> rv;
    for (auto it = VT.vs.rbegin(); it != VT.vs.rend(); ++it)
      if (it->k <= jmax)
        rv.push_back({q - it->k, it->v});
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
    F fe = Policy::copy_flip(flip);
    i64 Ko = own.V.K();
    if (fe.K() < Ko) {
      if (fe.K() + 1 <= Ko)
        fe.vs.push_back({fe.K() + 1, NEG_INF});
      if (fe.K() < Ko)
        fe.vs.push_back({Ko, NEG_INF});
      fe.normalize();
    }
    own.V.pointwise_max(fe);
  }

  opp.V = newVopp;

  // Neutral-state sync: V_own(0) and V_opp(0) are the same logical state.
  i64 best0 = std::max(own.V.eval(0), opp.V.eval(0));
  if (own.V.vs.front().v < best0)
    own.V.set_front_value(best0);
  if (opp.V.vs.front().v < best0)
    opp.V.set_front_value(best0);
}

template <bool Traceback = false, class F = PLF, class Policy = ReferencePolicy,
          typename ProfitFn>
auto sweep_solve(const std::vector<Event> &events, ProfitFn profit,
                 double radius) {
  BasicSideState<F> sides[2];
  // The false specialization has no traceback vectors or recording work.
  std::conditional_t<Traceback, BasicTraceStorage<F>, NoTraceStorage> trace;
  if constexpr (Traceback)
    trace.records.resize(events.size());
  for (size_t i = 0; i < events.size(); ++i) {
    const Event &e = events[i];
    Policy::event(e, profit);
    if (e.count <= 0)
      continue;
    if constexpr (Traceback) {
      auto &record = trace.records[i];
      record.end[0] = trace.end[0];
      record.end[1] = trace.end[1];
      sweep_step<true, Policy>(sides[e.side], sides[1 - e.side], e.pos, e.count,
                               profit, radius, &record, e.side);
      trace.end[e.side] += e.count;
      trace.blocks[e.side].push_back(
          {e.index == SIZE_MAX ? i : e.index, trace.end[e.side]});
    } else {
      sweep_step<false, Policy>(sides[e.side], sides[1 - e.side], e.pos,
                                e.count, profit, radius);
    }
  }
  i64 best = std::max(sides[0].V.max_value(), sides[1].V.max_value());
  if constexpr (!Traceback) {
    return std::max<i64>(best, 0);
  } else {
    TraceResult result;
    result.profit = std::max<i64>(best, 0);
    int side = sides[0].V.max_value() >= sides[1].V.max_value() ? 0 : 1;
    i64 pending = argmax(sides[side].V, 0, sides[side].V.K()).first;
    i64 value = result.profit;
    auto emit = [&](const Event &event, size_t current, int opposite, i64 from,
                    i64 to) {
      const auto &blocks = trace.blocks[opposite];
      auto it = std::upper_bound(
          blocks.begin(), blocks.end(), from,
          [](i64 rank, const typename BasicTraceStorage<F>::Block &block) {
            return rank < block.end;
          });
      while (from < to) {
        if (it == blocks.end())
          throw std::runtime_error("Invalid sweep traceback rank");
        const i64 take = std::min(to, it->end) - from;
        result.matches.push_back(event.side == 1
                                     ? Match{it->index, current, take}
                                     : Match{current, it->index, take});
        from += take;
        ++it;
      }
    };
    for (size_t i = events.size(); i-- > 0;) {
      const auto &e = events[i];
      if (e.count <= 0)
        continue;
      const auto &r = trace.records[i];
      ++result.records;
      result.vertices +=
          r.before[0].vs.size() + r.before[1].vs.size() + r.profits.vs.size();
      const int own = e.side, opp = 1 - own;
      F envelope = r.before[opp];
      envelope.suffix_max();
      envelope.truncate(r.profits.K());
      const F vt = add_pl(envelope, r.profits);
      auto stay = [&](i64 g) {
        auto choice = argmax(vt, g, std::min(vt.K(), g + e.count));
        choice.second -= r.profits.eval(g);
        return choice;
      };
      auto append = [&](i64 g) {
        if (g > r.own_limit + e.count)
          return std::pair<i64, i64>{0, NEG_INF};
        return argmax(r.before[own], std::max<i64>(0, g - e.count),
                      r.before[own].K());
      };
      auto flip = [&](i64 g) {
        if (g > e.count)
          return std::pair<i64, i64>{0, NEG_INF};
        // Some fresh pending units may be discarded after the flip.
        // Neutral sync represents this free-drop choice at zero.
        auto choice = argmax(vt, 0, std::min(vt.K(), e.count - g));
        return std::pair<i64, i64>{e.count - choice.first, choice.second};
      };
      // Neutral sync may select either side's transition at g == 0.
      if (!pending) {
        const i64 own_value = std::max(append(0).second, flip(0).second);
        const i64 opp_value = stay(0).second;
        side = own_value >= opp_value ? own : opp;
      }
      i64 k = 0, predecessor = 0;
      if (side == own) {
        const auto choice = append(pending);
        const auto flipped = flip(pending);
        if (choice.second >= flipped.second) {
          if (choice.second != value)
            throw std::runtime_error("Invalid sweep append traceback");
          pending = choice.first;
          value = r.before[own].eval(pending);
          continue;
        }
        if (flipped.second != value)
          throw std::runtime_error("Invalid sweep flip traceback at event " +
                                   std::to_string(i) +
                                   ", pending=" + std::to_string(pending) +
                                   ", value=" + std::to_string(value) +
                                   ", flip=" + std::to_string(flipped.second) +
                                   ", append=" + std::to_string(choice.second));
        k = e.count - flipped.first;
        const auto previous = argmax(r.before[opp], k, r.before[opp].K());
        predecessor = previous.first;
        const size_t index = e.index == SIZE_MAX ? i : e.index;
        emit(e, index, opp, r.end[opp] - k, r.end[opp]);
      } else {
        const auto choice = stay(pending);
        if (choice.second != value)
          throw std::runtime_error("Invalid sweep matching traceback");
        k = choice.first;
        predecessor = argmax(r.before[opp], k, r.before[opp].K()).first;
        const size_t index = e.index == SIZE_MAX ? i : e.index;
        emit(e, index, opp, r.end[opp] - k, r.end[opp] - pending);
      }
      side = opp;
      pending = predecessor;
      value = r.before[opp].eval(pending);
    }
    if (pending || value)
      throw std::runtime_error("Invalid sweep initial traceback state");
    return result;
  }
}

// ---------------------------------------------------------------------- //
// Marginals: forward pass with snapshots + adjoint (completion) pass.
// gain(z) = best profit improvement from ONE extra theoretical unit at
// position z, with the extra unit matched to an empirical unit on its LEFT
// (this orientation) or unmatched.  The caller runs both orientations
// (mirror positions) and takes max(gain_fwd, gain_mir, 0).
// Validated against oracle re-solves in prototypes/wp_sweep_prototype.py.
// ---------------------------------------------------------------------- //

template <class F> inline F reflect_pl(const F &f) {
  // r(k) = f(K - k)
  F r;
  i64 K = f.K();
  if constexpr (requires { f.vs.reflected(K); }) {
    r.vs = f.vs.reflected(K);
    return r;
  }
  r.vs.reserve(f.vs.size());
  for (auto it = f.vs.rbegin(); it != f.vs.rend(); ++it)
    r.vs.push_back({K - it->k, it->v});
  r.normalize();
  return r;
}

template <class F> inline F prefix_max_pl(const F &f) {
  // g(k) = max_{k' <= k} f(k')
  F r = reflect_pl(f);
  r.suffix_max();
  return reflect_pl(r);
}

// g(k) = max_{max(0, k-q) <= k' <= k} f(k')
template <class F> inline F windowed_max_left(const F &f, i64 q) {
  return reflect_pl(windowed_max(reflect_pl(f), q));
}

// g(m) = f(min(m + q, K)), domain [0, K]
template <class F> inline F shift_left_clamp(const F &f, i64 q) {
  if constexpr (requires(F out) { out.vs.shift_left_clamp(q); }) {
    F out = f;
    out.vs.shift_left_clamp(q);
    return out;
  }
  i64 K = f.K();
  F out;
  for (const auto &x : f.vs) {
    i64 m = x.k - q;
    if (m < 0)
      continue;
    out.vs.push_back({m, x.v});
  }
  if (out.vs.empty() || out.vs.front().k != 0)
    out.vs.insert(out.vs.begin(), {0, f.eval(std::min(q, K))});
  if (out.vs.back().k < K)
    out.vs.push_back({K, f.eval(K)});
  out.normalize();
  return out;
}

// Extend a profile's domain to K_new by repeating its last value.
template <class F> inline F extend_flat(const F &f, i64 K_new) {
  F out = f;
  if constexpr (requires { out.vs.extend_flat(K_new); }) {
    if (K_new > out.K())
      out.vs.extend_flat(K_new);
    return out;
  }
  if (K_new > out.K())
    out.vs.push_back({K_new, out.vs.back().v});
  out.normalize();
  return out;
}

template <class F> struct BasicSnapshot {
  // Post-prune, pre-transition state at each event (index i), plus final.
  BasicSideState<F> sides[2];
};

template <class F = PLF, class Policy = ReferencePolicy, typename ProfitFn>
i64 forward_store(const std::vector<Event> &events, ProfitFn profit,
                  double radius, std::vector<BasicSnapshot<F>> &snaps) {
  BasicSideState<F> sides[2];
  snaps.clear();
  snaps.reserve(events.size() + 1);
  auto prune_side = [&](BasicSideState<F> &st, double z) {
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
  for (const Event &e : events) {
    prune_side(sides[0], e.pos);
    prune_side(sides[1], e.pos);
    snaps.push_back({{sides[0], sides[1]}});
    Policy::event(e, profit);
    // transition (sweep_step body minus the prune it normally does; the
    // radius below never triggers because we just pruned at this z)
    sweep_step<false, Policy>(sides[e.side], sides[1 - e.side], e.pos, e.count,
                              profit, std::numeric_limits<double>::max());
  }
  snaps.push_back({{sides[0], sides[1]}});
  i64 best = std::max(sides[0].V.max_value(), sides[1].V.max_value());
  return std::max<i64>(best, 0);
}

// Backward completion pass; composes the extra-unit value at every
// theoretical event and records it into `composed` (same length as events,
// NEG_INF elsewhere).  C[side] entering iteration i is the completion
// function on snapshot(i+1) domains; the loop rewinds it to snapshot(i).
template <class F = PLF, typename ProfitFn>
void adjoint_compose(const std::vector<Event> &events, ProfitFn profit,
                     const std::vector<BasicSnapshot<F>> &snaps,
                     std::vector<i64> &composed) {
  const size_t n = events.size();
  composed.assign(n, NEG_INF);
  F C[2] = {F::constant(snaps[n].sides[0].V.K(), 0),
            F::constant(snaps[n].sides[1].V.K(), 0)};
  for (size_t i = n; i-- > 0;) {
    const Event &e = events[i];
    const int own_s = e.side, opp_s = 1 - e.side;
    const i64 q = e.count;
    const BasicSnapshot<F> &pre = snaps[i];
    const BasicSnapshot<F> &post = snaps[i + 1];
    // Prune adjoint between post-transition of event i and snapshot(i+1):
    // snapshot(i+1) was taken after the NEXT event's prune, so extend C
    // flat to the post-transition domains.
    const i64 K_own_post = pre.sides[own_s].V.K() + q;
    const i64 K_opp_post = pre.sides[opp_s].V.K();
    F C_own_after = extend_flat(C[own_s], K_own_post);
    F C_opp_after = extend_flat(C[opp_s], K_opp_post);

    // T over the OPP side's pendings before event i (post-prune).
    F T = rank_prefix_profits(pre.sides[opp_s], e.pos, profit);

    // ---- composition: one extra unit of side `1` (theoretical) at this
    // event, consumed here (matched to an opp pending or kept among the
    // fresh block's pendings up to q real slots).
    if (e.side == 1) {
      const i64 qh = q + 1;
      F Vt = pre.sides[opp_s].V;
      Vt.suffix_max();
      F VT = add_pl(Vt, T);
      // (A') stay on opp side: max_g [win_max_{[g,g+qh]}(VT) - T + C_opp]
      F X = windowed_max(VT, qh);
      F comp = add_pl(sub_pl(X, T), C_opp_after);
      i64 best = comp.max_value();
      // (B') flip: j matched in [1..min(qh, K)], keep m = qh - j <= q
      // (m <= q means the EXTRA unit was consumed, not left pending):
      // value = VT(j) + C_own_after(qh - j).  Vertex-based max.
      {
        i64 j_hi = std::min<i64>(qh, VT.K());
        i64 j_lo = std::max<i64>(1, qh - std::min<i64>(q, C_own_after.K()));
        if (j_lo <= j_hi) {
          std::vector<i64> js;
          for (auto &x : VT.vs)
            if (x.k >= j_lo && x.k <= j_hi)
              js.push_back(x.k);
          for (auto &x : C_own_after.vs) {
            i64 j = qh - x.k;
            if (j >= j_lo && j <= j_hi)
              js.push_back(j);
          }
          js.push_back(j_lo);
          js.push_back(j_hi);
          for (i64 j : js) {
            i64 v = VT.eval(j) + C_own_after.eval(qh - j);
            if (v > best)
              best = v;
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
    F CT = sub_pl(C_opp_after, T);
    F H = add_pl(T, windowed_max_left(CT, q));
    F C_opp_pre = prefix_max_pl(H);
    {
      // flip term as a PL profile over k: C_pre(k) >= max_{j <= min(q, k)}
      // D(j), D(j) = T(j) + C_own_after(q - j), reachable j in [j_lo, j_hi].
      i64 j_hi = std::min<i64>(q, T.K());
      i64 j_lo = std::max<i64>(0, q - C_own_after.K());
      if (j_lo <= j_hi) {
        // D on full domain [0, j_hi], NEG_INF below j_lo with the
        // adjacent-integer drop so interpolation stays honest.
        std::vector<i64> ks;
        for (auto &x : T.vs)
          if (x.k >= j_lo && x.k <= j_hi)
            ks.push_back(x.k);
        for (auto &x : C_own_after.vs) {
          i64 j = q - x.k;
          if (j >= j_lo && j <= j_hi)
            ks.push_back(j);
        }
        ks.push_back(j_lo);
        ks.push_back(j_hi);
        std::sort(ks.begin(), ks.end());
        ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
        F D;
        if (j_lo > 0) {
          D.vs.push_back({0, NEG_INF});
          if (j_lo > 1)
            D.vs.push_back({j_lo - 1, NEG_INF});
        }
        for (i64 j : ks)
          D.vs.push_back({j, T.eval(j) + C_own_after.eval(q - j)});
        D.normalize();
        // flipk(k) = PM(min(k, j_hi)) = flat extension of prefix max
        F flipk = extend_flat(prefix_max_pl(D), C_opp_pre.K());
        C_opp_pre.pointwise_max(flipk);
      }
    }
    C_opp_pre.truncate(pre.sides[opp_s].V.K());
    // Own side (the block was appended): C_pre(m) = max_{m' <= m+q} C(m')
    F C_own_pre = shift_left_clamp(prefix_max_pl(C_own_after), q);
    C_own_pre.truncate(pre.sides[own_s].V.K());

    C[own_s] = std::move(C_own_pre);
    C[opp_s] = std::move(C_opp_pre);
  }
}

using SideState = BasicSideState<PLF>;
using Snapshot = BasicSnapshot<PLF>;
using TraceRecord = BasicTraceRecord<PLF>;
using TraceStorage = BasicTraceStorage<PLF>;

} // namespace sweep_trials_kernel

#endif // WNET_TRIAL_SWEEP_KERNEL_HPP
