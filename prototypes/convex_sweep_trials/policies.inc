// Included inside the trial-kernel namespace after PL primitives.
struct ReferencePolicy {
  template <class E, class Profit> static void event(const E &, Profit &) {}
  template <class F, class Side, class Profit>
  static F curve(const Side &side, double z, const Profit &profit) {
    // Same code as production rank_prefix_profits; SideState is declared later.
    F out;
    TRIAL_COUNT(profit_knots, side.blocks.size());
    out.vs.push_back({0, 0});
    i64 k = 0, acc = 0;
    for (size_t i = side.blocks.size(); i-- > 0;) {
      const auto [pos, count] = side.blocks[i];
      acc += profit(pos, z) * count;
      k += count;
      out.vs.push_back({k, acc});
    }
    out.normalize();
    return out;
  }
  template <class F> static F materialize(const F &curve) { return curve; }
  template <class F> static F combine(const F &a, const F &b) {
    return add_pl(a, b);
  }
  template <class F> static F window(const F &f, i64 q, const F &t) {
    return sub_pl(windowed_max(f, q), t);
  }
  template <class F> static const F &copy_flip(F &f) { return f; }
};
template <class Curve> size_t curve_size(const Curve &t) {
  if constexpr (requires { t.knot_count(); })
    return t.knot_count();
  else
    return t.vs.size();
}
template <class Curve> i64 curve_key(const Curve &t, size_t i) {
  if constexpr (requires { t.knot(i); })
    return t.knot(i);
  else
    return t.vs[i].k;
}
// A single window pass emits max-window(f)-T directly. Candidate keys include
// T's knots, so subtracting a non-affine T never loses a required breakpoint.
template <class F, class Curve>
F exact_window_sub(const F &f, i64 q, const Curve &t) {
  auto w = exact_window(f, q);
  F out;
  size_t a = 0, b = 0;
  Walker<F> walk(w);
  while (a < w.vs.size() || b < curve_size(t)) {
    const i64 ka = a < w.vs.size() ? w.vs[a].k : INT64_MAX,
              kb = b < curve_size(t) ? curve_key(t, b) : INT64_MAX,
              k = std::min(ka, kb);
    if (ka == k)
      ++a;
    if (kb == k)
      ++b;
    out.vs.push_back({k, walk.eval(k) - t.eval(k)});
  }
  out.normalize();
  return out;
}
template <class F, class Curve>
F fused_window_sub(const F &f, i64 q, const Curve &t) {
  if constexpr (F::exact_window)
    return exact_window_sub(f, q, t);
  TRIAL_COUNT(window_passes, 1);
  static thread_local std::vector<i64> keys;
  keys.clear();
  const i64 K = f.K();
  size_t a = 0, b = 0, c = 0;
  while (b < f.vs.size() && f.vs[b].k < q)
    ++b;
  while (a < f.vs.size() || b < f.vs.size() || c < curve_size(t)) {
    const i64 ka =
        a < f.vs.size() ? f.vs[a].k : std::numeric_limits<i64>::max();
    const i64 kb =
        b < f.vs.size() ? f.vs[b].k - q : std::numeric_limits<i64>::max();
    const i64 kc =
        c < curve_size(t) ? curve_key(t, c) : std::numeric_limits<i64>::max();
    const i64 k = std::min({ka, kb, kc});
    if (ka == k)
      ++a;
    if (kb == k)
      ++b;
    if (kc == k)
      ++c;
    if (keys.empty() || keys.back() != k)
      keys.push_back(k);
  }
  F out;
  out.vs.reserve(keys.size() * 2);
  Walker<F> wlo(f), whi(f), wc1(f), wc2(f), wc3(f), wc4(f);
  std::deque<size_t> dq;
  size_t next = 0;
  i64 prev = -1;
  for (i64 g : keys) {
    const i64 hi = std::min(K, g + q);
    while (next < f.vs.size() && f.vs[next].k < hi) {
      while (!dq.empty() && f.vs[dq.back()].v <= f.vs[next].v)
        dq.pop_back();
      dq.push_back(next++);
    }
    // No vertex or shifted vertex lies strictly between consecutive keys.
    // This maximum is valid throughout that open interval.
    while (!dq.empty() && f.vs[dq.front()].k <= prev)
      dq.pop_front();
    const i64 interior = dq.empty() ? NEG_INF : f.vs[dq.front()].v;
    while (!dq.empty() && f.vs[dq.front()].k <= g)
      dq.pop_front();
    i64 value = std::max(wlo.eval(g), whi.eval(hi));
    if (!dq.empty())
      value = std::max(value, f.vs[dq.front()].v);
    if (prev >= 0 && g - prev > 1) {
      const i64 dl = wc1.eval(prev) - wc2.eval(std::min(K, prev + q));
      const i64 dr = wc3.eval(g) - wc4.eval(hi);
      if ((dl > 0 && dr < 0) || (dl < 0 && dr > 0)) {
        i64 lo = prev, right = g;
        while (right - lo > 1) {
          const i64 mid = lo + (right - lo) / 2,
                    d = f.eval(mid) - f.eval(std::min(K, mid + q));
          if (dl > 0 ? d > 0 : d < 0)
            lo = mid;
          else
            right = mid;
        }
        for (i64 k : {lo, right})
          out.vs.push_back(
              {k, std::max({f.eval(k), f.eval(std::min(K, k + q)), interior}) -
                      t.eval(k)});
      }
    }
    out.vs.push_back({g, value - t.eval(g)});
    prev = g;
  }
  out.normalize();
  return out;
}
struct FusedPolicy : ReferencePolicy {
  template <class F, class Curve>
  static F window(const F &f, i64 q, const Curve &t) {
    return fused_window_sub(f, q, t);
  }
  template <class F> static F &&copy_flip(F &f) { return std::move(f); }
};
struct MomentPolicy : FusedPolicy {
  template <class E, class Profit>
  static void event(const E &e, Profit &profit) {
    profit.bank->own = e.side;
    profit.bank->end[e.side] += e.count;
  }
  template <class F, class Side, class Profit>
  static auto curve(const Side &side, double z, const Profit &profit) {
    const int opp = 1 - profit.bank->own;
    return sweep_trials::MomentCurve(profit.bank->sides[opp],
                                     profit.bank->end[opp], side.V.K(), i64(z),
                                     profit.tau);
  }
  template <class F> static F materialize(const sweep_trials::MomentCurve &t) {
    F f;
    f.vs.reserve(t.knot_count());
    for (size_t i = 0; i < t.knot_count(); ++i) {
      const i64 k = t.knot(i);
      f.vs.push_back({k, t.eval(k)});
    }
    f.normalize();
    return f;
  }
  template <class F>
  static F combine(const F &f, const sweep_trials::MomentCurve &t) {
    F out;
    out.vs.reserve(f.vs.size() + t.knot_count());
    size_t a = 0, b = 0;
    Walker<F> walk(f);
    while (a < f.vs.size() || b < t.knot_count()) {
      const i64 ka =
          a < f.vs.size() ? f.vs[a].k : std::numeric_limits<i64>::max();
      const i64 kb =
          b < t.knot_count() ? t.knot(b) : std::numeric_limits<i64>::max();
      const i64 k = std::min(ka, kb);
      if (ka == k)
        ++a;
      if (kb == k)
        ++b;
      out.vs.push_back({k, walk.eval(k) + t.eval(k)});
    }
    out.normalize();
    return out;
  }
};

template <class Curve> i64 curve_value(const Curve &t, size_t i) {
  if constexpr (requires { t.knot_value(i); })
    return t.knot_value(i);
  else if constexpr (requires { t.vs[i].v; })
    return t.vs[i].v;
  else
    return t.eval(curve_key(t, i));
}
// Stream the existing window operation into the subtraction merge. T's knots
// are inserted by interpolation, rather than redoing window work at each one.
template <class F, class Curve>
F stream_window_sub(const F &f, i64 q, const Curve &t) {
  if constexpr (F::exact_window)
    return exact_window_sub(f, q, t);
  TRIAL_COUNT(window_passes, 1);
  static thread_local std::vector<i64> keys;
  keys.clear();
  size_t a = 0, b = 0;
  while (b < f.vs.size() && f.vs[b].k < q)
    ++b;
  while (a < f.vs.size() || b < f.vs.size()) {
    const i64 ka = a < f.vs.size() ? f.vs[a].k : INT64_MAX;
    const i64 kb = b < f.vs.size() ? f.vs[b].k - q : INT64_MAX;
    const i64 key = std::min(ka, kb);
    if (ka == key)
      ++a;
    if (kb == key)
      ++b;
    if (keys.empty() || keys.back() != key)
      keys.push_back(key);
  }
  F out;
  out.vs.reserve(keys.size() + curve_size(t));
  size_t cursor = 0;
  i64 previous = -1, previous_value = 0;
  auto emit = [&](i64 key, i64 value) {
    if (key == previous) {
      out.vs.back().v = std::max(out.vs.back().v, value - t.eval(key));
      previous_value = std::max(previous_value, value);
      return;
    }
    while (cursor < curve_size(t) && curve_key(t, cursor) < key) {
      const i64 k = curve_key(t, cursor);
      if (previous < 0)
        throw std::logic_error("stream subtraction missing origin");
      const i64 v =
          previous_value +
          mul_div_trunc(value - previous_value, k - previous, key - previous);
      out.vs.push_back({k, v - curve_value(t, cursor)});
      ++cursor;
    }
    i64 tv;
    if (cursor < curve_size(t) && curve_key(t, cursor) == key)
      tv = curve_value(t, cursor++);
    else
      tv = t.eval(key);
    out.vs.push_back({key, value - tv});
    previous = key;
    previous_value = value;
  };
  const i64 K = f.K();
  Walker<F> wlo(f), whi(f), wc1(f), wc2(f), wc3(f), wc4(f);
  std::deque<size_t> dq;
  size_t next = 0;
  i64 prev = -1;
  for (i64 g : keys) {
    const i64 hi = std::min(K, g + q);
    while (next < f.vs.size() && f.vs[next].k < hi) {
      while (!dq.empty() && f.vs[dq.back()].v <= f.vs[next].v)
        dq.pop_back();
      dq.push_back(next++);
    }
    while (!dq.empty() && f.vs[dq.front()].k <= prev)
      dq.pop_front();
    const i64 interior = dq.empty() ? NEG_INF : f.vs[dq.front()].v;
    while (!dq.empty() && f.vs[dq.front()].k <= g)
      dq.pop_front();
    i64 value = std::max(wlo.eval(g), whi.eval(hi));
    if (!dq.empty())
      value = std::max(value, f.vs[dq.front()].v);
    if (prev >= 0 && g - prev > 1) {
      const i64 dl = wc1.eval(prev) - wc2.eval(std::min(K, prev + q));
      const i64 dr = wc3.eval(g) - wc4.eval(hi);
      if ((dl > 0 && dr < 0) || (dl < 0 && dr > 0)) {
        i64 lo = prev, right = g;
        while (right - lo > 1) {
          const i64 mid = lo + (right - lo) / 2,
                    d = f.eval(mid) - f.eval(std::min(K, mid + q));
          if (dl > 0 ? d > 0 : d < 0)
            lo = mid;
          else
            right = mid;
        }
        for (i64 k : {lo, right})
          emit(k, std::max({f.eval(k), f.eval(std::min(K, k + q)), interior}));
      }
    }
    emit(g, value);
    prev = g;
  }
  out.normalize();
  return out;
}
struct StreamPolicy : FusedPolicy {
  template <class F, class Curve>
  static F window(const F &f, i64 q, const Curve &t) {
    return stream_window_sub(f, q, t);
  }
};
struct FastMomentPolicy : StreamPolicy {
  template <class E, class Profit> static void event(const E &e, Profit &p) {
    MomentPolicy::event(e, p);
  }
  template <class F, class Side, class Profit>
  static auto curve(const Side &side, double z, const Profit &p) {
    const int opp = 1 - p.bank->own;
    return sweep_trials::FastMomentCurve(p.bank->sides[opp], p.bank->end[opp],
                                         side.V.K(), i64(z), p.tau);
  }
  template <class F>
  static F materialize(const sweep_trials::FastMomentCurve &t) {
    F out;
    out.vs.reserve(t.knot_count());
    for (size_t i = 0; i < t.knot_count(); ++i)
      out.vs.push_back({t.knot(i), t.knot_value(i)});
    out.normalize();
    return out;
  }
  template <class F>
  static F combine(const F &f, const sweep_trials::FastMomentCurve &t) {
    F out;
    out.vs.reserve(f.vs.size() + t.knot_count());
    size_t a = 0, b = 0;
    Walker<F> walk(f);
    while (a < f.vs.size() || b < t.knot_count()) {
      const i64 ka = a < f.vs.size() ? f.vs[a].k : INT64_MAX,
                kb = b < t.knot_count() ? t.knot(b) : INT64_MAX,
                k = std::min(ka, kb);
      const i64 tv = kb == k ? t.knot_value(b) : t.eval(k);
      if (ka == k)
        ++a;
      if (kb == k)
        ++b;
      out.vs.push_back({k, walk.eval(k) + tv});
    }
    out.normalize();
    return out;
  }
};
