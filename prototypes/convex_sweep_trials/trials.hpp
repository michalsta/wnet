#ifndef SWEEP_TRIALS_HPP
#define SWEEP_TRIALS_HPP
#include "kernel.hpp"
#include "wnet/convex_sweep_dual.hpp"
#include "wnet/monge_cost.hpp"
#include <string>
namespace sweep_trials {
namespace k = sweep_trials_kernel;
using W = wide_int::wide_t;
struct Variant {
  bool group = false, linear = false, fused = false, moments = false,
       stream = false;
};
inline Variant variant(const std::string &name) {
  if (name == "baseline")
    return {};
  if (name == "group")
    return {true};
  if (name == "linear")
    return {false, true};
  if (name == "fused")
    return {false, false, true};
  if (name == "moments")
    return {false, false, true, true};
  if (name == "group-linear")
    return {true, true};
  if (name == "group-fused")
    return {true, false, true};
  if (name == "linear-fused")
    return {false, true, true};
  if (name == "group-linear-fused")
    return {true, true, true};
  if (name == "all")
    return {true, true, true, true};
  if (name == "stream")
    return {false, false, true, false, true};
  if (name == "moment-stream")
    return {false, false, true, true, true};
  if (name == "linear-stream")
    return {false, true, true, false, true};
  if (name == "group-stream")
    return {true, false, true, false, true};
  if (name == "group-linear-stream")
    return {true, true, true, false, true};
  if (name == "all-stream")
    return {true, true, true, true, true};
  throw std::invalid_argument("unknown trial variant");
}
struct Input {
  std::vector<double> pos;
  std::vector<I> e, t;
  std::vector<uint8_t> theo;
  monge_cost::GridPower model;
  I tau = 0;
  double p = 2;
  void grid(const std::vector<double> &real, I scale, double order, I reward) {
    p = order;
    tau = reward;
    model.reset(real, scale, order, pos);
  }
};
struct Part {
  size_t index;
  I count;
};
struct Events {
  std::vector<k::Event> events;
  std::vector<std::vector<Part>> parts;
};
inline Events prepare(const Input &in, bool grouped, bool probes = false,
                      bool mirror = false, bool mapping = true) {
  Events out;
  auto append = [&](size_t i, int side, I count) {
    const double pos = mirror ? -in.pos[i] : in.pos[i];
    // At a shared coordinate, groups of the two sides can interleave.
    size_t found = out.events.size();
    if (grouped)
      for (size_t j = out.events.size(); j-- > 0 && out.events[j].pos == pos;)
        if (out.events[j].side == side) {
          found = j;
          break;
        }
    if (found == out.events.size()) {
      out.events.push_back({pos, side, count, found});
      if (mapping)
        out.parts.push_back({{i, count}});
    } else {
      out.events[found].count += count;
      if (mapping)
        out.parts[found].push_back({i, count});
    }
  };
  auto emit = [&](size_t i) {
    if (in.e[i] > 0)
      append(i, 0, in.e[i]);
    if (in.t[i] > 0 || (probes && in.theo[i]))
      append(i, 1, in.t[i]);
  };
  if (mirror)
    for (size_t i = in.pos.size(); i-- > 0;)
      emit(i);
  else
    for (size_t i = 0; i < in.pos.size(); ++i)
      emit(i);
  return out;
}
struct Profit {
  const monge_cost::GridPower *model;
  I tau;
  MomentBank *bank = nullptr;
  I operator()(double a, double b) const { return tau - model->cost(a, b); }
};
inline std::vector<convex_sweep::Match>
expand(const Events &grouped, const std::vector<k::Match> &matches) {
  std::vector<convex_sweep::Match> out;
  std::vector<size_t> cursor(grouped.parts.size());
  std::vector<I> used(grouped.parts.size());
  for (const auto &m : matches) {
    I left = m.count;
    while (left) {
      auto get = [&](size_t id) -> const Part & {
        while (cursor[id] < grouped.parts[id].size() &&
               used[id] == grouped.parts[id][cursor[id]].count) {
          ++cursor[id];
          used[id] = 0;
        }
        if (cursor[id] >= grouped.parts[id].size())
          throw std::logic_error("group traceback exceeds supply");
        return grouped.parts[id][cursor[id]];
      };
      const auto &e = get(m.empirical);
      const auto &t = get(m.theoretical);
      const I take = std::min(
          {left, e.count - used[m.empirical], t.count - used[m.theoretical]});
      out.push_back({e.index, t.index, take});
      used[m.empirical] += take;
      used[m.theoretical] += take;
      left -= take;
    }
  }
  return out;
}
struct Result {
  I profit = 0;
  size_t events = 0, trace_vertices = 0;
  std::vector<convex_sweep::Match> matches;
};
template <class F, class Policy, bool Trace>
Result solve_impl(const Input &in, const Variant &v) {
  auto grouped = prepare(in, v.group, false, false, Trace);
  std::unique_ptr<MomentBank> bank;
  if (v.moments)
    bank = std::make_unique<MomentBank>(grouped.events);
  Profit profit{&in.model, in.tau, bank.get()};
  Result out;
  out.events = grouped.events.size();
  if constexpr (Trace) {
    auto r = k::sweep_solve<true, F, Policy>(grouped.events, profit,
                                             in.model.radius(in.tau));
    out.profit = r.profit;
    out.trace_vertices = r.vertices;
    out.matches = expand(grouped, r.matches);
  } else
    out.profit = k::sweep_solve<false, F, Policy>(grouped.events, profit,
                                                  in.model.radius(in.tau));
  return out;
}
template <class F, bool Trace>
Result solve_policy(const Input &in, const Variant &v) {
  if (v.moments) {
    if (in.p != 2)
      throw std::invalid_argument("moment policy requires p=2");
    if (v.stream)
      return solve_impl<F, k::FastMomentPolicy, Trace>(in, v);
    return solve_impl<F, k::MomentPolicy, Trace>(in, v);
  }
  if (v.fused) {
    if (v.stream)
      return solve_impl<F, k::StreamPolicy, Trace>(in, v);
    return solve_impl<F, k::FusedPolicy, Trace>(in, v);
  }
  return solve_impl<F, k::ReferencePolicy, Trace>(in, v);
}
template <bool Trace = false> Result solve(const Input &in, const Variant &v) {
  return v.linear ? solve_policy<k::LinearPLF, Trace>(in, v)
                  : solve_policy<k::PLF, Trace>(in, v);
}
inline auto dual(const Input &in, const Result &result) {
  auto cost = [&](double a, double b) { return in.model.cost(a, b); };
  return convex_sweep_dual::recover<I>(in.pos, in.e, in.t, in.tau,
                                       in.model.radius(in.tau), cost,
                                       result.matches, W(result.profit));
}
template <class F, class Policy>
std::vector<I> gains_impl(const Input &in, const Variant &v) {
  std::vector<I> gains(in.pos.size(), 0);
  for (bool mirror : {false, true}) {
    auto grouped = prepare(in, v.group, true, mirror);
    std::unique_ptr<MomentBank> bank;
    if (v.moments)
      bank = std::make_unique<MomentBank>(grouped.events);
    Profit profit{&in.model, in.tau, bank.get()};
    std::vector<k::BasicSnapshot<F>> snaps;
    std::vector<I> composed;
    const I base = k::forward_store<F, Policy>(grouped.events, profit,
                                               in.model.radius(in.tau), snaps);
    // Adjoint uses the existing explicit profit construction; only its
    // forward pass and profile primitives use the new policies here.
    k::adjoint_compose(grouped.events, profit, snaps, composed);
    for (size_t i = 0; i < grouped.events.size(); ++i)
      if (grouped.events[i].side == 1)
        for (const auto &part : grouped.parts[i])
          gains[part.index] = std::max(gains[part.index], composed[i] - base);
  }
  return gains;
}
template <class F>
std::vector<I> gains_policy(const Input &in, const Variant &v) {
  if (v.moments) {
    if (v.stream)
      return gains_impl<F, k::FastMomentPolicy>(in, v);
    return gains_impl<F, k::MomentPolicy>(in, v);
  }
  if (v.fused) {
    if (v.stream)
      return gains_impl<F, k::StreamPolicy>(in, v);
    return gains_impl<F, k::FusedPolicy>(in, v);
  }
  return gains_impl<F, k::ReferencePolicy>(in, v);
}
inline std::vector<I> gains(const Input &in, const Variant &v) {
  if (v.moments && in.p != 2)
    throw std::invalid_argument("moment policy requires p=2");
  // The existing adjoint is not invariant under merged event blocks on
  // real data. Preserve its original event sequence until that dependency
  // is resolved; grouping remains enabled for cost and certificate solves.
  Variant derivative = v;
  derivative.group = false;
  return v.linear ? gains_policy<k::LinearPLF>(in, derivative)
                  : gains_policy<k::PLF>(in, derivative);
}
} // namespace sweep_trials
#endif
