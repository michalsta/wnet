#include "trials.hpp"
#include <cassert>
#include <iostream>
#include <random>
using namespace sweep_trials;
const std::vector<std::string> variants{"baseline",
                                        "group",
                                        "linear",
                                        "fused",
                                        "group-linear",
                                        "group-fused",
                                        "linear-fused",
                                        "group-linear-fused",
                                        "moments",
                                        "all",
                                        "stream",
                                        "moment-stream",
                                        "linear-stream",
                                        "group-stream",
                                        "group-linear-stream",
                                        "all-stream"};
I oracle(const Input &in) {
  std::vector<size_t> emp, theo;
  for (size_t i = 0; i < in.pos.size(); ++i) {
    for (I j = 0; j < in.e[i]; ++j)
      emp.push_back(i);
    for (I j = 0; j < in.t[i]; ++j)
      theo.push_back(i);
  }
  std::vector<I> dp(size_t(1) << theo.size(), -1000000000);
  dp[0] = 0;
  for (auto e : emp) {
    auto next = dp;
    for (size_t mask = 0; mask < dp.size(); ++mask)
      if (dp[mask] >= 0)
        for (size_t t = 0; t < theo.size(); ++t)
          if (!(mask & (size_t(1) << t)))
            next[mask | (size_t(1) << t)] = std::max(
                next[mask | (size_t(1) << t)],
                dp[mask] +
                    static_cast<I>(std::max<W>(
                        W(0), W(in.tau) - in.model.wide_cost(
                                              in.pos[e], in.pos[theo[t]]))));
    dp.swap(next);
  }
  return *std::max_element(dp.begin(), dp.end());
}
void check(const Input &in, bool derivatives) {
  const I expected = oracle(in);
  auto source = prepare(in, false);
  std::vector<convex_sweep::Event> ev;
  for (auto e : source.events)
    ev.push_back({e.pos, e.side, e.count, e.index});
  auto benefit = [&](double a, double b) {
    return in.tau - in.model.cost(a, b);
  };
  assert(convex_sweep::sweep_solve(ev, benefit, in.model.radius(in.tau)) ==
         expected);
  for (const auto &name : variants) {
    const auto v = variant(name);
    if (v.moments && in.p != 2)
      continue;
    try {
      const auto got = solve(in, v);
      if (got.profit != expected)
        throw std::runtime_error("ordinary oracle mismatch");
      const auto trace = solve<true>(in, v);
      if (trace.profit != expected)
        throw std::runtime_error("trace oracle mismatch");
      auto [a, b] = dual(in, trace);
      W objective = 0;
      std::vector<I> e(in.e.size()), t(in.t.size());
      for (auto m : trace.matches) {
        e[m.empirical] += m.count;
        t[m.theoretical] += m.count;
      }
      for (size_t i = 0; i < in.pos.size(); ++i) {
        assert(e[i] <= in.e[i] && t[i] <= in.t[i]);
        objective += W(a[i]) * in.e[i] + W(b[i]) * in.t[i];
        for (size_t j = 0; j < in.pos.size(); ++j)
          assert(!(in.model.wide_cost(in.pos[i], in.pos[j]) - W(in.tau) <
                   W(a[i]) + W(b[j])));
      }
      assert(objective == -W(expected));
      if (derivatives) {
        const auto g = gains(in, v);
        for (size_t i = 0; i < in.pos.size(); ++i)
          if (in.theo[i]) {
            auto extra = in;
            ++extra.t[i];
            const I ref = oracle(extra) - expected;
            if (g[i] != ref)
              throw std::runtime_error(
                  "marginal oracle mismatch at " + std::to_string(i) + ": " +
                  std::to_string(g[i]) + " vs " + std::to_string(ref));
          }
      }
    } catch (const std::exception &e) {
      std::cerr << "variant=" << name << " p=" << in.p << " tau=" << in.tau
                << " error=" << e.what() << '\n';
      for (size_t i = 0; i < in.pos.size(); ++i)
        std::cerr << in.pos[i] << ',' << in.e[i] << ',' << in.t[i] << ' ';
      std::cerr << '\n';
      throw;
    }
  }
}
int main(int argc, char **argv) {
  const size_t trials = argc > 1 ? std::stoul(argv[1]) : 1000;
  std::mt19937 random(20261015);
  for (size_t trial = 0; trial < trials; ++trial) {
    const double p = std::vector<double>{1, 2, 1.5, 2, 3, 2}[trial % 6];
    const size_t n = 2 + random() % 5;
    std::vector<double> real(n);
    Input in;
    in.e.resize(n);
    in.t.resize(n);
    in.theo.resize(n, 1);
    for (size_t i = 0; i < n; ++i) {
      real[i] = (i ? real[i - 1] : 0) + (random() % 5) * .031;
      in.e[i] = random() % 3;
      in.t[i] = random() % 2;
      if (trial % 2) {
        const bool theoretical = random() % 2;
        in.theo[i] = theoretical;
        if (theoretical)
          in.e[i] = 0;
        else
          in.t[i] = 0;
      }
    }
    in.grid(real, 1 + random() % 1000, p, random() % 200);
    check(in, trial < 150);
  }
  // A zero-distance pair is not greedily cancelled: crossing alternatives
  // can be cheaper for squared distance. Group only within a side.
  Input zero;
  zero.e = {1, 1, 0};
  zero.t = {0, 1, 1};
  zero.theo = {1, 1, 1};
  zero.grid({0, 1, 2}, 1, 2, 10);
  check(zero, true);
  // Exact duplicate coordinates and zero-count theoretical probes.
  Input tied;
  tied.e = {2, 1, 0, 0, 1};
  tied.t = {0, 1, 1, 0, 0};
  tied.theo = {1, 1, 1, 1, 1};
  tied.grid({0, 0, 0, .5, 1}, 64, 2, 50);
  check(tied, true);
  // Wide moments are necessary despite locally small profitable costs.
  Input huge;
  huge.e = {1, 0, 1, 0};
  huge.t = {0, 1, 0, 1};
  huge.theo = {1, 1, 1, 1};
  huge.grid({0, 1e-8, 4e7, 4e7 + 1e-8}, 10000000000000000LL, 2, 100);
  check(huge, false);
  // Exercise the 256-bit moment index with a large mass and extreme origin.
  std::vector<k::Event> events{
      {0, 0, 1000000000000000000LL, 0},
      {4503599627370495., 0, 1000000000000000000LL, 1}};
  MomentIndex index(events, 0);
  assert(index.big);
  assert(index.benefit(1000000000000000000LL, 1000000000000000001LL,
                       4503599627370496LL, 100) == 99);
  // A production-budget-sized supply whose distant prefix moment needs
  // more than 128 bits even though only one profitable pair is present.
  Input cancellation;
  cancellation.e = {1000000000000000000LL, 1, 0};
  cancellation.t = {0, 0, 1};
  cancellation.theo = {1, 1, 1};
  cancellation.grid({0, 4503599627370495., 4503599627370495.}, 1, 2, 1);
  for (const auto &name : variants) {
    const auto v = variant(name);
    assert(solve(cancellation, v).profit == 1);
    auto traced = solve<true>(cancellation, v);
    assert(traced.profit == 1);
    dual(cancellation, traced);
    const auto g = gains(cancellation, v);
    assert(g[0] == 1 && g[1] == 0 && g[2] == 0);
  }
  std::cout
      << trials
      << " unrestricted-oracle cases; all variants, traces, dual inequalities, "
         "duplicate grids, wide moments and marginal probes passed\n";
}
