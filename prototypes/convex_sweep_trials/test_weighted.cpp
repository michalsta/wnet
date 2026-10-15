#include "trials.hpp"
#include <cassert>
#include <iostream>
#include <lemon/cost_scaling.h>
#include <lemon/list_graph.h>
#include <random>
using namespace sweep_trials;
I oracle(const Input &in) {
  using G = lemon::ListDigraph;
  G graph;
  auto source = graph.addNode(), sink = graph.addNode();
  std::vector<G::Node> e, t;
  for (size_t i = 0; i < in.pos.size(); ++i) {
    e.push_back(graph.addNode());
    t.push_back(graph.addNode());
  }
  G::ArcMap<I> capacity(graph), cost(graph);
  G::NodeMap<I> supply(graph, 0);
  I total = 0;
  auto arc = [&](G::Node a, G::Node b, I cap, I price) {
    auto edge = graph.addArc(a, b);
    capacity[edge] = cap;
    cost[edge] = price;
  };
  for (size_t i = 0; i < in.pos.size(); ++i) {
    total += in.e[i];
    arc(source, e[i], in.e[i], 0);
    arc(t[i], sink, in.t[i], 0);
    for (size_t j = 0; j < in.pos.size(); ++j) {
      const W c = in.model.wide_cost(in.pos[i], in.pos[j]);
      if (c < W(in.tau))
        arc(e[i], t[j], std::min(in.e[i], in.t[j]), static_cast<I>(c) - in.tau);
    }
  }
  arc(source, sink, total, 0);
  supply[source] = total;
  supply[sink] = -total;
  lemon::CostScaling<G, I, I> solver(graph);
  auto status = solver.upperMap(capacity).costMap(cost).supplyMap(supply).run();
  if (status != decltype(solver)::OPTIMAL)
    throw std::runtime_error("independent flow oracle failed");
  return -solver.totalCost<I>();
}
int main(int argc, char **argv) {
  size_t trials = argc > 1 ? std::stoul(argv[1]) : 2000;
  std::mt19937 random(71);
  for (size_t test = 0; test < trials; ++test) {
    Input in;
    const size_t n = 2 + random() % 12;
    std::vector<double> real(n);
    in.e.resize(n);
    in.t.resize(n);
    in.theo.resize(n, 1);
    for (size_t i = 0; i < n; ++i) {
      real[i] = (i ? real[i - 1] : 0) + (random() % 7) * .03;
      in.e[i] = random() % 100000000;
      in.t[i] = random() % 100000000;
      if (test % 2) {
        const bool theoretical = random() % 2;
        in.theo[i] = theoretical;
        if (theoretical)
          in.e[i] = 0;
        else
          in.t[i] = 0;
      }
    }
    const double p = std::vector<double>{1, 1.5, 2, 3}[(test / 2) % 4];
    in.grid(real, 1 + random() % 1000000, p, random() % 1000000);
    const I expected = oracle(in);
    for (auto name : {"baseline",
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
                      "all-stream",
                      "stream-cursor",
                      "moment-cursor",
                      "group-linear-cursor",
                      "all-cursor"}) {
      const auto v = variant(name);
      if (v.moments && p != 2)
        continue;
      try {
        const auto r = solve(in, v);
        if (r.profit != expected)
          throw std::runtime_error("profit mismatch: got " +
                                   std::to_string(r.profit) + " expected " +
                                   std::to_string(expected));
        auto trace = solve<true>(in, v);
        assert(trace.profit == expected);
        dual(in, trace);
        if (test < 40) {
          const auto g = gains(in, v);
          for (size_t i = 0; i < n; ++i)
            if (in.theo[i]) {
              auto more = in;
              ++more.t[i];
              if (g[i] != oracle(more) - expected)
                throw std::runtime_error("weighted marginal mismatch");
            }
        }
      } catch (const std::exception &e) {
        std::cerr << "case=" << test << " variant=" << name << " p=" << p
                  << " error=" << e.what() << '\n';
        return 1;
      }
    }
  }
  std::cout
      << trials
      << " large-capacity unrestricted min-cost-flow cases, all applicable "
         "variants/traces/duals and 40 marginal suites passed\n";
}
