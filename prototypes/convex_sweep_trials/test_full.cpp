// Optional full-fixture unrestricted flow oracle. This verifies the exact
// Monge model; it is not a proposed production backend or a repair mechanism.
#include "fixture.hpp"
#include <iostream>
#include <lemon/cost_scaling.h>
#include <lemon/list_graph.h>
#include <map>
using namespace sweep_trials;
int main(int argc, char **argv) {
  if (argc < 2)
    throw std::invalid_argument("test_full fixture [pair_limit]");
  Fixture fixture(argv[1]);
  const size_t limit = argc > 2 ? std::stoull(argv[2]) : 8000000;
  W total = 0;
  size_t total_pairs = 0;
  for (const auto &in : fixture.components) {
    std::map<I, I> em, tm;
    for (size_t i = 0; i < in.pos.size(); ++i) {
      if (in.e[i])
        em[I(in.pos[i])] += in.e[i];
      if (in.t[i])
        tm[I(in.pos[i])] += in.t[i];
    }
    std::vector<std::pair<I, I>> e(em.begin(), em.end()),
        t(tm.begin(), tm.end());
    if (e.empty() || t.empty())
      continue;
    const I radius = I(in.model.radius(in.tau));
    size_t pairs = 0;
    auto edges = [&](auto emit) {
      size_t lo = 0;
      for (size_t i = 0; i < e.size(); ++i) {
        while (lo < t.size() && t[lo].first < e[i].first - radius)
          ++lo;
        for (size_t j = lo; j < t.size() && t[j].first <= e[i].first + radius;
             ++j) {
          const W cost = in.model.wide_cost(e[i].first, t[j].first);
          if (cost < W(in.tau))
            emit(i, j, static_cast<I>(cost) - in.tau);
        }
      }
    };
    edges([&](size_t, size_t, I) { ++pairs; });
    if (pairs > limit)
      throw std::runtime_error("full oracle profitable-pair limit exceeded");
    total_pairs += pairs;
    using G = lemon::ListDigraph;
    G graph;
    graph.reserveNode(int(e.size() + t.size() + 2));
    graph.reserveArc(int(pairs + e.size() + t.size() + 1));
    auto source = graph.addNode(), sink = graph.addNode();
    std::vector<G::Node> en, tn;
    for (size_t i = 0; i < e.size(); ++i)
      en.push_back(graph.addNode());
    for (size_t j = 0; j < t.size(); ++j)
      tn.push_back(graph.addNode());
    G::ArcMap<I> capacity(graph), cost(graph);
    G::NodeMap<I> supply(graph, 0);
    I mass = 0;
    auto arc = [&](G::Node a, G::Node b, I cap, I price) {
      auto edge = graph.addArc(a, b);
      capacity[edge] = cap;
      cost[edge] = price;
    };
    for (size_t i = 0; i < e.size(); ++i) {
      arc(source, en[i], e[i].second, 0);
      mass += e[i].second;
    }
    for (size_t j = 0; j < t.size(); ++j)
      arc(tn[j], sink, t[j].second, 0);
    edges([&](size_t i, size_t j, I price) {
      arc(en[i], tn[j], std::min(e[i].second, t[j].second), price);
    });
    arc(source, sink, mass, 0);
    supply[source] = mass;
    supply[sink] = -mass;
    lemon::CostScaling<G, I, I> solver(graph);
    if (solver.upperMap(capacity).costMap(cost).supplyMap(supply).run() !=
        decltype(solver)::OPTIMAL)
      throw std::runtime_error("full oracle did not find optimum");
    total -= W(solver.totalCost<I>());
  }
  std::cout << "{\"profit\":" << static_cast<I>(total)
            << ",\"profitable_pairs\":" << total_pairs
            << ",\"status\":\"optimal\"}\n";
}
