#ifndef WNET_CONVEX_SWEEP_DUAL_HPP
#define WNET_CONVEX_SWEEP_DUAL_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
#include <utility>
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>
#include "wide_int.hpp"

namespace convex_sweep_dual {

// Lazy certificate oracle for the partial matching problem. Only profitable
// pairs need arcs: alpha,beta <= 0 automatically satisfy all other constraints.
// This deliberately uses an auxiliary simplex solve, rather than treating
// independent discrete marginals as a common feasible matching dual.
// All positions, including zero-capacity theoretical positions, participate:
// the resulting cut must remain valid when mixture weights change.
template<class I, class Cost>
auto recover(std::span<const double> positions, std::span<const I> empirical,
             std::span<const I> theoretical, I tau, double radius,
             Cost pair_cost, wide_int::wide_t expected_profit) {
    using Graph = lemon::StaticDigraph;
    using NS = lemon::NetworkSimplex<Graph, I, I>;
    using W = wide_int::wide_t;
    const size_t n = positions.size();
    if (empirical.size() != n || theoretical.size() != n || tau < 0)
        throw std::invalid_argument("inconsistent ConvexSweep dual inputs");
    if (n > (size_t(std::numeric_limits<int>::max()) - 2) / 2)
        throw std::overflow_error("ConvexSweep certificate has too many nodes");
    std::vector<I> alpha(n, 0), beta(n, 0);
    if (!tau) {
        if (expected_profit != W(0))
            throw std::runtime_error("ConvexSweep zero-reward objective mismatch");
        return std::make_pair(std::move(alpha), std::move(beta));
    }
    struct Arc { int from, to; I cost, capacity; };
    const int source = int(2 * n), sink = source + 1;
    const I unlimited = std::numeric_limits<I>::max();
    std::vector<Arc> arcs;
    for (size_t i = 0; i < n; ++i) {
        arcs.push_back({source, int(i), 0, empirical[i]});
        arcs.push_back({int(n + i), sink, 0, theoretical[i]});
        const double x = positions[i];
        const auto first = std::lower_bound(positions.begin(), positions.end(),
            std::nextafter(x - radius, -std::numeric_limits<double>::infinity()));
        const auto last = std::upper_bound(first, positions.end(),
            std::nextafter(x + radius, std::numeric_limits<double>::infinity()));
        for (auto it = first; it != last; ++it) {
            const size_t j = size_t(it - positions.begin());
            const I cost = pair_cost(x, *it);
            if (cost < tau) arcs.push_back({int(i), int(n + j), cost - tau, unlimited});
        }
    }
    arcs.push_back({sink, source, 0, unlimited});
    if (arcs.size() > size_t(std::numeric_limits<int>::max()))
        throw std::overflow_error("ConvexSweep certificate has too many arcs");
    std::sort(arcs.begin(), arcs.end(), [](const Arc& a, const Arc& b) {
        return a.from != b.from ? a.from < b.from : a.to < b.to;
    });
    std::vector<std::pair<int, int>> endpoints;
    endpoints.reserve(arcs.size());
    for (const auto& arc : arcs) endpoints.emplace_back(arc.from, arc.to);
    Graph graph;
    graph.build(sink + 1, endpoints.begin(), endpoints.end());
    Graph::ArcMap<I> costs(graph), capacities(graph);
    Graph::NodeMap<I> supplies(graph, 0);
    for (size_t i = 0; i < arcs.size(); ++i) {
        const auto arc = graph.arcFromId(int(i));
        costs[arc] = arcs[i].cost;
        capacities[arc] = arcs[i].capacity;
    }
    NS solver(graph);
    if (solver.costMap(costs).upperMap(capacities).supplyMap(supplies).run() != NS::OPTIMAL)
        throw std::runtime_error("ConvexSweep certificate recovery failed");
    const I ps = solver.potential(graph.nodeFromId(source));
    const I pt = solver.potential(graph.nodeFromId(sink));
    auto clip_difference = [tau](I a, I b) -> I {
        if (a >= b) return 0;
        // Unsigned subtraction represents the full magnitude even if the
        // signed difference would overflow. No ordering on software wide_t
        // is needed (Windows uses the software 128-bit accumulator).
        const uint64_t magnitude = uint64_t(b) - uint64_t(a);
        return magnitude >= uint64_t(tau) ? -tau : -I(magnitude);
    };
    W dual = 0;
    for (size_t i = 0; i < n; ++i) {
        alpha[i] = clip_difference(ps, solver.potential(graph.nodeFromId(int(i))));
        beta[i] = clip_difference(solver.potential(graph.nodeFromId(int(n + i))), pt);
        dual += W(alpha[i]) * empirical[i] + W(beta[i]) * theoretical[i];
    }
    // Validate feasibility independently of simplex's potential convention.
    // O(arcs), and nonprofitable pairs satisfy the inequality automatically.
    for (const auto& arc : arcs) {
        if (arc.from < int(n) && arc.to >= int(n) && arc.to < source &&
            alpha[arc.from] > arc.cost - beta[arc.to - int(n)])
            throw std::runtime_error("Invalid ConvexSweep matching dual");
    }
    W primal = 0;
    for (size_t i = 0; i < arcs.size(); ++i)
        primal += W(arcs[i].cost) * solver.flow(graph.arcFromId(int(i)));
    if (primal != dual || primal != -expected_profit)
        throw std::runtime_error("ConvexSweep primal-dual objective mismatch");
    return std::make_pair(std::move(alpha), std::move(beta));
}
}  // namespace convex_sweep_dual
#endif
