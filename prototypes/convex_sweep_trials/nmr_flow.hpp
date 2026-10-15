#ifndef WNET_NMR_TRIAL_FLOW_HPP
#define WNET_NMR_TRIAL_FLOW_HPP
#include "nmr_common.hpp"
#include <climits>
#include <lemon/cost_scaling.h>
#include <lemon/list_graph.h>
#include <queue>
namespace nmr_trials {
struct FlowResult {
    I profit = 0;
    Certificate certificate;
    std::vector<I> gains;
    size_t pairs = 0;
};
// Explicit profitable-pair graph, not a repair for rounded/non-Monge costs.
// Include zero-supply targets for sensitivity/cuts, and leave pair arcs with
// residual capacity even when their current flow saturates a target. Using
// min(e_i,t_j) as pair capacity gives wrong +1 sensitivities at zero targets.
inline FlowResult flow(const Input &in, bool derivatives, bool certificate, size_t pair_limit) {
    Coordinates c(in);
    using G = lemon::ListDigraph;
    const size_t n = c.pos.size();
    std::vector<size_t> ei, ti;
    I mass = 0;
    for (size_t i = 0; i < n; ++i) {
        if (c.e[i]) {
            ei.push_back(i);
            mass += c.e[i];
        }
        if (c.t[i] || ((derivatives || certificate) && c.theo[i]))
            ti.push_back(i);
    }
    size_t pairs = 0;
    const double radius = in.model.radius(in.tau);
    auto edges = [&](auto emit) {
        size_t lo = 0;
        for (size_t a = 0; a < ei.size(); ++a) {
            size_t i = ei[a];
            while (lo < ti.size() && c.pos[ti[lo]] < c.pos[i] - radius)
                ++lo;
            for (size_t b = lo; b < ti.size() && c.pos[ti[b]] <= c.pos[i] + radius; ++b) {
                I cost = in.model.cost(c.pos[i], c.pos[ti[b]]);
                if (cost < in.tau)
                    emit(a, b, cost - in.tau);
            }
        }
    };
    // GridPower.radius is the first integer distance with cost >= tau;
    // strict geometric bounds count edges in O(nodes), before any arc allocation.
    size_t low = 0, high = 0;
    for (auto i : ei) {
        while (low < ti.size() && c.pos[ti[low]] <= c.pos[i] - radius)
            ++low;
        high = std::max(high, low);
        while (high < ti.size() && c.pos[ti[high]] < c.pos[i] + radius)
            ++high;
        pairs += high - low;
    }
    if (pairs > pair_limit)
        throw std::length_error("profitable-pair budget exceeded");
    if (pairs + ei.size() + ti.size() + 1 >= size_t(INT_MAX))
        throw std::length_error("LEMON arc index budget exceeded");
    require(mass < INT64_MAX, "flow mass needs residual headroom");
    G graph;
    graph.reserveNode(int(ei.size() + ti.size() + 2));
    graph.reserveArc(int(pairs + ei.size() + ti.size() + 1));
    auto source = graph.addNode(), sink = graph.addNode();
    std::vector<G::Node> en, tn;
    for (auto i : ei) {
        (void)i;
        en.push_back(graph.addNode());
    }
    for (auto i : ti) {
        (void)i;
        tn.push_back(graph.addNode());
    }
    G::ArcMap<I> cap(graph), cost(graph);
    G::NodeMap<I> supply(graph, 0);
    auto add = [&](G::Node a, G::Node b, I capacity, I price) {
        auto arc = graph.addArc(a, b);
        cap[arc] = capacity;
        cost[arc] = price;
    };
    for (size_t a = 0; a < ei.size(); ++a)
        add(source, en[a], c.e[ei[a]], 0);
    for (size_t b = 0; b < ti.size(); ++b)
        add(tn[b], sink, c.t[ti[b]], 0);
    edges([&](size_t a, size_t b, I price) { add(en[a], tn[b], mass + 1, price); });
    add(source, sink, mass + 1, 0);
    supply[source] = mass;
    supply[sink] = -mass;
    lemon::CostScaling<G, I, I> solver(graph);
    if (solver.upperMap(cap).costMap(cost).supplyMap(supply).run() != decltype(solver)::OPTIMAL)
        throw std::runtime_error("cost scaling not optimal");
    FlowResult result;
    result.profit = -solver.totalCost<I>();
    result.pairs = pairs;
    const I anchor = solver.potential(sink);
    if (certificate) {
        std::vector<I> alpha(n, -in.tau), beta(n, -in.tau);
        for (size_t a = 0; a < ei.size(); ++a)
            alpha[ei[a]] = std::min<I>(0, anchor - solver.potential(en[a]));
        for (size_t b = 0; b < ti.size(); ++b)
            beta[ti[b]] = std::min<I>(0, solver.potential(tn[b]) - anchor);
        W dual = 0;
        for (size_t i = 0; i < n; ++i)
            dual += W(alpha[i]) * c.e[i] + W(beta[i]) * c.t[i];
        require(dual == -W(result.profit), "flow potential objective");
        for (auto i : c.original) {
            result.certificate.alpha.push_back(alpha[i]);
            result.certificate.beta.push_back(beta[i]);
        }
    }
    if (derivatives) {
        const I infinity = INT64_MAX / 4;
        G::NodeMap<I> dist(graph, infinity);
        dist[sink] = 0;
        using Entry = std::pair<I, int>;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
        queue.push({0, graph.id(sink)});
        auto relax = [&](G::Node u, G::Node v, I price) {
            I reduced = price + solver.potential(u) - solver.potential(v);
            require(reduced >= 0, "negative residual reduced cost");
            if (dist[u] <= infinity - reduced && dist[v] > dist[u] + reduced) {
                dist[v] = dist[u] + reduced;
                queue.push({dist[v], graph.id(v)});
            }
        };
        while (!queue.empty()) {
            auto [d, id] = queue.top();
            queue.pop();
            auto u = graph.nodeFromId(id);
            if (d != dist[u])
                continue;
            for (G::OutArcIt a(graph, u); a != lemon::INVALID; ++a)
                if (solver.flow(a) < cap[a])
                    relax(u, graph.target(a), cost[a]);
            for (G::InArcIt a(graph, u); a != lemon::INVALID; ++a)
                if (solver.flow(a) > 0)
                    relax(u, graph.source(a), -cost[a]);
        }
        std::vector<I> gain(n);
        for (size_t b = 0; b < ti.size(); ++b)
            if (dist[tn[b]] < infinity)
                gain[ti[b]] = std::max<I>(0, anchor - solver.potential(tn[b]) - dist[tn[b]]);
        result.gains.resize(in.pos.size());
        for (size_t i = 0; i < in.pos.size(); ++i)
            if (in.theo[i] || in.t[i])
                result.gains[i] = gain[c.original[i]];
    }
    return result;
}
} // namespace nmr_trials
#endif
