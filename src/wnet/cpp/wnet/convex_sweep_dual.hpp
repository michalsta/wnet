#ifndef WNET_CONVEX_SWEEP_DUAL_HPP
#define WNET_CONVEX_SWEEP_DUAL_HPP

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include "convex_sweep.hpp"

namespace convex_sweep_dual {

// Recover one common dual from the traced primal matching. Write alpha = x,
// beta = -y, with -tau <= x <= 0 and 0 <= y <= tau. Matching feasibility is
// x_i <= y_j + c_ij - tau. Positive matches add the reverse inequality;
// slack anchors pin x/y to zero. Propagate these difference constraints over
// implicit profitable pairs, without storing pair arcs or solving transport.
template<class I, class Cost>
auto recover(std::span<const double> positions, std::span<const I> empirical,
             std::span<const I> theoretical, I tau, double radius,
             Cost pair_cost, std::span<const convex_sweep::Match> matching,
             wide_int::wide_t expected_profit) {
    using W = wide_int::wide_t;
    const size_t n = positions.size();
    if (empirical.size() != n || theoretical.size() != n || tau < 0)
        throw std::invalid_argument("inconsistent ConvexSweep dual inputs");
    std::vector<I> used_e(n, 0), used_t(n, 0), x(n, 0), y(n, tau);
    struct Tight { size_t target; I profit; };
    std::vector<std::vector<Tight>> tight(n);
    W primal = 0;
    for (const auto& match : matching) {
        const auto i = match.empirical, j = match.theoretical;
        if (i >= n || j >= n || match.count <= 0 ||
            match.count > empirical[i] - used_e[i] ||
            match.count > theoretical[j] - used_t[j])
            throw std::runtime_error("Invalid ConvexSweep traced capacity");
        const I cost = pair_cost(positions[i], positions[j]);
        if (cost > tau) throw std::runtime_error("Unprofitable ConvexSweep traced match");
        const I profit = tau - cost;
        used_e[i] += match.count; used_t[j] += match.count;
        tight[i].push_back({j, profit});
        primal += -W(profit) * match.count;
    }
    if (primal != -expected_profit)
        throw std::runtime_error("ConvexSweep traceback objective mismatch");
    std::deque<size_t> queue;
    std::vector<bool> queued(2 * n, true);
    std::vector<size_t> depth(2 * n, 0);
    for (size_t j = 0; j < n; ++j)
        if (used_t[j] < theoretical[j]) y[j] = 0;
    for (size_t v = 0; v < 2 * n; ++v) queue.push_back(v);
    auto enqueue = [&](size_t to, size_t from) {
        depth[to] = depth[from] + 1;
        if (depth[to] >= 2 * n)
            throw std::runtime_error("Inconsistent ConvexSweep dual constraints");
        if (!queued[to]) { queue.push_back(to); queued[to] = true; }
    };
    auto window = [&](double position) {
        auto first = std::lower_bound(positions.begin(), positions.end(),
            std::nextafter(position - radius, -std::numeric_limits<double>::infinity()));
        auto last = std::upper_bound(first, positions.end(),
            std::nextafter(position + radius, std::numeric_limits<double>::infinity()));
        return std::make_pair(size_t(first - positions.begin()), size_t(last - positions.begin()));
    };
    while (!queue.empty()) {
        const size_t v = queue.front(); queue.pop_front(); queued[v] = false;
        if (v < n) {
            for (const auto& edge : tight[v]) {
                const I candidate = x[v] + edge.profit;
                if (candidate < y[edge.target]) {
                    if (candidate < 0)
                        throw std::runtime_error("Invalid ConvexSweep theoretical potential");
                    y[edge.target] = candidate;
                    enqueue(n + edge.target, v);
                }
            }
        } else {
            const size_t j = v - n;
            const auto [first, last] = window(positions[j]);
            for (size_t i = first; i < last; ++i) {
                const I cost = pair_cost(positions[i], positions[j]);
                if (cost >= tau) continue;
                const I candidate = y[j] + (cost - tau);
                if (candidate < x[i]) {
                    if (candidate < -tau || used_e[i] < empirical[i])
                        throw std::runtime_error("Invalid ConvexSweep empirical potential");
                    x[i] = candidate;
                    enqueue(i, v);
                }
            }
        }
    }
    W dual = 0;
    for (size_t i = 0; i < n; ++i) {
        dual += W(x[i]) * empirical[i] - W(y[i]) * theoretical[i];
        const auto [first, last] = window(positions[i]);
        for (size_t j = first; j < last; ++j) {
            const I cost = pair_cost(positions[i], positions[j]);
            if (cost < tau && x[i] > y[j] + (cost - tau))
                throw std::runtime_error("Invalid ConvexSweep matching dual");
        }
        for (const auto& edge : tight[i])
            if (y[edge.target] != x[i] + edge.profit)
                throw std::runtime_error("Invalid ConvexSweep match complementarity");
    }
    if (dual != primal)
        throw std::runtime_error("ConvexSweep primal-dual objective mismatch");
    for (auto& value : y) value = -value;
    return std::make_pair(std::move(x), std::move(y));
}
// Equal grid coordinates have identical cost rows/columns. Recover on their
// summed capacities, then broadcast the certificate to every original peak.
// Include zero-supply coordinates: their potentials constrain future weights.
// Matching indices refer to the sorted grouped sweep events, not peak indices.
template<class I, class Cost>
auto recover_grouped(std::span<const double> positions, std::span<const I> empirical,
                     std::span<const I> theoretical, I tau, double radius,
                     Cost pair_cost, std::span<const convex_sweep::Event> events,
                     std::vector<convex_sweep::Match> matching,
                     wide_int::wide_t expected_profit) {
    const size_t n = positions.size();
    if (empirical.size() != n || theoretical.size() != n || tau < 0)
        throw std::invalid_argument("inconsistent ConvexSweep dual inputs");
    const bool duplicates = std::adjacent_find(positions.begin(), positions.end()) != positions.end();
    std::vector<double> grouped_pos;
    std::vector<I> grouped_e, grouped_t;
    std::vector<size_t> original;
    if (duplicates) {
        grouped_pos.reserve(n);
        grouped_e.reserve(n);
        grouped_t.reserve(n);
        original.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (grouped_pos.empty() || grouped_pos.back() != positions[i]) {
                grouped_pos.push_back(positions[i]);
                grouped_e.push_back(0);
                grouped_t.push_back(0);
            }
            grouped_e.back() += empirical[i];
            grouped_t.back() += theoretical[i];
            original.push_back(grouped_pos.size() - 1);
        }
    }
    const std::span<const double> coords = duplicates ? std::span<const double>(grouped_pos) : positions;
    std::vector<size_t> event_coordinate;
    event_coordinate.reserve(events.size());
    size_t j = 0;
    for (const auto& event : events) {
        while (j < coords.size() && coords[j] < event.pos) ++j;
        if (j == coords.size() || coords[j] != event.pos)
            throw std::invalid_argument("inconsistent ConvexSweep event coordinates");
        event_coordinate.push_back(j);
    }
    for (auto& match : matching) {
        match.empirical = event_coordinate.at(match.empirical);
        match.theoretical = event_coordinate.at(match.theoretical);
    }
    // Avoid grouped capacity buffers and broadcasting when all coordinates
    // already differ (the usual case at fine precision).
    if (!duplicates)
        return recover<I>(positions, empirical, theoretical, tau, radius,
                          pair_cost, matching, expected_profit);
    auto [a, b] = recover<I>(grouped_pos, grouped_e, grouped_t, tau, radius,
                             pair_cost, matching, expected_profit);
    std::vector<I> alpha(n), beta(n);
    for (size_t i = 0; i < n; ++i) {
        alpha[i] = a[original[i]];
        beta[i] = b[original[i]];
    }
    return std::make_pair(std::move(alpha), std::move(beta));
}
}  // namespace convex_sweep_dual
#endif
