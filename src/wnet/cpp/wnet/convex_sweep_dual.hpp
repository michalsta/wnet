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
}  // namespace convex_sweep_dual
#endif
