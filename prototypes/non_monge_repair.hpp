// Historical benchmark only; production ConvexSweep preserves Monge instead.
#ifndef WNET_CONVEX_SWEEP_REPAIR_HPP
#define WNET_CONVEX_SWEEP_REPAIR_HPP
#include "wnet/convex_sweep_dual.hpp"
#include <queue>
#include <tuple>
#include <unordered_map>

namespace convex_sweep_dual {

// Rounding a convex distance can destroy the Monge inequality. Check adjacent
// active rows/possible columns once per geometry; four profitable-candidate
// costs suffice to find a violation. Outside the windows no match can improve
// the objective. Conservatively use exact repair if a boundary cost overflows.
template <class I, class Cost>
bool is_monge(std::span<const double> pos, std::span<const I> empirical,
              std::span<const unsigned char> possible_theoretical, I tau, double radius,
              Cost cost) {
    using W = wide_int::wide_t;
    std::vector<size_t> rows, cols;
    for (size_t k = 0; k < pos.size(); ++k) {
        if (empirical[k] > 0)
            rows.push_back(k);
        if (possible_theoretical[k])
            cols.push_back(k);
    }
    for (size_t a = 1; a < rows.size(); ++a) {
        const double x = pos[rows[a - 1]], y = pos[rows[a]];
        auto first = std::lower_bound(cols.begin(), cols.end(), x - radius,
                                      [&](size_t k, double z) { return pos[k] < z; });
        if (first != cols.begin())
            --first;
        for (auto j = first; j != cols.end() && std::next(j) != cols.end(); ++j) {
            const double z = pos[*j], w = pos[*std::next(j)];
            if (z > y + radius)
                break;
            if (w < x - radius)
                continue;
            try {
                const I c00 = cost(x, z), c01 = cost(x, w), c10 = cost(y, z), c11 = cost(y, w);
                if (std::min({c00, c01, c10, c11}) >= tau)
                    continue;
                if (W(c01) + c10 < W(c00) + c11)
                    return false;
            } catch (const std::overflow_error &) {
                return false;
            }
        }
    }
    return true;
}

// Exact native correction of a feasible sweep matching. The implicit residual
// network has only O(N+M) stored state; profitable forward pairs are enumerated
// in coordinate windows, and reverse arcs exist only for positive matches.
// Negative cycles exchange or augment matched chunks. Each exchange strictly
// improves the integer objective. No auxiliary transport graph/simplex is built.
template <class I, class Cost>
auto repair(std::span<const double> pos, std::span<const I> empirical,
            std::span<const I> theoretical, I tau, double radius, Cost cost,
            std::span<const convex_sweep::Match> initial) {
    using W = wide_int::wide_t;
    const size_t n = pos.size(), root = 2 * n, vertices = root + 1;
    if (empirical.size() != n || theoretical.size() != n || tau < 0)
        throw std::invalid_argument("inconsistent sweep repair inputs");
    std::vector<std::unordered_map<size_t, I>> matches(n);
    std::vector<I> used_e(n, 0), used_t(n, 0);
    for (auto m : initial) {
        if (m.empirical >= n || m.theoretical >= n || m.count <= 0 ||
            m.count > empirical[m.empirical] - used_e[m.empirical] ||
            m.count > theoretical[m.theoretical] - used_t[m.theoretical])
            throw std::runtime_error("Invalid sweep repair capacity");
        matches[m.empirical][m.theoretical] += m.count;
        used_e[m.empirical] += m.count;
        used_t[m.theoretical] += m.count;
    }
    auto window = [&](double z) {
        auto a =
            std::lower_bound(pos.begin(), pos.end(),
                             std::nextafter(z - radius, -std::numeric_limits<double>::infinity()));
        auto b = std::upper_bound(
            a, pos.end(), std::nextafter(z + radius, std::numeric_limits<double>::infinity()));
        return std::pair{size_t(a - pos.begin()), size_t(b - pos.begin())};
    };
    enum Kind { bound, add_match, remove_match, empirical_slack, theoretical_slack };
    struct Arc {
        size_t from = SIZE_MAX, to = SIZE_MAX;
        I price = 0, capacity = 0;
        Kind kind = bound;
    };
    auto arcs = [&](size_t from, auto visit) {
        constexpr I inf = std::numeric_limits<I>::max();
        if (from == root) {
            for (size_t i = 0; i < n; ++i) {
                if (visit(Arc{root, i, 0, inf, bound}))
                    return true;
                if (used_t[i] < theoretical[i] &&
                    visit(Arc{root, n + i, 0, theoretical[i] - used_t[i], theoretical_slack}))
                    return true;
            }
        } else if (from < n) {
            if (used_e[from] < empirical[from] &&
                visit(Arc{from, root, 0, empirical[from] - used_e[from], empirical_slack}))
                return true;
            for (auto [j, mass] : matches[from])
                if (mass &&
                    visit(Arc{from, n + j, tau - cost(pos[from], pos[j]), mass, remove_match}))
                    return true;
        } else {
            const size_t j = from - n;
            if (visit(Arc{from, root, 0, inf, bound}))
                return true;
            const auto [a, b] = window(pos[j]);
            for (size_t i = a; i < b; ++i) {
                const I c = cost(pos[i], pos[j]);
                if (c < tau && visit(Arc{from, i, c - tau, inf, add_match}))
                    return true;
            }
        }
        return false;
    };
    std::vector<W> distance(vertices, W(0));
    std::vector<Arc> predecessor(vertices);
    std::vector<bool> queued(vertices, true);
    std::deque<size_t> queue;
    for (size_t v = 0; v < vertices; ++v)
        queue.push_back(v);
    for (;;) {
        std::vector<Arc> cycle;
        while (!queue.empty() && cycle.empty()) {
            const size_t v = queue.front();
            queue.pop_front();
            queued[v] = false;
            arcs(v, [&](Arc arc) {
                const W candidate = distance[v] + arc.price;
                if (!(candidate < distance[arc.to]))
                    return false;
                distance[arc.to] = candidate;
                predecessor[arc.to] = arc;
                // The predecessor graph was a forest before this relaxation.
                // A new back-edge identifies a cycle immediately, rather than
                // waiting O(N) rounds for a small quantization cycle to repeat.
                size_t u = arc.from;
                std::vector<Arc> path{arc};
                while (u != arc.to && predecessor[u].from != SIZE_MAX) {
                    path.push_back(predecessor[u]);
                    u = predecessor[u].from;
                }
                if (u == arc.to) {
                    W total = 0;
                    for (auto edge : path)
                        total += edge.price;
                    if (total < W(0)) {
                        cycle = std::move(path);
                        return true;
                    }
                    predecessor[arc.to] = Arc{}; // stale zero-cost predecessor cycle
                }
                if (!queued[arc.to]) {
                    queue.push_back(arc.to);
                    queued[arc.to] = true;
                }
                return false;
            });
        }
        if (cycle.empty())
            break;
        I amount = std::numeric_limits<I>::max();
        bool finite = false;
        for (auto arc : cycle) {
            amount = std::min(amount, arc.capacity);
            finite |= arc.kind != bound && arc.kind != add_match;
        }
        if (amount <= 0 || !finite)
            throw std::runtime_error("Invalid sweep repair residual cycle");
        // Remove first, so row/column totals cannot overflow transiently when
        // a full capacity is exchanged for another pair on the same cycle.
        for (auto arc : cycle) {
            if (arc.kind == remove_match) {
                const size_t i = arc.from, j = arc.to - n;
                matches[i][j] -= amount;
                used_e[i] -= amount;
                used_t[j] -= amount;
                if (!matches[i][j])
                    matches[i].erase(j);
            }
        }
        for (auto arc : cycle)
            if (arc.kind == add_match) {
                const size_t i = arc.to, j = arc.from - n;
                matches[i][j] += amount;
                used_e[i] += amount;
                used_t[j] += amount;
            }
        for (size_t i = 0; i < n; ++i)
            if (used_e[i] < 0 || used_e[i] > empirical[i] || used_t[i] < 0 ||
                used_t[i] > theoretical[i])
                throw std::runtime_error("Invalid sweep repaired capacity");
        // Keep the labels and pending work. Only cycle endpoints gained new
        // residual arcs; restarting all windows after each exchange is costly.
        // Clear predecessor links that may refer to now-saturated arcs.
        std::fill(predecessor.begin(), predecessor.end(), Arc{});
        auto enqueue = [&](size_t v) {
            if (!queued[v]) {
                queued[v] = true;
                queue.push_back(v);
            }
        };
        enqueue(root);
        for (auto arc : cycle) {
            enqueue(arc.from);
            enqueue(arc.to);
        }
    }
    std::vector<convex_sweep::Match> matching;
    W profit = 0;
    for (size_t i = 0; i < n; ++i)
        for (auto [j, mass] : matches[i]) {
            matching.push_back({i, j, mass});
            profit += W(tau - cost(pos[i], pos[j])) * mass;
        }
    std::sort(matching.begin(), matching.end(), [](auto a, auto b) {
        return std::pair{a.empirical, a.theoretical} < std::pair{b.empirical, b.theoretical};
    });
    auto potentials = recover<I>(pos, empirical, theoretical, tau, radius, cost, matching, profit);
    return std::make_tuple(std::move(matching), profit, std::move(potentials.first),
                           std::move(potentials.second));
}

// One-unit theoretical gains in the corrected matching: shortest residual
// paths to the root, found simultaneously by reversed reduced-cost Dijkstra.
// This also handles probes at zero-supply theoretical positions.
template <class I, class Cost>
std::vector<I> repaired_gains(std::span<const double> pos, std::span<const I> empirical,
                              std::span<const I> theoretical, I tau, double radius, Cost cost,
                              std::span<const convex_sweep::Match> matches,
                              std::span<const I> alpha, std::span<const I> beta) {
    using W = wide_int::wide_t;
    const size_t n = pos.size(), root = 2 * n;
    std::vector<I> used_e(n, 0), used_t(n, 0);
    std::vector<std::vector<size_t>> incoming(n);
    for (auto m : matches) {
        used_e[m.empirical] += m.count;
        used_t[m.theoretical] += m.count;
        incoming[m.theoretical].push_back(m.empirical);
    }
    struct Item {
        W distance;
        size_t v;
    };
    struct Greater {
        bool operator()(const Item &a, const Item &b) const { return b.distance < a.distance; }
    };
    std::priority_queue<Item, std::vector<Item>, Greater> queue;
    std::vector<W> distance(2 * n + 1, W(0));
    std::vector<bool> reached(2 * n + 1, false);
    reached[root] = true;
    queue.push({W(0), root});
    auto visit = [&](size_t v, W value) {
        if (!reached[v] || value < distance[v]) {
            distance[v] = value;
            reached[v] = true;
            queue.push({value, v});
        }
    };
    while (!queue.empty()) {
        const auto item = queue.top();
        queue.pop();
        const size_t v = item.v;
        if (item.distance != distance[v])
            continue;
        if (v == root) {
            for (size_t i = 0; i < n; ++i) {
                visit(n + i, W(-beta[i]));
                if (used_e[i] < empirical[i])
                    visit(i, W(0));
            }
        } else if (v < n) {
            const auto a = std::lower_bound(pos.begin(), pos.end(),
                                            std::nextafter(pos[v] - radius, -INFINITY));
            const auto b =
                std::upper_bound(a, pos.end(), std::nextafter(pos[v] + radius, INFINITY));
            for (size_t j = size_t(a - pos.begin()); j < size_t(b - pos.begin()); ++j) {
                const I c = cost(pos[v], pos[j]);
                if (c >= tau)
                    continue;
                const W reduced = W(c) - tau - beta[j] - alpha[v];
                if (reduced < W(0))
                    throw std::runtime_error("Invalid sweep reduced cost");
                visit(n + j, item.distance + reduced);
            }
        } else {
            const size_t j = v - n;
            for (size_t i : incoming[j])
                visit(i, item.distance);
        }
    }
    std::vector<I> gains(n, 0);
    for (size_t j = 0; j < n; ++j)
        gains[j] = -beta[j] - I(distance[n + j]);
    return gains;
}
} // namespace convex_sweep_dual
#endif
