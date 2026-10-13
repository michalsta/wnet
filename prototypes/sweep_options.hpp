#ifndef WNET_PROTOTYPE_SWEEP_OPTIONS_HPP
#define WNET_PROTOTYPE_SWEEP_OPTIONS_HPP
// Experimental alternatives; production dispatch is deliberately unchanged.
#include "wnet/decompositable_graph.hpp"
#include "non_monge_repair.hpp"
#include <chrono>
#include <fstream>
#include <lemon/list_graph.h>
#include <random>
#include <unordered_set>

namespace sweep_options {
using I = int64_t;
using W = wide_int::wide_t;
struct Input {
    std::vector<double> pos;
    std::vector<I> e, t;
    I tau = 0, scale = 1;
    int p = 1;
    double radius() const { return std::pow((double(tau) + 1) / scale, 1.0 / p); }
    I cost(double a, double b) const {
        double d = std::abs(a - b);
        return std::llround(scale * (p == 1 ? d : d * d));
    }
    std::vector<convex_sweep::Event> events() const {
        std::vector<convex_sweep::Event> ev;
        for (size_t i = 0; i < pos.size(); ++i) {
            if (e[i])
                ev.push_back({pos[i], 0, e[i], i});
            if (t[i])
                ev.push_back({pos[i], 1, t[i], i});
        }
        return ev;
    }
};
struct Result {
    W profit = 0, dual_profit = 0;
    std::vector<convex_sweep::Match> matches;
    std::vector<I> alpha, beta;
    size_t pairs = 0, exchanges = 0;
    I epsilon = 0;
    long double model_cost = 0;
    W original_profit = 0;
};
inline W matching_profit(const Input &in, std::span<const convex_sweep::Match> matches) {
    W value = 0;
    for (auto m : matches)
        value += W(in.tau - in.cost(in.pos[m.empirical], in.pos[m.theoretical])) * m.count;
    return value;
}
inline W dual_profit(const Input &in, std::span<const I> a, std::span<const I> b) {
    W value = 0;
    for (size_t i = 0; i < in.pos.size(); ++i)
        value += -(W(a[i]) * in.e[i] + W(b[i]) * in.t[i]);
    return value;
}
inline void validate(const Input &in, const Result &r, bool exact) {
    const size_t n = in.pos.size();
    if (r.alpha.size() != n || r.beta.size() != n)
        throw std::runtime_error("missing certificate");
    std::vector<I> e(n), t(n);
    for (auto m : r.matches) {
        if (m.empirical >= n || m.theoretical >= n || m.count <= 0)
            throw std::runtime_error("bad match");
        if (m.count > in.e[m.empirical] - e[m.empirical] ||
            m.count > in.t[m.theoretical] - t[m.theoretical])
            throw std::runtime_error("bad capacity");
        e[m.empirical] += m.count;
        t[m.theoretical] += m.count;
    }
    for (size_t i = 0; i < n; ++i) {
        if (r.alpha[i] > 0 || r.beta[i] > 0)
            throw std::runtime_error("bad dual sign");
        auto first =
            std::lower_bound(in.pos.begin(), in.pos.end(), in.pos[i] - in.radius() - 1e-10);
        auto last = std::upper_bound(first, in.pos.end(), in.pos[i] + in.radius() + 1e-10);
        for (size_t j = size_t(first - in.pos.begin()); j < size_t(last - in.pos.begin()); ++j)
            if (W(in.cost(in.pos[i], in.pos[j])) - in.tau < W(r.alpha[i]) + r.beta[j])
                throw std::runtime_error("bad dual inequality");
    }
    if (matching_profit(in, r.matches) != r.profit ||
        dual_profit(in, r.alpha, r.beta) != r.dual_profit || r.dual_profit < r.profit ||
        (exact && r.profit != r.dual_profit))
        throw std::runtime_error("bad objective/bounds");
}
inline Result traced(const Input &in) {
    auto profit = [&](double a, double b) { return in.tau - in.cost(a, b); };
    auto trace = convex_sweep::sweep_solve<true>(in.events(), profit, in.radius());
    Result r;
    r.matches = std::move(trace.matches);
    r.profit = trace.profit;
    return r;
}
inline Result baseline(const Input &in) {
    auto r = traced(in);
    auto cost = [&](double a, double b) { return in.cost(a, b); };
    auto [m, p, a, b] =
        convex_sweep_dual::repair<I>(in.pos, in.e, in.t, in.tau, in.radius(), cost, r.matches);
    r.matches = std::move(m);
    r.profit = p;
    r.alpha = std::move(a);
    r.beta = std::move(b);
    r.dual_profit = p;
    return r;
}
template <class I, class Cost>
auto cached_repair(std::span<const double> pos, std::span<const I> empirical,
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
    std::vector<std::vector<std::pair<size_t, I>>> candidates(n);
    size_t pair_count = 0;
    for (size_t j = 0; j < n; ++j)
        if (theoretical[j] > 0) {
            auto [a, b] = window(pos[j]);
            for (size_t i = a; i < b; ++i)
                if (empirical[i] > 0) {
                    I c = cost(pos[i], pos[j]);
                    if (c < tau) {
                        if (++pair_count > 8000000)
                            throw std::length_error("profitable pair budget exceeded");
                        candidates[j].push_back({i, c});
                    }
                }
        }
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
                if (empirical[i] > 0 && visit(Arc{root, i, 0, inf, bound}))
                    return true;
                if (used_t[i] < theoretical[i] &&
                    visit(Arc{root, n + i, 0, theoretical[i] - used_t[i], theoretical_slack}))
                    return true;
            }
        } else if (from < n) {
            if (empirical[from] == 0)
                return false;
            if (used_e[from] < empirical[from] &&
                visit(Arc{from, root, 0, empirical[from] - used_e[from], empirical_slack}))
                return true;
            for (auto [j, mass] : matches[from])
                if (mass &&
                    visit(Arc{from, n + j, tau - cost(pos[from], pos[j]), mass, remove_match}))
                    return true;
        } else {
            const size_t j = from - n;
            if (theoretical[j] == 0)
                return false;
            if (visit(Arc{from, root, 0, inf, bound}))
                return true;
            for (auto [i, c] : candidates[j]) {
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
    auto potentials = convex_sweep_dual::recover<I>(pos, empirical, theoretical, tau, radius, cost,
                                                    matching, profit);
    return std::make_tuple(std::move(matching), profit, std::move(potentials.first),
                           std::move(potentials.second));
}

inline Result cached(const Input &in) {
    auto r = traced(in);
    auto cost = [&](double a, double b) { return in.cost(a, b); };
    auto [m, p, a, b] = cached_repair<I>(in.pos, in.e, in.t, in.tau, in.radius(), cost, r.matches);
    r.matches = std::move(m);
    r.profit = p;
    r.alpha = std::move(a);
    r.beta = std::move(b);
    r.dual_profit = p;
    return r;
}
// Retain rounded pair costs, but allow epsilon slack in complementary
// equalities/unused anchors. Pair inequalities stay EXACT. No flow correction.
inline Result approximate(const Input &in) {
    auto r = traced(in);
    const size_t n = in.pos.size();
    std::vector<I> ue(n), ut(n);
    std::vector<std::vector<std::pair<size_t, I>>> tight(n);
    for (auto m : r.matches) {
        ue[m.empirical] += m.count;
        ut[m.theoretical] += m.count;
        tight[m.empirical].push_back(
            {m.theoretical, in.tau - in.cost(in.pos[m.empirical], in.pos[m.theoretical])});
    }
    for (I eps = 1; eps <= 65536; eps *= 2) {
        std::vector<I> x(n, 0), y(n, in.tau), depth(2 * n);
        std::vector<bool> queued(2 * n, true);
        std::deque<size_t> queue;
        for (size_t j = 0; j < n; ++j)
            if (ut[j] < in.t[j])
                y[j] = std::min(eps, in.tau);
        for (size_t v = 0; v < 2 * n; ++v)
            queue.push_back(v);
        bool failed = false;
        auto push = [&](size_t v, size_t from) {
            depth[v] = depth[from] + 1;
            if (depth[v] >= I(2 * n + 1)) {
                failed = true;
                return;
            }
            if (!queued[v]) {
                queued[v] = true;
                queue.push_back(v);
            }
        };
        while (!queue.empty() && !failed) {
            size_t v = queue.front();
            queue.pop_front();
            queued[v] = false;
            if (v < n) {
                for (auto [j, price] : tight[v]) {
                    I value = std::min(in.tau, x[v] + price + eps);
                    if (value < y[j]) {
                        if (value < 0) {
                            failed = true;
                            break;
                        }
                        y[j] = value;
                        push(n + j, v);
                    }
                }
            } else {
                size_t j = v - n;
                auto a =
                    std::lower_bound(in.pos.begin(), in.pos.end(), in.pos[j] - in.radius() - 1e-10);
                auto b = std::upper_bound(a, in.pos.end(), in.pos[j] + in.radius() + 1e-10);
                for (size_t i = size_t(a - in.pos.begin()); i < size_t(b - in.pos.begin()); ++i) {
                    I c = in.cost(in.pos[i], in.pos[j]);
                    if (c >= in.tau)
                        continue;
                    I value = y[j] + c - in.tau;
                    if (value < x[i]) {
                        if (value < -in.tau || (ue[i] < in.e[i] && value < -eps)) {
                            failed = true;
                            break;
                        }
                        x[i] = value;
                        push(i, v);
                    }
                }
            }
        }
        if (failed)
            continue;
        r.alpha = std::move(x);
        r.beta.resize(n);
        for (size_t j = 0; j < n; ++j)
            r.beta[j] = -y[j];
        r.dual_profit = dual_profit(in, r.alpha, r.beta);
        r.epsilon = eps;
        validate(in, r, false);
        return r;
    }
    throw std::runtime_error("epsilon recovery exhausted");
}
// Materialize profitable pairs only between positive supplies. Cost scaling
// is an independent exact fallback, with matching duals recovered afterwards.
inline Result cost_scaling(const Input &in, size_t pair_limit = 8000000, bool active = false) {
    using G = lemon::ListDigraph;
    G g;
    auto root = g.addNode();
    size_t n = in.pos.size();
    std::vector<G::Node> e(n, lemon::INVALID), t(n, lemon::INVALID);
    G::ArcMap<I> cap(g), price(g);
    auto arc = [&](G::Node a, G::Node b, I c, I p) {
        auto edge = g.addArc(a, b);
        cap[edge] = c;
        price[edge] = p;
        return edge;
    };
    I mass = 0;
    for (I x : in.e)
        mass += x;
    std::vector<size_t> columns;
    for (size_t i = 0; i < n; ++i) {
        if (in.e[i]) {
            e[i] = g.addNode();
            arc(root, e[i], in.e[i], 0);
        }
        if (in.t[i]) {
            t[i] = g.addNode();
            arc(t[i], root, in.t[i], 0);
            columns.push_back(i);
        }
    }
    struct Pair {
        size_t i, j;
        G::Arc edge;
    };
    std::vector<Pair> pairs;
    std::vector<std::unordered_set<size_t>> existing(n);
    auto add = [&](size_t i, size_t j, I c) {
        if (c >= in.tau || existing[i].contains(j))
            return;
        if (pairs.size() >= pair_limit)
            throw std::length_error("profitable pair budget exceeded");
        existing[i].insert(j);
        pairs.push_back({i, j, arc(e[i], t[j], mass + 1, c - in.tau)});
    };
    if (active) {
        for (size_t i = 0; i < n; ++i)
            if (in.e[i]) {
                size_t at = std::lower_bound(columns.begin(), columns.end(), in.pos[i],
                                             [&](size_t j, double x) { return in.pos[j] < x; }) -
                            columns.begin();
                for (size_t k = at > 4 ? at - 4 : 0; k < std::min(columns.size(), at + 4); ++k) {
                    size_t j = columns[k];
                    add(i, j, in.cost(in.pos[i], in.pos[j]));
                }
            }
    } else {
        for (size_t j : columns) {
            auto a =
                std::lower_bound(in.pos.begin(), in.pos.end(), in.pos[j] - in.radius() - 1e-10);
            auto b = std::upper_bound(a, in.pos.end(), in.pos[j] + in.radius() + 1e-10);
            for (size_t i = size_t(a - in.pos.begin()); i < size_t(b - in.pos.begin()); ++i)
                if (in.e[i])
                    add(i, j, in.cost(in.pos[i], in.pos[j]));
        }
    }
    for (size_t round = 0; round < 128; ++round) {
        lemon::CostScaling<G, I, I> solver(g);
        if (solver.upperMap(cap).costMap(price).run() != decltype(solver)::OPTIMAL)
            throw std::runtime_error("cost scaling failed");
        Result r;
        r.pairs = pairs.size();
        r.exchanges = round + 1;
        r.alpha.assign(n, -in.tau);
        r.beta.assign(n, -in.tau);
        I root_pi = solver.potential(root);
        for (size_t i = 0; i < n; ++i) {
            if (in.e[i])
                r.alpha[i] = std::clamp(root_pi - solver.potential(e[i]), -in.tau, I(0));
            if (in.t[i])
                r.beta[i] = std::clamp(solver.potential(t[i]) - root_pi, -in.tau, I(0));
        }
        std::vector<std::tuple<size_t, size_t, I>> additions;
        if (active) {
            for (size_t i = 0; i < n; ++i)
                if (in.e[i]) {
                    auto first = std::lower_bound(
                        columns.begin(), columns.end(), in.pos[i] - in.radius() - 1e-10,
                        [&](size_t j, double x) { return in.pos[j] < x; });
                    I worst = 0;
                    size_t worst_j = 0;
                    I worst_c = 0;
                    for (auto it = first;
                         it != columns.end() && in.pos[*it] <= in.pos[i] + in.radius() + 1e-10;
                         ++it) {
                        size_t j = *it;
                        I c = in.cost(in.pos[i], in.pos[j]);
                        I violation = r.alpha[i] + r.beta[j] + in.tau - c;
                        if (violation > worst) {
                            worst = violation;
                            worst_j = j;
                            worst_c = c;
                        }
                    }
                    if (worst)
                        additions.push_back({i, worst_j, worst_c});
                }
        }
        if (!additions.empty()) {
            for (auto [i, j, c] : additions) {
                if (existing[i].contains(j))
                    throw std::runtime_error("restricted dual infeasible");
                add(i, j, c);
            }
            continue;
        }
        for (auto pair : pairs) {
            I count = solver.flow(pair.edge);
            if (count)
                r.matches.push_back({pair.i, pair.j, count});
        }
        r.profit = matching_profit(in, r.matches);
        r.dual_profit = dual_profit(in, r.alpha, r.beta);
        if (r.profit != r.dual_profit)
            throw std::runtime_error("cost scaling dual mismatch");
        return r;
    }
    throw std::runtime_error("active pricing iteration budget exceeded");
}
// Reuse the native slope recurrence, backtrack selected row/column masses,
// then greedily couple them. Recover the chain matching dual in linear work.
inline Result slope(const Input &in, const std::vector<I> &q, I tau) {
    size_t n = q.size();
    slopedp_detail::ConvexPL<I> dp;
    I wall = std::numeric_limits<I>::max() / 1024;
    dp.insL(0, wall);
    dp.insR(0, wall);
    std::vector<I> lo(n), hi(n), u(n), y(n), s(n ? n - 1 : 0), gap(n ? n - 1 : 0);
    for (size_t i = 0; i < n; ++i) {
        auto [a, b] = dp.convV(tau, in.e[i] - in.t[i], in.t[i]);
        lo[i] = a;
        hi[i] = b;
        dp.minval += -W(tau) * in.t[i];
        if (i + 1 < n) {
            gap[i] = q[i + 1] - q[i];
            dp.addAbs(0, gap[i]);
        }
    }
    W profit = -dp.evalAt(0);
    I next = 0;
    for (size_t i = n; i-- > 0;) {
        I prev = std::clamp(next - in.e[i] + in.t[i], lo[i], hi[i]);
        if (next - prev < -in.t[i])
            prev = next + in.t[i];
        if (next - prev > in.e[i])
            prev = next - in.e[i];
        if (i == 0)
            prev = 0;
        I net = next - prev;
        y[i] = std::clamp(in.e[i] - net, I(0), in.t[i]);
        u[i] = std::clamp(net + y[i], I(0), in.e[i]);
        if (i)
            s[i - 1] = prev;
        next = prev;
    }
    Result r;
    r.profit = profit;
    auto remu = u, remy = y;
    size_t i = 0, j = 0;
    while (i < n && j < n) {
        if (!remu[i]) {
            ++i;
            continue;
        }
        if (!remy[j]) {
            ++j;
            continue;
        }
        I m = std::min(remu[i], remy[j]);
        r.matches.push_back({i, j, m});
        remu[i] -= m;
        remy[j] -= m;
    }
    auto pi = slopedp_dual::recover<I>(in.e, in.t, u, y, s, gap, tau);
    r.alpha = pi;
    r.beta.resize(n);
    for (size_t k = 0; k < n; ++k)
        r.beta[k] = -tau - pi[k];
    r.dual_profit = profit;
    return r;
}
// Map a changed-model certificate to original integer costs. Subtract each
// row's worst violation, including zero-supply columns. This is feasibility
// projection, not complementary recovery and not an exact optimality claim.
inline void project(const Input &in, Result &r, long double ratio) {
    for (auto &v : r.alpha)
        v = std::min<I>(0, I(std::floor(v * ratio)));
    for (auto &v : r.beta)
        v = std::min<I>(0, I(std::floor(v * ratio)));
    size_t n = in.pos.size();
    for (size_t i = 0; i < n; ++i) {
        I violation = 0;
        auto a = std::lower_bound(in.pos.begin(), in.pos.end(), in.pos[i] - in.radius() - 1e-10);
        auto b = std::upper_bound(a, in.pos.end(), in.pos[i] + in.radius() + 1e-10);
        for (size_t j = size_t(a - in.pos.begin()); j < size_t(b - in.pos.begin()); ++j)
            violation = std::max(violation,
                                 r.alpha[i] + r.beta[j] + in.tau - in.cost(in.pos[i], in.pos[j]));
        r.alpha[i] = std::max(-in.tau, r.alpha[i] - violation);
    }
    r.profit = matching_profit(in, r.matches);
    r.dual_profit = dual_profit(in, r.alpha, r.beta);
}
// Linear-time conservative projection for p=1. Prefix/suffix maxima bound
// max_j(beta_j - S*|x_i-x_j|). Rounding contributes at most half a cost unit;
// a conservative floating guard also covers evaluation of the original cost.
inline void project_p1_fast(const Input &in, Result &r, long double ratio) {
    if (in.p != 1)
        throw std::invalid_argument("linear projection requires p=1");
    size_t n = in.pos.size();
    std::vector<long double> left(n), right(n);
    for (auto &v : r.alpha)
        v = std::min<I>(0, I(std::floor(v * ratio)));
    for (auto &v : r.beta)
        v = std::min<I>(0, I(std::floor(v * ratio)));
    long double biggest = 1;
    for (size_t i = 0; i < n; ++i) {
        long double x = (long double)in.scale * in.pos[i];
        left[i] = r.beta[i] + x;
        right[i] = r.beta[i] - x;
        biggest = std::max(biggest, std::abs(x) + std::abs((long double)r.beta[i]) + in.tau);
    }
    for (size_t i = 1; i < n; ++i)
        left[i] = std::max(left[i], left[i - 1]);
    for (size_t i = n; i > 1; --i)
        right[i - 2] = std::max(right[i - 2], right[i - 1]);
    long double guard = 32 * std::numeric_limits<double>::epsilon() * biggest;
    for (size_t i = 0; i < n; ++i) {
        long double x = (long double)in.scale * in.pos[i];
        long double value = r.alpha[i] + in.tau + std::max(left[i] - x, right[i] + x) + .5L + guard;
        I delta = std::max<I>(0, I(std::ceil(value)));
        r.alpha[i] = std::max(-in.tau, r.alpha[i] - delta);
    }
    r.profit = matching_profit(in, r.matches);
    r.dual_profit = dual_profit(in, r.alpha, r.beta);
}
inline Result monge(const Input &in, double h, bool original_certificate = true,
                    bool force_sweep = false, bool fast_certificate = false) {
    if (!std::isfinite(h) || h <= 0)
        throw std::invalid_argument("grid step must be finite and positive");
    std::vector<I> q(in.pos.size());
    std::vector<double> positions(in.pos.size());
    for (size_t i = 0; i < q.size(); ++i) {
        if (!std::isfinite(in.pos[i] / h) || std::abs(in.pos[i] / h) > std::ldexp(1., 51))
            throw std::overflow_error("grid coordinates exceed exact double integer range");
        q[i] = std::llround(in.pos[i] / h);
        positions[i] = double(q[i]);
    }
    const long double scale = in.p == 1 ? 1.L / h : 1.L / (h * h);
    I tau = std::llround((long double)in.tau / in.scale * scale);
    if (W(std::numeric_limits<I>::max() / 8) <
        W(tau) * std::accumulate(in.e.begin(), in.e.end(), I(0)))
        throw std::overflow_error("grid objective budget exceeded");
    Result r;
    if (in.p == 1 && !force_sweep)
        r = slope(in, q, tau);
    else {
        auto cost = [p = in.p](double a, double b) -> I {
            I d = std::llround(a - b);
            return p == 1 ? std::abs(d) : d * d;
        };
        auto profit = [&](double a, double b) { return tau - cost(a, b); };
        auto ev = in.events();
        for (auto &event : ev)
            event.pos = positions[event.index];
        auto trace =
            convex_sweep::sweep_solve<true>(ev, profit, std::pow(double(tau) + 1, 1.0 / in.p));
        auto [a, b] = convex_sweep_dual::recover<I>(positions, in.e, in.t, tau,
                                                    std::pow(double(tau) + 1, 1.0 / in.p), cost,
                                                    trace.matches, W(trace.profit));
        r.profit = trace.profit;
        r.dual_profit = r.profit;
        r.matches = std::move(trace.matches);
        r.alpha = std::move(a);
        r.beta = std::move(b);
    }
    r.model_cost = ((long double)tau *
                        (std::accumulate(in.e.begin(), in.e.end(), I(0)) +
                         std::accumulate(in.t.begin(), in.t.end(), I(0))) /
                        2 -
                    (long double)I(r.profit)) /
                   scale;
    r.original_profit = matching_profit(in, r.matches);
    if (original_certificate) {
        if (fast_certificate)
            project_p1_fast(in, r, (long double)in.scale / scale);
        else
            project(in, r, (long double)in.scale / scale);
    }
    return r;
}
inline Result slope_chain(const Input &in, bool original_certificate = true,
                          bool fast_certificate = false) {
    if (in.p != 1)
        throw std::invalid_argument("SlopeDP requires p=1");
    std::vector<I> q(in.pos.size());
    for (size_t i = 1; i < q.size(); ++i)
        q[i] = q[i - 1] + std::llround(in.scale * (in.pos[i] - in.pos[i - 1]));
    auto r = slope(in, q, in.tau);
    r.original_profit = matching_profit(in, r.matches);
    if (original_certificate) {
        if (fast_certificate)
            project_p1_fast(in, r, 1);
        else
            project(in, r, 1);
    }
    return r;
}
} // namespace sweep_options
#endif
