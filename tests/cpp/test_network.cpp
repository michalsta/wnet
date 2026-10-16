// Network-level oracle suite for the C++ core (decompositable_graph.hpp):
// factories, every backend, warm restarts, derivatives, dual cuts, position
// updates, and concurrent independent networks.  Standalone (no Python), so
// it runs under every sanitizer and valgrind lane; instances are kept small
// so the slowest of those (valgrind, MSan) stays within a few minutes.
//
// The oracles are agreements that must hold exactly in integer units:
//   * every NetworkSimplex warm mode and pivot rule, and the three scaling /
//     canceling solvers, reach the same optimal value on the same network;
//   * on integer 1-D positions the chain factory (NetworkSimplex, every
//     SlopeDP variant) and the dense factory give the same real cost, and so
//     does ConvexSweep against dense NetworkSimplex at p = 2;
//   * a warm position update equals a fresh build at the new positions;
//   * a network solved on another thread equals its sequential result.
//
// Build (from the repository root):
//   c++ -std=c++20 -O1 -g -pthread -Isrc/wnet/cpp -Isrc/wnet/cpp/wnet -I"$(python -m pylmcf --include)" tests/cpp/test_network.cpp
#include "wnet/decompositable_graph.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

void check(bool ok, const char* what, const char* where) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("FAIL %s: %s\n", where, what);
    }
}

bool close(double a, double b, double rel = 1e-9, double abs_tol = 1e-9) {
    return std::fabs(a - b) <= abs_tol + rel * std::max(std::fabs(a), std::fabs(b));
}

template <size_t D>
using Dist = VectorDistribution<D, double, double>;
using Net = WassersteinNetwork<int64_t, double>;
using Factory = WassersteinNetworkFactory<int64_t>;

// Random distribution; integer positions when `integral` so that p = 1 and
// p = 2 costs are exact on every factory.
template <size_t D>
Dist<D> random_dist(std::mt19937_64& rng, size_t n, double extent, bool integral) {
    std::uniform_real_distribution<double> pos(0.0, extent), inten(0.2, 2.0);
    std::vector<std::array<double, D>> p(n);
    std::vector<double> w(n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t d = 0; d < D; ++d) {
            const double x = pos(rng);
            p[i][d] = integral ? std::floor(x) : x;
        }
        w[i] = inten(rng);
    }
    return Dist<D>(std::move(p), std::move(w));
}

enum class Trash { Simple, Both, Independent };

void add_trash(Net& net, Trash t) {
    switch (t) {
        case Trash::Simple: net.add_simple_trash(3.0); break;
        case Trash::Both: net.add_experimental_trash(2.0); net.add_theoretical_trash(3.5); break;
        case Trash::Independent: net.add_independent_asymmetric_trash(1.5, 2.0); break;
    }
}

template <size_t D>
Net make(const Dist<D>& emp, const std::vector<const Dist<D>*>& theos, bool chain,
         double max_dist, double p, Trash trash, const SolverConfig& cfg,
         int64_t cost_scale = 0) {
    std::vector<Dist<D>*> t;
    for (const auto* x : theos) t.push_back(const_cast<Dist<D>*>(x));
    Net net = [&] {
        if constexpr (D == 1) {
            if (chain) return Factory::create_1d(&emp, t, DistanceMetric::L1, max_dist, p);
        }
        return Factory::create(&emp, t, DistanceMetric::L1, max_dist, p);
    }();
    net.set_intensity_scale(1000.0);
    if (cost_scale > 0) net.set_cost_scaling(cost_scale);
    add_trash(net, trash);
    net.build(cfg);
    return net;
}

double real_cost(const Net& net) {
    return (double)net.total_cost() / ((double)net.scale_factor() * net.intensity_scale_factor());
}

std::vector<std::vector<double>> points(size_t k) {
    std::vector<std::vector<double>> pts;
    std::mt19937_64 rng(77);
    std::uniform_real_distribution<double> u(0.3, 1.7);
    for (int i = 0; i < 8; ++i) {
        std::vector<double> w(k);
        for (auto& x : w) x = u(rng);
        pts.push_back(w);
    }
    pts.push_back(std::vector<double>(k, 0.0));   // everything to trash
    pts.push_back(std::vector<double>(k, 1.0));
    return pts;
}

NetworkSimplexConfig ns(NSWarmMode warm, NSPivotRule pivot = NSPivotRule::BLOCK_SEARCH) {
    NetworkSimplexConfig c;
    c.warm = warm;
    c.pivot = pivot;
    return c;
}

// Integer-unit costs of one network over the shared point sequence.
std::vector<int64_t> scaled_costs(Net& net, const std::vector<std::vector<double>>& pts) {
    std::vector<int64_t> out;
    for (const auto& w : pts) {
        net.solve(w);
        out.push_back(net.total_cost());
    }
    return out;
}

// ── 1. dense 2-D: every NetworkSimplex mode and solver agree ───────────────
void dense_solvers_agree() {
    std::mt19937_64 rng(1);
    auto emp = random_dist<2>(rng, 40, 12.0, false);
    auto t1 = random_dist<2>(rng, 30, 12.0, false);
    auto t2 = random_dist<2>(rng, 25, 12.0, false);
    const auto pts = points(2);
    for (Trash trash : {Trash::Simple, Trash::Both, Trash::Independent}) {
        Net ref = make<2>(emp, {&t1, &t2}, false, 3.0, 1.0, trash, ns(NSWarmMode::None), 1000);
        const auto expect = scaled_costs(ref, pts);
        std::vector<SolverConfig> cfgs;
        for (auto warm : {NSWarmMode::Simple, NSWarmMode::Dual, NSWarmMode::Primal,
                          NSWarmMode::DualRatio, NSWarmMode::DualGreedy, NSWarmMode::LinkCut})
            cfgs.push_back(ns(warm));
        for (auto pivot : {NSPivotRule::FIRST_ELIGIBLE, NSPivotRule::BEST_ELIGIBLE,
                           NSPivotRule::CANDIDATE_LIST, NSPivotRule::ALTERING_LIST})
            cfgs.push_back(ns(NSWarmMode::DualRatio, pivot));
        NetworkSimplexConfig gated = ns(NSWarmMode::DualRatio);
        gated.repair_max_nodes = 0;
        cfgs.push_back(gated);
        cfgs.push_back(CostScalingConfig{});
        cfgs.push_back(CycleCancelingConfig{});
        cfgs.push_back(CapacityScalingConfig{});
        for (const auto& cfg : cfgs) {
            Net net = make<2>(emp, {&t1, &t2}, false, 3.0, 1.0, trash, cfg, 1000);
            check(scaled_costs(net, pts) == expect, "solver disagrees with cold NetworkSimplex",
                  "dense_solvers_agree");
        }
    }
}

// ── 2. 1-D: chain factory, SlopeDP and dense factory agree ─────────────────
void chain_backends_agree() {
    std::mt19937_64 rng(2);
    auto emp = random_dist<1>(rng, 60, 40.0, true);
    auto t1 = random_dist<1>(rng, 50, 40.0, true);
    auto t2 = random_dist<1>(rng, 45, 40.0, true);
    const auto pts = points(2);
    for (Trash trash : {Trash::Simple, Trash::Both, Trash::Independent}) {
        // Dense factory with an unlimited window: the chain's semantics.
        Net dense = make<1>(emp, {&t1, &t2}, false, 1e9, 1.0, trash, ns(NSWarmMode::None), 1);
        std::vector<double> expect;
        for (const auto& w : pts) { dense.solve(w); expect.push_back(real_cost(dense)); }
        std::vector<SolverConfig> cfgs;
        for (const char* store : {"flat", "tree", "augmented"})
            for (const char* mode : {"lazy", "cluster"})
                cfgs.push_back(SlopeDPConfig(mode, store));
        if (trash != Trash::Independent) {   // chain hop arcs cannot carry the shift
            cfgs.push_back(ns(NSWarmMode::DualRatio));
            cfgs.push_back(ns(NSWarmMode::None));
        }
        for (const auto& cfg : cfgs) {
            Net chain = make<1>(emp, {&t1, &t2}, true, 1e9, 1.0, trash, cfg, 1);
            for (size_t i = 0; i < pts.size(); ++i) {
                chain.solve(pts[i]);
                check(close(real_cost(chain), expect[i]), "chain backend disagrees with dense",
                      "chain_backends_agree");
            }
        }
    }
}

// ── 3. p = 2: ConvexSweep against dense NetworkSimplex ─────────────────────
void sweep_matches_dense() {
    std::mt19937_64 rng(3);
    auto emp = random_dist<1>(rng, 40, 30.0, true);
    auto t1 = random_dist<1>(rng, 35, 30.0, true);
    const auto pts = points(1);
    for (Trash trash : {Trash::Simple, Trash::Both}) {
        Net dense = make<1>(emp, {&t1}, false, 1e9, 2.0, trash, ns(NSWarmMode::DualRatio), 1);
        for (const char* store : {"flat", "tree", "augmented"}) {
            Net sweep = make<1>(emp, {&t1}, true, 1e9, 2.0, trash, ConvexSweepConfig(store), 1);
            for (const auto& w : pts) {
                dense.solve(w);
                sweep.solve(w, /*with_duals=*/true);
                check(close(real_cost(sweep), real_cost(dense)), "ConvexSweep disagrees with dense",
                      "sweep_matches_dense");
            }
        }
    }
}

// ── 4. derivatives and dual-cut bounds agree across backends ───────────────
void derivatives_and_cuts() {
    std::mt19937_64 rng(4);
    auto emp = random_dist<1>(rng, 50, 30.0, true);
    auto t1 = random_dist<1>(rng, 40, 30.0, true);
    auto t2 = random_dist<1>(rng, 40, 30.0, true);
    const auto pts = points(2);
    Net dense = make<1>(emp, {&t1, &t2}, false, 1e9, 1.0, Trash::Both, ns(NSWarmMode::DualRatio), 1);
    Net chain = make<1>(emp, {&t1, &t2}, true, 1e9, 1.0, Trash::Both, ns(NSWarmMode::DualRatio), 1);
    Net sdp = make<1>(emp, {&t1, &t2}, true, 1e9, 1.0, Trash::Both, SlopeDPConfig(), 1);
    for (const auto& w : pts) {
        for (Net* net : {&dense, &chain, &sdp}) net->solve(w);
        const auto d0 = dense.spectrum_proportion_derivatives();
        for (Net* net : {&chain, &sdp}) {
            const auto d = net->spectrum_proportion_derivatives();
            check(d.size() == d0.size(), "derivative count", "derivatives_and_cuts");
            for (size_t k = 0; k < d.size() && k < d0.size(); ++k)
                check(d[k].first == d0[k].first && close(d[k].second, d0[k].second, 1e-9, 1e-6),
                      "proportion derivative disagrees with dense", "derivatives_and_cuts");
        }
        for (Net* net : {&dense, &chain, &sdp}) {
            auto [intercept, slopes, upper, error, rounded] = net->dual_cut();
            double lower = intercept;
            for (size_t k = 0; k < slopes.size(); ++k) lower += slopes[k] * w[k];
            // The cut bounds the continuous-supply objective; rounded is the
            // truncated-supply cost and may lie below it, upper may not.
            check(lower <= upper + 1e-9 * std::max(1.0, std::fabs(upper)), "cut above the upper bound",
                  "derivatives_and_cuts");
            check(upper >= rounded && error >= 0, "upper bound below the cost", "derivatives_and_cuts");
            auto [cost_err, rho, lower_slack] = net->cost_quantization_bounds();
            check(cost_err >= 0 && rho >= 0 && rho <= 1 && lower_slack >= 0,
                  "cost quantization bounds out of range", "derivatives_and_cuts");
        }
    }
}

// ── 5. warm position updates equal fresh builds ────────────────────────────
void position_updates() {
    std::mt19937_64 rng(5);
    auto emp = random_dist<2>(rng, 40, 12.0, false);
    auto t1 = random_dist<2>(rng, 35, 12.0, false);
    // Unlimited window: a position update keeps the build-time edge set, so a
    // fresh build at the new positions is the same problem only if no pair
    // can fall in or out of the window.
    Net net = make<2>(emp, {&t1}, false, 1e9, 1.0, Trash::Simple, ns(NSWarmMode::DualRatio), 1000);
    net.solve(std::vector<double>{1.0});
    std::normal_distribution<double> jitter(0.0, 0.05);
    auto pos = emp.get_positions();
    for (int step = 0; step < 5; ++step) {
        for (auto& p : pos) for (auto& x : p) x += jitter(rng);
        Dist<2> moved(pos, std::vector<double>(emp.intensities().begin(), emp.intensities().end()));
        net.update_positions_and_solve(&moved, std::vector<Dist<2>*>{&t1}, DistanceMetric::L1);
        Net fresh = make<2>(moved, {&t1}, false, 1e9, 1.0, Trash::Simple, ns(NSWarmMode::None), 1000);
        fresh.solve(std::vector<double>{1.0});
        check(net.total_cost() == fresh.total_cost(),
              "warm position update disagrees with a fresh build", "position_updates");
    }
}

// ── 6. independent networks on concurrent threads ──────────────────────────
void concurrent_networks() {
    const auto pts = points(2);
    auto build_and_solve = [&](int seed, std::vector<int64_t>& out) {
        std::mt19937_64 rng(seed);
        auto emp = random_dist<2>(rng, 30, 10.0, false);
        auto t1 = random_dist<2>(rng, 25, 10.0, false);
        auto t2 = random_dist<2>(rng, 25, 10.0, false);
        Net net = make<2>(emp, {&t1, &t2}, false, 3.0, 1.0, Trash::Both, ns(NSWarmMode::DualRatio), 1000);
        out = scaled_costs(net, pts);
        std::mt19937_64 rng1(seed + 1000);
        auto e1 = random_dist<1>(rng1, 40, 30.0, true);
        auto s1 = random_dist<1>(rng1, 35, 30.0, true);
        Net sweep = make<1>(e1, {&s1}, true, 1e9, 2.0, Trash::Simple, ConvexSweepConfig(), 1);
        Net sdp = make<1>(e1, {&s1}, true, 1e9, 1.0, Trash::Both, SlopeDPConfig(), 1);
        for (const auto& w : points(1)) {
            sweep.solve(w, true);
            sdp.solve(w);
            out.push_back(sweep.total_cost());
            out.push_back(sdp.total_cost());
        }
    };
    constexpr int N = 4;
    std::vector<std::vector<int64_t>> seq(N), par(N);
    for (int i = 0; i < N; ++i) build_and_solve(100 + i, seq[i]);
    std::vector<std::thread> threads;
    for (int i = 0; i < N; ++i) threads.emplace_back(build_and_solve, 100 + i, std::ref(par[i]));
    for (auto& t : threads) t.join();
    for (int i = 0; i < N; ++i)
        check(par[i] == seq[i], "threaded result differs from sequential", "concurrent_networks");
}

}  // namespace

int main() {
    dense_solvers_agree();
    chain_backends_agree();
    sweep_matches_dense();
    derivatives_and_cuts();
    position_updates();
    concurrent_networks();
    std::printf("%d checks, %d failures\n", checks, failures);
    std::printf("RESULT: %s\n", failures ? "FAILED" : "PASSED");
    return failures ? 1 : 0;
}
