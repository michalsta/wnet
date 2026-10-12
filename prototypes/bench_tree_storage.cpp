// g++ -std=c++20 -O3 -Iwnet/src/wnet/cpp -I<installed-pylmcf>/cpp
//     wnet/prototypes/bench_tree_storage.cpp -o /tmp/bench_tree_storage
// /tmp/bench_tree_storage [position_count=1000] [profitable_radius=8]
#include "wnet/decompositable_graph.hpp"
#include <chrono>
#include <cstdio>
#include <random>

using Clock = std::chrono::steady_clock;
using I = int64_t;

template<class Storage>
auto slope(const std::vector<I>& e, const std::vector<I>& t) {
    slopedp_detail::ConvexPL<I, Storage> dp;
    const I wall = std::numeric_limits<I>::max() / 1024;
    dp.insL(0, wall); dp.insR(0, wall);
    for (size_t i = 0; i < e.size(); ++i) {
        dp.convV(50, e[i] - t[i], t[i]);
        dp.minval += -slopedp_detail::wide_t(50) * t[i];
        if (i + 1 < e.size()) dp.addAbs(0, 1);
    }
    return dp.evalAt(0);
}

template<class Fn>
auto timed(const char* label, Fn fn) {
    const auto start = Clock::now();
    auto result = fn();
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::printf("%-20s %.3f ms\n", label, ms);
    return result;
}

int main(int argc, char** argv) {
    const size_t n = argc > 1 ? std::stoull(argv[1]) : 1000;
    const double radius = argc > 2 ? std::stod(argv[2]) : 8;
    std::mt19937 rng(42);
    std::vector<I> e(n), t(n);
    std::vector<convex_sweep::Event> events;
    for (size_t i = 0; i < n; ++i) {
        e[i] = 1 + rng() % 7; t[i] = 1 + rng() % 7;
        events.push_back({double(i), 0, e[i], i});
        events.push_back({double(i) + 0.25, 1, t[i], i});
    }
    const I tau = I(radius * radius);
    auto profit = [tau](double a, double b) -> I { return tau - std::llround((a-b)*(a-b)); };
    const double bound = std::sqrt(double(tau) + 1);
    const auto sf = timed("SlopeDP flat", [&] { return slope<breakpoint_storage::FlatSlopeStorage>(e,t); });
    const auto st = timed("SlopeDP std::map", [&] { return slope<breakpoint_storage::MapStorage>(e,t); });
    if (sf != st) throw std::runtime_error("SlopeDP storage mismatch");
    const auto cf = timed("ConvexSweep flat", [&] { return convex_sweep::sweep_solve<false>(events,profit,bound); });
    const auto ct = timed("ConvexSweep std::map", [&] { return convex_sweep::sweep_solve<false,convex_sweep::TreePLF>(events,profit,bound); });
    if (cf != ct) throw std::runtime_error("ConvexSweep storage mismatch");
    const auto tf = timed("Sweep trace flat", [&] { return convex_sweep::sweep_solve<true>(events,profit,bound); });
    const auto tt = timed("Sweep trace std::map", [&] { return convex_sweep::sweep_solve<true,convex_sweep::TreePLF>(events,profit,bound); });
    if (tf.profit != tt.profit) throw std::runtime_error("Traced storage mismatch");
    std::printf("trace records=%zu vertices=%zu matched_pairs=%zu\n", tt.records,tt.vertices,tt.matches.size());
}
