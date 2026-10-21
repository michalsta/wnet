// Standalone ConvexSweep vertex-count probe on a dense synthetic profile.
// Build from src/wnet/cpp: g++ -std=c++20 -O2 -I. ../../../prototypes/sweep_vertex_probe.cpp
// The slimming variants in docs/convex_sweep_vertex_growth.md used temporary
// hooks in sweep_step() (normalize() after every step; dropping near-collinear
// vertices) that were not kept.
#include <cstdio>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include "wnet/wide_int.hpp"
#include "wnet/ordered_breakpoints.hpp"
#include "wnet/convex_sweep.hpp"
using namespace convex_sweep;
int main(int argc, char** argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 2000;
    const double radius = argc > 2 ? atof(argv[2]) : 0.2236;
    std::mt19937_64 rng(5);
    std::vector<Event> ev;
    const double step = 10.0 / n;
    for (int i = 0; i < n; ++i) {
        for (int side = 0; side < 2; ++side) {
            Event e; e.side = side; e.pos = i * step; e.count = 1 + rng() % 100000; e.index = SIZE_MAX;
            ev.push_back(e);
        }
    }
    const double S = 1e8;
    auto profit = [&](double a, double b) -> i64 { double d = a - b; return (i64)std::llround(S * (radius * radius - d * d)); };
    auto t0 = std::chrono::steady_clock::now();
    auto r = sweep_solve<true>(ev, profit, radius);
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("n=%d radius=%.4f W~%.0f profit=%lld records=%zu vertices/event=%.1f time=%.3fs\n", n, radius, 2 * radius / step,
           (long long)r.profit, (size_t)r.records, (double)r.vertices / r.records, dt);
}
