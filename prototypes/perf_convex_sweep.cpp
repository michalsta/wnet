// Perf probe for convex_sweep.hpp at profile-NMR scale, with vertex stats.
// Build: g++ -std=c++20 -O2 -I../src/wnet/cpp perf_convex_sweep.cpp -o perf_kernel

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "wnet/convex_sweep.hpp"

using namespace convex_sweep;

int main(int argc, char** argv) {
    int K = argc > 1 ? atoi(argv[1]) : 2000;
    std::mt19937_64 rng(1);
    const double grid = 9.64e-4;
    const i64 scale = 1099511627776;  // 2^40-ish like auto cost scale
    const double tau_real = 0.1109;
    const i64 tau_q = (i64)llround(tau_real * scale);
    const double R = std::sqrt((double)(tau_q + 1) / (double)scale);
    std::vector<Event> ev;
    for (int i = 0; i < K; ++i) {
        double pos = i * grid;
        ev.push_back({pos, 0, (i64)(1 + rng() % 29)});
        ev.push_back({pos + grid / 3, 1, (i64)(1 + rng() % 29)});
    }
    std::sort(ev.begin(), ev.end(), [](const Event& a, const Event& b) {
        if (a.pos != b.pos) return a.pos < b.pos;
        return a.side < b.side;
    });
    auto profit = [&](double a, double z) -> i64 {
        const double d = z > a ? z - a : a - z;
        return tau_q - (i64)llround(std::pow(d, 2.0) * (double)scale);
    };
    // instrumented sweep: track vertex counts
    SideState sides[2];
    size_t max_v = 0, sum_v = 0, steps = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (const Event& e : ev) {
        sweep_step(sides[e.side], sides[1 - e.side], e.pos, e.count, profit, R);
        max_v = std::max(max_v, std::max(sides[0].V.vs.size(), sides[1].V.vs.size()));
        sum_v += sides[0].V.vs.size() + sides[1].V.vs.size();
        ++steps;
    }
    auto t1 = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    std::printf("K=%d events=%zu time=%.2fs max_vertices=%zu avg_vertices=%.0f\n",
                K, ev.size(), dt, max_v, (double)sum_v / (2 * steps));
    return 0;
}
