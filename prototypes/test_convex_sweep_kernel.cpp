// Standalone fuzz harness for convex_sweep.hpp: sweep kernel vs a direct
// O(N*M) unit-level alignment DP oracle, integer profits.
// Build: g++ -std=c++20 -O2 -I../src/wnet/cpp test_convex_sweep_kernel.cpp
//        -o test_kernel && ./test_kernel

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "wnet/convex_sweep.hpp"

using namespace convex_sweep;

static i64 quantize(double real, i64 scale) {
    return (i64)llround(real * (double)scale);
}

// Oracle: unit-level alignment DP.
static i64 alignment_dp(const std::vector<double>& eu,
                        const std::vector<double>& tu,
                        i64 tau_q, double p, i64 scale) {
    size_t N = eu.size(), M = tu.size();
    std::vector<i64> dp(M + 1, 0), prev;
    for (size_t i = 1; i <= N; ++i) {
        prev = dp;
        dp[0] = prev[0];
        for (size_t j = 1; j <= M; ++j) {
            i64 pi = tau_q - quantize(std::pow(std::abs(eu[i-1] - tu[j-1]), p), scale);
            i64 best = dp[j - 1];
            if (prev[j] > best) best = prev[j];
            if (prev[j - 1] + pi > best) best = prev[j - 1] + pi;
            dp[j] = best;
        }
    }
    i64 r = dp[M];
    return r > 0 ? r : 0;
}

int main() {
    std::mt19937_64 rng(12345);
    const i64 scale = 1 << 20;
    const double p_list[] = {2.0, 3.0, 1.5};
    int fails = 0;
    for (double p : p_list) {
        const double c_exp = 0.5, c_theo = 0.8125;
        const double tau_real = c_exp + c_theo;
        const i64 tau_q = quantize(tau_real, scale);
        const double R = std::pow(tau_real, 1.0 / p);
        for (int trial = 0; trial < 400; ++trial) {
            auto side = [&](std::vector<Event>& ev, int s,
                            std::vector<double>& units) {
                int n = 1 + (int)(rng() % 10);
                double last = -1;
                for (int i = 0; i < n; ++i) {
                    double pos = (double)(rng() % 80) * 0.25;
                    if (pos <= last) continue;
                    last = pos;
                    i64 cnt = 1 + (i64)(rng() % 5);
                    ev.push_back({pos, s, cnt});
                    for (i64 c = 0; c < cnt; ++c) units.push_back(pos);
                }
            };
            std::vector<Event> ev;
            std::vector<double> eu, tu;
            side(ev, 0, eu);
            side(ev, 1, tu);
            std::sort(ev.begin(), ev.end(),
                      [](const Event& a, const Event& b) {
                          if (a.pos != b.pos) return a.pos < b.pos;
                          return a.side < b.side;
                      });
            std::sort(eu.begin(), eu.end());
            std::sort(tu.begin(), tu.end());
            auto profit = [&](double pending_pos, double z) -> i64 {
                return tau_q - quantize(std::pow(std::abs(z - pending_pos), p), scale);
            };
            i64 got = sweep_solve(ev, profit, R);
            i64 got_noprune = sweep_solve(
                ev, profit, std::numeric_limits<double>::max());
            i64 ref = alignment_dp(eu, tu, tau_q, p, scale);
            if (got != ref || got_noprune != ref) {
                ++fails;
                if (fails <= 5) {
                    std::printf("FAIL p=%.1f trial %d: sweep=%lld "
                                "noprune=%lld ref=%lld\n",
                                p, trial, (long long)got,
                                (long long)got_noprune, (long long)ref);
                    for (auto& e : ev)
                        std::printf("  ev pos=%.2f side=%d cnt=%lld\n",
                                    e.pos, e.side, (long long)e.count);
                }
            }
        }
        std::printf("p=%.1f: 400 trials done\n", p);
    }
    std::printf("total failures: %d\n", fails);
    return fails ? 1 : 0;
}
