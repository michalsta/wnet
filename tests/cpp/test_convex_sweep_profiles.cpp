#include "wnet/convex_sweep.hpp"
#include <iostream>
#include <random>
#include <thread>
using namespace convex_sweep;
void require(bool ok, const char *why) {
    if (!ok)
        throw std::runtime_error(why);
}
template <class F> void test(std::mt19937_64 &rng, size_t trials) {
    for (size_t it = 0; it < trials; ++it) {
        F f, t;
        i64 k = 0, v = 10000;
        f.vs.push_back({0, v});
        for (size_t j = 0, n = 1 + rng() % 12; j < n; ++j) {
            i64 len = 1 + rng() % 8, slope = i64(rng() % 41) - 20;
            k += len;
            v += len * slope;
            f.vs.push_back({k, v});
        }
        t.vs.push_back({0, 0});
        i64 at = 0, value = 0;
        while (at < k) {
            i64 span = std::min<i64>(1 + rng() % 8, k - at);
            value += span * i64(rng() % 20);
            at += span;
            t.vs.push_back({at, value});
        }
        f.normalize();
        t.normalize();
        const i64 q = rng() % (k + 10);
        auto w = windowed_max(f, q), fused = window_sub(f, q, t), separate = sub_pl(w, t);
        auto suffix = f;
        suffix.suffix_max();
        auto truncate = f;
        auto cut = i64(rng() % (k + 1));
        truncate.truncate(cut);
        auto shifted = f;
        shifted.shift_append(q);
        auto maxed = f;
        maxed.pointwise_max(t);
        auto self = f;
        self.pointwise_max(self);
        ProfitCursor cursor(t);
        for (i64 x = 0; x <= k; ++x) {
            i64 expected = NEG_INF, suf = NEG_INF;
            for (i64 z = x; z <= k; ++z) {
                suf = std::max(suf, f.eval(z));
                if (z <= x + q)
                    expected = std::max(expected, f.eval(z));
            }
            require(w.eval(x) == expected, "window max");
            require(suffix.eval(x) == suf, "suffix max");
            require(fused.eval(x) == expected - t.eval(x), "fused subtraction");
            require(fused.eval(x) == separate.eval(x), "fusion parity");
            require(cursor.eval(x) == t.eval(x), "profit cursor");
            require(maxed.eval(x) == std::max(f.eval(x), t.eval(x)), "pointwise max");
            require(self.eval(x) == f.eval(x), "self-aliased max");
            if (x <= cut)
                require(truncate.eval(x) == f.eval(x), "truncate");
        }
        for (i64 x = 0; x <= k + q; ++x)
            require(shifted.eval(x) == f.eval(std::clamp<i64>(x - q, 0, k)), "shift append");
        // Unequal domains: the shorter function stops contributing at K+1.
        auto shortf = F::constant(k / 2, 20000);
        auto longf = F::constant(k, 0);
        shortf.pointwise_max(longf);
        for (i64 x = 0; x <= k; ++x)
            require(shortf.eval(x) == (x <= k / 2 ? 20000 : 0), "unequal-domain pointwise max");
    }
}
int main(int argc, char **argv) {
    size_t n = argc > 1 ? std::stoul(argv[1]) : 10000;
    std::mt19937_64 rng(52);
    // Edge/interior plateau crossing missed by the old generic window.
    PLF f;
    f.vs = {{0, 200}, {2, 182}, {4, 184}, {12, 112}};
    require(windowed_max(f, 10).eval(1) == 191, "plateau regression");
    // Cursor fallback for nonintegral slopes, and exact wide interpolation.
    for (i64 end : {i64(3), i64(-3)}) {
        PLF f;
        f.vs = {{0, 0}, {10, end}};
        ProfitCursor cursor(f);
        for (i64 k = 0; k <= 10; ++k)
            require(cursor.eval(k) == f.eval(k), "fractional cursor");
    }
    PLF wide;
    wide.vs = {{0, 0}, {1000000000000LL, 3000000000000000000LL}};
    ProfitCursor cursor(wide);
    for (i64 k : {i64(0), i64(1), i64(999999999999LL), i64(1000000000000LL)})
        require(cursor.eval(k) == wide.eval(k), "wide cursor");
    std::thread a([] {
        std::mt19937_64 r(1);
        test<PLF>(r, 200);
    });
    std::thread b([] {
        std::mt19937_64 r(2);
        test<PLF>(r, 200);
    });
    a.join();
    b.join();
    test<PLF>(rng, n);
    test<TreePLF>(rng, n / 10);
    test<AugmentedPLF>(rng, n / 10);
    std::cout << n << " flat and " << n / 10
              << " cases per tree backend: exact windows, fusion, cursor, suffix, shift, truncate "
                 "and maxima passed\n";
}
