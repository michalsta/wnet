// Fuzz harness for the software 128-bit path in wnet/wide_int.hpp (the one
// MSVC builds, since it has no __int128) against native __int128, which this
// TU may still use regardless of WNET_FORCE_SOFT_INT128.
// Build: g++ -std=c++20 -O2 -DWNET_FORCE_SOFT_INT128 -I../src/wnet/cpp
//        test_wide_int.cpp -o test_wide_int && ./test_wide_int

#include <cstdio>
#include <cstdlib>
#include <random>

#include "wnet/wide_int.hpp"

static_assert(WNET_NATIVE_INT128 == 0,
              "build with -DWNET_FORCE_SOFT_INT128 to test the software path");

int main() {
    std::mt19937_64 rng(12345);
    auto pick = [&](int bits) -> int64_t {
        const int64_t v = (int64_t)(rng() >> (64 - bits));
        return (rng() & 1) ? -v : v;
    };
    const int widths[] = {1, 2, 8, 20, 31, 32, 33, 40, 52, 62, 63};
    long long cases = 0, bad = 0;

    for (int wa : widths) for (int wb : widths) for (int wc : widths) {
        for (int it = 0; it < 4000; ++it) {
            const int64_t a = pick(wa), b = pick(wb), c = pick(wc);
            if (c == 0) continue;
            const __int128 q = ((__int128)a * b) / c;
            // The callers' contract: the quotient fits in int64.
            if (q > (__int128)INT64_MAX || q < (__int128)INT64_MIN) continue;
            ++cases;

            const int64_t got = wide_int::mul_div_trunc(a, b, c);
            if (got != (int64_t)q && ++bad < 10)
                printf("mul_div_trunc(%lld, %lld, %lld) = %lld, want %lld\n",
                       (long long)a, (long long)b, (long long)c,
                       (long long)got, (long long)q);

            const int64_t d = pick(wc);
            const bool want = ((__int128)a * b == (__int128)c * d);
            if (wide_int::products_equal(a, b, c, d) != want && ++bad < 20)
                printf("products_equal(%lld, %lld, %lld, %lld) != %d\n",
                       (long long)a, (long long)b, (long long)c,
                       (long long)d, (int)want);

            if (!wide_int::products_equal(a, b, a, b)) {
                ++bad;
                printf("products_equal is not reflexive at %lld, %lld\n",
                       (long long)a, (long long)b);
            }
            // Opposite signs compare equal only when the product is zero.
            if (wide_int::products_equal(a, b, -a, b) != ((__int128)a * b == 0)) {
                ++bad;
                printf("products_equal sign handling at %lld, %lld\n",
                       (long long)a, (long long)b);
            }
        }
    }

    // wide_t accumulator: (wide)a * b, +, -, ==, narrowing cast.
    for (int it = 0; it < 200000; ++it) {
        const int64_t a = pick(52), b = pick(52);
        const wide_int::wide_t w = wide_int::wide_t(a) * b;
        if ((int64_t)(w + w - w) != (int64_t)((__int128)a * b)) {
            ++bad;
            printf("wide_t accumulate at %lld, %lld\n",
                   (long long)a, (long long)b);
        }
        if (!(w + w - w == w)) {
            ++bad;
            printf("wide_t compare at %lld, %lld\n",
                   (long long)a, (long long)b);
        }
    }

    printf("%lld cases, %lld failures\n", cases, bad);
    return bad ? 1 : 0;
}
