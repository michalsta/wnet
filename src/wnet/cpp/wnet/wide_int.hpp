#ifndef WNET_WIDE_INT_HPP
#define WNET_WIDE_INT_HPP

#include <bit>
#include <cstdint>

// 128-bit signed integer support for the chain-native solvers.  Products of
// two int64 quantities (mass x distance in the slope DP, profit x span in the
// convex sweep) overflow int64 even when every final result fits, so the
// intermediates must be 128-bit.
//
// GCC and Clang provide __int128; MSVC does not, so a software fallback
// implements exactly the operations the solvers use:
//   * wide_t     - accumulator: construction from int64, (wide)a * b, +=, +,
//                  binary and unary -, ==/!=, explicit narrowing cast.  No
//                  ordering, division or shifts.
//   * mul_div_trunc(a, b, c)     - (a * b) / c truncated toward zero, with the
//                                  quotient assumed to fit in int64.
//   * products_equal(a, b, c, d) - exact a * b == c * d.
//
// Define WNET_FORCE_SOFT_INT128 to build the fallback on a compiler that has
// __int128, which is how the test suite exercises it off Windows.
namespace wide_int {

#if defined(__SIZEOF_INT128__) && !defined(WNET_FORCE_SOFT_INT128)
#define WNET_NATIVE_INT128 1
#else
#define WNET_NATIVE_INT128 0
#endif

namespace detail {

inline constexpr uint64_t abs_u64(int64_t v) {
    return v < 0 ? 0u - static_cast<uint64_t>(v) : static_cast<uint64_t>(v);
}

struct u128 { uint64_t hi, lo; };

// 64x64 -> 128 unsigned, schoolbook on 32-bit limbs.
inline constexpr u128 umul128(uint64_t a, uint64_t b) {
    const uint64_t a0 = a & 0xffffffffu, a1 = a >> 32;
    const uint64_t b0 = b & 0xffffffffu, b1 = b >> 32;
    const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffu) + (p10 & 0xffffffffu);
    return {p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32),
            (p00 & 0xffffffffu) | (mid << 32)};
}

// (hi:lo) / d, normalized schoolbook long division on 32-bit limbs (Knuth
// algorithm D, the two-limb case).  Requires d != 0 and hi < d, i.e. the
// quotient fits in 64 bits.
inline uint64_t udiv128by64(uint64_t hi, uint64_t lo, uint64_t d) {
    constexpr uint64_t B = uint64_t(1) << 32;
    const int s = std::countl_zero(d);
    d <<= s;
    const uint64_t d1 = d >> 32, d0 = d & 0xffffffffu;
    const uint64_t n32 = (s == 0) ? hi : (hi << s) | (lo >> (64 - s));
    const uint64_t n10 = lo << s;
    const uint64_t n1 = n10 >> 32, n0 = n10 & 0xffffffffu;

    uint64_t q1 = n32 / d1, r = n32 - q1 * d1;
    while (q1 >= B || q1 * d0 > B * r + n1) {
        --q1;
        r += d1;
        if (r >= B) break;
    }
    const uint64_t n21 = n32 * B + n1 - q1 * d;
    uint64_t q0 = n21 / d1;
    r = n21 - q0 * d1;
    while (q0 >= B || q0 * d0 > B * r + n0) {
        --q0;
        r += d1;
        if (r >= B) break;
    }
    return q1 * B + q0;
}

}  // namespace detail

#if WNET_NATIVE_INT128

using wide_t = __int128;

inline int64_t mul_div_trunc(int64_t a, int64_t b, int64_t c) {
    return static_cast<int64_t>((static_cast<__int128>(a) * b) / c);
}

inline bool products_equal(int64_t a, int64_t b, int64_t c, int64_t d) {
    return static_cast<__int128>(a) * b == static_cast<__int128>(c) * d;
}

#else

struct wide_t {
    uint64_t lo = 0;
    int64_t  hi = 0;

    constexpr wide_t() = default;
    constexpr wide_t(long long v)                    // implicit: `W x = 0`
        : lo(static_cast<uint64_t>(v)), hi(v < 0 ? -1 : 0) {}

    constexpr wide_t& operator+=(const wide_t& b) {
        lo += b.lo;
        hi += b.hi + (lo < b.lo ? 1 : 0);
        return *this;
    }
    friend constexpr wide_t operator+(wide_t a, const wide_t& b) { return a += b; }
    friend constexpr wide_t operator-(const wide_t& a) {
        wide_t r;
        r.lo = ~a.lo;
        r.hi = ~a.hi;
        r.lo += 1;
        if (r.lo == 0) r.hi += 1;
        return r;
    }
    friend constexpr wide_t operator-(const wide_t& a, const wide_t& b) {
        return a + (-b);
    }
    friend constexpr bool operator==(const wide_t& a, const wide_t& b) {
        return a.lo == b.lo && a.hi == b.hi;
    }
    friend constexpr bool operator!=(const wide_t& a, const wide_t& b) {
        return !(a == b);
    }
    // (wide)a * b with |*this| known to fit in int64 — always true at the
    // DP's call sites, where the left factor is a freshly promoted int64.
    constexpr wide_t operator*(long long b) const {
        const auto a = static_cast<long long>(lo);   // value fits: hi is sign fill
        const bool neg = (a < 0) != (b < 0);
        const detail::u128 p = detail::umul128(detail::abs_u64(a), detail::abs_u64(b));
        wide_t r;
        r.lo = p.lo;
        r.hi = static_cast<int64_t>(p.hi);
        return neg ? -r : r;
    }
    template <typename T>
    explicit constexpr operator T() const {          // narrowing, like (V)__int128
        return static_cast<T>(static_cast<long long>(lo));
    }
};

inline int64_t mul_div_trunc(int64_t a, int64_t b, int64_t c) {
    const bool neg = ((a < 0) != (b < 0)) != (c < 0);
    const detail::u128 p = detail::umul128(detail::abs_u64(a), detail::abs_u64(b));
    const uint64_t q = detail::udiv128by64(p.hi, p.lo, detail::abs_u64(c));
    return neg ? -static_cast<int64_t>(q) : static_cast<int64_t>(q);
}

inline bool products_equal(int64_t a, int64_t b, int64_t c, int64_t d) {
    const detail::u128 p = detail::umul128(detail::abs_u64(a), detail::abs_u64(b));
    const detail::u128 q = detail::umul128(detail::abs_u64(c), detail::abs_u64(d));
    if (p.hi != q.hi || p.lo != q.lo) return false;
    if ((p.hi | p.lo) == 0) return true;             // +0 == -0
    return ((a < 0) != (b < 0)) == ((c < 0) != (d < 0));
}

#endif

}  // namespace wide_int

#endif  // WNET_WIDE_INT_HPP
