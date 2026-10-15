#ifndef WNET_NMR_TRIAL_VIEWS_HPP
#define WNET_NMR_TRIAL_VIEWS_HPP
#include "moments.hpp"
namespace nmr_views {
using I = int64_t;
template <class C> size_t count(const C &c) {
    if constexpr (requires { c.knot_count(); })
        return c.knot_count();
    else
        return c.vs.size();
}
template <class C> I key(const C &c, size_t i) {
    if constexpr (requires { c.knot(i); })
        return c.knot(i);
    else
        return c.vs[i].k;
}
template <class C> I value(const C &c, size_t i) {
    if constexpr (requires { c.knot_value(i); })
        return c.knot_value(i);
    else
        return c.vs[i].v;
}
template <class View> size_t hint(const View &f) {
    if constexpr (requires { f.vs.capacity_hint(); })
        return f.vs.capacity_hint();
    else
        return f.vs.size();
}
template <class C> struct Cursor {
    const C &c;
    size_t at = 0;
    bool cached = false, integral = false;
    I left = 0, right = 0, vl = 0, vr = 0, slope = 0;
    explicit Cursor(const C &c) : c(c) {}
    I eval(I k) {
        if (k <= 0)
            return value(c, 0);
        if (k >= c.K())
            return value(c, count(c) - 1);
        size_t next = at;
        while (next + 1 < count(c) && key(c, next + 1) <= k)
            ++next;
        if (!cached || next != at) {
            at = next;
            left = key(c, at);
            right = key(c, at + 1);
            vl = value(c, at);
            vr = value(c, at + 1);
            I delta = vr - vl, span = right - left;
            integral = delta % span == 0;
            slope = delta / span;
            cached = true;
        }
        return vl + (integral ? slope * (k - left)
                              : wide_int::mul_div_trunc(vr - vl, k - left, right - left));
    }
};
// Read-only union of two knot streams. No array of V+T values is constructed.
// The iterator keeps its position in both source streams, making its vertex
// evaluation O(1); random crossing queries delegate to the underlying curves.
template <class F, class C, bool Fast = false> struct Sum {
    using V = typename F::V;
    const F &f;
    const C &c;
    V first{}, last{};
    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using value_type = V;
        using difference_type = std::ptrdiff_t;
        using reference = V;
        using pointer = const V *;
        const Sum *owner = nullptr;
        size_t a = 0, b = 0;
        mutable V cache{};
        mutable bool valid = false;
        I current() const {
            return std::min(a < count(owner->f) ? key(owner->f, a) : INT64_MAX,
                            b < count(owner->c) ? key(owner->c, b) : INT64_MAX);
        }
        template <class Curve> I at(const Curve &curve, size_t j, I k) const {
            if (j == count(curve))
                return value(curve, j - 1);
            if (key(curve, j) == k)
                return value(curve, j);
            const I l = key(curve, j - 1), r = key(curve, j), v = value(curve, j - 1);
            const I delta = value(curve, j) - v, span = r - l;
            if constexpr (Fast)
                if (delta % span == 0)
                    return v + (delta / span) * (k - l);
            return v + wide_int::mul_div_trunc(delta, k - l, span);
        }
        V operator*() const {
            if constexpr (Fast)
                if (valid)
                    return cache;
            I k = current();
            V out{k, at(owner->f, a, k) + at(owner->c, b, k)};
            if constexpr (Fast) {
                cache = out;
                valid = true;
            }
            return out;
        }
        pointer operator->() const {
            cache = operator*();
            return &cache;
        }
        Iterator &operator++() {
            I k = current();
            valid = false;
            if (a < count(owner->f) && key(owner->f, a) == k)
                ++a;
            if (b < count(owner->c) && key(owner->c, b) == k)
                ++b;
            return *this;
        }
        Iterator operator++(int) {
            auto old = *this;
            ++*this;
            return old;
        }
        bool operator==(const Iterator &o) const {
            return owner == o.owner && a == o.a && b == o.b;
        }
    };
    struct Vertices {
        const Sum *owner;
        Iterator begin() const { return {owner, 0, 0}; }
        Iterator end() const { return {owner, count(owner->f), count(owner->c)}; }
        size_t capacity_hint() const { return count(owner->f) + count(owner->c); }
        V front() const { return {0, owner->eval(0)}; }
        V back() const { return {owner->K(), owner->eval(owner->K())}; }
    } vs;
    Sum(const F &f, const C &c) : f(f), c(c), vs{this} {
        if constexpr (Fast) {
            first = {0, f.eval(0) + c.eval(0)};
            I end = K();
            last = {end, f.eval(end) + c.eval(end)};
        }
    }
    Sum(const Sum &o) : f(o.f), c(o.c), first(o.first), last(o.last), vs{this} {}
    Sum(Sum &&o) : f(o.f), c(o.c), first(o.first), last(o.last), vs{this} {}
    I K() const { return std::max(f.K(), c.K()); }
    I eval(I k) const {
        if constexpr (Fast) {
            if (k <= 0)
                return first.v;
            if (k >= K())
                return last.v;
        }
        return f.eval(k) + c.eval(k);
    }
};
} // namespace nmr_views
#endif
