#ifndef WNET_MONGE_COST_HPP
#define WNET_MONGE_COST_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
#include "wide_int.hpp"

namespace monge_cost {

// All event coordinates are integers, exactly representable as doubles. The
// even distance cost is convex by construction: linear/quadratic for p=1/2,
// otherwise a maximum of integer affine functions. Thus its translated cost
// matrix is Monge, including ties, zero supplies and mirrored derivative runs.
// Floating arithmetic constructs the model, never rounds individual pairs.
class GridPower {
    using I = int64_t;
    using W = wide_int::wide_t;
    struct Line { I slope, intercept, start; };
    std::vector<Line> hull_;
    long double denominator_ = 1;
    I extent_ = 0;
    double p_ = 1;
    long double error_ = 0;

    static I checked(long double value) {
        if (!std::isfinite(value) || value < std::numeric_limits<I>::min()
            || value >= static_cast<long double>(std::numeric_limits<I>::max()))
            throw std::overflow_error("Monge cost exceeds integer arithmetic budget");
        return static_cast<I>(value);
    }
    static W evaluate(const Line& line, I q) {
        return W(line.slope) * q + W(line.intercept);
    }
    W value(I q) const {
        if (p_ == 1) return W(q);
        if (p_ == 2) return W(q) * q;
        auto it = std::upper_bound(hull_.begin(), hull_.end(), q,
            [](I x, const Line& line) { return x < line.start; });
        return evaluate(*std::prev(it), q);
    }

public:
    void reset(std::span<const double> positions, I scale, double p,
               std::vector<double>& grid_positions) {
        if (scale < 1 || !std::isfinite(p) || p < 1)
            throw std::invalid_argument("Invalid Monge grid parameters");
        p_ = p; error_ = 0;
        denominator_ = p == 1 ? static_cast<long double>(scale)
                              : std::sqrt(static_cast<long double>(scale));
        grid_positions.clear(); grid_positions.reserve(positions.size());
        for (double x : positions) {
            const long double q = std::round(static_cast<long double>(x) * denominator_);
            if (!std::isfinite(q) || q < 0 || q > 4503599627370496.0L)
                throw std::overflow_error("Monge grid coordinate exceeds exact-double range");
            grid_positions.push_back(static_cast<double>(q));
        }
        extent_ = grid_positions.empty() ? 0 : static_cast<I>(grid_positions.back());
        hull_.clear(); hull_.push_back({0, 0, 0});
        // Work in real distance units when constructing the hull. Computing
        // q^p and M^p separately can overflow even when (q/M)^p is finite.
        auto real_power = [&](long double q) { return scale * std::pow(q / denominator_, p); };
        auto derivative = [&](long double q) {
            return p * (static_cast<long double>(scale) / denominator_)
                * std::pow(q / denominator_, p - 1);
        };
        auto stationary = [&](I slope) {
            return denominator_ * std::pow((static_cast<long double>(slope) * denominator_)
                / (p * static_cast<long double>(scale)), 1.0L / (p - 1));
        };
        if (p != 1 && p != 2 && extent_ > 0) {
            // Curvature-adapted knots, capped to keep setup memory bounded.
            // The hull, rather than independent rounded samples, defines the
            // model. Its measured uniform cost error is exposed below.
            const size_t count = static_cast<size_t>(std::min(32768.0L,
                std::max(32.0L, std::ceil(4 * std::sqrt(p * static_cast<long double>(extent_))))));
            std::vector<Line> lines;
            lines.reserve(count + 1); lines.push_back({0, 0, 0});
            for (size_t k = 1; k <= count; ++k) {
                const long double q = extent_ * std::pow(static_cast<long double>(k) / count, 2.0L / p);
                const I slope = checked(std::round(derivative(q)));
                if (!slope) continue;
                // Conjugate of A*q^p gives the tangent with this integer
                // slope. Round its intercept down; evaluate in wide integers.
                const long double end_slope = derivative(extent_);
                const long double tangent_q = slope >= end_slope ? extent_
                    : stationary(slope);
                const long double conjugate = slope >= end_slope
                    ? slope * tangent_q - real_power(tangent_q)
                    : (p - 1) / p * slope * tangent_q;
                lines.push_back({slope, checked(-std::ceil(conjugate)), 0});
            }
            std::sort(lines.begin(), lines.end(), [](const Line& a, const Line& b) {
                return a.slope < b.slope || (a.slope == b.slope && a.intercept < b.intercept);
            });
            hull_.clear();
            for (auto line : lines) {
                if (!hull_.empty() && line.slope == hull_.back().slope) hull_.pop_back();
                bool useful = true;
                while (!hull_.empty()) {
                    const auto& prev = hull_.back();
                    if (!(evaluate(prev, extent_) < evaluate(line, extent_))) {
                        useful = false; break;
                    }
                    I lo = 0, hi = extent_;
                    while (lo < hi) {
                        const I mid = lo + (hi - lo) / 2;
                        if (!(evaluate(line, mid) < evaluate(prev, mid))) hi = mid;
                        else lo = mid + 1;
                    }
                    line.start = lo;
                    if (line.start > prev.start) break;
                    hull_.pop_back();
                }
                if (useful) {
                    if (hull_.empty()) line.start = 0;
                    hull_.push_back(line);
                }
            }
            // f(q)-line(q) is convex on each hull interval: its maximum
            // occurs at an endpoint, its minimum at f'(q)=line.slope.
            // Check these candidates to bound the model error on every
            // integer distance, without enumerating all pair distances.
            for (size_t k = 0; k < hull_.size(); ++k) {
                const I first = hull_[k].start;
                const I last = k + 1 < hull_.size() ? hull_[k + 1].start - 1 : extent_;
                auto check = [&](I q) {
                    const long double exact = real_power(q);
                    const long double approx = static_cast<long double>(static_cast<I>(evaluate(hull_[k], q)));
                    error_ = std::max(error_, std::abs(exact - approx));
                };
                check(first); check(last);
                const long double critical = hull_[k].slope >= derivative(extent_)
                    ? extent_ : stationary(hull_[k].slope);
                for (long double q : {std::floor(critical), std::ceil(critical)})
                    if (q >= first && q <= last) check(static_cast<I>(q));
            }
        }
        // Distance changes by at most one grid step when both endpoints round.
        const long double span = (static_cast<long double>(extent_) + 1) / denominator_;
        error_ /= scale;
        error_ += p * std::pow(span, p - 1) / denominator_;
        error_ += 128 * std::numeric_limits<double>::epsilon()
            * std::max(1.0L, std::pow(span, p));
        // Quadratic costs on distant, unprofitable pairs may exceed int64.
        // Keep their exact convex definition in wide arithmetic; the solver
        // evaluates only its model-derived profitable window.
        if (p != 2) cost(0, static_cast<double>(extent_));
    }

    W wide_cost(double a, double b) const {
        const I q = static_cast<I>(std::abs(a - b));
        if (q > extent_) throw std::out_of_range("Pair outside Monge grid domain");
        return value(q);
    }
    I cost(double a, double b) const {
        const W c = wide_cost(a, b);
        if (c < W(0) || W(std::numeric_limits<I>::max()) < c)
            throw std::overflow_error("Monge pair cost exceeds solver cost type");
        return static_cast<I>(c);
    }
    double radius(I tau) const {
        // Conservative integer window for this model, not an inverse of the
        // unquantized power. This also covers flat zero-cost hull intervals.
        I lo = 0, hi = extent_;
        if (value(hi) < W(tau)) return static_cast<double>(hi + 1);
        while (lo < hi) {
            const I mid = lo + (hi - lo) / 2;
            if (value(mid) < W(tau)) lo = mid + 1;
            else hi = mid;
        }
        return static_cast<double>(lo);
    }
    double error() const { return static_cast<double>(error_); }
};
} // namespace monge_cost
#endif
