#ifndef WNET_DISTANCES_HPP
#define WNET_DISTANCES_HPP

#include <array>
#include <cmath>
#include <cstdlib>

// Runtime tag for the Python API.  The three policy structs below (L1Metric,
// L2Metric, LinfMetric) are the compile-time counterparts used as template
// parameters throughout the C++ internals.
enum class DistanceMetric {
    L1,
    L2,
    LINF
};

// ---------------------------------------------------------------------------
// Distance policy structs.
//
// Each struct exposes two [[gnu::always_inline]] static methods:
//   dist(p1, p2)   — the scalar distance value  (same semantics as the old
//                    l1_distance / l2_distance / linf_distance free functions)
//   grad_x(p1, p2) — gradient of dist w.r.t. p1 (used for position derivatives)
//
// Both methods are function templates deducing DIM and position_type from the
// std::array arguments, so no explicit template arguments are needed at call
// sites.  The structs carry a `metric` constexpr tag so generic code can
// recover the runtime enum value when needed.
// ---------------------------------------------------------------------------

struct L1Metric {
    static constexpr DistanceMetric metric = DistanceMetric::L1;

    template<size_t DIM, typename position_type = double>
    [[gnu::always_inline]] static double dist(
        const std::array<position_type, DIM>& p1,
        const std::array<position_type, DIM>& p2
    ) {
        if constexpr (DIM == 0) {
            return 0.0;
        } else if constexpr (DIM == 1) {
            return std::abs(p1[0] - p2[0]);
        } else {
            // Fold expression unrolls the sum at compile time.
            return [&]<size_t... Is>(std::index_sequence<Is...>) {
                return (std::abs(p1[Is] - p2[Is]) + ...);
            }(std::make_index_sequence<DIM>{});
        }
    }

    // Subgradient w.r.t. p1: sign(p1[d] - p2[d]) per dimension.
    // Returns 0 at ties (non-differentiable point).
    template<size_t DIM, typename position_type = double>
    [[gnu::always_inline]] static std::array<double, DIM> grad_x(
        const std::array<position_type, DIM>& p1,
        const std::array<position_type, DIM>& p2
    ) {
        if constexpr (DIM == 0) {
            return {};
        } else {
            return [&]<size_t... Is>(std::index_sequence<Is...>) -> std::array<double, DIM> {
                return {((p1[Is] > p2[Is]) ? 1.0 : (p1[Is] < p2[Is] ? -1.0 : 0.0))...};
            }(std::make_index_sequence<DIM>{});
        }
    }
};

struct L2Metric {
    static constexpr DistanceMetric metric = DistanceMetric::L2;

    template<size_t DIM, typename position_type = double>
    [[gnu::always_inline]] static double dist(
        const std::array<position_type, DIM>& p1,
        const std::array<position_type, DIM>& p2
    ) {
        if constexpr (DIM == 0) {
            return 0.0;
        } else if constexpr (DIM == 1) {
            return std::abs(p1[0] - p2[0]);
        } else {
            return std::sqrt([&]<size_t... Is>(std::index_sequence<Is...>) {
                return (((p1[Is] - p2[Is]) * (p1[Is] - p2[Is])) + ...);
            }(std::make_index_sequence<DIM>{}));
        }
    }

    // Gradient w.r.t. p1: (p1 - p2) / ||p1 - p2||_2.
    // Returns zero vector at coincident points (undefined gradient).
    template<size_t DIM, typename position_type = double>
    [[gnu::always_inline]] static std::array<double, DIM> grad_x(
        const std::array<position_type, DIM>& p1,
        const std::array<position_type, DIM>& p2
    ) {
        if constexpr (DIM == 0) {
            return {};
        } else if constexpr (DIM == 1) {
            const double d = static_cast<double>(p1[0]) - static_cast<double>(p2[0]);
            const double n = std::abs(d);
            return {n > 1e-300 ? d / n : 0.0};
        } else {
            const double norm_sq = [&]<size_t... Is>(std::index_sequence<Is...>) {
                return (((static_cast<double>(p1[Is]) - static_cast<double>(p2[Is])) *
                         (static_cast<double>(p1[Is]) - static_cast<double>(p2[Is]))) + ...);
            }(std::make_index_sequence<DIM>{});
            if (norm_sq < 1e-300) return {};
            const double inv = 1.0 / std::sqrt(norm_sq);
            return [&]<size_t... Is>(std::index_sequence<Is...>) -> std::array<double, DIM> {
                return {((static_cast<double>(p1[Is]) - static_cast<double>(p2[Is])) * inv)...};
            }(std::make_index_sequence<DIM>{});
        }
    }
};

struct LinfMetric {
    static constexpr DistanceMetric metric = DistanceMetric::LINF;

    template<size_t DIM, typename position_type = double>
    [[gnu::always_inline]] static double dist(
        const std::array<position_type, DIM>& p1,
        const std::array<position_type, DIM>& p2
    ) {
        if constexpr (DIM == 0) {
            return 0.0;
        } else if constexpr (DIM == 1) {
            return std::abs(p1[0] - p2[0]);
        } else {
            return [&]<size_t... Is>(std::index_sequence<Is...>) {
                return std::max({std::abs(p1[Is] - p2[Is])...});
            }(std::make_index_sequence<DIM>{});
        }
    }

    // Subgradient w.r.t. p1: 1 along the argmax-abs coordinate.
    // Returns zero vector if all components are equal (p1 == p2).
    template<size_t DIM, typename position_type = double>
    [[gnu::always_inline]] static std::array<double, DIM> grad_x(
        const std::array<position_type, DIM>& p1,
        const std::array<position_type, DIM>& p2
    ) {
        std::array<double, DIM> g{};
        if constexpr (DIM == 0) return g;
        size_t k = 0;
        double mx = 0.0;
        for (size_t d = 0; d < DIM; ++d) {
            const double v = std::abs(static_cast<double>(p1[d]) - static_cast<double>(p2[d]));
            if (v > mx) { mx = v; k = d; }
        }
        if (mx > 0.0) g[k] = (p1[k] > p2[k]) ? 1.0 : -1.0;
        return g;
    }
};

// ---------------------------------------------------------------------------
// Legacy free functions — thin wrappers around the policy structs so any
// direct call site outside of distribution.hpp continues to compile unchanged.
// ---------------------------------------------------------------------------

template<size_t DIM, typename position_type = double>
[[gnu::always_inline]] inline double l1_distance(
    const std::array<position_type, DIM>& p1,
    const std::array<position_type, DIM>& p2
) { return L1Metric::dist(p1, p2); }

template<size_t DIM, typename position_type = double>
[[gnu::always_inline]] inline double l2_distance(
    const std::array<position_type, DIM>& p1,
    const std::array<position_type, DIM>& p2
) { return L2Metric::dist(p1, p2); }

template<size_t DIM, typename position_type = double>
[[gnu::always_inline]] inline double linf_distance(
    const std::array<position_type, DIM>& p1,
    const std::array<position_type, DIM>& p2
) { return LinfMetric::dist(p1, p2); }

#endif // WNET_DISTANCES_HPP