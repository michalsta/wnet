#ifndef WNET_DISTANCES_HPP
#define WNET_DISTANCES_HPP

#include <array>
#include <cmath>
#include <cstdlib>

template<size_t DIM, typename position_type = double>
inline double l1_distance(
    const std::array<position_type, DIM>& p1,
    const std::array<position_type, DIM>& p2
) {
    if constexpr (DIM == 0) {
        return 0.0;
    } else if constexpr (DIM == 1) {
        return std::abs(p1[0] - p2[0]);
    } else {
        // Use fold expression to unroll the loop at compile time
        return [&]<size_t... Is>(std::index_sequence<Is...>) {
            return (std::abs(p1[Is] - p2[Is]) + ...);
        }(std::make_index_sequence<DIM>{});
    }
}

template<size_t DIM, typename position_type = double>
inline double l2_distance(
    const std::array<position_type, DIM>& p1,
    const std::array<position_type, DIM>& p2
) {
    if constexpr (DIM == 0) {
        return 0.0;
    } else if constexpr (DIM == 1) {
        return std::abs(p1[0] - p2[0]);
    } else {
        // Use fold expression to unroll the loop at compile time
        return std::sqrt([&]<size_t... Is>(std::index_sequence<Is...>) {
            return (((p1[Is] - p2[Is]) * (p1[Is] - p2[Is])) + ...);
        }(std::make_index_sequence<DIM>{}));
    }
}

#endif // WNET_DISTANCES_HPP