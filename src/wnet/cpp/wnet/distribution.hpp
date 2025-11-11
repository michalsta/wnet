#ifndef WNET_DISTRIBUTION_HPP
#define WNET_DISTRIBUTION_HPP

#include <array>
#include <functional>
#include <vector>
#include <stdexcept>
#include <random>


#include "pylmcf/basics.hpp"
//#include "py_support.h"


#ifdef INCLUDE_NANOBIND_STUFF
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
namespace nb = nanobind;



template<typename T>
std::span<const T> numpy_to_span(const nb::ndarray<T, nb::shape<-1>>& array) {
    return std::span<const T>(static_cast<T*>(array.data()), array.shape(0));
}

class Distribution {
    const nb::ndarray<> py_positions;
    const nb::ndarray<LEMON_INT, nb::shape<-1>> py_intensities;
public:
    using Point_t = std::pair<const nb::ndarray<>*, size_t>;
    using distance_fun_t = nb::callable;
    const std::span<const LEMON_INT> intensities;

    Distribution(nb::ndarray<> positions, nb::ndarray<LEMON_INT, nb::shape<-1>> intensities)
        : py_positions(positions), py_intensities(intensities), intensities(numpy_to_span(intensities)) {
        if (positions.shape(1) != intensities.shape(0)) {
            throw std::invalid_argument("Positions and intensities must have the same size");
        }
    }

    size_t size() const {
        return intensities.size();
    }

    Point_t get_point(size_t idx) const {
        if (idx >= size()) {
            throw std::out_of_range("Index out of range");
        }
        return {&py_positions, idx};
    }

    const nb::ndarray<> get_positions() const {
        return py_positions;
    }

    const nb::ndarray<LEMON_INT, nb::shape<-1>> get_intensities() const {
        return py_intensities;
    }

    std::pair<std::vector<size_t>, std::vector<LEMON_INT>> closer_than(
        const Point_t point,
        const distance_fun_t wrapped_dist_fun,
        LEMON_INT max_dist
    ) const
    {
        std::vector<size_t> indices;
        std::vector<LEMON_INT> distances;

        nb::object distances_obj = (wrapped_dist_fun)(point, py_positions);
        nb::ndarray<LEMON_INT, nb::shape<-1>> distances_array = nb::cast<nb::ndarray<LEMON_INT, nb::shape<-1>>>(distances_obj);
        LEMON_INT* distances_ptr = static_cast<LEMON_INT*>(distances_array.data());
        // if (distances_info.ndim != 1) {
        //     throw std::invalid_argument("Only 1D arrays are supported");
        // }
        for (size_t ii = 0; ii < size(); ++ii) {
            if(distances_ptr[ii] <= max_dist) {
                indices.push_back(ii);
                distances.push_back(distances_ptr[ii]);
            }
        }
        return {indices, distances};
    }

    const nb::ndarray<>& py_get_positions() const {
        return py_positions;
    }

    const nb::ndarray<LEMON_INT, nb::shape<-1>>& py_get_intensities() const {
        return py_intensities;
    }
};

#endif // INCLUDE_NANOBIND_STUFF

template<size_t DIM, typename position_type_ = double, typename intensity_type_ = LEMON_INT>
class VectorDistribution {
    std::vector<std::array<position_type_, DIM>> positions;
    std::vector<intensity_type_> intensities_vector;
public:
    using intensity_type = intensity_type_;
    using position_type = position_type_;
    using Point_t = std::array<position_type, DIM>;
    using distance_fun_t = std::function<intensity_type(const Point_t&, const Point_t&)>;

    const std::span<const intensity_type> intensities;

    VectorDistribution(
        const std::vector<std::array<position_type, DIM>>& positions_,
        const std::vector<intensity_type>& intensities_
    ) : positions(positions_), intensities_vector(intensities_), intensities(intensities_vector) {
        if (positions.size() != intensities.size()) {
            throw std::invalid_argument("Positions and intensities must have the same size");
        }
    }

    VectorDistribution(
        std::vector<std::array<position_type, DIM>>&& positions_,
        std::vector<intensity_type>&& intensities_
    ) : positions(std::move(positions_)), intensities_vector(std::move(intensities_)), intensities(intensities_vector) {
        if (positions.size() != intensities.size()) {
            throw std::invalid_argument("Positions and intensities must have the same size");
        }
    }

    size_t size() const {
        return intensities.size();
    }

    const Point_t& get_point(size_t idx) const {
        return positions[idx];
    }

    const std::vector<std::array<position_type, DIM>>& get_positions() const {
        return positions;
    }

    const std::vector<intensity_type>& get_intensities() const {
        return intensities;
    }

    std::pair<std::vector<size_t>, std::vector<intensity_type>> closer_than(
        const Point_t& point,
        const distance_fun_t dist_fun,
        intensity_type max_dist
    ) const
    {
        std::vector<size_t> indices;
        std::vector<intensity_type> distances;

        for (size_t ii = 0; ii < size(); ++ii) {
            intensity_type dist = dist_fun(point, positions[ii]);
            if(dist <= max_dist) {
                indices.push_back(ii);
                distances.push_back(dist);
            }
        }
        return {indices, distances};
    }

    static VectorDistribution CreateRandom(size_t no_points,
                                           position_type position_range,
                                           intensity_type intensity_range,
                                           std::mt19937& rng) {
        std::uniform_real_distribution<position_type> pos_dist(0, position_range);
        std::uniform_int_distribution<intensity_type> int_dist(1, intensity_range);

        std::vector<std::array<position_type, DIM>> positions;
        std::vector<intensity_type> intensities;
        positions.reserve(no_points);
        intensities.reserve(no_points);

        for (size_t i = 0; i < no_points; ++i) {
            std::array<position_type, DIM> pos;
            for (size_t d = 0; d < DIM; ++d) {
                pos[d] = pos_dist(rng);
            }
            positions.push_back(pos);
            intensities.push_back(int_dist(rng));
        }
        return VectorDistribution(std::move(positions), std::move(intensities));
    }
};

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

#endif // WNET_DISTRIBUTION_HPP