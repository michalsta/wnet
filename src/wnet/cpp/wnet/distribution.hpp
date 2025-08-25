#ifndef WNET_DISTRIBUTION_HPP
#define WNET_DISTRIBUTION_HPP

#include <array>

#include "pylmcf/basics.hpp"
//#include "py_support.h"

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
        const nb::callable* wrapped_dist_fun,
        LEMON_INT max_dist
    ) const
    {
        std::vector<size_t> indices;
        std::vector<LEMON_INT> distances;

        nb::object distances_obj = (*wrapped_dist_fun)(point, py_positions);
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

#endif // WNET_DISTRIBUTION_HPP