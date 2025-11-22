#ifndef WNET_DISTRIBUTION_HPP
#define WNET_DISTRIBUTION_HPP

#include <array>
#include <functional>
#include <vector>
#include <stdexcept>
#include <random>


#ifdef INCLUDE_NANOBIND_STUFF
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include "misc.hpp"
namespace nb = nanobind;
#endif // INCLUDE_NANOBIND_STUFF

#include "pylmcf/basics.hpp"
#include "py_support.hpp"
#include "distances.hpp"

template<typename intensity_type_>
class Distribution;


template<size_t DIM, typename position_type_ = double, typename intensity_type_ = LEMON_INT>
class VectorDistribution {
    std::vector<std::array<position_type_, DIM>> positions;
    std::vector<intensity_type_> intensities_vector;
public:
    using intensity_type = intensity_type_;
    using position_type = position_type_;
    using Point_t = std::array<position_type, DIM>;
    // using distance_fun_t = std::function<intensity_type(const Point_t&, const Point_t&)>;

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

    #if defined(INCLUDE_NANOBIND_STUFF)
    VectorDistribution(const nb::ndarray<position_type_, nb::shape<DIM, -1>>& positions_arg, const nb::ndarray<intensity_type_, nb::shape<-1>>& intensities_arg) :
        positions(numpy_to_vector_of_arrays<position_type_ , DIM>(positions_arg)),
        intensities_vector(numpy_to_vector<intensity_type_>(intensities_arg)),
        intensities(intensities_vector)
    {
        if (positions.size() != intensities_vector.size()) {
            throw std::invalid_argument("Positions and intensities must have the same size");
        }
    }

    VectorDistribution(const nb::ndarray<position_type_, nb::shape<-1, -1>>& positions_arg, const nb::ndarray<intensity_type_, nb::shape<-1>>& intensities_arg) :
        positions(numpy_to_vector_of_arrays<position_type_ , DIM>(positions_arg)),
        intensities_vector(numpy_to_vector<intensity_type_>(intensities_arg)),
        intensities(intensities_vector)
    {
        if (positions.size() != intensities_vector.size()) {
            throw std::invalid_argument("Positions and intensities must have the same size");
        }
    }


    VectorDistribution(const Distribution<intensity_type_>& dist)
    : VectorDistribution(
        dist.get_positions(),
        dist.get_intensities()
    ) {};

    nb::ndarray<nb::numpy, position_type_, nb::shape<DIM, -1>> py_get_positions() const {
        return vector_of_arrays_to_numpy<position_type_, DIM>(positions);
    }
    nb::ndarray<nb::numpy, intensity_type_, nb::shape<-1>> py_get_intensities() const {
        return span_to_numpy<intensity_type_>(intensities);
    }

    #endif // INCLUDE_NANOBIND_STUFF



    size_t size() const {
        return intensities_vector.size();
    }

    const Point_t& get_point(size_t idx) const {
        return positions[idx];
    }

    const std::vector<std::array<position_type, DIM>>& get_positions() const {
        return positions;
    }

    const std::vector<intensity_type>& get_intensities() const {
        return intensities_vector;
    }

    /* std::pair<std::vector<size_t>, std::vector<intensity_type>> closer_than(
        const Point_t& point,
        const DistanceMetric dist_fun,
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
    }*/

    class CloserThanIterator {
        const VectorDistribution<DIM, position_type, intensity_type>& distribution;
        const Point_t& point;
        const DistanceMetric dist_fun;
        intensity_type max_dist;
        size_t current_index;
        intensity_type current_distance;

    public:
        CloserThanIterator(
            const VectorDistribution<DIM, position_type, intensity_type>& distribution_,
            const Point_t& point_,
            const DistanceMetric dist_fun_,
            intensity_type max_dist_
        ) : distribution(distribution_),
            point(point_),
            dist_fun(dist_fun_),
            max_dist(max_dist_),
            current_index(-1)
        {}
        inline bool advance() {
            current_index++;
            while (current_index < distribution.size()) {
                if(dist_fun == DistanceMetric::L1) {
                    current_distance = l1_distance<DIM, position_type>(point, distribution.get_point(current_index));
                } else if(dist_fun == DistanceMetric::L2) {
                    current_distance = l2_distance<DIM, position_type>(point, distribution.get_point(current_index));
                } else if(dist_fun == DistanceMetric::LINF) {
                    current_distance = linf_distance<DIM, position_type>(point, distribution.get_point(current_index));
                } else {
                    throw std::runtime_error("Unsupported distance metric.");
                }
                if (current_distance <= max_dist) {
                    return true;
                }
                ++current_index;
            }
            return false;
        }
        inline size_t get_index() const {
            return current_index;
        }
        inline intensity_type get_distance() const {
            return current_distance;
        }
    };

    CloserThanIterator closer_than_iter(
        const Point_t& point,
        const DistanceMetric dist_fun,
        intensity_type max_dist
    ) const {
        return CloserThanIterator(
            *this,
            point,
            dist_fun,
            max_dist
        );
    };

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


#ifdef INCLUDE_NANOBIND_STUFF
template<typename T>
std::span<const T> numpy_to_span(const nb::ndarray<T, nb::shape<-1>>& array) {
    return std::span<const T>(static_cast<T*>(array.data()), array.shape(0));
}

template<typename intensity_type_>
class Distribution {
    const nb::ndarray<nb::shape<-1, -1>> py_positions;
    const nb::ndarray<intensity_type_, nb::shape<-1>> py_intensities;
public:
    using intensity_type = intensity_type_;
    using Point_t = std::pair<const nb::ndarray<nb::shape<-1, -1>>*, size_t>;
    using distance_fun_t = nb::callable;
    const std::span<const intensity_type_> intensities;

    Distribution(nb::ndarray<nb::shape<-1, -1>> positions, nb::ndarray<intensity_type_, nb::shape<-1>> intensities)
        : py_positions(positions), py_intensities(intensities), intensities(numpy_to_span(intensities)) {
        if (positions.shape(1) != intensities.shape(0)) {
            throw std::invalid_argument("Positions and intensities must have the same size");
        }
    }

    size_t size() const {
        return intensities.size();
    }

    size_t dimension() const {
        return py_positions.shape(0);
    }

    Point_t get_point(size_t idx) const {
        if (idx >= size()) {
            throw std::out_of_range("Index out of range");
        }
        return {&py_positions, idx};
    }

    const nb::ndarray<nb::shape<-1, -1>> get_positions() const {
        return py_positions;
    }

    const nb::ndarray<intensity_type_, nb::shape<-1>> get_intensities() const {
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

    class CloserThanIterator {
        size_t current_index;
        nb::ndarray<LEMON_INT, nb::shape<-1>> distances_array;
        LEMON_INT* distances_ptr;
        LEMON_INT max_dist;
        size_t distribution_size;
    public:
        CloserThanIterator(
            const nb::ndarray<nb::shape<-1, -1>> py_positions_,
            const Point_t point_,
            const distance_fun_t& dist_fun_,
            LEMON_INT max_dist_
        ) : current_index(-1),
            max_dist(max_dist_)
        {
            nb::object distances_obj = (dist_fun_)(point_, py_positions_);
            distances_array = nb::cast<nb::ndarray<LEMON_INT, nb::shape<-1>>>(distances_obj);
            distances_ptr = static_cast<LEMON_INT*>(distances_array.data());
            distribution_size = distances_array.shape(0);
        }

        bool advance() {
            current_index++;
            while (current_index < distribution_size) {
                if (distances_ptr[current_index] <= max_dist) {
                    return true;
                }
                ++current_index;
            }
            return false;
        }
        size_t get_index() const {
            return current_index;
        }
        LEMON_INT get_distance() const {
            return distances_ptr[current_index];
        }
    };

    CloserThanIterator closer_than_iter(
        const Point_t point,
        const distance_fun_t& dist_fun,
        LEMON_INT max_dist
    ) const {
        return CloserThanIterator(
            py_positions,
            point,
            dist_fun,
            max_dist
        );
    };

    const nb::ndarray<nb::shape<-1, -1>>& py_get_positions() const {
        return py_positions;
    }

    const nb::ndarray<intensity_type_, nb::shape<-1>>& py_get_intensities() const {
        return py_intensities;
    }

    template<size_t DIM>
    VectorDistribution<DIM, double, intensity_type_> to_vector_distribution() const {
        return VectorDistribution<DIM, double, intensity_type_>(
            py_positions, py_intensities
        );
    }
};

#endif // INCLUDE_NANOBIND_STUFF


#endif // WNET_DISTRIBUTION_HPP