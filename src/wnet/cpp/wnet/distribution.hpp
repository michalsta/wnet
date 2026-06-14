#ifndef WNET_DISTRIBUTION_HPP
#define WNET_DISTRIBUTION_HPP

#include <array>
#include <functional>
#include <limits>
#include <numeric>
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
    std::vector<size_t> sorted_indices; // indices of the peaks sorted by first dimension of the position, used for faster distance calculations
public:
    using intensity_type = intensity_type_;
    using position_type = position_type_;
    using Point_t = std::array<position_type, DIM>;
    // using distance_fun_t = std::function<intensity_type(const Point_t&, const Point_t&)>;

    // View over the owning intensities vector. Computed on access rather than
    // cached as a member so the class retains default value semantics
    // (copy/move/assign); a cached span member would alias the source object.
    std::span<const intensity_type> intensities() const { return intensities_vector; }

    VectorDistribution(
        const std::vector<std::array<position_type, DIM>>& positions_,
        const std::vector<intensity_type>& intensities_
    ) : positions(positions_), intensities_vector(intensities_) {
        init();
    }

    VectorDistribution(
        std::vector<std::array<position_type, DIM>>&& positions_,
        std::vector<intensity_type>&& intensities_
    ) : positions(std::move(positions_)), intensities_vector(std::move(intensities_)) {
        init();
    }

    #if defined(INCLUDE_NANOBIND_STUFF)
    VectorDistribution(const nb::ndarray<position_type_, nb::shape<DIM, -1>>& positions_arg, const nb::ndarray<intensity_type_, nb::shape<-1>>& intensities_arg) :
        positions(numpy_to_vector_of_arrays<position_type_ , DIM>(positions_arg)),
        intensities_vector(numpy_to_vector<intensity_type_>(intensities_arg))
    {
        init();
    }

    VectorDistribution(const nb::ndarray<position_type_, nb::shape<-1, -1>>& positions_arg, const nb::ndarray<intensity_type_, nb::shape<-1>>& intensities_arg) :
        positions(numpy_to_vector_of_arrays<position_type_ , DIM>(positions_arg)),
        intensities_vector(numpy_to_vector<intensity_type_>(intensities_arg))
    {
        init();
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
        return span_to_numpy<intensity_type_>(intensities());
    }

    #endif // INCLUDE_NANOBIND_STUFF

    void init()
    {
        if (positions.size() != intensities_vector.size()) {
            throw std::invalid_argument("Positions and intensities must have the same size");
        }
        sorted_indices.resize(positions.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                  [this](size_t i1, size_t i2) {
                      return positions[i1][0] < positions[i2][0];
                  });
    }


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

    double sum_intensities() const {
        return std::accumulate(intensities_vector.begin(), intensities_vector.end(), 0.0);
    }

    // Returns a copy with intensities multiplied by `factor` and positions
    // unchanged (mirrors the Python Distribution.scaled()).  NOTE: for an
    // integer intensity_type the product truncates toward zero.
    VectorDistribution scaled(double factor) const {
        std::vector<std::array<position_type, DIM>> pos = positions;
        std::vector<intensity_type> scaled_int;
        scaled_int.reserve(intensities_vector.size());
        for (const auto& v : intensities_vector)
            scaled_int.push_back(static_cast<intensity_type>(static_cast<double>(v) * factor));
        return VectorDistribution(std::move(pos), std::move(scaled_int));
    }

    // Returns a copy with intensities scaled to sum to 1 (mirrors the Python
    // Distribution.normalized()).  NOTE: for an integer intensity_type the
    // normalized values (< 1) truncate toward zero, so this is only meaningful
    // for the double-intensity instantiation.
    VectorDistribution normalized() const {
        const double total = sum_intensities();
        if (total == 0.0)
            throw std::runtime_error("Cannot normalize a distribution with zero total intensity.");
        std::vector<std::array<position_type, DIM>> pos = positions;
        std::vector<intensity_type> norm;
        norm.reserve(intensities_vector.size());
        for (const auto& v : intensities_vector)
            norm.push_back(static_cast<intensity_type>(static_cast<double>(v) / total));
        return VectorDistribution(std::move(pos), std::move(norm));
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

    template<typename DistMetric>
    class CloserThanIteratorPoint {
        const VectorDistribution<DIM, position_type, intensity_type>& distribution;
        const Point_t& point;
        intensity_type max_dist;
        size_t current_index;
        intensity_type current_distance;

    public:
        CloserThanIteratorPoint(
            const VectorDistribution<DIM, position_type, intensity_type>& distribution_,
            const Point_t& point_,
            intensity_type max_dist_
        ) : distribution(distribution_),
            point(point_),
            max_dist(max_dist_),
            current_index(std::numeric_limits<decltype(current_index)>::max())
        {}
        inline bool advance() {
            current_index++;
            while (current_index < distribution.size()) [[likely]] {
                current_distance = DistMetric::dist(point, distribution.get_point(current_index));
                if (current_distance <= max_dist) [[likely]] {
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

    template<typename DistMetric>
    CloserThanIteratorPoint<DistMetric> closer_than_iter_point(
        const Point_t& point,
        intensity_type max_dist
    ) const {
        return CloserThanIteratorPoint<DistMetric>(
            *this,
            point,
            max_dist
        );
    };

    template<typename DistMetric>
    class CloserThanIter {
        const VectorDistribution<DIM, position_type, intensity_type>& distribution;
        const VectorDistribution<DIM, position_type, intensity_type>& other_distribution;
        intensity_type max_dist;
        size_t current_index;
        size_t other_current_index;
        size_t last_window_start_index;
        intensity_type current_distance;
    public:
        CloserThanIter(
            const VectorDistribution<DIM, position_type, intensity_type>& distribution_,
            const VectorDistribution<DIM, position_type, intensity_type>& other_distribution_,
            intensity_type max_dist_
        ) : distribution(distribution_),
            other_distribution(other_distribution_),
            max_dist(max_dist_),
            current_index(0),
            other_current_index(std::numeric_limits<size_t>::max()),
            last_window_start_index(0)
        {}
        inline bool advance() {
            while (true) {
                other_current_index++;
                if(other_current_index >= other_distribution.size())
                {
                    current_index++;
                    if(current_index >= distribution.size())
                        return false;
                    other_current_index = last_window_start_index - 1;
                    continue;
                }
                position_type other_pos0 = other_distribution.get_point(other_distribution.sorted_indices[other_current_index])[0];
                position_type this_pos0 = distribution.get_point(distribution.sorted_indices[current_index])[0];
                if(other_pos0 < this_pos0 - static_cast<position_type>(max_dist))
                {
                    last_window_start_index = other_current_index + 1;
                    continue;
                }
                if(other_pos0 > this_pos0 + static_cast<position_type>(max_dist))
                {
                    current_index++;
                    if(current_index >= distribution.size())
                        return false;
                    other_current_index = last_window_start_index - 1;
                    continue;
                }
                current_distance = DistMetric::dist(
                    distribution.get_point(distribution.sorted_indices[current_index]),
                    other_distribution.get_point(other_distribution.sorted_indices[other_current_index]));
                if (current_distance <= max_dist) [[likely]] {
                    return true;
                }
            }
        }
        std::pair<size_t, size_t> get_indices() const {
            return {distribution.sorted_indices[current_index], other_distribution.sorted_indices[other_current_index]};
        }
        intensity_type get_distance() const {
            return current_distance;
        }
    };

    template<typename DistMetric>
    CloserThanIter<DistMetric> closer_than_iter(
        const VectorDistribution<DIM, position_type, intensity_type>& other_distribution,
        intensity_type max_dist
    ) const {
        return CloserThanIter<DistMetric>(*this, other_distribution, max_dist);
    };

    VectorDistribution n_highest(size_t n) const {
        if (n >= size()) return *this;
        std::vector<size_t> idx(size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + n, idx.end(),
            [this](size_t a, size_t b) {
                return intensities_vector[a] > intensities_vector[b];
            });
        idx.resize(n);
        std::vector<std::array<position_type, DIM>> new_positions;
        std::vector<intensity_type> new_intensities;
        new_positions.reserve(n);
        new_intensities.reserve(n);
        for (size_t i : idx) {
            new_positions.push_back(positions[i]);
            new_intensities.push_back(intensities_vector[i]);
        }
        return VectorDistribution(std::move(new_positions), std::move(new_intensities));
    }

    VectorDistribution p_trim(double p) const {
        if (p <= 0.0) return VectorDistribution(
            std::vector<std::array<position_type, DIM>>{},
            std::vector<intensity_type>{});
        if (p >= 1.0) return *this;
        std::vector<size_t> idx(size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
            [this](size_t a, size_t b) {
                return intensities_vector[a] > intensities_vector[b];
            });
        double total = 0.0;
        for (auto& v : intensities_vector) total += static_cast<double>(v);
        double threshold = p * total;
        std::vector<std::array<position_type, DIM>> new_positions;
        std::vector<intensity_type> new_intensities;
        double cumsum = 0.0;
        for (size_t i : idx) {
            cumsum += static_cast<double>(intensities_vector[i]);
            new_positions.push_back(positions[i]);
            new_intensities.push_back(intensities_vector[i]);
            if (cumsum >= threshold) break;
        }
        return VectorDistribution(std::move(new_positions), std::move(new_intensities));
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

    class CloserThanIteratorPoint {
        size_t current_index;
        nb::ndarray<LEMON_INT, nb::shape<-1>> distances_array;
        LEMON_INT* distances_ptr;
        LEMON_INT max_dist;
        size_t distribution_size;
    public:
        CloserThanIteratorPoint(
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

    CloserThanIteratorPoint closer_than_iter_point(
        const Point_t point,
        const distance_fun_t& dist_fun,
        LEMON_INT max_dist
    ) const {
        return CloserThanIteratorPoint(
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
