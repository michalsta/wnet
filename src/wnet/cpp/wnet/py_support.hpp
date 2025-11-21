#ifndef PY_SUPPORT_HPP
#define PY_SUPPORT_HPP

#include <vector>
#include <array>
#include <stdexcept>


#ifdef INCLUDE_NANOBIND_STUFF
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;

template<typename T>
std::vector<T> numpy_to_vector(const nb::ndarray<T, nb::shape<-1>>& array) {
    std::vector<T> result;
    result.reserve(array.shape(0));
    T* data_ptr = static_cast<T*>(array.data());
    for (size_t ii = 0; ii < array.shape(0); ++ii) {
        result.push_back(data_ptr[ii]);
    }
    return result;
}

template<typename T, size_t DIM>
std::vector<std::array<T, DIM>> numpy_to_vector_of_arrays(const nb::ndarray<T, nb::shape<DIM, -1>>& array) {
    if( array.shape(0) != DIM) {
        throw std::invalid_argument("Array first dimension does not match template parameter DIM");
    }
    std::vector<std::array<T, DIM>> result;
    result.reserve(array.shape(1));
    for (size_t ii = 0; ii < array.shape(1); ++ii) {
        result.emplace_back();
        for (size_t d = 0; d < DIM; ++d) {
            result.back()[d] = array(d, ii);
        }
    }
    return result;
}

template<typename T , size_t DIM>
std::vector<std::array<T, DIM>> numpy_to_vector_of_arrays(const nb::ndarray<T, nb::shape<-1, -1>>& array) {
    if( array.shape(0) != DIM) {
        throw std::invalid_argument("Array first dimension does not match template parameter DIM");
    }
    std::vector<std::array<T, DIM>> result;
    result.reserve(array.shape(1));
    for (size_t ii = 0; ii < array.shape(1); ++ii) {
        result.emplace_back();
        for (size_t d = 0; d < DIM; ++d) {
            result.back()[d] = array(d, ii);
        }
    }
    return result;
}



#endif // INCLUDE_NANOBIND_STUFF
#endif // PY_SUPPORT_HPP