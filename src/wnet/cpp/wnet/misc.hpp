#ifndef WNET_MISC_HPP
#define WNET_MISC_HPP

#include <string>
#include <vector>
#include <cstring> // for std::memcpy

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
namespace nb = nanobind;

inline std::string get_type_str(const nb::object &obj) {
    nb::handle type = obj.type();
    std::string name = nb::cast<std::string>(type.attr("__name__"));
    std::string module = nb::cast<std::string>(type.attr("__module__"));
    return module + "." + name;
}

template<typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> vector_to_numpy(const std::vector<T>& vec) {
    // Create a 1D NumPy array from the vector
    T* data = new T[vec.size()];
    std::memcpy(data, vec.data(), vec.size() * sizeof(T));
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(
        data,
        { vec.size() },
        owner
    );
}

template<typename T, size_t DIM>
nb::ndarray<nb::numpy, T, nb::shape<DIM, -1>> vector_of_arrays_to_numpy(const std::vector<std::array<T, DIM>>& vec) {
    // Create a 2D NumPy array from the vector of arrays, stored in dim-major
    // order so that shape (DIM, n_points) with default C strides is correct.
    T* data = new T[vec.size() * DIM];
    for (size_t i = 0; i < vec.size(); ++i) {
        for (size_t d = 0; d < DIM; ++d) {
            data[d * vec.size() + i] = vec[i][d];
        }
    }
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::ndarray<nb::numpy, T, nb::shape<DIM, -1>>(
        data,
        { DIM, vec.size() },
        owner
    );
}

template<typename T>
nb::ndarray<nb::numpy, T, nb::shape<-1>> span_to_numpy(const std::span<const T>& span)
{
    // Create a 1D NumPy array from the span
    T* data = new T[span.size()];
    std::memcpy(data, span.data(), span.size() * sizeof(T));
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::ndarray<nb::numpy, T, nb::shape<-1>>(
        data,
        { span.size() },
        owner
    );
}


#endif // WNET_MISC_HPP