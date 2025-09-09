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

#endif // WNET_MISC_HPP