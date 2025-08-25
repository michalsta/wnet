#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>




NB_MODULE(wnetalign_cpp, m) {
    m.doc() = "WNet C++ imlementation module";

    // Define a simple function to demonstrate the module
    m.def("wnet_cpp_hello", []() {
        std::cout << "Hello from WNet (C++)!" << std::endl;
    }, "A simple hello world function for the WNet (C++) extension");
}