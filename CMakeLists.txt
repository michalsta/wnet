cmake_minimum_required(VERSION 3.15)

execute_process(
    COMMAND python -m pylmcf --include
    OUTPUT_VARIABLE PYLMCF_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)


project(wnet LANGUAGES CXX)
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 20)

# Max VectorDistribution dimension to instantiate. Defaults to 20 (and stays 20
# for CI/wheel builds); reinstall.sh lowers it for faster local builds.
if(NOT DEFINED WNET_MAX_DIM)
    set(WNET_MAX_DIM 20)
endif()

# Each dimension's heavy template instantiation lives in its own TU
# (build_stubs/dim_register_<N>.cpp) so the build parallelizes across cores. The
# stubs are tiny `#define WNET_DIM <N>` + `#include` shims; the real binding
# logic lives in register_dim.inc. Only the 1..WNET_MAX_DIM stubs are compiled,
# matching the WNET_MAX_DIM dispatch in wnet.cpp.
set(WNET_SOURCES
    src/wnet/cpp/wnet/wnet.cpp
    src/wnet/cpp/wnet/bind_network_ii.cpp
    src/wnet/cpp/wnet/bind_network_if.cpp)
foreach(d RANGE 1 ${WNET_MAX_DIM})
    list(APPEND WNET_SOURCES src/wnet/cpp/wnet/build_stubs/dim_register_${d}.cpp)
endforeach()

nanobind_add_module(wnet_cpp
    NB_STATIC NOMINSIZE
    ${WNET_SOURCES})

target_include_directories(wnet_cpp PRIVATE ${PYLMCF_INCLUDE_DIRS})
target_compile_definitions(wnet_cpp PRIVATE INCLUDE_NANOBIND_STUFF)

# Pass WNET_MAX_DIM unconditionally so wnet.cpp's #if dispatch matches the set of
# dim_register_<N>.cpp files CMake actually compiled.
target_compile_definitions(wnet_cpp PRIVATE WNET_MAX_DIM=${WNET_MAX_DIM})

# Precompile the shared heavy-header umbrella once and reuse it across every TU.
# register_dim.hpp pulls in nanobind plus the expensive C++ headers
# (decompositable_graph.hpp ~131KB, distribution.hpp, scaling.hpp, ...); without
# a PCH each of the 20+ per-dimension TUs re-parses all of them, which is the
# bulk of the redundant compile work. The PCH is built first (before any TU) and
# force-included into each source. WNET_DIM is *not* referenced by the header
# itself (only by register_dim.inc, after the force-include), so precompiling is
# safe and dimension-independent.
target_precompile_headers(wnet_cpp PRIVATE src/wnet/cpp/wnet/register_dim.hpp)

# Add -Wall -Wextra for Unix-style compilers (GCC/Clang)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(wnet_cpp PRIVATE -Wall -Wextra)
endif()



if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(wnet_cpp PRIVATE DEBUG_MODE)
    target_compile_options(wnet_cpp PRIVATE -g -Og -DWNET_DO_ASSERTS -D_GLIBCXX_ASSERTIONS -D_GLIBCXX_SANITIZE_VECTOR)
endif()


install(TARGETS wnet_cpp LIBRARY DESTINATION wnet)
install(DIRECTORY src/wnet/cpp/wnet DESTINATION wnet/cpp
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")

# Uncomment the following line to enable detailed debug prints in the C++ code
# target_compile_definitions(wnet_cpp PRIVATE DO_TONS_OF_PRINTS)
