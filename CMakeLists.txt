cmake_minimum_required(VERSION 3.15)

project(wnet LANGUAGES CXX)
find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind REQUIRED CONFIG)

# --- nanobind build mode ------------------------------------------------------
# Split mode (nanobind >= 3) leaves no nanobind library code in the extension and
# resolves a shared backend module at import time. Every extension sharing that
# backend has full mutual visibility of the others' registered types -- which is
# what lets wnetalign nb::cast a class registered inside wnet_cpp.so without the
# two being pinned to a common nanobind ABI version.
#
# Split mode requires CPython, and on a free-threaded interpreter it requires the
# 'abi3t' stable ABI of Python 3.15+. Fall back to the classic linked build
# everywhere else, and whenever WNET_NB_LINKED is set: sanitizer builds need that,
# because the published backend is an uninstrumented libstdc++ binary and the
# frontend/backend pair must agree on compiler, C++ runtime and debug mode.
option(WNET_NB_LINKED "Force a linked (NB_STATIC) nanobind build instead of split mode" OFF)

if (WNET_NB_LINKED)
    set(NB_MODE NB_STATIC)
    set(NB_MODE_DESC "linked (WNET_NB_LINKED=ON)")
    # A linked nanobind cannot target the 3.10/3.11 stable ABI -- that is what
    # split mode is for -- and nanobind turns the combination into a hard
    # configure error. WNET_NB_LINKED exists for local sanitizer and debug
    # builds, not for producing distributable wheels, so drop the abi3 request
    # rather than making the caller pass a second incantation.
    if (DEFINED SKBUILD_SABI_VERSION AND NOT SKBUILD_SABI_VERSION STREQUAL ""
        AND SKBUILD_SABI_VERSION VERSION_LESS "3.12")
        message(WARNING
            "WNET_NB_LINKED=ON overrides wheel.py-api=${SKBUILD_SABI_VERSION}: "
            "a linked build is specific to Python ${Python_VERSION}. Any wheel "
            "produced from this build is NOT abi3 and must not be published.")
        set(SKBUILD_SABI_VERSION "")
    endif()
elseif (NOT Python_INTERPRETER_ID STREQUAL "Python")
    set(NB_MODE NB_STATIC)
    set(NB_MODE_DESC "linked (${Python_INTERPRETER_ID} is not CPython)")
elseif (NB_ABI MATCHES "[0-9]t" AND Python_VERSION VERSION_LESS 3.15)
    set(NB_MODE NB_STATIC)
    set(NB_MODE_DESC "linked (free-threaded Python ${Python_VERSION} predates abi3t)")
else()
    set(NB_MODE BACKEND_MODULE nanobind_backend BACKEND_PYPI nanobind-backend)
    set(NB_MODE_DESC "split (backend module: nanobind_backend)")
endif()
message(STATUS "wnet: nanobind build mode: ${NB_MODE_DESC}")


# Resolved after find_package(Python) so the probe uses the interpreter CMake
# actually selected, not whatever bare `python` happens to be on PATH.
execute_process(
   COMMAND ${Python_EXECUTABLE} -m pylmcf --include
   OUTPUT_VARIABLE PYLMCF_INCLUDE_DIRS
   OUTPUT_STRIP_TRAILING_WHITESPACE
   RESULT_VARIABLE PYLMCF_INCLUDE_RESULT
)
if(NOT PYLMCF_INCLUDE_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to get include path from pylmcf (is it installed for ${Python_EXECUTABLE}?)")
endif()

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
    ${NB_MODE} NOMINSIZE
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
