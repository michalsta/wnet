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

# Which platforms have no backend to reach? Ask _wnet_metadata -- the very
# predicates the metadata provider uses to decide whether to declare
# nanobind-backend -- rather than reimplementing the tests here, so the build
# mode and the declared dependency can never disagree about the platform. A
# failure is fatal on purpose: guessing would mean silently building a mode that
# cannot be installed, and the module ships in the sdist right beside this file,
# so it being unimportable means the build is already broken.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import sys;sys.path.insert(0,sys.argv[1]);import _wnet_metadata as m;print(int(m.is_musl()),int(m.is_win32()))"
            "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE WNET_PLATFORM_RESULT
    OUTPUT_VARIABLE WNET_PLATFORM_PROBE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE WNET_PLATFORM_ERROR)
if (NOT WNET_PLATFORM_RESULT EQUAL 0)
    message(FATAL_ERROR
        "wnet: could not determine the target platform via _wnet_metadata "
        "(is_musl / is_win32): ${WNET_PLATFORM_ERROR}")
endif()
separate_arguments(WNET_PLATFORM_PROBE)
list(GET WNET_PLATFORM_PROBE 0 WNET_IS_MUSL)
list(GET WNET_PLATFORM_PROBE 1 WNET_IS_WIN32)

if (WNET_NB_LINKED)
    set(NB_MODE NB_STATIC)
    set(NB_LINKED TRUE)
    set(NB_MODE_DESC "linked (WNET_NB_LINKED=ON)")
elseif (NOT Python_INTERPRETER_ID STREQUAL "Python")
    set(NB_MODE NB_STATIC)
    set(NB_LINKED TRUE)
    set(NB_MODE_DESC "linked (${Python_INTERPRETER_ID} is not CPython)")
elseif (WNET_IS_MUSL)
    # nanobind-backend ships manylinux, macOS and Windows wheels only -- no
    # musllinux wheel, and no sdist to build one from -- because auditwheel
    # bundles a private libstdc++ while split mode needs the C++ runtime shared
    # for exceptions to propagate. So split mode is unsatisfiable on musl by any
    # route, and this branch must be tested before the free-threaded ones: a
    # free-threaded musl interpreter has no backend to reach either.
    set(NB_MODE NB_STATIC)
    set(NB_LINKED TRUE)
    set(NB_MODE_DESC "linked (musl libc: nanobind-backend publishes no musllinux wheel)")
elseif (WNET_IS_WIN32)
    # Same hole, different platform: nanobind-backend ships win_amd64 and
    # win_arm64 and nothing 32-bit, at any Python version. Split mode here
    # produces a wheel that builds cleanly and then cannot be installed on the
    # platform it was built for, because it depends on a distribution that does
    # not exist there -- pylmcf 1.2.0 shipped exactly that. 32-bit Windows gets
    # no wheel from build_wheels.yml (CIBW_SKIP), so this branch is what makes
    # the sdist install path work there.
    set(NB_MODE NB_STATIC)
    set(NB_LINKED TRUE)
    set(NB_MODE_DESC "linked (32-bit Windows: nanobind-backend publishes no win32 wheel)")
elseif (NB_ABI MATCHES "[0-9]t" AND Python_VERSION VERSION_LESS 3.15)
    set(NB_MODE NB_STATIC)
    set(NB_LINKED TRUE)
    set(NB_MODE_DESC "linked (free-threaded Python ${Python_VERSION} predates abi3t)")
elseif (NB_ABI MATCHES "[0-9]t")
    # Free-threaded 3.15+. Split mode here is abi3t-only -- the classic Stable
    # ABI does not exist for a free-threaded interpreter -- and nanobind makes
    # FREE_THREADED mandatory for it, refusing to configure otherwise. That
    # option declares Py_MOD_GIL_NOT_USED, so the module must not depend on the
    # GIL for its own internal state.
    # Verified for wnet_cpp: the only file-scope mutable symbol across all 23
    # objects is the function-local `static nb::exception<InfeasibleException>
    # infeasible_error` in wnet.cpp, constructed once during module exec and
    # only read afterwards; everything else in .data/.bss is a vtable, typeinfo
    # or libstdc++ fixture, and all solver state is per-network. Re-check with
    # `nm -C --defined-only` on the objects if file-scope state is ever added.
    # What this does NOT promise is that one WassersteinNetwork may be driven
    # from two threads at once -- solve() mutates the instance and takes no
    # lock, so a *shared* object stays the caller's business.
    #
    # This branch must also stay in step with pylmcf and the rest of the stack:
    # mode is what decides whether these extensions can see each other's
    # registered types, so one package falling back here while another goes
    # split would break type sharing on that interpreter. is_nanobind_split()
    # and the import-time check in __init__.py are the guard on that.
    set(NB_MODE BACKEND_MODULE nanobind_backend BACKEND_PYPI nanobind-backend FREE_THREADED)
    set(NB_LINKED FALSE)
    set(NB_MODE_DESC "split, free-threaded abi3t (backend module: nanobind_backend)")
else()
    set(NB_MODE BACKEND_MODULE nanobind_backend BACKEND_PYPI nanobind-backend)
    set(NB_LINKED FALSE)
    set(NB_MODE_DESC "split (backend module: nanobind_backend)")
endif()
message(STATUS "wnet: nanobind build mode: ${NB_MODE_DESC}")

# A linked nanobind cannot target the classic 3.10/3.11 stable ABI -- that is
# precisely what split mode is for -- and nanobind turns the combination into a
# hard configure error. Every linked path above can walk into it, not just
# WNET_NB_LINKED: musl is the one that bites in practice, because there
# wheel.py-api is honoured. (PyPy and free-threaded builds escape only because
# scikit-build-core discards the classic abi3 request for them anyway.) Drop the
# request rather than making the platform pass a second incantation.
if (NB_LINKED AND DEFINED SKBUILD_SABI_VERSION
    AND NOT SKBUILD_SABI_VERSION STREQUAL ""
    AND SKBUILD_SABI_VERSION VERSION_LESS "3.12")
    message(WARNING
        "wnet: dropping the wheel.py-api=${SKBUILD_SABI_VERSION} stable-ABI "
        "request -- this is a ${NB_MODE_DESC} build, which is specific to "
        "Python ${Python_VERSION}. scikit-build-core still tags the wheel abi3, "
        "so the wheel produced here is NOT portable across Python versions: "
        "install it and discard it, never publish or reuse it.")
    set(SKBUILD_SABI_VERSION "")
endif()


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
