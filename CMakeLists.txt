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

nanobind_add_module(wnet_cpp
    NB_STATIC NOMINSIZE
    src/wnet/cpp/wnet/wnet.cpp)

target_include_directories(wnet_cpp PRIVATE ${PYLMCF_INCLUDE_DIRS})
target_compile_definitions(wnet_cpp PRIVATE INCLUDE_NANOBIND_STUFF)

# Add -Wall -Wextra for Unix-style compilers (GCC/Clang)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(wnet_cpp PRIVATE -Wall -Wextra)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(wnet_cpp PRIVATE DEBUG_MODE)
    target_compile_options(wnet_cpp PRIVATE -g -Og -fsanitize=address -D_GLIBCXX_ASSERTIONS -D_GLIBCXX_SANITIZE_VECTOR -DGLIBCXX_DEBUG -D_LIBCPP_ENABLE_ASSERTIONS=1)
    #target_compile_options(wnet_cpp PRIVATE -g Og -D_GLIBCXX_ASSERTIONS -D_LIBCPP_ENABLE_ASSERTIONS=1)
endif()


install(TARGETS wnet_cpp LIBRARY DESTINATION wnet)

# Uncomment the following line to enable detailed debug prints in the C++ code
# target_compile_definitions(wnet_cpp PRIVATE DO_TONS_OF_PRINTS)
