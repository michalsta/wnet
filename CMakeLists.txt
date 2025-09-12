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

install(TARGETS wnet_cpp LIBRARY DESTINATION wnet)

# Uncomment the following line to enable detailed debug prints in the C++ code
# target_compile_definitions(wnet_cpp PRIVATE DO_TONS_OF_PRINTS)
