#ifndef WNET_REGISTER_DIM_HPP
#define WNET_REGISTER_DIM_HPP

// Shared preamble + per-dimension registration macros.
//
// The per-dimension VectorDistribution / *Scaler / factory-create / update-solve
// instantiations used to live inline in wnet.cpp, making it a single huge
// translation unit. They are now emitted one-dimension-per-TU via the stub
// files dim_register_<N>.cpp (each `#define WNET_DIM <N>` then
// `#include "register_dim.inc"`), so ninja can compile them in parallel.
//
// This header carries the include preamble (so every TU sees identical
// NB_MAKE_OPAQUE / opaque-type ABI declarations), the registration macros, and
// the forward declarations of the register_dim_<N>() functions that wnet.cpp
// calls.

#include <iostream>
#include <memory>
#include <vector>
#include <array>
#include <unordered_map>
#include <variant>

#include <nanobind/nanobind.h>

NB_MAKE_OPAQUE(std::vector<int32_t>);
NB_MAKE_OPAQUE(std::vector<int64_t>);
NB_MAKE_OPAQUE(std::vector<uint32_t>);
NB_MAKE_OPAQUE(std::vector<uint64_t>);
NB_MAKE_OPAQUE(std::vector<double>);
NB_MAKE_OPAQUE(std::unordered_map<int32_t, int64_t>);


#include <nanobind/ndarray.h>

#include <nanobind/stl/string.h>
//#include <nanobind/stl/vector.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/variant.h>

// Declare the type as opaque to avoid conflicts
NB_MAKE_OPAQUE(std::pair<const nanobind::ndarray<nanobind::detail::shape<-1, -1>> *, size_t>);

#include "decompositable_graph.hpp"
#include "graph_elements.hpp"
#include "distribution.hpp"
#include "scaling.hpp"
#include "misc.hpp"
#include <new>


#ifndef WNET_MAX_DIM
#define WNET_MAX_DIM 20
#endif

// Type aliases avoid commas inside macro arguments (which the preprocessor
// would miscount as argument separators).
using WNetII = WassersteinNetwork<int64_t, int64_t>;
using WNetIF = WassersteinNetwork<int64_t, double>;
using WNetFactory = WassersteinNetworkFactory<int64_t>;

// Two-level indirection: callers pass WNET_DIM (a macro). The forwarding macro
// forces it to expand to its numeric value *before* the IMPL macro stringizes
// (#DIM) and token-pastes (##DIM) it, so each dimension gets a distinct Python
// class name ("CVectorDistribution20") rather than the literal "...WNET_DIM".
#define EXPOSE_VECTOR_DISTRIBUTION(DIM) EXPOSE_VECTOR_DISTRIBUTION_IMPL(DIM)
#define EXPOSE_VECTOR_DISTRIBUTION_IMPL(DIM) \
    using VectorDistribution_##DIM = VectorDistribution<DIM, double, LEMON_INT>; \
    nb::class_<VectorDistribution_##DIM>(m, "CVectorDistribution" #DIM) \
        /*.def(nb::init<const std::vector<std::array<double, DIM>>&, const std::vector<LEMON_INT>&>()) \
        .def(nb::init<std::vector<std::array<double, DIM>>&&,  std::vector<LEMON_INT>&&>()) */ \
        .def(nb::init<const nb::ndarray<double, nb::shape<DIM, -1>>&, const nb::ndarray<LEMON_INT, nb::shape<-1>>&>()) \
        .def("size", &VectorDistribution_##DIM::size) \
        .def("py_get_positions", &VectorDistribution_##DIM::py_get_positions) \
        .def("py_get_intensities", &VectorDistribution_##DIM::py_get_intensities) \
        .def("positions_view", [](nb::object self_obj) { \
            auto& d = nb::cast<VectorDistribution_##DIM&>(self_obj); \
            const auto& pos = d.get_positions(); \
            return nb::ndarray<nb::numpy, double>( \
                (void*)pos.data(), {pos.size(), (size_t)DIM}, self_obj); \
        }) \
        .def("intensities_view", [](nb::object self_obj) { \
            auto& d = nb::cast<VectorDistribution_##DIM&>(self_obj); \
            const auto& ints = d.get_intensities(); \
            return nb::ndarray<nb::numpy, LEMON_INT>( \
                (void*)ints.data(), {ints.size()}, self_obj); \
        }) \
        .def("get_point", &VectorDistribution_##DIM::get_point) \
        .def("n_highest", &VectorDistribution_##DIM::n_highest) \
        .def("p_trim", &VectorDistribution_##DIM::p_trim) \
        .def("scaled", &VectorDistribution_##DIM::scaled) \
        .def("add", &VectorDistribution_##DIM::add) \
        .def("binned", &VectorDistribution_##DIM::binned) \
        .def("sorted_by_positions", &VectorDistribution_##DIM::sorted_by_positions) \
        .def_static("linear_combination", [](const std::vector<VectorDistribution_##DIM>& dists, const nb::ndarray<double, nb::shape<-1>>& weights) { \
            return VectorDistribution_##DIM::linear_combination(dists, numpy_to_vector<double>(weights)); \
        }) \
        .def("__len__", &VectorDistribution_##DIM::size); \
    using VectorDistributionFloat_##DIM = VectorDistribution<DIM, double, double>; \
    nb::class_<VectorDistributionFloat_##DIM>(m, "CVectorDistributionFloat" #DIM) \
        /*.def(nb::init<const std::vector<std::array<double, DIM>>&, const std::vector<double>&>()) \
        .def(nb::init<std::vector<std::array<double, DIM>>&&,  std::vector<double>&&>()) */ \
        .def(nb::init<const nb::ndarray<double, nb::shape<DIM, -1>>&, const nb::ndarray<double, nb::shape<-1>>&>()) \
        .def("size", &VectorDistributionFloat_##DIM::size) \
        .def("py_get_positions", &VectorDistributionFloat_##DIM::py_get_positions) \
        .def("py_get_intensities", &VectorDistributionFloat_##DIM::py_get_intensities) \
        .def("positions_view", [](nb::object self_obj) { \
            auto& d = nb::cast<VectorDistributionFloat_##DIM&>(self_obj); \
            const auto& pos = d.get_positions(); \
            return nb::ndarray<nb::numpy, double>( \
                (void*)pos.data(), {pos.size(), (size_t)DIM}, self_obj); \
        }) \
        .def("intensities_view", [](nb::object self_obj) { \
            auto& d = nb::cast<VectorDistributionFloat_##DIM&>(self_obj); \
            const auto& ints = d.get_intensities(); \
            return nb::ndarray<nb::numpy, double>( \
                (void*)ints.data(), {ints.size()}, self_obj); \
        }) \
        .def("get_point", &VectorDistributionFloat_##DIM::get_point) \
        .def("n_highest", &VectorDistributionFloat_##DIM::n_highest) \
        .def("p_trim", &VectorDistributionFloat_##DIM::p_trim) \
        .def("scaled", &VectorDistributionFloat_##DIM::scaled) \
        .def("add", &VectorDistributionFloat_##DIM::add) \
        .def("binned", &VectorDistributionFloat_##DIM::binned) \
        .def("sorted_by_positions", &VectorDistributionFloat_##DIM::sorted_by_positions) \
        .def_static("linear_combination", [](const std::vector<VectorDistributionFloat_##DIM>& dists, const nb::ndarray<double, nb::shape<-1>>& weights) { \
            return VectorDistributionFloat_##DIM::linear_combination(dists, numpy_to_vector<double>(weights)); \
        }) \
        .def("__len__", &VectorDistributionFloat_##DIM::size); \
    using WNetAlignScaler_##DIM = WNetAlignScaler<DIM>; \
    nb::class_<WNetAlignScaler_##DIM>(m, "CWNetAlignScalerFloat" #DIM) \
        .def("__init__", [](WNetAlignScaler_##DIM* self, \
                const VectorDistributionFloat_##DIM& empirical, \
                const std::vector<VectorDistributionFloat_##DIM>& theoretical, \
                DistanceMetric metric, double max_distance, \
                const nb::ndarray<double, nb::shape<-1>>& trash_costs, \
                double max_int) { \
            std::vector<const VectorDistributionFloat_##DIM*> ptrs; \
            ptrs.reserve(theoretical.size()); \
            for (const auto& t : theoretical) ptrs.push_back(&t); \
            std::vector<double> tc = numpy_to_vector<double>(trash_costs); \
            new (self) WNetAlignScaler_##DIM(empirical, ptrs, metric, max_distance, tc, max_int); \
        }) \
        .def("sf_distance", &WNetAlignScaler_##DIM::sf_distance) \
        .def("sf_intensity", &WNetAlignScaler_##DIM::sf_intensity) \
        .def("scale_factor", &WNetAlignScaler_##DIM::scale_factor) \
        .def("ftol",         &WNetAlignScaler_##DIM::ftol); \
    using WNetDeconvScaler_##DIM = WNetDeconvScaler<DIM>; \
    nb::class_<WNetDeconvScaler_##DIM>(m, "CWNetDeconvScalerFloat" #DIM) \
        .def("__init__", [](WNetDeconvScaler_##DIM* self, \
                const VectorDistributionFloat_##DIM& empirical, \
                const std::vector<VectorDistributionFloat_##DIM>& theoretical, \
                DistanceMetric metric, double max_distance, \
                const nb::ndarray<double, nb::shape<-1>>& trash_costs, \
                double max_int, double max_dropped_fraction) { \
            std::vector<const VectorDistributionFloat_##DIM*> ptrs; \
            ptrs.reserve(theoretical.size()); \
            for (const auto& t : theoretical) ptrs.push_back(&t); \
            std::vector<double> tc = numpy_to_vector<double>(trash_costs); \
            new (self) WNetDeconvScaler_##DIM(empirical, ptrs, metric, max_distance, tc, \
                max_int, max_dropped_fraction); \
        }) \
        .def("sf_distance", &WNetDeconvScaler_##DIM::sf_distance) \
        .def("sf_intensity", &WNetDeconvScaler_##DIM::sf_intensity) \
        .def("scale_factor", &WNetDeconvScaler_##DIM::scale_factor) \
        .def("ftol",         &WNetDeconvScaler_##DIM::ftol); \
    using FineGridScaler_##DIM = FineGridScaler<DIM>; \
    nb::class_<FineGridScaler_##DIM>(m, "CFineGridScalerFloat" #DIM) \
        .def("__init__", [](FineGridScaler_##DIM* self, \
                const VectorDistributionFloat_##DIM& empirical, \
                const std::vector<VectorDistributionFloat_##DIM>& theoretical, \
                DistanceMetric metric, double max_distance, \
                const nb::ndarray<double, nb::shape<-1>>& trash_costs, \
                double p, double max_int) { \
            std::vector<const VectorDistributionFloat_##DIM*> ptrs; \
            ptrs.reserve(theoretical.size()); \
            for (const auto& t : theoretical) ptrs.push_back(&t); \
            std::vector<double> tc = numpy_to_vector<double>(trash_costs); \
            new (self) FineGridScaler_##DIM(empirical, ptrs, metric, max_distance, tc, p, max_int); \
        }) \
        .def("sf_distance", &FineGridScaler_##DIM::sf_distance) \
        .def("sf_intensity", &FineGridScaler_##DIM::sf_intensity) \
        .def("scale_factor", &FineGridScaler_##DIM::scale_factor) \
        .def("ftol",         &FineGridScaler_##DIM::ftol); \
    using GenericScaler_##DIM = GenericScaler<DIM>; \
    nb::class_<GenericScaler_##DIM>(m, "CGenericScalerFloat" #DIM) \
        .def("__init__", [](GenericScaler_##DIM* self, \
                const VectorDistributionFloat_##DIM& empirical, \
                const std::vector<VectorDistributionFloat_##DIM>& theoretical, \
                DistanceMetric metric, double max_distance, \
                const nb::ndarray<double, nb::shape<-1>>& trash_costs, \
                double p95_frac, double rounding_tol, \
                double max_dropped_frac, double max_int) { \
            std::vector<const VectorDistributionFloat_##DIM*> ptrs; \
            ptrs.reserve(theoretical.size()); \
            for (const auto& t : theoretical) ptrs.push_back(&t); \
            std::vector<double> tc = numpy_to_vector<double>(trash_costs); \
            new (self) GenericScaler_##DIM(empirical, ptrs, metric, max_distance, tc, \
                p95_frac, rounding_tol, max_dropped_frac, max_int); \
        }) \
        .def("sf_distance", &GenericScaler_##DIM::sf_distance) \
        .def("sf_intensity", &GenericScaler_##DIM::sf_intensity) \
        .def("scale_factor", &GenericScaler_##DIM::scale_factor) \
        .def("ftol",         &GenericScaler_##DIM::ftol);

// Bind update_positions_and_solve for one (network alias, intensity type, dimension) triple.
#define BIND_UPDATE_AND_SOLVE(NET_ALIAS, INTENSITY_TYPE, DIM) \
    .def("update_positions_and_solve", \
         [](NET_ALIAS& self, \
            const VectorDistribution<DIM, double, INTENSITY_TYPE>* new_emp, \
            const std::vector<VectorDistribution<DIM, double, INTENSITY_TYPE>*>& new_theo, \
            DistanceMetric metric) \
         { self.update_positions_and_solve(new_emp, new_theo, metric); }, \
         nb::arg("new_empirical"), nb::arg("new_theoretical"), nb::arg("metric"))

// Bind update_positions_and_get_gradient for one triple.
// Returns (emp_grad [N_emp, DIM], list of theo_grad [N_k, DIM]) as numpy arrays.
#define BIND_UPDATE_AND_GET_GRADIENT(NET_ALIAS, INTENSITY_TYPE, DIM) \
    .def("update_positions_and_get_gradient", \
         [](NET_ALIAS& self, \
            const VectorDistribution<DIM, double, INTENSITY_TYPE>* new_emp, \
            const std::vector<VectorDistribution<DIM, double, INTENSITY_TYPE>*>& new_theo, \
            DistanceMetric metric) { \
             const size_t n_emp = new_emp->size(); \
             std::unique_ptr<double[]> emp_buf(new double[n_emp * DIM]()); \
             std::span<double> emp_span(emp_buf.get(), n_emp * DIM); \
             std::vector<std::unique_ptr<double[]>> theo_bufs; \
             std::vector<std::span<double>> theo_spans; \
             std::vector<size_t> theo_sizes; \
             for (auto* t : new_theo) { \
                 const size_t n_k = t->size(); \
                 theo_bufs.push_back(std::unique_ptr<double[]>(new double[n_k * DIM]())); \
                 theo_spans.emplace_back(theo_bufs.back().get(), n_k * DIM); \
                 theo_sizes.push_back(n_k); \
             } \
             self.update_positions_and_get_gradient<VectorDistribution<DIM, double, INTENSITY_TYPE>>( \
                 new_emp, new_theo, emp_span, theo_spans, metric); \
             double* emp_raw = emp_buf.release(); \
             nb::capsule emp_cap(emp_raw, [](void* p) noexcept { delete[] static_cast<double*>(p); }); \
             size_t emp_shape[2] = {n_emp, (size_t)DIM}; \
             auto emp_arr = nb::ndarray<nb::numpy, double, nb::ndim<2>>(emp_raw, 2, emp_shape, emp_cap); \
             nb::list theo_list; \
             for (size_t s = 0; s < theo_bufs.size(); ++s) { \
                 double* raw = theo_bufs[s].release(); \
                 nb::capsule cap(raw, [](void* p) noexcept { delete[] static_cast<double*>(p); }); \
                 size_t theo_shape[2] = {theo_sizes[s], (size_t)DIM}; \
                 theo_list.append(nb::ndarray<nb::numpy, double, nb::ndim<2>>(raw, 2, theo_shape, cap)); \
             } \
             return nb::make_tuple(std::move(emp_arr), std::move(theo_list)); \
         }, \
         nb::arg("new_empirical"), nb::arg("new_theoretical"), nb::arg("metric"))

// Common signature of every per-dimension registration function. The classes
// are constructed (with their non-templated base methods) in wnet.cpp and
// passed by reference so each register_dim_<N>() can append its dimension's
// overloads / class registrations to the same module.
#define REGISTER_DIM_PARAMS \
    nb::module_& m, nb::class_<WNetII>& net_ii, nb::class_<WNetIF>& net_if, \
    nb::class_<WNetFactory>& factory

#define REGISTER_DIM_NAME_(N) register_dim_##N
#define REGISTER_DIM_NAME(N) REGISTER_DIM_NAME_(N)

// Network base bindings (the two heaviest classes in the module) live in their
// own TUs, split by intensity type, so neither sits on the build's critical
// path. Each builds the non-templated base methods and returns the network
// class for register_dim_<N>() to extend with per-dimension update_* overloads.
nb::class_<WNetII> bind_network_ii(nb::module_& m);
nb::class_<WNetIF> bind_network_if(nb::module_& m);

// Forward declarations for the whole 1..20 range. Declaring a function that is
// never defined/called is harmless; the calls (in wnet.cpp) and the compilation
// of the matching build_stubs/dim_register_<N>.cpp are both gated on WNET_MAX_DIM.
void register_dim_1(REGISTER_DIM_PARAMS);
void register_dim_2(REGISTER_DIM_PARAMS);
void register_dim_3(REGISTER_DIM_PARAMS);
void register_dim_4(REGISTER_DIM_PARAMS);
void register_dim_5(REGISTER_DIM_PARAMS);
void register_dim_6(REGISTER_DIM_PARAMS);
void register_dim_7(REGISTER_DIM_PARAMS);
void register_dim_8(REGISTER_DIM_PARAMS);
void register_dim_9(REGISTER_DIM_PARAMS);
void register_dim_10(REGISTER_DIM_PARAMS);
void register_dim_11(REGISTER_DIM_PARAMS);
void register_dim_12(REGISTER_DIM_PARAMS);
void register_dim_13(REGISTER_DIM_PARAMS);
void register_dim_14(REGISTER_DIM_PARAMS);
void register_dim_15(REGISTER_DIM_PARAMS);
void register_dim_16(REGISTER_DIM_PARAMS);
void register_dim_17(REGISTER_DIM_PARAMS);
void register_dim_18(REGISTER_DIM_PARAMS);
void register_dim_19(REGISTER_DIM_PARAMS);
void register_dim_20(REGISTER_DIM_PARAMS);

#endif // WNET_REGISTER_DIM_HPP
