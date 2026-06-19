// Base bindings for the double-intensity Wasserstein network, split into its own
// TU to keep wnet.cpp off the build's critical path. The per-dimension
// update_*/get_gradient overloads are appended later by register_dim_<N>().
#include "register_dim.hpp"

nb::class_<WNetIF> bind_network_if(nb::module_& m) {
    return nb::class_<WassersteinNetwork<int64_t, double>>(m, "CWassersteinNetworkFloat")
        //.def(nb::init<const Distribution<LEMON_INT>*, const std::vector<Distribution<LEMON_INT>*>&, const nb::callable, LEMON_INT>())
        .def("add_simple_trash", &WassersteinNetwork<int64_t, double>::add_simple_trash)
        .def("add_experimental_trash", &WassersteinNetwork<int64_t, double>::add_experimental_trash)
        .def("add_theoretical_trash", &WassersteinNetwork<int64_t, double>::add_theoretical_trash)
        .def("build", &WassersteinNetwork<int64_t, double>::build, nb::arg("config") = NetworkSimplexConfig{})
        .def("solve",
             [](WassersteinNetwork<int64_t, double>& self) { self.solve(); })
        .def("solve",
             [](WassersteinNetwork<int64_t, double>& self, const std::vector<double>& point) { self.solve(point); },
             nb::arg("point"))
        .def("total_cost", &WassersteinNetwork<int64_t, double>::total_cost)
        .def("scale_factor", &WassersteinNetwork<int64_t, double>::scale_factor)
        .def("intensity_scale_factor", &WassersteinNetwork<int64_t, double>::intensity_scale_factor)
        .def("set_intensity_scale", &WassersteinNetwork<int64_t, double>::set_intensity_scale, nb::arg("scale"))
        .def("set_cost_scaling", &WassersteinNetwork<int64_t, double>::set_cost_scaling, nb::arg("scale") = 0)
        .def("get_subgraph", &WassersteinNetwork<int64_t, double>::get_subgraph, nb::rv_policy::reference)
        .def("__str__", &WassersteinNetwork<int64_t, double>::to_string)
        .def("lemon_to_string", &WassersteinNetwork<int64_t, double>::lemon_to_string)
        .def("no_subgraphs", &WassersteinNetwork<int64_t, double>::no_subgraphs)
        .def("flows_for_target", [](WassersteinNetwork<int64_t, double>& self, size_t target_id) {
            auto [empirical_peak_indices, theoretical_peak_indices, flows] = self.flows_for_target(target_id);
            return std::make_tuple(vector_to_numpy<LEMON_INDEX>(empirical_peak_indices),
                                   vector_to_numpy<LEMON_INDEX>(theoretical_peak_indices),
                                   vector_to_numpy<int64_t>(flows));
        }, nb::rv_policy::move)
        .def("count_empirical_nodes", &WassersteinNetwork<int64_t, double>::count_nodes_of_type<EmpiricalNode<double>>)
        .def("count_theoretical_nodes", &WassersteinNetwork<int64_t, double>::count_nodes_of_type<TheoreticalNode<double>>)
        .def("count_matching_edges", &WassersteinNetwork<int64_t, double>::count_edges_of_type<MatchingEdge>)
        .def("count_theoretical_to_sink_edges", &WassersteinNetwork<int64_t,  double>::count_edges_of_type<TheoreticalToSinkEdge>)
        .def("count_src_to_empirical_edges", &WassersteinNetwork<int64_t, double>::count_edges_of_type<SrcToEmpiricalEdge>)
        .def("count_simple_trash_edges", &WassersteinNetwork<int64_t, double>::count_edges_of_type<SimpleTrashEdge>)
        .def("count_chain_edges", &WassersteinNetwork<int64_t, double>::count_edges_of_type<ChainEdge>)
        .def("matching_density", &WassersteinNetwork<int64_t, double>::matching_density)
        .def("no_theoretical_spectra", &WassersteinNetwork<int64_t, double>::no_theoretical_spectra)
        .def("theoretical_spectra_sizes", &WassersteinNetwork<int64_t, double>::theoretical_spectra_sizes)
        .def_static("value_type_size", &WassersteinNetwork<int64_t, double>::value_type_size)
        .def_static("index_type_size", &WassersteinNetwork<int64_t, double>::index_type_size)
        .def_static("max_value", &WassersteinNetwork<int64_t, double>::max_value)
        .def_static("max_index", &WassersteinNetwork<int64_t, double>::max_index)
        .def("signal_part_derivatives", &WassersteinNetwork<int64_t, double>::signal_part_derivatives)
        .def("spectrum_proportion_derivatives", &WassersteinNetwork<int64_t, double>::spectrum_proportion_derivatives)
        .def("signal_part_derivatives_fast_approx", &WassersteinNetwork<int64_t, double>::signal_part_derivatives_fast_approx)
        .def("spectrum_proportion_derivatives_fast_approx", &WassersteinNetwork<int64_t, double>::spectrum_proportion_derivatives_fast_approx)
        .def("warm_start_count", &WassersteinNetwork<int64_t, double>::warm_start_count)
        .def("cold_start_count", &WassersteinNetwork<int64_t, double>::cold_start_count)
        .def("dual_repair_count", &WassersteinNetwork<int64_t, double>::dual_repair_count)
        .def("primal_repair_count", &WassersteinNetwork<int64_t, double>::primal_repair_count);
}
