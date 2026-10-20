// Base bindings for the int64-intensity Wasserstein network, split into its own
// TU to keep wnet.cpp off the build's critical path. The per-dimension
// update_*/get_gradient overloads are appended later by register_dim_<N>().
#include "register_dim.hpp"

nb::class_<WNetII> bind_network_ii(nb::module_& m) {
    return nb::class_<WNetII>(m, "CWassersteinNetwork")
        //.def(nb::init<const Distribution<LEMON_INT>*, const std::vector<Distribution<LEMON_INT>*>&, const nb::callable, LEMON_INT>())
        .def("add_simple_trash", &WassersteinNetwork<int64_t, int64_t>::add_simple_trash)
        .def("add_experimental_trash", &WassersteinNetwork<int64_t, int64_t>::add_experimental_trash)
        .def("add_theoretical_trash", &WassersteinNetwork<int64_t, int64_t>::add_theoretical_trash)
        .def("add_independent_asymmetric_trash", &WassersteinNetwork<int64_t, int64_t>::add_independent_asymmetric_trash, nb::arg("C_exp"), nb::arg("C_theo"))
        .def("build", &WassersteinNetwork<int64_t, int64_t>::build, nb::arg("config") = NetworkSimplexConfig{})
        .def("solve",
             [](WassersteinNetwork<int64_t, int64_t>& self, bool with_duals) { self.solve(with_duals); },
             nb::arg("with_duals") = false)
        .def("solve",
             [](WassersteinNetwork<int64_t, int64_t>& self, const std::vector<double>& point, bool with_duals) { self.solve(point, with_duals); },
             nb::arg("point"), nb::arg("with_duals") = false)
        .def("total_cost", &WassersteinNetwork<int64_t, int64_t>::total_cost)
        .def("dual_cut", [](const WassersteinNetwork<int64_t, int64_t>& self) {
            auto [intercept, slopes, upper, error, rounded_cost] = self.dual_cut();
            nb::dict out; out["intercept"] = intercept;
            out["gradient"] = vector_to_numpy<double>(slopes);
            out["upper_bound"] = upper; out["rounding_error"] = error;
            out["rounded_cost"] = rounded_cost;
            auto [cost_err, rho, lower_slack] = self.cost_quantization_bounds();
            out["cost_rounding_error"] = cost_err;
            out["cost_round_up_ratio"] = rho;
            out["cost_lower_slack"] = lower_slack;
            auto [t_rho, t_sigma, t_abs] = self.cost_transfer_bounds();
            out["transfer_round_up_ratio"] = t_rho;
            out["transfer_round_down_ratio"] = t_sigma;
            out["transfer_absolute_error"] = t_abs;
            return out;
        })
        .def("scale_factor", &WassersteinNetwork<int64_t, int64_t>::scale_factor)
        .def("intensity_scale_factor", &WassersteinNetwork<int64_t, int64_t>::intensity_scale_factor)
        .def("set_intensity_scale", &WassersteinNetwork<int64_t, int64_t>::set_intensity_scale, nb::arg("scale"))
        .def("set_flow_budget", &WassersteinNetwork<int64_t, int64_t>::set_flow_budget, nb::arg("flow"))
        .def("get_subgraph", &WassersteinNetwork<int64_t, int64_t>::get_subgraph, nb::rv_policy::reference_internal)
        .def("__str__", &WassersteinNetwork<int64_t, int64_t>::to_string)
        .def("lemon_to_string", &WassersteinNetwork<int64_t, int64_t>::lemon_to_string)
        .def("no_subgraphs", &WassersteinNetwork<int64_t, int64_t>::no_subgraphs)
        .def("flows_for_target", [](WassersteinNetwork<int64_t, int64_t>& self, size_t target_id) {
            auto [empirical_peak_indices, theoretical_peak_indices, flows] = self.flows_for_target(target_id);
            return std::make_tuple(vector_to_numpy<LEMON_INDEX>(empirical_peak_indices),
                                   vector_to_numpy<LEMON_INDEX>(theoretical_peak_indices),
                                   vector_to_numpy<int64_t>(flows));
        }, nb::rv_policy::move)
        .def("count_empirical_nodes", &WassersteinNetwork<int64_t, int64_t>::count_nodes_of_type<EmpiricalNode<int64_t>>)
        .def("count_theoretical_nodes", &WassersteinNetwork<int64_t, int64_t>::count_nodes_of_type<TheoreticalNode<int64_t>>)
        .def("count_matching_edges", &WassersteinNetwork<int64_t, int64_t>::count_edges_of_type<MatchingEdge>)
        .def("count_theoretical_to_sink_edges", &WassersteinNetwork<int64_t,  int64_t>::count_edges_of_type<TheoreticalToSinkEdge>)
        .def("count_src_to_empirical_edges", &WassersteinNetwork<int64_t, int64_t>::count_edges_of_type<SrcToEmpiricalEdge>)
        .def("count_simple_trash_edges", &WassersteinNetwork<int64_t, int64_t>::count_edges_of_type<SimpleTrashEdge>)
        .def("count_chain_edges", &WassersteinNetwork<int64_t, int64_t>::count_edges_of_type<ChainEdge>)
        .def("matching_density", &WassersteinNetwork<int64_t, int64_t>::matching_density)
        .def("no_theoretical_spectra", &WassersteinNetwork<int64_t, int64_t>::no_theoretical_spectra)
        .def("theoretical_spectra_sizes", &WassersteinNetwork<int64_t, int64_t>::theoretical_spectra_sizes)
        .def_static("value_type_size", &WassersteinNetwork<int64_t, int64_t>::value_type_size)
        .def_static("index_type_size", &WassersteinNetwork<int64_t, int64_t>::index_type_size)
        .def_static("max_value", &WassersteinNetwork<int64_t, int64_t>::max_value)
        .def_static("max_index", &WassersteinNetwork<int64_t, int64_t>::max_index)
        .def("signal_part_derivatives", &WassersteinNetwork<int64_t, int64_t>::signal_part_derivatives)
        .def("spectrum_proportion_derivatives", &WassersteinNetwork<int64_t, int64_t>::spectrum_proportion_derivatives)
        .def("signal_part_derivatives_fast_approx", &WassersteinNetwork<int64_t, int64_t>::signal_part_derivatives_fast_approx)
        .def("spectrum_proportion_derivatives_fast_approx", &WassersteinNetwork<int64_t, int64_t>::spectrum_proportion_derivatives_fast_approx)
        .def("warm_start_count", &WassersteinNetwork<int64_t, int64_t>::warm_start_count)
        .def("set_warm_policy", &WassersteinNetwork<int64_t, int64_t>::set_warm_policy, nb::arg("policy"))
        .def("ns_stats", [](const WassersteinNetwork<int64_t, int64_t>& self) {
            auto [work, policy_cold, viol, mass, supply] = self.ns_stats();
            nb::dict out; out["work_units"] = work; out["policy_cold"] = policy_cold;
            out["violations"] = viol; out["violation_mass"] = mass; out["positive_supply"] = supply;
            return out;
        })
        .def("cold_start_count", &WassersteinNetwork<int64_t, int64_t>::cold_start_count)
        .def("dual_repair_count", &WassersteinNetwork<int64_t, int64_t>::dual_repair_count)
        .def("primal_repair_count", &WassersteinNetwork<int64_t, int64_t>::primal_repair_count);
}
