// Main module TU. The per-dimension VectorDistribution / *Scaler / factory
// create / update-solve instantiations live in the dim_register_<N>.cpp stubs
// (see register_dim.hpp / register_dim.inc) so ninja can compile them in
// parallel. This TU keeps NB_MODULE and the non-templated base bindings, builds
// the three classes that the stubs extend, and dispatches to register_dim_<N>().

#include "register_dim.hpp"

NB_MODULE(wnet_cpp, m) {
    m.doc() = "WNet C++ imlementation module";
    m.def("wnet_cpp_hello", []() {
        std::cout << "Hello from WNet (C++)!" << std::endl;
    }, "A simple hello world function for the WNet (C++) extension");
    // Bind the classes to the module
    nb::bind_vector<std::vector<int32_t>>(m, "std_vector_int32_t");
    nb::bind_vector<std::vector<int64_t>>(m, "std_vector_int64_t");
    nb::bind_vector<std::vector<uint32_t>>(m, "std_vector_uint32_t");
    nb::bind_vector<std::vector<uint64_t>>(m, "std_vector_uint64_t");
    nb::bind_vector<std::vector<double>>(m, "std_vector_double");
    nb::bind_map<std::unordered_map<int32_t, int64_t>>(m, "std_unordered_map_int32_t_int64_t");

    nb::enum_<DistanceMetric>(m, "DistanceMetric")
        .value("L1", DistanceMetric::L1)
        .value("L2", DistanceMetric::L2)
        .value("LINF", DistanceMetric::LINF);

    nb::enum_<NSPivotRule>(m, "NSPivotRule")
        .value("FIRST_ELIGIBLE", NSPivotRule::FIRST_ELIGIBLE)
        .value("BEST_ELIGIBLE",  NSPivotRule::BEST_ELIGIBLE)
        .value("BLOCK_SEARCH",   NSPivotRule::BLOCK_SEARCH)
        .value("CANDIDATE_LIST", NSPivotRule::CANDIDATE_LIST)
        .value("ALTERING_LIST",  NSPivotRule::ALTERING_LIST);

    nb::enum_<CSMethod>(m, "CSMethod")
        .value("PUSH",            CSMethod::PUSH)
        .value("AUGMENT",         CSMethod::AUGMENT)
        .value("PARTIAL_AUGMENT", CSMethod::PARTIAL_AUGMENT);

    nb::enum_<CCMethod>(m, "CCMethod")
        .value("SIMPLE_CYCLE_CANCELING",       CCMethod::SIMPLE_CYCLE_CANCELING)
        .value("MINIMUM_MEAN_CYCLE_CANCELING", CCMethod::MINIMUM_MEAN_CYCLE_CANCELING)
        .value("CANCEL_AND_TIGHTEN",           CCMethod::CANCEL_AND_TIGHTEN);

    // Python keyword `None` cannot be an attribute name, so the None mode is
    // exposed as WarmMode.NONE.
    nb::enum_<NSWarmMode>(m, "WarmMode")
        .value("NONE",   NSWarmMode::None)
        .value("Simple", NSWarmMode::Simple)
        .value("Dual",   NSWarmMode::Dual)
        .value("Primal", NSWarmMode::Primal)
        .value("DualRatio", NSWarmMode::DualRatio)
        .value("DualGreedy", NSWarmMode::DualGreedy)
        .value("LinkCut", NSWarmMode::LinkCut);


    nb::class_<NetworkSimplexConfig>(m, "NetworkSimplex")
        .def(nb::init<>())
        .def_rw("pivot", &NetworkSimplexConfig::pivot)
        .def_rw("warm",  &NetworkSimplexConfig::warm);

    nb::class_<CostScalingConfig>(m, "CostScaling")
        .def(nb::init<>())
        .def_rw("method", &CostScalingConfig::method)
        .def_rw("factor", &CostScalingConfig::factor);

    nb::class_<CycleCancelingConfig>(m, "CycleCanceling")
        .def(nb::init<>())
        .def_rw("method", &CycleCancelingConfig::method);

    nb::class_<CapacityScalingConfig>(m, "CapacityScaling")
        .def(nb::init<>())
        .def_rw("factor", &CapacityScalingConfig::factor);

    nb::class_<FlowNode<int64_t>>(m, "FlowNode")
        .def(nb::init<LEMON_INDEX, SourceNode>())
        .def(nb::init<LEMON_INDEX, SinkNode>())
        .def(nb::init<LEMON_INDEX, EmpiricalNode<int64_t>>())
        .def(nb::init<LEMON_INDEX, TheoreticalNode<int64_t>>())
        .def("get_id", &FlowNode<int64_t>::get_id)
        .def("get_type", &FlowNode<int64_t>::get_type)
        .def("layer", &FlowNode<int64_t>::layer)
        .def("type_str", &FlowNode<int64_t>::type_str)
        .def("__str__", &FlowNode<int64_t>::to_string);

    nb::class_<FlowEdge<int64_t>>(m, "FlowEdge")
        .def(nb::init<LEMON_INDEX, const FlowNode<int64_t>&, const FlowNode<int64_t>&, FlowEdgeType>())
        .def("get_id", &FlowEdge<int64_t>::get_id)
        .def("get_start_node", &FlowEdge<int64_t>::get_start_node)
        .def("get_end_node", &FlowEdge<int64_t>::get_end_node)
        .def("get_start_node_id", &FlowEdge<int64_t>::get_start_node_id)
        .def("get_end_node_id", &FlowEdge<int64_t>::get_end_node_id)
        .def("get_type", &FlowEdge<int64_t>::get_type)
        .def("get_cost", &FlowEdge<int64_t>::get_cost)
        .def("get_base_capacity", &FlowEdge<int64_t>::get_base_capacity)
        .def("to_string", &FlowEdge<int64_t>::to_string);

    // Double-intensity variants used by CWassersteinNetworkSubgraphFloat.
    nb::class_<FlowNode<double>>(m, "FlowNodeFloat")
        .def("get_id", &FlowNode<double>::get_id)
        .def("layer", &FlowNode<double>::layer)
        .def("type_str", &FlowNode<double>::type_str)
        .def("__str__", &FlowNode<double>::to_string);

    nb::class_<FlowEdge<double>>(m, "FlowEdgeFloat")
        .def("get_id", &FlowEdge<double>::get_id)
        .def("get_start_node_id", &FlowEdge<double>::get_start_node_id)
        .def("get_end_node_id", &FlowEdge<double>::get_end_node_id)
        .def("get_type", &FlowEdge<double>::get_type)
        .def("get_cost", &FlowEdge<double>::get_cost)
        .def("get_base_capacity", &FlowEdge<double>::get_base_capacity)
        .def("to_string", &FlowEdge<double>::to_string);

    nb::class_<WassersteinNetworkSubgraph<int64_t, int64_t>>(m, "CWassersteinNetworkSubgraph")
        .def(nb::init<const std::vector<LEMON_INDEX>&, const std::vector<FlowNode<int64_t>>&, const std::vector<FlowEdge<int64_t>*>&, size_t>())
        .def("add_simple_trash", &WassersteinNetworkSubgraph<int64_t, int64_t>::add_simple_trash)
        .def("add_experimental_trash", &WassersteinNetworkSubgraph<int64_t, int64_t>::add_experimental_trash)
        .def("add_theoretical_trash", &WassersteinNetworkSubgraph<int64_t, int64_t>::add_theoretical_trash)
        .def("build", &WassersteinNetworkSubgraph<int64_t, int64_t>::build, nb::arg("config") = NetworkSimplexConfig{})
        .def("set_point", &WassersteinNetworkSubgraph<int64_t, int64_t>::set_point)
        .def("total_cost", &WassersteinNetworkSubgraph<int64_t, int64_t>::total_cost)
        .def("simple_trash_cost", &WassersteinNetworkSubgraph<int64_t, int64_t>::simple_trash_cost)
        .def("to_string", &WassersteinNetworkSubgraph<int64_t, int64_t>::to_string)
        .def("lemon_to_string", &WassersteinNetworkSubgraph<int64_t, int64_t>::lemon_to_string)
        .def("no_nodes", &WassersteinNetworkSubgraph<int64_t, int64_t>::no_nodes)
        .def("no_edges", &WassersteinNetworkSubgraph<int64_t, int64_t>::no_edges)
        .def("get_nodes", &WassersteinNetworkSubgraph<int64_t, int64_t>::get_nodes)
        .def("get_edges", &WassersteinNetworkSubgraph<int64_t, int64_t>::get_edges)
        .def("get_flow_map", &WassersteinNetworkSubgraph<int64_t, int64_t>::get_flow_map)
        .def("is_solved", &WassersteinNetworkSubgraph<int64_t, int64_t>::is_solved)
        .def("signal_part_derivatives", &WassersteinNetworkSubgraph<int64_t, int64_t>::signal_part_derivatives)
        .def("spectrum_proportion_derivatives", &WassersteinNetworkSubgraph<int64_t, int64_t>::spectrum_proportion_derivatives)
        .def("signal_part_derivatives_fast_approx", &WassersteinNetworkSubgraph<int64_t, int64_t>::signal_part_derivatives_fast_approx)
        .def("spectrum_proportion_derivatives_fast_approx", &WassersteinNetworkSubgraph<int64_t, int64_t>::spectrum_proportion_derivatives_fast_approx)
        .def("warm_start_count", &WassersteinNetworkSubgraph<int64_t, int64_t>::warm_start_count)
        .def("cold_start_count", &WassersteinNetworkSubgraph<int64_t, int64_t>::cold_start_count)
        .def("dual_repair_count", &WassersteinNetworkSubgraph<int64_t, int64_t>::dual_repair_count)
        .def("primal_repair_count", &WassersteinNetworkSubgraph<int64_t, int64_t>::primal_repair_count)
        .def("count_empirical_nodes", &WassersteinNetworkSubgraph<int64_t, int64_t>::count_nodes_of_type<EmpiricalNode<int64_t>>)
        .def("count_theoretical_nodes", &WassersteinNetworkSubgraph<int64_t, int64_t>::count_nodes_of_type<TheoreticalNode<int64_t>>)
        .def("count_matching_edges", &WassersteinNetworkSubgraph<int64_t, int64_t>::count_edges_of_type<MatchingEdge>)
        .def("count_chain_edges", &WassersteinNetworkSubgraph<int64_t, int64_t>::count_edges_of_type<ChainEdge>)
        .def("count_src_to_empirical_edges", &WassersteinNetworkSubgraph<int64_t, int64_t>::count_edges_of_type<SrcToEmpiricalEdge>)
        .def("count_theoretical_to_sink_edges", &WassersteinNetworkSubgraph<int64_t, int64_t>::count_edges_of_type<TheoreticalToSinkEdge>)
        .def("count_simple_trash_edges", &WassersteinNetworkSubgraph<int64_t, int64_t>::count_edges_of_type<SimpleTrashEdge>)
        .def("matching_density", &WassersteinNetworkSubgraph<int64_t, int64_t>::matching_density)
        .def("theoretical_spectra_involved", &WassersteinNetworkSubgraph<int64_t, int64_t>::theoretical_spectra_involved);

        nb::class_<WassersteinNetworkSubgraph<int64_t, double>>(m, "CWassersteinNetworkSubgraphFloat")
        .def(nb::init<const std::vector<LEMON_INDEX>&, const std::vector<FlowNode<double>>&, const std::vector<FlowEdge<double>*>&, size_t>())
        .def("add_simple_trash", &WassersteinNetworkSubgraph<int64_t, double>::add_simple_trash)
        .def("add_experimental_trash", &WassersteinNetworkSubgraph<int64_t, double>::add_experimental_trash)
        .def("add_theoretical_trash", &WassersteinNetworkSubgraph<int64_t, double>::add_theoretical_trash)
        .def("build", &WassersteinNetworkSubgraph<int64_t, double>::build, nb::arg("config") = NetworkSimplexConfig{})
        .def("set_point", &WassersteinNetworkSubgraph<int64_t, double>::set_point)
        .def("total_cost", &WassersteinNetworkSubgraph<int64_t, double>::total_cost)
        .def("simple_trash_cost", &WassersteinNetworkSubgraph<int64_t, double>::simple_trash_cost)
        .def("to_string", &WassersteinNetworkSubgraph<int64_t, double>::to_string)
        .def("lemon_to_string", &WassersteinNetworkSubgraph<int64_t, double>::lemon_to_string)
        .def("no_nodes", &WassersteinNetworkSubgraph<int64_t, double>::no_nodes)
        .def("no_edges", &WassersteinNetworkSubgraph<int64_t, double>::no_edges)
        .def("get_nodes", &WassersteinNetworkSubgraph<int64_t, double>::get_nodes)
        .def("get_edges", &WassersteinNetworkSubgraph<int64_t, double>::get_edges)
        .def("get_flow_map", &WassersteinNetworkSubgraph<int64_t, double>::get_flow_map)
        .def("is_solved", &WassersteinNetworkSubgraph<int64_t, double>::is_solved)
        .def("signal_part_derivatives", &WassersteinNetworkSubgraph<int64_t, double>::signal_part_derivatives)
        .def("spectrum_proportion_derivatives", &WassersteinNetworkSubgraph<int64_t, double>::spectrum_proportion_derivatives)
        .def("signal_part_derivatives_fast_approx", &WassersteinNetworkSubgraph<int64_t, double>::signal_part_derivatives_fast_approx)
        .def("spectrum_proportion_derivatives_fast_approx", &WassersteinNetworkSubgraph<int64_t, double>::spectrum_proportion_derivatives_fast_approx)
        .def("warm_start_count", &WassersteinNetworkSubgraph<int64_t, double>::warm_start_count)
        .def("cold_start_count", &WassersteinNetworkSubgraph<int64_t, double>::cold_start_count)
        .def("dual_repair_count", &WassersteinNetworkSubgraph<int64_t, double>::dual_repair_count)
        .def("primal_repair_count", &WassersteinNetworkSubgraph<int64_t, double>::primal_repair_count)
        .def("count_empirical_nodes", &WassersteinNetworkSubgraph<int64_t, double>::count_nodes_of_type<EmpiricalNode<double>>)
        .def("count_theoretical_nodes", &WassersteinNetworkSubgraph<int64_t, double>::count_nodes_of_type<TheoreticalNode<double>>)
        .def("count_matching_edges", &WassersteinNetworkSubgraph<int64_t, double>::count_edges_of_type<MatchingEdge>)
        .def("count_chain_edges", &WassersteinNetworkSubgraph<int64_t, double>::count_edges_of_type<ChainEdge>)
        .def("count_src_to_empirical_edges", &WassersteinNetworkSubgraph<int64_t, double>::count_edges_of_type<SrcToEmpiricalEdge>)
        .def("count_theoretical_to_sink_edges", &WassersteinNetworkSubgraph<int64_t, double>::count_edges_of_type<TheoreticalToSinkEdge>)
        .def("count_simple_trash_edges", &WassersteinNetworkSubgraph<int64_t, double>::count_edges_of_type<SimpleTrashEdge>)
        .def("matching_density", &WassersteinNetworkSubgraph<int64_t, double>::matching_density)
        .def("theoretical_spectra_involved", &WassersteinNetworkSubgraph<int64_t, double>::theoretical_spectra_involved);

    // Networks whose per-dimension update_* overloads are appended by the
    // register_dim_<N>() functions. Build the non-templated base methods here,
    // then hand the class objects to the per-dim stubs.
    auto net_ii = nb::class_<WNetII>(m, "CWassersteinNetwork")
        //.def(nb::init<const Distribution<LEMON_INT>*, const std::vector<Distribution<LEMON_INT>*>&, const nb::callable, LEMON_INT>())
        .def("add_simple_trash", &WassersteinNetwork<int64_t, int64_t>::add_simple_trash)
        .def("add_experimental_trash", &WassersteinNetwork<int64_t, int64_t>::add_experimental_trash)
        .def("add_theoretical_trash", &WassersteinNetwork<int64_t, int64_t>::add_theoretical_trash)
        .def("build", &WassersteinNetwork<int64_t, int64_t>::build, nb::arg("config") = NetworkSimplexConfig{})
        .def("solve",
             [](WassersteinNetwork<int64_t, int64_t>& self) { self.solve(); })
        .def("solve",
             [](WassersteinNetwork<int64_t, int64_t>& self, const std::vector<double>& point) { self.solve(point); },
             nb::arg("point"))
        .def("total_cost", &WassersteinNetwork<int64_t, int64_t>::total_cost)
        .def("scale_factor", &WassersteinNetwork<int64_t, int64_t>::scale_factor)
        .def("intensity_scale_factor", &WassersteinNetwork<int64_t, int64_t>::intensity_scale_factor)
        .def("set_intensity_scale", &WassersteinNetwork<int64_t, int64_t>::set_intensity_scale, nb::arg("scale"))
        .def("get_subgraph", &WassersteinNetwork<int64_t, int64_t>::get_subgraph, nb::rv_policy::reference)
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
        .def("cold_start_count", &WassersteinNetwork<int64_t, int64_t>::cold_start_count)
        .def("dual_repair_count", &WassersteinNetwork<int64_t, int64_t>::dual_repair_count)
        .def("primal_repair_count", &WassersteinNetwork<int64_t, int64_t>::primal_repair_count);

    auto net_if = nb::class_<WassersteinNetwork<int64_t, double>>(m, "CWassersteinNetworkFloat")
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

    nb::class_<Distribution<LEMON_INT>>(m, "CDistribution")
        .def(nb::init<nb::ndarray<nb::shape<-1, -1>>, nb::ndarray<LEMON_INT, nb::shape<-1>>>(), nb::arg().noconvert(), nb::arg().noconvert())
        .def("size", &Distribution<LEMON_INT>::size)
        .def("get_positions", &Distribution<LEMON_INT>::get_positions)
        .def("get_intensities", &Distribution<LEMON_INT>::get_intensities)
        .def("get_point", &Distribution<LEMON_INT>::get_point)
        //.def("closer_than", &Distribution<LEMON_INT>::closer_than)
        .def("__len__", &Distribution<LEMON_INT>::size);

    nb::class_<Distribution<LEMON_INT>::Point_t>(m, "DistributionPoint")
        .def_ro("positions", &Distribution<LEMON_INT>::Point_t::first)
        .def_ro("index", &Distribution<LEMON_INT>::Point_t::second);

    // Factory whose per-dimension create() overloads are appended by the
    // register_dim_<N>() functions; create_1d is added afterwards (distinct
    // method name, order-independent).
    auto factory = nb::class_<WNetFactory>(m, "CWassersteinNetworkFactory");

    // Per-dimension instantiations live in dim_register_<N>.cpp (compiled in
    // parallel). The call set is gated on WNET_MAX_DIM to match the source files
    // CMake actually compiled.
#if WNET_MAX_DIM >= 1
    register_dim_1(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 2
    register_dim_2(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 3
    register_dim_3(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 4
    register_dim_4(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 5
    register_dim_5(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 6
    register_dim_6(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 7
    register_dim_7(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 8
    register_dim_8(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 9
    register_dim_9(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 10
    register_dim_10(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 11
    register_dim_11(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 12
    register_dim_12(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 13
    register_dim_13(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 14
    register_dim_14(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 15
    register_dim_15(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 16
    register_dim_16(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 17
    register_dim_17(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 18
    register_dim_18(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 19
    register_dim_19(m, net_ii, net_if, factory);
#endif
#if WNET_MAX_DIM >= 20
    register_dim_20(m, net_ii, net_if, factory);
#endif

    factory
        .def_static("create_1d", &WNetFactory::create_1d<LEMON_INT>, nb::arg("empirical_spectrum"), nb::arg("theoretical_spectra"), nb::arg("distance_metric"), nb::arg("max_dist"), nb::arg("p") = 1.0)
        .def_static("create_1d", &WNetFactory::create_1d<double>, nb::arg("empirical_spectrum"), nb::arg("theoretical_spectra"), nb::arg("distance_metric"), nb::arg("max_dist"), nb::arg("p") = 1.0);

/*
    m.def("WnetViaVectorDistribution", [](
        const Distribution<LEMON_INT>* empirical_dist,
        const std::vector<Distribution<LEMON_INT>*>& theoretical_dists,
        LEMON_INT max_distance)
    {
        VectorDistribution<2, double, LEMON_INT> empirical_vec_dist(*empirical_dist);
    }
);*/
    // Export SourceNode
    nb::class_<SourceNode>(m, "SourceNode");

    // Export SinkNode
    nb::class_<SinkNode>(m, "SinkNode");

    // Export EmpiricalNode
    nb::class_<EmpiricalNode<int64_t>>(m, "EmpiricalNode")
        .def(nb::init<LEMON_INDEX, int64_t>())
        .def("get_peak_index", &EmpiricalNode<int64_t>::get_peak_index)
        .def("get_intensity", &EmpiricalNode<int64_t>::get_intensity);

    // Export TheoreticalNode
    nb::class_<TheoreticalNode<int64_t>>(m, "TheoreticalNode")
        .def(nb::init<size_t, LEMON_INDEX, int64_t>())
        .def("get_spectrum_id", &TheoreticalNode<int64_t>::get_spectrum_id)
        .def("get_peak_index", &TheoreticalNode<int64_t>::get_peak_index)
        .def("get_intensity", &TheoreticalNode<int64_t>::get_intensity);

    // Export MatchingEdge
    nb::class_<MatchingEdge>(m, "MatchingEdge")
        .def(nb::init<double>())
        .def("get_cost", &MatchingEdge::get_cost);

    // Export SrcToEmpiricalEdge
    nb::class_<SrcToEmpiricalEdge>(m, "SrcToEmpiricalEdge");

    // Export TheoreticalToSinkEdge
    nb::class_<TheoreticalToSinkEdge>(m, "TheoreticalToSinkEdge");

    // Export SimpleTrashEdge
    nb::class_<SimpleTrashEdge>(m, "SimpleTrashEdge")
        .def(nb::init<double>())
        .def("get_cost", &SimpleTrashEdge::get_cost);

    // Export ChainEdge (1D chain-optimization adjacency edge).
    nb::class_<ChainEdge>(m, "ChainEdge")
        .def(nb::init<double>())
        .def("get_cost", &ChainEdge::get_cost);

    // Export EmpiricalTrashEdge (asymmetric trash: EmpiricalNode -> Sink).
    nb::class_<EmpiricalTrashEdge>(m, "EmpiricalTrashEdge")
        .def(nb::init<double>())
        .def("get_cost", &EmpiricalTrashEdge::get_cost);

    // Export TheoreticalTrashEdge (asymmetric trash: Source -> TheoreticalNode).
    nb::class_<TheoreticalTrashEdge>(m, "TheoreticalTrashEdge")
        .def(nb::init<double>())
        .def("get_cost", &TheoreticalTrashEdge::get_cost);
}
