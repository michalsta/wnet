#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

#include "decompositable_graph.hpp"
#include "graph_elements.hpp"
#include "distribution.hpp"
#include "misc.hpp"


NB_MODULE(wnet_cpp, m) {
    m.doc() = "WNet C++ imlementation module";
    m.def("wnet_cpp_hello", []() {
        std::cout << "Hello from WNet (C++)!" << std::endl;
    }, "A simple hello world function for the WNet (C++) extension");
    // Bind the classes to the module

    nb::class_<FlowNode>(m, "FlowNode")
        .def(nb::init<LEMON_INDEX, SourceNode>())
        .def(nb::init<LEMON_INDEX, SinkNode>())
        .def(nb::init<LEMON_INDEX, EmpiricalNode>())
        .def(nb::init<LEMON_INDEX, TheoreticalNode>())
        .def("get_id", &FlowNode::get_id)
        .def("get_type", &FlowNode::get_type)
        .def("layer", &FlowNode::layer)
        .def("type_str", &FlowNode::type_str)
        .def("__str__", &FlowNode::to_string);

    nb::class_<FlowEdge>(m, "FlowEdge")
        .def(nb::init<LEMON_INDEX, const FlowNode&, const FlowNode&, FlowEdgeType>())
        .def("get_id", &FlowEdge::get_id)
        .def("get_start_node", &FlowEdge::get_start_node)
        .def("get_end_node", &FlowEdge::get_end_node)
        .def("get_start_node_id", &FlowEdge::get_start_node_id)
        .def("get_end_node_id", &FlowEdge::get_end_node_id)
        .def("get_type", &FlowEdge::get_type)
        .def("get_cost", &FlowEdge::get_cost)
        .def("get_base_capacity", &FlowEdge::get_base_capacity)
        .def("to_string", &FlowEdge::to_string);

    nb::class_<WassersteinNetworkSubgraph<int64_t>>(m, "CWassersteinNetworkSubgraph")
        .def(nb::init<const std::vector<LEMON_INDEX>&, const std::vector<FlowNode>&, const std::vector<FlowEdge*>&, size_t>())
        .def("add_simple_trash", &WassersteinNetworkSubgraph<int64_t>::add_simple_trash)
        .def("build", &WassersteinNetworkSubgraph<int64_t>::build)
        .def("set_point", &WassersteinNetworkSubgraph<int64_t>::set_point)
        .def("total_cost", &WassersteinNetworkSubgraph<int64_t>::total_cost)
        .def("to_string", &WassersteinNetworkSubgraph<int64_t>::to_string)
        .def("lemon_to_string", &WassersteinNetworkSubgraph<int64_t>::lemon_to_string)
        .def("no_nodes", &WassersteinNetworkSubgraph<int64_t>::no_nodes)
        .def("no_edges", &WassersteinNetworkSubgraph<int64_t>::no_edges)
        .def("get_nodes", &WassersteinNetworkSubgraph<int64_t>::get_nodes)
        .def("get_edges", &WassersteinNetworkSubgraph<int64_t>::get_edges);

    nb::class_<WassersteinNetwork<int64_t, Distribution>>(m, "CWassersteinNetwork")
        .def(nb::init<const Distribution*, const std::vector<Distribution*>&, const nb::callable, LEMON_INT>())
        .def("add_simple_trash", &WassersteinNetwork<int64_t, Distribution>::add_simple_trash)
        .def("build", &WassersteinNetwork<int64_t, Distribution>::build)
        .def("solve", nb::overload_cast<>(&WassersteinNetwork<int64_t, Distribution>::solve))
        .def("solve", nb::overload_cast<const std::vector<double>&>(&WassersteinNetwork<int64_t, Distribution>::solve))
        .def("total_cost", &WassersteinNetwork<int64_t, Distribution>::total_cost)
        .def("get_subgraph", &WassersteinNetwork<int64_t, Distribution>::get_subgraph, nb::rv_policy::reference)
        .def("__str__", &WassersteinNetwork<int64_t, Distribution>::to_string)
        .def("lemon_to_string", &WassersteinNetwork<int64_t, Distribution>::lemon_to_string)
        .def("no_subgraphs", &WassersteinNetwork<int64_t, Distribution>::no_subgraphs)
        .def("lemon_to_string", &WassersteinNetwork<int64_t, Distribution>::lemon_to_string)
        .def("flows_for_target", [](WassersteinNetwork<int64_t, Distribution>& self, size_t target_id) {
            auto [empirical_peak_indices, theoretical_peak_indices, flows] = self.flows_for_target(target_id);
            return std::make_tuple(vector_to_numpy<LEMON_INDEX>(empirical_peak_indices),
                                   vector_to_numpy<LEMON_INDEX>(theoretical_peak_indices),
                                   vector_to_numpy<int64_t>(flows));
        }, nb::rv_policy::move)
        .def("count_empirical_nodes", &WassersteinNetwork<int64_t, Distribution>::count_nodes_of_type<EmpiricalNode>)
        .def("count_theoretical_nodes", &WassersteinNetwork<int64_t, Distribution>::count_nodes_of_type<TheoreticalNode>)
        .def("count_matching_edges", &WassersteinNetwork<int64_t, Distribution>::count_edges_of_type<MatchingEdge>)
        .def("count_theoretical_to_sink_edges", &WassersteinNetwork<int64_t, Distribution>::count_edges_of_type<TheoreticalToSinkEdge>)
        .def("count_src_to_empirical_edges", &WassersteinNetwork<int64_t, Distribution>::count_edges_of_type<SrcToEmpiricalEdge>)
        .def("count_simple_trash_edges", &WassersteinNetwork<int64_t, Distribution>::count_edges_of_type<SimpleTrashEdge>)
        .def("matching_density", &WassersteinNetwork<int64_t, Distribution>::matching_density)
        .def_static("value_type_size", &WassersteinNetwork<int64_t, Distribution>::value_type_size)
        .def_static("index_type_size", &WassersteinNetwork<int64_t, Distribution>::index_type_size)
        .def_static("max_value", &WassersteinNetwork<int64_t, Distribution>::max_value)
        .def_static("max_index", &WassersteinNetwork<int64_t, Distribution>::max_index);

    nb::class_<Distribution>(m, "CDistribution")
        .def(nb::init<nb::ndarray<>, nb::ndarray<LEMON_INT, nb::shape<-1>>>())
        .def("size", &Distribution::size)
        .def("get_positions", &Distribution::get_positions)
        .def("get_intensities", &Distribution::get_intensities)
        .def("get_point", &Distribution::get_point)
        .def("closer_than", &Distribution::closer_than)
        .def("__len__", &Distribution::size);

    nb::class_<Distribution::Point_t>(m, "DistributionPoint")
        .def_ro("positions", &Distribution::Point_t::first)
        .def_ro("index", &Distribution::Point_t::second);
}