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


#define EXPOSE_VECTOR_DISTRIBUTION(DIM) \
    using VectorDistribution_##DIM = VectorDistribution<DIM, double, LEMON_INT>; \
    nb::class_<VectorDistribution_##DIM>(m, "CVectorDistribution" #DIM) \
        .def(nb::init<const std::vector<std::array<double, DIM>>&, const std::vector<LEMON_INT>&>()) \
        .def(nb::init<std::vector<std::array<double, DIM>>&&,  std::vector<LEMON_INT>&&>()) \
        .def("size", &VectorDistribution_##DIM::size) \
        .def("get_positions", &VectorDistribution_##DIM::get_positions) \
        .def("get_intensities", &VectorDistribution_##DIM::get_intensities) \
        .def("get_point", &VectorDistribution_##DIM::get_point) \
        .def("closer_than", &VectorDistribution_##DIM::closer_than) \
        .def("__len__", &VectorDistribution_##DIM::size);

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

    nb::class_<WassersteinNetwork<int64_t>>(m, "CWassersteinNetwork")
        .def(nb::init<const Distribution<LEMON_INT>*, const std::vector<Distribution<LEMON_INT>*>&, const nb::callable, LEMON_INT>())
        .def("add_simple_trash", &WassersteinNetwork<int64_t>::add_simple_trash)
        .def("build", &WassersteinNetwork<int64_t>::build)
        .def("solve", nb::overload_cast<>(&WassersteinNetwork<int64_t>::solve))
        .def("solve", nb::overload_cast<const std::vector<double>&>(&WassersteinNetwork<int64_t>::solve))
        .def("total_cost", &WassersteinNetwork<int64_t>::total_cost)
        .def("get_subgraph", &WassersteinNetwork<int64_t>::get_subgraph, nb::rv_policy::reference)
        .def("__str__", &WassersteinNetwork<int64_t>::to_string)
        .def("lemon_to_string", &WassersteinNetwork<int64_t>::lemon_to_string)
        .def("no_subgraphs", &WassersteinNetwork<int64_t>::no_subgraphs)
        .def("lemon_to_string", &WassersteinNetwork<int64_t>::lemon_to_string)
        .def("flows_for_target", [](WassersteinNetwork<int64_t>& self, size_t target_id) {
            auto [empirical_peak_indices, theoretical_peak_indices, flows] = self.flows_for_target(target_id);
            return std::make_tuple(vector_to_numpy<LEMON_INDEX>(empirical_peak_indices),
                                   vector_to_numpy<LEMON_INDEX>(theoretical_peak_indices),
                                   vector_to_numpy<int64_t>(flows));
        }, nb::rv_policy::move)
        .def("count_empirical_nodes", &WassersteinNetwork<int64_t>::count_nodes_of_type<EmpiricalNode>)
        .def("count_theoretical_nodes", &WassersteinNetwork<int64_t>::count_nodes_of_type<TheoreticalNode>)
        .def("count_matching_edges", &WassersteinNetwork<int64_t>::count_edges_of_type<MatchingEdge>)
        .def("count_theoretical_to_sink_edges", &WassersteinNetwork<int64_t>::count_edges_of_type<TheoreticalToSinkEdge>)
        .def("count_src_to_empirical_edges", &WassersteinNetwork<int64_t>::count_edges_of_type<SrcToEmpiricalEdge>)
        .def("count_simple_trash_edges", &WassersteinNetwork<int64_t>::count_edges_of_type<SimpleTrashEdge>)
        .def("matching_density", &WassersteinNetwork<int64_t>::matching_density)
        .def_static("value_type_size", &WassersteinNetwork<int64_t>::value_type_size)
        .def_static("index_type_size", &WassersteinNetwork<int64_t>::index_type_size)
        .def_static("max_value", &WassersteinNetwork<int64_t>::max_value)
        .def_static("max_index", &WassersteinNetwork<int64_t>::max_index);

    nb::class_<Distribution<LEMON_INT>>(m, "CDistribution")
        .def(nb::init<nb::ndarray<nb::shape<-1, -1>>, nb::ndarray<LEMON_INT, nb::shape<-1>>>(), nb::arg().noconvert(), nb::arg().noconvert())
        .def("size", &Distribution<LEMON_INT>::size)
        .def("get_positions", &Distribution<LEMON_INT>::get_positions)
        .def("get_intensities", &Distribution<LEMON_INT>::get_intensities)
        .def("get_point", &Distribution<LEMON_INT>::get_point)
        .def("closer_than", &Distribution<LEMON_INT>::closer_than)
        .def("__len__", &Distribution<LEMON_INT>::size);

    nb::class_<Distribution<LEMON_INT>::Point_t>(m, "DistributionPoint")
        .def_ro("positions", &Distribution<LEMON_INT>::Point_t::first)
        .def_ro("index", &Distribution<LEMON_INT>::Point_t::second);


    EXPOSE_VECTOR_DISTRIBUTION(1)
    EXPOSE_VECTOR_DISTRIBUTION(2)
    EXPOSE_VECTOR_DISTRIBUTION(3)
    EXPOSE_VECTOR_DISTRIBUTION(4)
    EXPOSE_VECTOR_DISTRIBUTION(5)
    EXPOSE_VECTOR_DISTRIBUTION(6)
    EXPOSE_VECTOR_DISTRIBUTION(7)
    EXPOSE_VECTOR_DISTRIBUTION(8)
    EXPOSE_VECTOR_DISTRIBUTION(9)
    EXPOSE_VECTOR_DISTRIBUTION(10)
    EXPOSE_VECTOR_DISTRIBUTION(11)
    EXPOSE_VECTOR_DISTRIBUTION(12)
    EXPOSE_VECTOR_DISTRIBUTION(13)
    EXPOSE_VECTOR_DISTRIBUTION(14)
    EXPOSE_VECTOR_DISTRIBUTION(15)
    EXPOSE_VECTOR_DISTRIBUTION(16)
    EXPOSE_VECTOR_DISTRIBUTION(17)
    EXPOSE_VECTOR_DISTRIBUTION(18)
    EXPOSE_VECTOR_DISTRIBUTION(19)
    EXPOSE_VECTOR_DISTRIBUTION(20)

}