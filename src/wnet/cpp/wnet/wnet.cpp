#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "decompositable_graph.hpp"
#include "graph_elements.hpp"


NB_MODULE(wnetalign_cpp, m) {
    m.doc() = "WNet C++ imlementation module";

    // Define a simple function to demonstrate the module
    m.def("wnet_cpp_hello", []() {
        std::cout << "Hello from WNet (C++)!" << std::endl;
    }, "A simple hello world function for the WNet (C++) extension");
    // Bind the classes to the module
    nb::class_<FlowNode>(m, "FlowNode")
        .def(nb::init<LEMON_INT, SourceNode>())
        .def(nb::init<LEMON_INT, SinkNode>())
        .def(nb::init<LEMON_INT, EmpiricalNode>())
        .def(nb::init<LEMON_INT, TheoreticalNode>())
        .def("get_id", &FlowNode::get_id)
        .def("get_type", &FlowNode::get_type)
        .def("layer", &FlowNode::layer)
        .def("type_str", &FlowNode::type_str)
        .def("to_string", &FlowNode::to_string);
    nb::class_<FlowEdge>(m, "FlowEdge")
        .def(nb::init<LEMON_INT, const FlowNode&, const FlowNode&, FlowEdgeType>())
        .def("get_id", &FlowEdge::get_id)
        .def("get_start_node", &FlowEdge::get_start_node)
        .def("get_end_node", &FlowEdge::get_end_node)
        .def("get_start_node_id", &FlowEdge::get_start_node_id)
        .def("get_end_node_id", &FlowEdge::get_end_node_id)
        .def("get_type", &FlowEdge::get_type)
        .def("to_string", &FlowEdge::to_string);
    nb::class_<WassersteinNetworkSubgraph>(m, "WassersteinNetworkSubgraph")
        .def(nb::init<const std::vector<size_t>&, const std::vector<FlowNode>&, const std::vector<FlowEdge>&, size_t>())
        .def("add_simple_trash", &WassersteinNetworkSubgraph::add_simple_trash)
        .def("build", &WassersteinNetworkSubgraph::build)
        .def("set_point", &WassersteinNetworkSubgraph::set_point)
        .def("total_cost", &WassersteinNetworkSubgraph::total_cost)
        .def("to_string", &WassersteinNetworkSubgraph::to_string)
        .def("lemon_to_string", &WassersteinNetworkSubgraph::lemon_to_string)
        .def("no_nodes", &WassersteinNetworkSubgraph::no_nodes)
        .def("no_edges", &WassersteinNetworkSubgraph::no_edges)
        .def("get_nodes", &WassersteinNetworkSubgraph::get_nodes)
        .def("get_edges", &WassersteinNetworkSubgraph::get_edges);
    nb::class_<WassersteinNetwork>(m, "WassersteinNetwork")
        .def(nb::init<const Distribution*, const std::vector<Distribution*>&, const nb::callable*, LEMON_INT>())
        .def("add_simple_trash", &WassersteinNetwork::add_simple_trash)
        .def("build", &WassersteinNetwork::build)
        .def("set_point", &WassersteinNetwork::set_point)
        .def("total_cost", &WassersteinNetwork::total_cost)
        .def("no_subgraphs", &WassersteinNetwork::no_subgraphs)
        .def("get_subgraph", &WassersteinNetwork::get_subgraph, nb::rv_policy::reference)
        .def("to_string", &WassersteinNetwork::to_string)
        .def("lemon_to_string", &WassersteinNetwork::lemon_to_string)
        .def("flows_for_spectrum", &WassersteinNetwork::flows_for_spectrum);
    nb::class_<Distribution>(m, "Distribution")
        .def(nb::init<nb::ndarray<>, nb::ndarray<LEMON_INT, nb::shape<-1>>>())
        .def("size", &Distribution::size)
        .def("get_point", &Distribution::get_point)
        .def("closer_than", &Distribution::closer_than);
}