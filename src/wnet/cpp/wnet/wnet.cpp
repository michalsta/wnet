#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/optional.h>
#include <nanobind/ndarray.h>

#include "decompositable_graph.hpp"
#include "graph_elements.hpp"
#include "distribution.hpp"
#include "misc.hpp"


#define EXPOSE_VECTOR_DISTRIBUTION(DIM) \
    using VectorDistribution_##DIM = VectorDistribution<DIM, double, LEMON_INT>; \
    nb::class_<VectorDistribution_##DIM>(m, "CVectorDistribution" #DIM) \
        /*.def(nb::init<const std::vector<std::array<double, DIM>>&, const std::vector<LEMON_INT>&>()) \
        .def(nb::init<std::vector<std::array<double, DIM>>&&,  std::vector<LEMON_INT>&&>()) */ \
        .def(nb::init<const nb::ndarray<double, nb::shape<DIM, -1>>&, const nb::ndarray<LEMON_INT, nb::shape<-1>>&>()) \
        .def("size", &VectorDistribution_##DIM::size) \
        .def("py_get_positions", &VectorDistribution_##DIM::py_get_positions) \
        .def("py_get_intensities", &VectorDistribution_##DIM::py_get_intensities) \
        .def("get_point", &VectorDistribution_##DIM::get_point) \
        .def("__len__", &VectorDistribution_##DIM::size); \
    using VectorDistributionFloat_##DIM = VectorDistribution<DIM, double, double>; \
    nb::class_<VectorDistributionFloat_##DIM>(m, "CVectorDistributionFloat" #DIM) \
        /*.def(nb::init<const std::vector<std::array<double, DIM>>&, const std::vector<double>&>()) \
        .def(nb::init<std::vector<std::array<double, DIM>>&&,  std::vector<double>&&>()) */ \
        .def(nb::init<const nb::ndarray<double, nb::shape<DIM, -1>>&, const nb::ndarray<double, nb::shape<-1>>&>()) \
        .def("size", &VectorDistributionFloat_##DIM::size) \
        .def("py_get_positions", &VectorDistributionFloat_##DIM::py_get_positions) \
        .def("py_get_intensities", &VectorDistributionFloat_##DIM::py_get_intensities) \
        .def("get_point", &VectorDistributionFloat_##DIM::get_point) \
        .def("__len__", &VectorDistributionFloat_##DIM::size);

NB_MODULE(wnet_cpp, m) {
    m.doc() = "WNet C++ imlementation module";
    m.def("wnet_cpp_hello", []() {
        std::cout << "Hello from WNet (C++)!" << std::endl;
    }, "A simple hello world function for the WNet (C++) extension");
    // Bind the classes to the module

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

    nb::class_<WassersteinNetworkSubgraph<int64_t, int64_t>>(m, "CWassersteinNetworkSubgraph")
        .def(nb::init<const std::vector<LEMON_INDEX>&, const std::vector<FlowNode<int64_t>>&, const std::vector<FlowEdge<int64_t>*>&, size_t>())
        .def("add_simple_trash", &WassersteinNetworkSubgraph<int64_t, int64_t>::add_simple_trash)
        .def("build", &WassersteinNetworkSubgraph<int64_t, int64_t>::build)
        .def("set_point", &WassersteinNetworkSubgraph<int64_t, int64_t>::set_point)
        .def("total_cost", &WassersteinNetworkSubgraph<int64_t, int64_t>::total_cost)
        .def("to_string", &WassersteinNetworkSubgraph<int64_t, int64_t>::to_string)
        .def("lemon_to_string", &WassersteinNetworkSubgraph<int64_t, int64_t>::lemon_to_string)
        .def("no_nodes", &WassersteinNetworkSubgraph<int64_t, int64_t>::no_nodes)
        .def("no_edges", &WassersteinNetworkSubgraph<int64_t, int64_t>::no_edges)
        .def("get_nodes", &WassersteinNetworkSubgraph<int64_t, int64_t>::get_nodes)
        .def("get_edges", &WassersteinNetworkSubgraph<int64_t, int64_t>::get_edges);


        nb::class_<WassersteinNetworkSubgraph<int64_t, double>>(m, "CWassersteinNetworkSubgraphFloat")
        .def(nb::init<const std::vector<LEMON_INDEX>&, const std::vector<FlowNode<double>>&, const std::vector<FlowEdge<double>*>&, size_t>())
        .def("add_simple_trash", &WassersteinNetworkSubgraph<int64_t, double>::add_simple_trash)
        .def("build", &WassersteinNetworkSubgraph<int64_t, double>::build)
        .def("set_point", &WassersteinNetworkSubgraph<int64_t, double>::set_point)
        .def("total_cost", &WassersteinNetworkSubgraph<int64_t, double>::total_cost)
        .def("to_string", &WassersteinNetworkSubgraph<int64_t, double>::to_string)
        .def("lemon_to_string", &WassersteinNetworkSubgraph<int64_t, double>::lemon_to_string)
        .def("no_nodes", &WassersteinNetworkSubgraph<int64_t, double>::no_nodes)
        .def("no_edges", &WassersteinNetworkSubgraph<int64_t, double>::no_edges)
        .def("get_nodes", &WassersteinNetworkSubgraph<int64_t, double>::get_nodes)
        .def("get_edges", &WassersteinNetworkSubgraph<int64_t, double>::get_edges);

    nb::class_<WassersteinNetwork<int64_t, int64_t>>(m, "CWassersteinNetwork")
        //.def(nb::init<const Distribution<LEMON_INT>*, const std::vector<Distribution<LEMON_INT>*>&, const nb::callable, LEMON_INT>())
        .def("add_simple_trash", &WassersteinNetwork<int64_t, int64_t>::add_simple_trash)
        .def("build", &WassersteinNetwork<int64_t, int64_t>::build)
        .def("solve", nb::overload_cast<>(&WassersteinNetwork<int64_t, int64_t>::solve))
        .def("solve", nb::overload_cast<const std::vector<double>&>(&WassersteinNetwork<int64_t, int64_t>::solve))
        .def("total_cost", &WassersteinNetwork<int64_t, int64_t>::total_cost)
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
        .def("matching_density", &WassersteinNetwork<int64_t, int64_t>::matching_density)
        .def_static("value_type_size", &WassersteinNetwork<int64_t, int64_t>::value_type_size)
        .def_static("index_type_size", &WassersteinNetwork<int64_t, int64_t>::index_type_size)
        .def_static("max_value", &WassersteinNetwork<int64_t, int64_t>::max_value)
        .def_static("max_index", &WassersteinNetwork<int64_t, int64_t>::max_index);

    nb::class_<WassersteinNetwork<int64_t, double>>(m, "CWassersteinNetworkFloat")
        //.def(nb::init<const Distribution<LEMON_INT>*, const std::vector<Distribution<LEMON_INT>*>&, const nb::callable, LEMON_INT>())
        .def("add_simple_trash", &WassersteinNetwork<int64_t, double>::add_simple_trash)
        .def("build", &WassersteinNetwork<int64_t, double>::build)
        .def("solve", nb::overload_cast<>(&WassersteinNetwork<int64_t, double>::solve))
        .def("solve", nb::overload_cast<const std::vector<double>&>(&WassersteinNetwork<int64_t, double>::solve))
        .def("total_cost", &WassersteinNetwork<int64_t, double>::total_cost)
        .def("get_subgraph", &WassersteinNetwork<int64_t, double>::get_subgraph, nb::rv_policy::reference)
        .def("__str__", &WassersteinNetwork<int64_t, double>::to_string)
        .def("lemon_to_string", &WassersteinNetwork<int64_t, double>::lemon_to_string)
        .def("no_subgraphs", &WassersteinNetwork<int64_t, double>::no_subgraphs)
        .def("lemon_to_string", &WassersteinNetwork<int64_t, double>::lemon_to_string)
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
        .def("matching_density", &WassersteinNetwork<int64_t, double>::matching_density)
        .def_static("value_type_size", &WassersteinNetwork<int64_t, double>::value_type_size)
        .def_static("index_type_size", &WassersteinNetwork<int64_t, double>::index_type_size)
        .def_static("max_value", &WassersteinNetwork<int64_t, double>::max_value)
        .def_static("max_index", &WassersteinNetwork<int64_t, double>::max_index);

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

    nb::class_<WassersteinNetworkFactory<int64_t>>(m, "CWassersteinNetworkFactory")
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<1, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<2, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<3, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<4, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<5, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<6, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<7, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<8, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<9, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<10, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<11, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<12, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<13, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<14, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<15, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<16, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<17, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<18, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<19, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<20, double, LEMON_INT>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<1, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<2, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<3, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<4, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<5, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<6, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<7, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<8, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<9, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<10, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<11, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<12, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<13, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<14, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<15, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<16, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<17, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<18, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<19, double, double>>)
        .def_static("create", &WassersteinNetworkFactory<int64_t>::create<VectorDistribution<20, double, double>>);

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
/*
    m.def("WnetViaVectorDistribution", [](
        const Distribution<LEMON_INT>* empirical_dist,
        const std::vector<Distribution<LEMON_INT>*>& theoretical_dists,
        LEMON_INT max_distance)
    {
        VectorDistribution<2, double, LEMON_INT> empirical_vec_dist(*empirical_dist);
    }
);*/
    nb::enum_<DistanceMetric>(m, "DistanceMetric")
        .value("L1", DistanceMetric::L1)
        .value("L2", DistanceMetric::L2)
        .value("LINF", DistanceMetric::LINF);
}