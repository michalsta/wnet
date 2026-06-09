#ifndef WNET_GRAPH_ELEMENTS_HPP
#define WNET_GRAPH_ELEMENTS_HPP

#include <iostream>
#include <vector>
#include <span>
#include <algorithm>
#include <stdexcept>
#include <variant>

#include <pylmcf/basics.hpp>

class SourceNode {};
class SinkNode {};

template<typename intensity_type>
class EmpiricalNode {
    const LEMON_INDEX peak_index;
    const intensity_type intensity;
public:
    EmpiricalNode() = delete;
    EmpiricalNode(LEMON_INDEX peak_index, intensity_type intensity)
        : peak_index(peak_index), intensity(intensity) {}
    LEMON_INDEX get_peak_index() const { return peak_index; }
    intensity_type get_intensity() const { return intensity; }
};

template<typename intensity_type>
class TheoreticalNode {
    const size_t spectrum_id;
    const LEMON_INDEX peak_index;
    const intensity_type intensity;
public:
    TheoreticalNode() = delete;
    TheoreticalNode(size_t spectrum_id, LEMON_INDEX peak_index, intensity_type intensity)
        : spectrum_id(spectrum_id), peak_index(peak_index), intensity(intensity) {}
    size_t get_spectrum_id() const { return spectrum_id; }
    LEMON_INDEX get_peak_index() const { return peak_index; }
    intensity_type get_intensity() const { return intensity; }
};

template<typename intensity_type>
using FlowNodeType = std::variant<SourceNode, SinkNode, EmpiricalNode<intensity_type>, TheoreticalNode<intensity_type>>;

template<typename intensity_type>
class FlowNode {
    const LEMON_INDEX id;
    const FlowNodeType<intensity_type> type;
public:
    FlowNode() = delete;
    FlowNode(LEMON_INDEX id, SourceNode n) : id(id), type(n) {};
    FlowNode(LEMON_INDEX id, SinkNode n) : id(id), type(n) {};
    FlowNode(LEMON_INDEX id, EmpiricalNode<intensity_type> n) : id(id), type(n) {};
    FlowNode(LEMON_INDEX id, TheoreticalNode<intensity_type> n) : id(id), type(n) {};
    FlowNode(LEMON_INDEX id, FlowNodeType<intensity_type> n) : id(id), type(n) {};
    LEMON_INDEX get_id() const { return id; };
    const FlowNodeType<intensity_type>& get_type() const { return type; };
    size_t layer() const {
        if (std::holds_alternative<SourceNode>(type)) return 0;
        if (std::holds_alternative<SinkNode>(type)) return 3;
        if (std::holds_alternative<EmpiricalNode<intensity_type>>(type)) return 1;
        if (std::holds_alternative<TheoreticalNode<intensity_type>>(type)) return 2;
        throw std::runtime_error("Invalid FlowNode type");
    };

    std::string type_str() const {
        return std::visit([](const auto& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, SourceNode>) {
                return "SourceNode";
            } else if constexpr (std::is_same_v<T, SinkNode>) {
                return "SinkNode";
            } else if constexpr (std::is_same_v<T, EmpiricalNode<intensity_type>>) {
                return "EmpiricalNode";
            } else if constexpr (std::is_same_v<T, TheoreticalNode<intensity_type>>) {
                return "TheoreticalNode";
            }
        }, type);
    };

    std::string to_string() const {
        std::string result = type_str() + "(" + std::to_string(id);
        std::visit([&result](const auto& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, SourceNode>) { }
            else if constexpr (std::is_same_v<T, SinkNode>) { }
            else if constexpr (std::is_same_v<T, EmpiricalNode<intensity_type>>) {
                result += ", peak_idx: " + std::to_string(arg.get_peak_index()) + ", intensity: " + std::to_string(arg.get_intensity());
            } else if constexpr (std::is_same_v<T, TheoreticalNode<intensity_type>>) {
                result += ", spectrum_id: " + std::to_string(arg.get_spectrum_id()) + ", peak_idx: " + std::to_string(arg.get_peak_index()) + ", intensity: " + std::to_string(arg.get_intensity());
            }
        }, type);
        result += ")";
        return result;
    };
};


// Cost-bearing edges hold the *real* (unscaled, possibly fractional) cost as a
// double.  The integer cost handed to the LEMON solver is produced later, at
// build time, by quantize_cost() (round(scale * real) for p != 1, truncate for
// p == 1).  See decompositable_graph.hpp.
class MatchingEdge
{
    const double cost;
public:
    MatchingEdge() = delete;
    MatchingEdge(double cost)
        : cost(cost) {}
    double get_cost() const { return cost; }
};

class SrcToEmpiricalEdge {};
class TheoreticalToSinkEdge {};
class SimpleTrashEdge {
    const double cost;
public:
    SimpleTrashEdge() = delete;
    SimpleTrashEdge(double cost)
        : cost(cost) {}
    double get_cost() const { return cost; }
};

// Adjacency edge used by the 1D chain optimization: cost equals the gap
// between two adjacent sorted peak positions. Unlike MatchingEdge, it does
// not correspond to a specific (empirical, theoretical) pair; it is used
// as part of a chain through which flow is routed, with accumulated gap-cost
// equal to the metric distance in 1D.
class ChainEdge {
    const double cost;
public:
    ChainEdge() = delete;
    ChainEdge(double cost)
        : cost(cost) {}
    double get_cost() const { return cost; }
};

// Asymmetric trash: EmpiricalNode → Sink, skipping the theoretical layer.
// Allows empirical signal to be discarded at a per-unit cost without needing
// a matching theoretical peak. Exclusive with SimpleTrashEdge.
class EmpiricalTrashEdge {
    const double cost;
public:
    EmpiricalTrashEdge() = delete;
    EmpiricalTrashEdge(double cost) : cost(cost) {}
    double get_cost() const { return cost; }
};

// Asymmetric trash: Source → TheoreticalNode, skipping the empirical layer.
// Allows theoretical capacity to be phantom-filled at a per-unit cost without
// a matching empirical peak. Exclusive with SimpleTrashEdge.
class TheoreticalTrashEdge {
    const double cost;
public:
    TheoreticalTrashEdge() = delete;
    TheoreticalTrashEdge(double cost) : cost(cost) {}
    double get_cost() const { return cost; }
};

using FlowEdgeType = std::variant<MatchingEdge, SrcToEmpiricalEdge, TheoreticalToSinkEdge, SimpleTrashEdge, ChainEdge, EmpiricalTrashEdge, TheoreticalTrashEdge>;

template<typename intensity_type>
class FlowEdge {
    const LEMON_INDEX id;
    const FlowNode<intensity_type>& start_node;
    const FlowNode<intensity_type>& end_node;
    const FlowEdgeType type;
public:
    FlowEdge() = delete;
    FlowEdge(LEMON_INDEX id, const FlowNode<intensity_type>& start_node, const FlowNode<intensity_type>& end_node, FlowEdgeType type)
        : id(id), start_node(start_node), end_node(end_node), type(type) {}
    LEMON_INDEX get_id() const { return id; }
    const FlowNode<intensity_type>& get_start_node() const { return start_node; }
    const FlowNode<intensity_type>& get_end_node() const { return end_node; }
    LEMON_INDEX get_start_node_id() const { return start_node.get_id(); }
    LEMON_INDEX get_end_node_id() const { return end_node.get_id(); }
    const FlowEdgeType& get_type() const { return type; }

    std::string to_string() const {
        std::string result = "FlowEdge(" + std::to_string(id) + ", " + start_node.to_string() + ", " + end_node.to_string() + ")";
        return result;
    };

    // Real (unscaled) cost.  Quantised to the solver's integer cost at build.
    double get_cost() const {
        return std::visit([](const auto& arg) -> double {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, MatchingEdge>) {
                return arg.get_cost();
            } else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) {
                return 0.0;
            } else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) {
                return 0.0;
            } else if constexpr (std::is_same_v<T, SimpleTrashEdge>) {
                return arg.get_cost();
            } else if constexpr (std::is_same_v<T, ChainEdge>) {
                return arg.get_cost();
            } else if constexpr (std::is_same_v<T, EmpiricalTrashEdge>) {
                return arg.get_cost();
            } else if constexpr (std::is_same_v<T, TheoreticalTrashEdge>) {
                return arg.get_cost();
            } else {
                throw std::runtime_error("Invalid FlowEdge type");
            }
        }, type);
    };

    std::optional<intensity_type> get_base_capacity() const {
        return std::visit([&](const auto& arg) -> std::optional<intensity_type> {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, MatchingEdge>) {
                return std::nullopt; // Unlimited capacity
            } else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) {
                return std::get<EmpiricalNode<intensity_type>>(this->get_end_node().get_type()).get_intensity();
            } else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) {
                return std::get<TheoreticalNode<intensity_type>>(this->get_start_node().get_type()).get_intensity();
            } else if constexpr (std::is_same_v<T, SimpleTrashEdge>) {
                return std::nullopt; // Unlimited capacity
            } else if constexpr (std::is_same_v<T, ChainEdge>) {
                return std::nullopt; // Unlimited capacity
            } else if constexpr (std::is_same_v<T, EmpiricalTrashEdge>) {
                return std::get<EmpiricalNode<intensity_type>>(this->get_start_node().get_type()).get_intensity();
            } else if constexpr (std::is_same_v<T, TheoreticalTrashEdge>) {
                return std::nullopt; // Effectively unbounded; TheoreticalToSinkEdge is the binding cap
            } else {
                throw std::runtime_error("Invalid FlowEdge type");
            }
        }, type);
    };
};

#endif // wNET_GRAPH_ELEMENTS_HPP
