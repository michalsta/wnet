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
class EmpiricalNode {
    const LEMON_INDEX peak_index;
    const LEMON_INT intensity;
public:
    EmpiricalNode() = delete;
    EmpiricalNode(LEMON_INDEX peak_index, LEMON_INT intensity)
        : peak_index(peak_index), intensity(intensity) {}
    LEMON_INDEX get_peak_index() const { return peak_index; }
    LEMON_INT get_intensity() const { return intensity; }
};

class TheoreticalNode {
    const size_t spectrum_id;
    const LEMON_INDEX peak_index;
    const LEMON_INT intensity;
public:
    TheoreticalNode() = delete;
    TheoreticalNode(size_t spectrum_id, LEMON_INDEX peak_index, LEMON_INT intensity)
        : spectrum_id(spectrum_id), peak_index(peak_index), intensity(intensity) {}
    size_t get_spectrum_id() const { return spectrum_id; }
    LEMON_INDEX get_peak_index() const { return peak_index; }
    LEMON_INT get_intensity() const { return intensity; }
};

using FlowNodeType = std::variant<SourceNode, SinkNode, EmpiricalNode, TheoreticalNode>;

class FlowNode {
    const LEMON_INDEX id;
    const FlowNodeType type;
public:
    FlowNode() = delete;
    FlowNode(LEMON_INDEX id, SourceNode n) : id(id), type(n) {};
    FlowNode(LEMON_INDEX id, SinkNode n) : id(id), type(n) {};
    FlowNode(LEMON_INDEX id, EmpiricalNode n) : id(id), type(n) {};
    FlowNode(LEMON_INDEX id, TheoreticalNode n) : id(id), type(n) {};
    FlowNode(LEMON_INDEX id, FlowNodeType n) : id(id), type(n) {};
    LEMON_INDEX get_id() const { return id; };
    const FlowNodeType& get_type() const { return type; };
    size_t layer() const {
        if (std::holds_alternative<SourceNode>(type)) return 0;
        if (std::holds_alternative<SinkNode>(type)) return 3;
        if (std::holds_alternative<EmpiricalNode>(type)) return 1;
        if (std::holds_alternative<TheoreticalNode>(type)) return 2;
        throw std::runtime_error("Invalid FlowNode type");
    };

    std::string type_str() const {
        return std::visit([](const auto& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, SourceNode>) {
                return "SourceNode";
            } else if constexpr (std::is_same_v<T, SinkNode>) {
                return "SinkNode";
            } else if constexpr (std::is_same_v<T, EmpiricalNode>) {
                return "EmpiricalNode";
            } else if constexpr (std::is_same_v<T, TheoreticalNode>) {
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
            else if constexpr (std::is_same_v<T, EmpiricalNode>) {
                result += ", peak_idx: " + std::to_string(arg.get_peak_index()) + ", intensity: " + std::to_string(arg.get_intensity());
            } else if constexpr (std::is_same_v<T, TheoreticalNode>) {
                result += ", spectrum_id: " + std::to_string(arg.get_spectrum_id()) + ", peak_idx: " + std::to_string(arg.get_peak_index()) + ", intensity: " + std::to_string(arg.get_intensity());
            }
        }, type);
        result += ")";
        return result;
    };
};



class MatchingEdge
{
    const LEMON_INT cost;
public:
    MatchingEdge() = delete;
    MatchingEdge(LEMON_INT cost)
        : cost(cost) {}
    LEMON_INT get_cost() const { return cost; }
};

class SrcToEmpiricalEdge {};
class TheoreticalToSinkEdge {};
class SimpleTrashEdge {
    const LEMON_INT cost;
public:
    SimpleTrashEdge() = delete;
    SimpleTrashEdge(LEMON_INT cost)
        : cost(cost) {}
    LEMON_INT get_cost() const { return cost; }
};

using FlowEdgeType = std::variant<MatchingEdge, SrcToEmpiricalEdge, TheoreticalToSinkEdge, SimpleTrashEdge>;

class FlowEdge {
    const LEMON_INDEX id;
    const FlowNode& start_node;
    const FlowNode& end_node;
    const FlowEdgeType type;
public:
    FlowEdge() = delete;
    FlowEdge(LEMON_INDEX id, const FlowNode& start_node, const FlowNode& end_node, FlowEdgeType type)
        : id(id), start_node(start_node), end_node(end_node), type(type) {}
    LEMON_INDEX get_id() const { return id; }
    const FlowNode& get_start_node() const { return start_node; }
    const FlowNode& get_end_node() const { return end_node; }
    LEMON_INDEX get_start_node_id() const { return start_node.get_id(); }
    LEMON_INDEX get_end_node_id() const { return end_node.get_id(); }
    const FlowEdgeType& get_type() const { return type; }

    std::string to_string() const {
        std::string result = "FlowEdge(" + std::to_string(id) + ", " + start_node.to_string() + ", " + end_node.to_string() + ")";
        return result;
    };

    LEMON_INT get_cost() const {
        return std::visit([](const auto& arg) -> LEMON_INT {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, MatchingEdge>) {
                return arg.get_cost();
            } else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) {
                return 0;
            } else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) {
                return 0;
            } else if constexpr (std::is_same_v<T, SimpleTrashEdge>) {
                return arg.get_cost();
            } else {
                throw std::runtime_error("Invalid FlowEdge type");
            }
        }, type);
    };

    std::optional<LEMON_INT> get_base_capacity() const {
        return std::visit([&](const auto& arg) -> std::optional<LEMON_INT> {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, MatchingEdge>) {
                return std::nullopt; // Unlimited capacity
            } else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) {
                return std::get<EmpiricalNode>(this->get_end_node().get_type()).get_intensity();
            } else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) {
                return std::get<TheoreticalNode>(this->get_start_node().get_type()).get_intensity();
            } else if constexpr (std::is_same_v<T, SimpleTrashEdge>) {
                return std::nullopt; // Unlimited capacity
            } else {
                throw std::runtime_error("Invalid FlowEdge type");
            }
        }, type);
    };
};

#endif // wNET_GRAPH_ELEMENTS_HPP
