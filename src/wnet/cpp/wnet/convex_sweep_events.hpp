#ifndef WNET_CONVEX_SWEEP_EVENTS_HPP
#define WNET_CONVEX_SWEEP_EVENTS_HPP

#include "convex_sweep.hpp"

namespace convex_sweep {

// Aggregate already-quantized supplies, never floating intensities. Mapping
// storage is absent from the cost-only instantiation. Opposite sides stay
// separate even at identical coordinates.
template <bool Mapping> struct GroupedEvents {
    struct Part {
        size_t index;
        i64 count;
    };
    std::vector<Event> events;
    std::conditional_t<Mapping, std::vector<std::vector<Part>>, NoTraceStorage> parts;

    void add(double pos, int side, i64 count, size_t index) {
        size_t found = events.size();
        for (size_t j = events.size(); j > 0 && events[j - 1].pos == pos; --j)
            if (events[j - 1].side == side) {
                found = j - 1;
                break;
            }
        if (found == events.size()) {
            events.push_back({pos, side, count});
            if constexpr (Mapping)
                parts.push_back({{index, count}});
        } else {
            events[found].count += count;
            if constexpr (Mapping)
                parts[found].push_back({index, count});
        }
    }

    void finish() {
        // The adjoint composes an extra theoretical unit matched leftward;
        // its mirrored pass covers strictly rightward matches. Equal-position
        // empirical supply MUST precede theoretical supply in BOTH passes.
        // Reversing arbitrary original peak order does not guarantee this.
        for (size_t i = 1; i < events.size(); ++i)
            if (events[i - 1].pos == events[i].pos && events[i - 1].side == 1 &&
                events[i].side == 0) {
                std::swap(events[i - 1], events[i]);
                if constexpr (Mapping)
                    std::swap(parts[i - 1], parts[i]);
            }
        for (size_t i = 0; i < events.size(); ++i)
            events[i].index = i;
    }

    std::vector<Match> expand(const std::vector<Match> &matches) const
        requires Mapping
    {
        std::vector<Match> out;
        std::vector<size_t> cursor(parts.size());
        std::vector<i64> used(parts.size());
        for (const auto &m : matches) {
            i64 left = m.count;
            while (left) {
                auto get = [&](size_t id) -> const Part & {
                    while (cursor[id] < parts[id].size() &&
                           used[id] == parts[id][cursor[id]].count) {
                        ++cursor[id];
                        used[id] = 0;
                    }
                    if (cursor[id] == parts[id].size())
                        throw std::logic_error("Grouped sweep traceback exceeds supply");
                    return parts[id][cursor[id]];
                };
                const auto &e = get(m.empirical);
                const auto &t = get(m.theoretical);
                const i64 take =
                    std::min({left, e.count - used[m.empirical], t.count - used[m.theoretical]});
                out.push_back({e.index, t.index, take});
                used[m.empirical] += take;
                used[m.theoretical] += take;
                left -= take;
            }
        }
        return out;
    }
};
} // namespace convex_sweep
#endif
