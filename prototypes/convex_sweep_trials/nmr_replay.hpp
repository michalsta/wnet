#ifndef WNET_NMR_TRIAL_REPLAY_HPP
#define WNET_NMR_TRIAL_REPLAY_HPP
#include "nmr_common.hpp"
#include "nmr_kernel.hpp"
namespace nmr_trials {
namespace k = nmr_kernel;
struct Profit {
    const monge_cost::GridPower *model;
    I tau;
    sweep_trials::MomentBank *bank = nullptr;
    I operator()(double a, double b) const { return tau - model->cost(a, b); }
};
template <class Policy> struct Engine {
    using F = k::PLF;
    k::BasicSideState<F> sides[2];
    I end[2] = {0, 0};
    std::unique_ptr<sweep_trials::MomentBank> bank;
    Profit profit;
    double radius;
    Engine(const Input &in, const std::vector<k::Event> &ev)
        : profit{&in.model, in.tau}, radius(in.model.radius(in.tau)) {
        if constexpr (std::is_base_of_v<k::QuadraticPolicy, Policy>) {
            if (in.p != 2)
                throw std::invalid_argument("quadratic requires p=2");
            bank = std::make_unique<sweep_trials::MomentBank>(ev);
            profit.bank = bank.get();
        }
    }
    void prune(double z) {
        for (auto &st : sides) {
            I drop = 0;
            while (!st.blocks.empty() && z - st.blocks.front().first > radius) {
                drop += st.blocks.front().second;
                st.blocks.pop_front();
            }
            if (drop) {
                st.V.suffix_max();
                st.V.truncate(std::max<I>(0, st.V.K() - drop));
            }
        }
    }
    template <bool Trace> void step(const k::Event &e, k::BasicTraceRecord<F> *record = nullptr) {
        if (record) {
            record->end[0] = end[0];
            record->end[1] = end[1];
        }
        if (bank) {
            bank->own = e.side;
            bank->end[e.side] += e.count;
        }
        k::sweep_step<Trace, Policy>(sides[e.side], sides[1 - e.side], e.pos, e.count, profit,
                                     radius, record, e.side);
        end[e.side] += e.count;
    }
    I best() const { return std::max<I>({0, sides[0].V.max_value(), sides[1].V.max_value()}); }
    struct Save {
        size_t event;
        k::BasicSideState<F> sides[2];
        I end[2];
    };
    Save save(size_t i) const { return {i, {sides[0], sides[1]}, {end[0], end[1]}}; }
    void restore(const Save &s) {
        sides[0] = s.sides[0];
        sides[1] = s.sides[1];
        end[0] = s.end[0];
        end[1] = s.end[1];
        if (bank) {
            bank->end[0] = end[0];
            bank->end[1] = end[1];
        }
    }
};
inline void reverse_trace(const std::vector<k::Event> &events,
                          const std::vector<k::TraceRecord> &records, const k::TraceStorage &trace,
                          k::TraceResult &result, int &side, I &pending, I &value) {
    using namespace nmr_kernel;
    using F = k::PLF;
    auto emit = [&](const Event &event, size_t current, int opposite, i64 from, i64 to) {
        const auto &blocks = trace.blocks[opposite];
        auto it = std::upper_bound(blocks.begin(), blocks.end(), from,
                                   [](i64 rank, const typename BasicTraceStorage<F>::Block &block) {
                                       return rank < block.end;
                                   });
        while (from < to) {
            if (it == blocks.end())
                throw std::runtime_error("Invalid sweep traceback rank");
            const i64 take = std::min(to, it->end) - from;
            result.matches.push_back(event.side == 1 ? Match{it->index, current, take}
                                                     : Match{current, it->index, take});
            from += take;
            ++it;
        }
    };
    for (size_t i = events.size(); i-- > 0;) {
        const auto &e = events[i];
        if (e.count <= 0)
            continue;
        const auto &r = records[i];
        ++result.records;
        result.vertices += r.before[0].vs.size() + r.before[1].vs.size() + r.profits.vs.size();
        const int own = e.side, opp = 1 - own;
        F envelope = r.before[opp];
        envelope.suffix_max();
        envelope.truncate(r.profits.K());
        const F vt = add_pl(envelope, r.profits);
        auto stay = [&](i64 g) {
            auto choice = argmax(vt, g, std::min(vt.K(), g + e.count));
            choice.second -= r.profits.eval(g);
            return choice;
        };
        auto append = [&](i64 g) {
            if (g > r.own_limit + e.count)
                return std::pair<i64, i64>{0, NEG_INF};
            return argmax(r.before[own], std::max<i64>(0, g - e.count), r.before[own].K());
        };
        auto flip = [&](i64 g) {
            if (g > e.count)
                return std::pair<i64, i64>{0, NEG_INF};
            // Some fresh pending units may be discarded after the flip.
            // Neutral sync represents this free-drop choice at zero.
            auto choice = argmax(vt, 0, std::min(vt.K(), e.count - g));
            return std::pair<i64, i64>{e.count - choice.first, choice.second};
        };
        // Neutral sync may select either side's transition at g == 0.
        if (!pending) {
            const i64 own_value = std::max(append(0).second, flip(0).second);
            const i64 opp_value = stay(0).second;
            side = own_value >= opp_value ? own : opp;
        }
        i64 k = 0, predecessor = 0;
        if (side == own) {
            const auto choice = append(pending);
            const auto flipped = flip(pending);
            if (choice.second >= flipped.second) {
                if (choice.second != value)
                    throw std::runtime_error("Invalid sweep append traceback");
                pending = choice.first;
                value = r.before[own].eval(pending);
                continue;
            }
            if (flipped.second != value)
                throw std::runtime_error(
                    "Invalid sweep flip traceback at event " + std::to_string(i) +
                    ", pending=" + std::to_string(pending) + ", value=" + std::to_string(value) +
                    ", flip=" + std::to_string(flipped.second) +
                    ", append=" + std::to_string(choice.second));
            k = e.count - flipped.first;
            const auto previous = argmax(r.before[opp], k, r.before[opp].K());
            predecessor = previous.first;
            const size_t index = e.index == SIZE_MAX ? i : e.index;
            emit(e, index, opp, r.end[opp] - k, r.end[opp]);
        } else {
            const auto choice = stay(pending);
            if (choice.second != value)
                throw std::runtime_error("Invalid sweep matching traceback");
            k = choice.first;
            predecessor = argmax(r.before[opp], k, r.before[opp].K()).first;
            const size_t index = e.index == SIZE_MAX ? i : e.index;
            emit(e, index, opp, r.end[opp] - k, r.end[opp] - pending);
        }
        side = opp;
        pending = predecessor;
        value = r.before[opp].eval(pending);
    }
}
template <class Policy> I ordinary(const Input &in, const std::vector<k::Event> &ev) {
    Engine<Policy> engine(in, ev);
    for (const auto &e : ev)
        engine.template step<false>(e);
    return engine.best();
}
template <class Policy>
k::TraceResult traced(const Input &in, const std::vector<k::Event> &ev, size_t stride) {
    Engine<Policy> engine(in, ev);
    using Save = typename Engine<Policy>::Save;
    std::vector<Save> checkpoints;
    k::TraceStorage trace;
    std::vector<k::TraceRecord> full;
    // Block-rank metadata is linear in events, independent of profile sizes.
    I ends[2] = {0, 0};
    for (const auto &e : ev) {
        if (e.count) {
            ends[e.side] += e.count;
            trace.blocks[e.side].push_back({e.index, ends[e.side]});
        }
    }
    if (!stride)
        full.resize(ev.size());
    for (size_t i = 0; i < ev.size(); ++i) {
        if (stride && i % stride == 0)
            checkpoints.push_back(engine.save(i));
        if (stride)
            engine.template step<false>(ev[i]);
        else
            engine.template step<true>(ev[i], &full[i]);
    }
    k::TraceResult out;
    out.profit = engine.best();
    int side = engine.sides[0].V.max_value() >= engine.sides[1].V.max_value() ? 0 : 1;
    I pending = k::argmax(engine.sides[side].V, 0, engine.sides[side].V.K()).first,
      value = out.profit;
    if (!stride)
        reverse_trace(ev, full, trace, out, side, pending, value);
    else
        for (size_t block = checkpoints.size(); block-- > 0;) {
            const size_t first = checkpoints[block].event, last = block + 1 < checkpoints.size()
                                                                      ? checkpoints[block + 1].event
                                                                      : ev.size();
            engine.restore(checkpoints[block]);
            std::vector<k::Event> local(ev.begin() + first, ev.begin() + last);
            std::vector<k::TraceRecord> records(local.size());
            for (size_t j = 0; j < local.size(); ++j)
                engine.template step<true>(local[j], &records[j]);
            reverse_trace(local, records, trace, out, side, pending, value);
        }
    require(!pending && !value, "checkpoint traceback initial state");
    return out;
}
// Save checkpoints after pruning at each boundary. The replay block's final
// snapshot must include the NEXT event's prune, matching the global adjoint's
// snapshot domains. Omitting that boundary prune gives subtly wrong gains.
template <class Policy>
std::vector<I> marginal_orientation(const Input &in, const std::vector<k::Event> &ev,
                                    size_t stride) {
    Engine<Policy> engine(in, ev);
    using Save = typename Engine<Policy>::Save;
    std::vector<Save> checkpoints;
    std::vector<k::Snapshot> full;
    for (size_t i = 0; i < ev.size(); ++i) {
        engine.prune(ev[i].pos);
        if (stride && i % stride == 0)
            checkpoints.push_back(engine.save(i));
        if (!stride)
            full.push_back({{engine.sides[0], engine.sides[1]}});
        engine.template step<false>(ev[i]);
    }
    const I base = engine.best();
    k::PLF C[2] = {k::PLF::constant(engine.sides[0].V.K(), 0),
                   k::PLF::constant(engine.sides[1].V.K(), 0)};
    std::vector<I> result(ev.size(), 0), composed;
    if (!stride) {
        full.push_back({{engine.sides[0], engine.sides[1]}});
        k::adjoint_range(ev, engine.profit, full, composed, C);
        for (size_t i = 0; i < ev.size(); ++i)
            if (ev[i].side == 1)
                result[i] = std::max<I>(0, composed[i] - base);
    } else
        for (size_t block = checkpoints.size(); block-- > 0;) {
            const size_t first = checkpoints[block].event, last = block + 1 < checkpoints.size()
                                                                      ? checkpoints[block + 1].event
                                                                      : ev.size();
            engine.restore(checkpoints[block]);
            std::vector<k::Event> local(ev.begin() + first, ev.begin() + last);
            std::vector<k::Snapshot> snaps;
            snaps.reserve(local.size() + 1);
            for (const auto &e : local) {
                engine.prune(e.pos);
                snaps.push_back({{engine.sides[0], engine.sides[1]}});
                engine.template step<false>(e);
            }
            if (last < ev.size())
                engine.prune(ev[last].pos);
            snaps.push_back({{engine.sides[0], engine.sides[1]}});
            k::adjoint_range(local, engine.profit, snaps, composed, C);
            for (size_t j = 0; j < local.size(); ++j)
                if (local[j].side == 1)
                    result[first + j] = std::max<I>(0, composed[j] - base);
        }
    return result;
}
template <class Policy> std::vector<I> gains(const Input &in, size_t stride) {
    std::vector<I> out(in.pos.size());
    for (bool mirror : {false, true}) {
        auto g = events<true>(in, mirror, true);
        auto values = marginal_orientation<Policy>(in, g.events, stride);
        for (size_t i = 0; i < g.events.size(); ++i)
            if (g.events[i].side == 1)
                for (auto part : g.parts[i])
                    out[part.index] = std::max(out[part.index], values[i]);
    }
    return out;
}
} // namespace nmr_trials
#endif
