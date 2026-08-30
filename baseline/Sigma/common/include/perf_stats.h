#ifndef PERF_STATS_H
#define PERF_STATS_H

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <mutex>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace perf_stats
{
    struct Entry
    {
        uint64_t calls = 0;
        uint64_t total_ns = 0;
    };

    inline bool &enabled_flag()
    {
        static bool enabled = false;
        return enabled;
    }

    inline std::mutex &stats_mutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    inline std::unordered_map<std::string, Entry> &stats_map()
    {
        static std::unordered_map<std::string, Entry> stats;
        return stats;
    }

    inline uint32_t &max_depth_flag()
    {
        static uint32_t max_depth = 0;
        return max_depth;
    }

    inline void enable(bool enabled)
    {
        enabled_flag() = enabled;
    }

    inline bool is_enabled()
    {
        return enabled_flag();
    }

    inline void set_max_depth(uint32_t max_depth)
    {
        max_depth_flag() = max_depth;
    }

    inline uint32_t max_depth()
    {
        return max_depth_flag();
    }

    inline void reset()
    {
        std::lock_guard<std::mutex> lock(stats_mutex());
        stats_map().clear();
    }

    inline void record(const std::string &name, uint64_t elapsed_ns)
    {
        if (!enabled_flag())
        {
            return;
        }

        std::lock_guard<std::mutex> lock(stats_mutex());
        Entry &entry = stats_map()[name];
        entry.calls += 1;
        entry.total_ns += elapsed_ns;
    }

    class Scope
    {
    public:
        explicit Scope(const char *name)
            : name_(name), active_(enabled_flag()), start_(active_ ? Clock::now() : Clock::time_point())
        {
        }

        ~Scope()
        {
            if (!active_)
            {
                return;
            }

            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_).count();
            record(name_, static_cast<uint64_t>(elapsed));
        }

    private:
        using Clock = std::chrono::steady_clock;

        const char *name_;
        bool active_;
        Clock::time_point start_;
    };

    inline void print_report(std::ostream &os, uint64_t outer_loops = 1)
    {
        std::vector<std::pair<std::string, Entry>> entries;
        {
            std::lock_guard<std::mutex> lock(stats_mutex());
            entries.reserve(stats_map().size());
            for (const auto &item : stats_map())
            {
                entries.emplace_back(item.first, item.second);
            }
        }

        const auto configured_max_depth = max_depth();
        if (configured_max_depth != 0)
        {
            entries.erase(std::remove_if(entries.begin(), entries.end(), [&](const auto &entry) {
                uint32_t depth = 0;
                const std::string &name = entry.first;
                for (size_t pos = 0; pos + 1 < name.size(); ++pos)
                {
                    if (name[pos] == ':' && name[pos + 1] == ':')
                    {
                        ++depth;
                        ++pos;
                    }
                }
                return depth > configured_max_depth;
            }), entries.end());
        }

        if (entries.empty())
        {
            os << "Perf breakdown: no samples" << std::endl;
            return;
        }

        std::sort(entries.begin(), entries.end(), [](const auto &lhs, const auto &rhs) {
            return lhs.second.total_ns > rhs.second.total_ns;
        });

        uint64_t total_ns = 0;
        for (const auto &entry : entries)
        {
            total_ns += entry.second.total_ns;
        }

        const uint64_t normalized_loops = std::max<uint64_t>(1, outer_loops);

        os << "Perf breakdown per round:" << std::endl;
        os << std::left << std::setw(40) << "Component"
           << std::right << std::setw(14) << "Total(ms)"
           << std::setw(10) << "Calls"
           << std::setw(14) << "AvgCall(ms)"
           << std::setw(15) << "PerRound(ms)"
           << std::setw(11) << "Share(%)" << std::endl;
        os << std::string(104, '-') << std::endl;
        os << std::fixed << std::setprecision(3);

        for (const auto &entry : entries)
        {
            const double total_ms = static_cast<double>(entry.second.total_ns) / 1e6;
            const double avg_ms = entry.second.calls == 0 ? 0.0 : total_ms / static_cast<double>(entry.second.calls);
            const double per_round_ms = total_ms / static_cast<double>(normalized_loops);
            const double pct = total_ns == 0 ? 0.0 : 100.0 * static_cast<double>(entry.second.total_ns) / static_cast<double>(total_ns);
            os << std::left << std::setw(40) << entry.first
               << std::right << std::setw(14) << total_ms
               << std::setw(10) << entry.second.calls
               << std::setw(14) << avg_ms
               << std::setw(15) << per_round_ms
               << std::setw(11) << pct << std::endl;
        }

        os << std::defaultfloat;
    }
}

#endif