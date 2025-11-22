#include "experiment.hpp"

#include <ostream>
#include <format>
#include <vector>
#include <string>
#include <string_view>

namespace {
  struct Row {
    std::string name;
    float min_ms;
    float max_ms;
    float avg_ms;
    float best_gbps;   // from min time
    float worst_gbps;  // from max time
    float avg_gbps;    // from avg time
  };

  Row make_row(const gpu_lab::ExperimentResult& e) {
    auto gbps_from_ms = [&](float ms) {
      const float sec = ms / 1000.0f;
      return (e.bytes_moved / sec) / 1e9f; // GB/s
    };

    return {
      e.name,
      e.min_ms,
      e.max_ms,
      e.avg_ms,
      gbps_from_ms(e.min_ms), // fastest -> highest BW
      gbps_from_ms(e.max_ms),  // slowest -> lowest BW
      gbps_from_ms(e.avg_ms),
    };
  }
} // namespace (anonymous)

namespace gpu_lab {
  void print_bandwidth_table(const std::vector<ExperimentResult>& results, std::ostream& out) {
    {
      std::string_view sep =
        "+--------------+----------+----------+----------+----------+------------+------------+------------+\n";

      out << sep;
      out << "| Experiment   |   MiB    |  Min ms  |  Max ms  |  Avg ms  | Best GB/s  | Worst GB/s |  Avg GB/s  |\n";
      out << sep;
    
      for (const auto& e : results) {
        const float mib = static_cast<float>(e.bytes_moved) / (1024.0f * 1024.0f);
    
        auto gbps_from_ms = [&](float ms) {
          const float sec = ms / 1000.0f;
          return (static_cast<float>(e.bytes_moved) / sec) / 1.0e9f; // GB/s
        };
    
        const float best_gbps  = gbps_from_ms(e.min_ms); // fastest -> highest BW
        const float worst_gbps = gbps_from_ms(e.max_ms); // slowest -> lowest BW
        const float avg_gbps   = gbps_from_ms(e.avg_ms);
    
        out << std::format(
          "| {:<12} | {:8.2f} | {:8.3f} | {:8.3f} | {:8.3f} | {:10.2f} | {:10.2f} | {:10.2f} |\n",
          e.name,
          mib,
          e.min_ms,
          e.max_ms,
          e.avg_ms,
          best_gbps,
          worst_gbps,
          avg_gbps
        );
      }
    
      out << sep;
    }
  }

  std::vector<ExperimentResult> run_bandwidth_experiments(
    const std::vector<std::pair<std::string, TimedExperimentFunc>>& funcs,
    std::size_t bytes_moved)
  {
    std::vector<ExperimentResult> results;
    results.reserve(funcs.size());
    for (const auto& [name, f] : funcs) {
      const auto [min_ms, max_ms, avg_ms] = f();
      results.push_back(ExperimentResult{name, bytes_moved, min_ms, max_ms, avg_ms});
    }
    return results;
  }
}