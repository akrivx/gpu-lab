#include "experiment.hpp"

#include <cstdio>
#include <sstream>
#include <vector>
#include <string>

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

  std::string format_bandwidth_table(const std::vector<gpu_lab::ExperimentResult>& results) {
    std::ostringstream oss;

    const char* sep =
      "+--------------+----------+----------+----------+----------+----------+----------+----------+\n";

    oss << sep;
    oss << "| Experiment   |   MiB    |  Min ms  |  Max ms  |  Avg ms  | Best GB/s| Avg GB/s | Worst GB/s|\n";
    oss << sep;

    for (const auto& e : results) {
      const float mib = e.bytes_moved / (1024.0f * 1024.0f);
      auto gbps_from_ms = [&](float ms) {
        const float sec = ms / 1000.0f;
        return (e.bytes_moved / sec) / 1.0e9f; // GB/s
      };

      const float best_gbps  = gbps_from_ms(e.min_ms); // fastest → highest BW
      const float avg_gbps   = gbps_from_ms(e.avg_ms);
      const float worst_gbps = gbps_from_ms(e.max_ms); // slowest → lowest BW

      char line[256];
      std::snprintf(
        line, sizeof(line),
        "| %-12s | %8.2f | %8.3f | %8.3f | %8.3f | %8.2f | %8.2f | %8.2f |\n",
        e.name.c_str(),
        mib,
        e.min_ms,
        e.max_ms,
        e.avg_ms,
        best_gbps,
        avg_gbps,
        worst_gbps
      );
      oss << line;
    }

    oss << sep;
    return std::move(oss).str();
  }
} // namespace (anonymous)

namespace gpu_lab {
  void print_bandwidth_table(const std::vector<ExperimentResult>& results, std::ostream& out) {
    out << ::format_bandwidth_table(results);
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