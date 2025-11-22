#include "experiment.hpp"

#include <ostream>
#include <vector>
#include <string>
#include <iomanip>

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

  inline Row make_row(const gpu_lab::ExperimentResult& e) {
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
    using std::left;
    using std::right;
    using std::setw;
    using std::fixed;
    using std::setprecision;

    std::vector<::Row> rows;
    rows.reserve(results.size());
    for (const auto& e : results) {
      rows.push_back(::make_row(e));
    }

    const char* sep =
      "+----------------+------------+------------+------------+------------+------------+------------+\n";

    out << sep;
    out << "| " << left  << setw(14) << "Experiment"
        << " | " << right << setw(10) << "Min ms"
        << " | " << right << setw(10) << "Max ms"
        << " | " << right << setw(10) << "Avg ms"
        << " | " << right << setw(10) << "Best GB/s"
        << " | " << right << setw(10) << "Worst GB/s"
        << " | " << right << setw(10) << "Avg GB/s"
        << " |\n";
    out << sep;

    out << fixed << setprecision(3);
    for (const auto& r : rows) {
      out << "| " << left  << setw(14) << r.name
          << " | " << right << setw(10) << r.min_ms
          << " | " << right << setw(10) << r.max_ms
          << " | " << right << setw(10) << r.avg_ms;

      out << setprecision(1);

      out << " | " << right << setw(10) << r.best_gbps
          << " | " << right << setw(10) << r.avg_gbps
          << " | " << right << setw(10) << r.worst_gbps
          << " |\n";

      out << setprecision(3); // restore for next time values
    }

    out << sep;
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