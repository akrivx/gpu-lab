#pragma once

#include <iosfwd>
#include <cstddef>
#include <string>
#include <utility>
#include <functional>

namespace gpu_lab {
  struct ExperimentResult {
    std::string name;
    std::size_t bytes_moved;
    float min_ms;
    float max_ms;
    float avg_ms;
  };

  void print_bandwidth_table(const std::vector<ExperimentResult>& results, std::ostream& out);

  // A function that returns a (min_ms, max_ms, avg_ms) tuple.
  using TimedExperimentFunc = std::function<std::tuple<float, float, float>()>;

  std::vector<ExperimentResult> run_bandwidth_experiments(
    const std::vector<std::pair<std::string, TimedExperimentFunc>>& funcs,
    std::size_t bytes_moved);

  inline void run_bandwidth_experiments(
    const std::vector<std::pair<std::string, TimedExperimentFunc>>& funcs,
    std::size_t bytes_moved,
    std::ostream& out)
  {
    const auto results = run_bandwidth_experiments(funcs, bytes_moved);
    print_bandwidth_table(results, out);
  }
}
