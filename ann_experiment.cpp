#include <iostream>
#include <sched.h>
#include <unistd.h>
#include <vector>

#include <filesystem>



void setAffinity() {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  int num_cpus = std::min(128, static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN)));
  for (int cpu = 0; cpu < num_cpus; ++cpu) {
    CPU_SET(cpu, &mask);
  }
  if (sched_setaffinity(0, sizeof(mask), &mask)) {
    std::cerr << "Warning: thread pinning failed" << std::endl;
  }
}

int main(int argc, const char **argv) {
  setAffinity();

  std::string base_path = argv[1];
  std::string queries_path = argv[2];
  std::string gt_path = argv[3];
  std::string output_dir = argv[4];
  int64_t num_base = std::stoll(argv[5]);
  int num_neighbors = std::stoi(argv[6]);

  std::vector<int> shard_counts;
  for (int i = 7; i < argc; ++i) {
    shard_counts.push_back(std::stoi(argv[i]));
  }

  std::filesystem::create_directories(output_dir);

  std::cout << "=== ANN Experiment ===" << std::endl;
  std::cout << "Base:        " << base_path << " (first " << num_base << ")"
            << std::endl;
  std::cout << "Queries:     " << queries_path << std::endl;
  std::cout << "GT:          " << gt_path << std::endl;
  std::cout << "Output:      " << output_dir << std::endl;
  return 0;
}