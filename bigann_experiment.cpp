#include <filesystem>
#include <fstream>
#include <iostream>
#include <sched.h>
#include <string>
#include <unistd.h>
#include <vector>

#include "dist.h"
#include "knn_graph.h"
#include "metis_io.h"
#include "npy_io.h"
#include "partitioning.h"
#include "recall.h"
#include "route_search_combination.h"

#include <parlay/parallel.h>

void SetAffinity() {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  int num_cpus = std::min(255, static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN)));
  for (int cpu = 0; cpu < num_cpus; ++cpu) {
    CPU_SET(cpu, &mask);
  }
  if (sched_setaffinity(0, sizeof(mask), &mask)) {
    std::cerr << "Warning: thread pinning failed" << std::endl;
  }
}

// Compute GP-ANN ground truth format (NNVec with distances) from ID-only npy
// ground truth.
std::vector<NNVec> ComputeGTDistances(PointSet &points, PointSet &queries,
                                      const npy::GroundTruthIds &gt_ids) {
  std::cout << "Computing ground truth distances..." << std::endl;
  Timer timer;
  timer.Start();

  std::vector<NNVec> ground_truth(gt_ids.num_queries,
                                  NNVec(gt_ids.num_neighbors));

  parlay::parallel_for(0, gt_ids.num_queries, [&](size_t q) {
    float *query = queries.GetPoint(q);
    for (size_t j = 0; j < gt_ids.num_neighbors; ++j) {
      uint32_t point_id = gt_ids.Get(q, j);
      float dist = distance(points.GetPoint(point_id), query, points.d);
      ground_truth[q][j] = {dist, point_id};
    }
  });

  std::cout << "GT distances computed in " << timer.Stop() << "s" << std::endl;
  return ground_truth;
}

int main(int argc, const char *argv[]) {
  if (argc < 8) {
    std::cerr << "Usage: ./BigANNExperiment base-npy queries-npy gt-npy "
                 "output-dir num-base num-neighbors shard1 [shard2 ...]\n"
              << "\n"
              << "Example:\n"
              << "  ./BigANNExperiment \\\n"
              << "      /scratch/brc7/bigann/train.npy \\\n"
              << "      /scratch/brc7/bigann/queries.npy \\\n"
              << "      /scratch/brc7/bigann/ground_truth_100m.npy \\\n"
              << "      /scratch/brc7/partitioning 100000000 10 50 100 500 "
                 "1000 2000\n";
    return 1;
  }

  SetAffinity();

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

  std::cout << "=== BigANN Experiment ===" << std::endl;
  std::cout << "Base:        " << base_path << " (first " << num_base << ")"
            << std::endl;
  std::cout << "Queries:     " << queries_path << std::endl;
  std::cout << "GT:          " << gt_path << std::endl;
  std::cout << "Output:      " << output_dir << std::endl;
  std::cout << "Neighbors:   " << num_neighbors << std::endl;
  std::cout << "Shard counts:";
  for (int K : shard_counts)
    std::cout << " " << K;
  std::cout << std::endl;
  std::cout << "Threads:     " << parlay::num_workers() << std::endl;
  std::cout << std::endl;

  // ==== Load data ====
  std::cout << "=== Loading data ===" << std::endl;
  Timer timer;

  timer.Start();
  PointSet points = npy::ReadUint8AsPointSet(base_path, num_base);
  std::cout << "Loaded " << points.n << " x " << points.d << " base points in "
            << timer.Stop() << "s\n"
            << std::endl;

  timer.Start();
  PointSet queries = npy::ReadUint8AsPointSet(queries_path);
  std::cout << "Loaded " << queries.n << " x " << queries.d << " queries in "
            << timer.Stop() << "s\n"
            << std::endl;

  timer.Start();
  npy::GroundTruthIds gt_raw = npy::ReadUint32_2D(gt_path);
  // Verify GT IDs are within the base set
  for (size_t i = 0; i < gt_raw.ids.size(); ++i) {
    if (gt_raw.ids[i] >= static_cast<uint32_t>(num_base)) {
      std::cerr << "ERROR: Ground truth ID " << gt_raw.ids[i] << " >= num_base "
                << num_base << std::endl;
      return 1;
    }
  }
  std::vector<NNVec> ground_truth = ComputeGTDistances(points, queries, gt_raw);
  std::cout << "Ground truth ready in " << timer.Stop() << "s\n" << std::endl;

  std::vector<float> distance_to_kth_neighbor =
      ConvertGroundTruthToDistanceToKthNeighbor(ground_truth, num_neighbors,
                                                points, queries);
  std::cout << "Distance-to-kth-neighbor computed\n" << std::endl;

  // ==== Build approximate k-NN graph ONCE ====
  std::cout << "=== Building approximate k-NN graph ===" << std::endl;
  timer.Start();
  ApproximateKNNGraphBuilder graph_builder;
  AdjGraph knn_graph =
      graph_builder.BuildApproximateNearestNeighborGraph(points, 10);
  double knn_time = timer.Stop();
  std::cout << "k-NN graph (" << points.n << " nodes, k=10) built in "
            << knn_time << "s\n"
            << std::endl;

  // ==== Summary CSV ====
  std::string summary_path = output_dir + "/bigann_" +
                             std::to_string(num_base / 1'000'000) +
                             "m.experiment_summary.csv";
  {
    std::ofstream summary(summary_path, std::ios::trunc);
    summary << "num_shards,num_shards_actual,partition_time_s,"
               "routing_time_s,shard_search_time_s,"
               "knn_graph_time_s,"
               "max_shard_size,min_shard_size,avg_shard_size,imbalance,"
               "num_routing_configs,num_search_configs"
            << std::endl;
  }
  std::cout << "Summary CSV: " << summary_path << std::endl;

  // ==== Iterate over shard counts ====
  for (int K : shard_counts) {
    std::cout
        << "\n============================================================"
        << std::endl;
    std::cout << "=== K=" << K << " shards  (" << points.n / K
              << " vectors/shard avg) ===" << std::endl;
    std::cout
        << "============================================================\n"
        << std::endl;

    std::string prefix =
        output_dir + "/bigann_" + std::to_string(num_base / 1'000'000) + "m";
    std::string part_file =
        prefix + ".partition.k=" + std::to_string(K) + ".GP";
    std::string output_prefix = prefix + ".GP.k=" + std::to_string(K);

    // -- Graph partitioning --
    std::cout << "--- Graph Partitioning ---" << std::endl;
    timer.Start();
    Partition partition = PartitionAdjListGraph(
        knn_graph, K, /*epsilon=*/0.05,
        std::min(64, static_cast<int>(parlay::num_workers())),
        /*strong=*/false, /*quiet=*/false);
    double part_time = timer.Stop();

    Clusters clusters = ConvertPartitionToClusters(partition);
    int num_shards = static_cast<int>(clusters.size());
    WriteClusters(clusters, part_file);

    // Log partition stats
    size_t max_shard_size = 0, min_shard_size = SIZE_MAX;
    for (const auto &c : clusters) {
      max_shard_size = std::max(max_shard_size, c.size());
      min_shard_size = std::min(min_shard_size, c.size());
    }
    double avg_shard_size = static_cast<double>(points.n) / num_shards;
    double imbalance = max_shard_size / avg_shard_size;

    std::cout << "Partitioning: K=" << K << " num_shards=" << num_shards
              << " time=" << part_time << "s"
              << " max_shard=" << max_shard_size
              << " min_shard=" << min_shard_size << " imbalance=" << imbalance
              << std::endl;

    // -- Routing index construction + evaluation --
    std::cout << "\n--- Routing ---" << std::endl;
    timer.Start();
    KMeansTreeRouterOptions router_options;
    router_options.budget = points.n / K;

    std::vector<RoutingConfig> routes = IterateRoutingConfigs(
        points, queries, clusters, num_shards, router_options, ground_truth,
        num_neighbors, part_file + ".routing_index", "", "");
    double route_time = timer.Stop();
    std::cout << "Routing: " << routes.size() << " configs generated in "
              << route_time << "s" << std::endl;
    SerializeRoutes(routes, output_prefix + ".routes");

    // -- Per-shard HNSW index construction + search --
    std::cout << "\n--- Shard Searches ---" << std::endl;
    timer.Start();
    std::vector<ShardSearch> shard_searches =
        RunInShardSearches(points, queries, HNSWParameters(), num_neighbors,
                           clusters, num_shards, distance_to_kth_neighbor);
    double search_time = timer.Stop();
    std::cout << "Shard searches: " << shard_searches.size() << " configs in "
              << search_time << "s" << std::endl;
    SerializeShardSearches(shard_searches, output_prefix + ".searches");

    // -- Combine routes x searches and emit results --
    std::cout << "\n--- Results ---" << std::endl;
    PrintCombinationsOfRoutesAndSearches(routes, shard_searches, output_prefix,
                                         num_neighbors, queries.n, num_shards,
                                         K, "GP");

    std::cout << "\n[K=" << K << " summary]"
              << " partition=" << part_time << "s"
              << " routing=" << route_time << "s"
              << " search=" << search_time << "s" << std::endl;

    // Append row to summary CSV (append mode so partial results survive)
    {
      std::ofstream summary(summary_path, std::ios::app);
      summary << K << "," << num_shards << "," << part_time << "," << route_time
              << "," << search_time << "," << knn_time << "," << max_shard_size
              << "," << min_shard_size << "," << avg_shard_size << ","
              << imbalance << "," << routes.size() << ","
              << shard_searches.size() << std::endl;
    }
  }

  std::cout << "\n=== All experiments complete ===" << std::endl;
  std::cout << "k-NN graph build (one-time): " << knn_time << "s" << std::endl;
  std::cout << "Summary CSV: " << summary_path << std::endl;
  std::cout << "Results in: " << output_dir << std::endl;
  return 0;
}
