# BigANN 100M: Graph Partitioning Scaling Experiment

## Goal

Measure how **graph partitioning (GP)** and **HNSW routing cost** scale as we
increase the number of shards from 50 to 2000 on the BigANN-100M dataset.

## Dataset

- **Base vectors**: First 100M vectors from BigANN-1B (`train.npy`)
  - 100,000,000 vectors, 128 dimensions, uint8
- **Queries**: 10,000 vectors, 128 dimensions, uint8 (`queries.npy`)
- **Ground truth**: 100 nearest neighbors per query for the 100M subset (`ground_truth_100m.npy`)
- **Distance metric**: Squared L2 (Euclidean)
- **Source**: `/scratch/brc7/bigann/`

## Shard Counts

| Shards | Avg vectors/shard |
|--------|------------------|
| 50     | 2,000,000        |
| 100    | 1,000,000        |
| 500    | 200,000          |
| 1,000  | 100,000          |
| 2,000  | 50,000           |

## Running the Experiment

A single C++ executable (`BigANNExperiment`) handles the entire pipeline:
data loading (reads `.npy` directly), graph partitioning, routing index
construction, shard searches, and result combination.

```bash
cd release_l2

# Full experiment (all shard counts)
./BigANNExperiment \
    /scratch/brc7/bigann/train.npy \
    /scratch/brc7/bigann/queries.npy \
    /scratch/brc7/bigann/ground_truth_100m.npy \
    /scratch/brc7/partitioning \
    100000000 10 50 100 500 1000 2000

# Single shard count for testing
./BigANNExperiment \
    /scratch/brc7/bigann/train.npy \
    /scratch/brc7/bigann/queries.npy \
    /scratch/brc7/bigann/ground_truth_100m.npy \
    /scratch/brc7/partitioning \
    100000000 10 50
```

### Arguments

```
./BigANNExperiment base-npy queries-npy gt-npy output-dir num-base num-neighbors shard1 [shard2 ...]
```

| Argument       | Description                                     |
|----------------|-------------------------------------------------|
| base-npy       | Path to base vectors (.npy, uint8)              |
| queries-npy    | Path to query vectors (.npy, uint8)             |
| gt-npy         | Path to ground truth IDs (.npy, uint32)         |
| output-dir     | Directory for all output files                  |
| num-base       | Number of base vectors to use (e.g. 100000000)  |
| num-neighbors  | k for k-NN recall (typically 10)                |
| shard1 ...     | One or more shard counts to evaluate            |

## Pipeline

### 1. Data Loading

Reads `.npy` files directly (no format conversion needed). The npy reader
(`src/npy_io.h`) handles uint8 base/query vectors (converted to float32 on
the fly) and uint32 ground truth IDs. Squared L2 distances for the ground
truth are computed in parallel after loading.

### 2. k-NN Graph Construction (one-time)

Builds an approximate 10-NN graph over all base vectors using recursive
random-projection sketching + brute-force within buckets (3 repetitions,
fanout 3). **This is done once and reused for all shard counts** — the most
expensive step (~30-60 min at 100M scale).

### 3. Graph Partitioning (per K)

For each shard count K, the pre-built k-NN graph is symmetrized, converted to
CSR, and partitioned using KaMinPar with epsilon=0.05 balance constraint.
Since the graph is already built, each partition call only takes the
symmetrize + CSR + KaMinPar time.

Output: `bigann_100m.partition.k={K}.GP`

### 4. Routing Index + Evaluation (per K)

Trains KMeansTree routers at multiple centroid budgets {20K..10M}, extracts
centroids, builds HNSW routing index (M=32, ef_construction=200), and
evaluates with varying ef_search {20..500} across four strategies (HNSW,
Pyramid, SPANN, Frequency).

Output: `bigann_100m.GP.k={K}.routes`

### 5. Shard Searches (per K)

Builds per-shard HNSW indexes and queries at multiple ef_search values
{50..500}.

Output: `bigann_100m.GP.k={K}.searches`

### 6. Result Combination (per K)

Cross-product of routing configs x shard search configs. Reports recall, QPS,
latency, n_probes.

Output: `bigann_100m.GP.k={K}` (CSV)

## MPI

**Not needed.** The entire codebase uses shared-memory parallelism via
[parlaylib](https://github.com/cmuparlay/parlaylib). All computation happens
in a single process. The machine has 255 cores and 1.5 TB RAM, which is
sufficient.

## Resource Estimates

| Phase | Peak RAM | Wall Time (est.) |
|-------|----------|------------------|
| Data loading | ~50 GB | ~1 min |
| k-NN graph (one-time) | ~80-120 GB | 30-60 min |
| Partitioning (per K) | ~20-30 GB (extra) | 5-30 min |
| Routing + search (per K) | ~60-100 GB | 30 min - 2 hr |

Peak memory during k-NN graph construction: ~48 GB for float32 points + ~8 GB
for the graph + working memory.

## Key Metrics to Compare Across Shard Counts

1. **Partitioning time**: How long does KaMinPar take as K grows?
2. **Partition balance**: Imbalance ratio (max shard / avg shard) logged per K
3. **Routing cost**: Distance computations per query for HNSW routing
   (logged as `routing_distance_calcs`)
4. **First-shard recall**: How often does the router pick the correct shard
   first? (logged during routing)
5. **Recall vs QPS**: The Pareto frontier from the combined route+search CSV

## Build

```bash
cd release_l2
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target BigANNExperiment -j 32
```

## Files

| File | Purpose |
|------|---------|
| `bigann_experiment.cpp` | Main experiment executable |
| `src/npy_io.h` | NumPy .npy file reader |
| `experiment.md` | This document |
| `convert_npy_to_gpann.py` | Standalone Python converter (alternative) |
| `run_bigann_100m.sh` | Shell script runner (alternative) |
