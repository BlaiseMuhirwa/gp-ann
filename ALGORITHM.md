# KaMinPar Graph Partitioning Algorithm

This document explains how the KaMinPar shared-memory graph partitioning algorithm works, tracing the full pipeline from input to output, and pointing to the relevant source files.

## Overview

KaMinPar is a **multilevel graph partitioner**. Given a graph G = (V, E) and a target number of blocks k, it computes a partition of V into k roughly balanced blocks while minimizing the number (or weight) of edges cut between blocks. The algorithm follows the classic multilevel paradigm:

```
Input Graph
  │
  ▼
┌─────────────┐
│  Coarsening  │  ← Repeatedly shrink the graph by clustering nodes
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│  Initial Partitioning │  ← Partition the tiny coarsened graph
└──────┬───────────────┘
       │
       ▼
┌───────────────────────┐
│  Uncoarsening/Refine  │  ← Project back and improve at each level
└──────┬────────────────┘
       │
       ▼
Output Partition
```

**Entry point:** `KaMinPar::compute_partition()` in `kaminpar-shm/kaminpar.cc` (line ~295).

---

## 1. Graph Data Structures

KaMinPar supports two internal graph representations, unified behind a type-erasing `Graph` wrapper.

### CSR (Compressed Sparse Row)

The standard uncompressed format. A graph with n nodes and m edges is stored as:

- `nodes[n+1]` — offset array where `nodes[u]` is the index of the first edge of node u
- `edges[m]` — target node IDs for all edges
- `node_weights[n]` (optional) — per-node weights
- `edge_weights[m]` (optional) — per-edge weights

Accessing the neighbors of node u means iterating `edges[nodes[u] .. nodes[u+1])`.

**Source:** `kaminpar-shm/datastructures/csr_graph.h`

### Compressed Graph

An advanced representation using variable-length integer encoding, gap encoding, interval encoding (for consecutive neighbor ranges), and high-degree node splitting. Achieves 3-5x compression on large graphs with SIMD-accelerated decoding (StreamVByte).

**Source:** `kaminpar-shm/datastructures/compressed_graph.h`
**Compression library:** `kaminpar-common/graph_compression/compressed_neighborhoods.h`

### Graph Wrapper and Partitioned Graph

- `Graph` — a `std::variant<CSRGraph, CompressedGraph>` with dynamic dispatch via `reified()` helper templates. Avoids virtual call overhead in inner loops.
- `PartitionedGraph` — wraps a `Graph` with a block assignment array (`partition[u] = block_id`) and per-block weight tracking. Supports thread-safe atomic moves.

**Source:**
- `kaminpar-shm/datastructures/graph.h`
- `kaminpar-shm/datastructures/partitioned_graph.h`

---

## 2. Partitioning Modes

KaMinPar offers four top-level partitioning strategies, selected via `PartitioningMode`:

| Mode | Class | Description |
|------|-------|-------------|
| `DEEP` | `DeepMultilevelPartitioner` | Deep multilevel with recursive bipartitioning during uncoarsening (default) |
| `VCYCLE` | `VcycleDeepMultilevelPartitioner` | Deep multilevel with multiple V-cycles for higher quality |
| `RB` | `RBMultilevelPartitioner` | Classical recursive bisection at the top level |
| `KWAY` | `KWayMultilevelPartitioner` | Direct k-way multilevel (coarsen, partition into k, uncoarsen) |

The factory in `kaminpar-shm/factories.cc` instantiates the chosen partitioner.

**Source:**
- `kaminpar-shm/partitioning/deep/deep_multilevel.h` / `.cc`
- `kaminpar-shm/partitioning/deep/vcycle_deep_multilevel.h` / `.cc`
- `kaminpar-shm/partitioning/rb/rb_multilevel.h` / `.cc`
- `kaminpar-shm/partitioning/kway/kway_multilevel.h` / `.cc`
- `kaminpar-shm/partitioning/partitioner.h` (interface)

---

## 3. Phase 1: Coarsening

Coarsening repeatedly shrinks the graph by grouping nodes into clusters and contracting each cluster into a single supernode. This builds a hierarchy of progressively smaller graphs. Coarsening stops when the graph is small enough (below `contraction_limit`, default ~2000 nodes) or the graph stops shrinking (convergence threshold).

### 3.1 Clustering

The primary clustering algorithm is **Label Propagation (LP)**. Each node adopts the cluster label of the plurality of its neighbors (weighted by edge weight), subject to a maximum cluster weight constraint. This runs for a configurable number of iterations with randomized node ordering.

**Source:** `kaminpar-shm/coarsening/clustering/lp_clusterer.h` / `.cc`

### 3.2 Contraction

Once a clustering is computed, the graph is contracted: all nodes in the same cluster become one coarse node, and edges between clusters are merged (weights summed). Three contraction strategies exist:

| Strategy | Description |
|----------|-------------|
| `BUFFERED` | Thread-local edge buffers, then merge (default, best parallelism) |
| `UNBUFFERED` | Direct atomic writes (lower memory, higher contention) |
| `UNBUFFERED_NAIVE` | Simple reference implementation |

**Source:**
- `kaminpar-shm/coarsening/contraction/cluster_contraction.h` / `.cc` (dispatcher)
- `kaminpar-shm/coarsening/contraction/buffered_cluster_contraction.h` / `.cc`
- `kaminpar-shm/coarsening/contraction/unbuffered_cluster_contraction.h`
- `kaminpar-shm/coarsening/contraction/cluster_contraction_preprocessing.h` / `.cc`

### 3.3 Coarsener Variants

| Coarsener | Description |
|-----------|-------------|
| `BasicClusterCoarsener` | Single LP clustering per level, then contract (default) |
| `OverlayClusterCoarsener` | Computes multiple independent clusterings, overlays them, then contracts |
| `SparsificationClusterCoarsener` | Clustering + edge sparsification to control density growth |
| `NoopCoarsener` | Disables coarsening (single-level partitioning) |

All cluster-based coarseners inherit from `AbstractClusterCoarsener`, which manages the graph hierarchy and clustering algorithm selection.

**Source:**
- `kaminpar-shm/coarsening/coarsener.h` (interface)
- `kaminpar-shm/coarsening/abstract_cluster_coarsener.h` / `.cc`
- `kaminpar-shm/coarsening/{basic,overlay,sparsification}_cluster_coarsener.h` / `.cc`

---

## 4. Phase 2: Initial Partitioning

Once the graph is small enough, KaMinPar computes an initial partition on the coarsest graph. This uses a **portfolio bipartitioning** approach: multiple algorithms run on the small graph, and the best result is kept.

### 4.1 Bipartitioning Algorithms

| Algorithm | Mechanism |
|-----------|-----------|
| **BFS** (5 variants) | Find two far-apart seed nodes via BFS, then grow two blocks. Variants differ in block selection strategy (alternating, lighter block, sequential, longer/shorter queue). |
| **Greedy Graph Growing (GGG)** | Priority-queue-driven expansion: always add the node with maximum edge-weight gain to the current block. |
| **Random** | Randomly assign nodes to blocks while respecting balance. |

All algorithms are run multiple times (configurable repetitions) via the `InitialPoolBipartitioner`, which adaptively skips algorithms that are unlikely to produce the best result.

**Source:**
- `kaminpar-shm/initial_partitioning/bipartitioning/initial_pool_bipartitioner.h` / `.cc`
- `kaminpar-shm/initial_partitioning/bipartitioning/initial_bfs_bipartitioner.h` / `.cc`
- `kaminpar-shm/initial_partitioning/bipartitioning/initial_ggg_bipartitioner.h` / `.cc`
- `kaminpar-shm/initial_partitioning/bipartitioning/initial_random_bipartitioner.h` / `.cc`

### 4.2 Initial Partitioning Refinement

After each bipartition attempt, a 2-way refinement can be applied:

- **2-way FM** — Fiduccia-Mattheyses local search with two priority queues (simple or adaptive stopping)
- **2-way Flow** — Min-cut flow-based refinement
- **Noop** — No refinement

**Source:**
- `kaminpar-shm/initial_partitioning/refinement/initial_fm_refiner.h`
- `kaminpar-shm/initial_partitioning/refinement/initial_twoway_flow_refiner.h`
- `kaminpar-shm/initial_partitioning/refinement/initial_noop_refiner.h`

### 4.3 Initial Partitioning Coarsening

The initial partitioning phase itself runs a **nested multilevel scheme** — the small graph is coarsened further using sequential label propagation, then bipartitioned, then uncoarsened with refinement. This is managed by `InitialMultilevelBipartitioner`.

**Source:**
- `kaminpar-shm/initial_partitioning/initial_multilevel_bipartitioner.h` / `.cc`
- `kaminpar-shm/initial_partitioning/coarsening/initial_coarsener.h` / `.cc`

### 4.4 Thread Pool

Initial bipartitioning is parallelized via `InitialBipartitionerWorkerPool`, which maintains thread-local `InitialMultilevelBipartitioner` instances to avoid allocation overhead.

**Source:** `kaminpar-shm/initial_partitioning/initial_bipartitioner_worker_pool.h`

---

## 5. Phase 3: Uncoarsening and Refinement

After initial partitioning, the algorithm walks back up the coarsening hierarchy. At each level:

1. **Project** — map the partition from the coarse graph to the finer graph using the stored fine-to-coarse mapping.
2. **Refine** — improve the partition using local search algorithms.
3. **Extend** (deep multilevel only) — if the target k has not been reached, further bipartition blocks via recursive bipartitioning.

### 5.1 Refinement Algorithms

Multiple refiners can be chained via `MultiRefiner`, executing in sequence:

| Algorithm | Class | Description |
|-----------|-------|-------------|
| **Label Propagation** | `LabelPropagationRefiner` | Fast parallel k-way LP: each node considers moving to its neighbors' blocks. |
| **k-way FM** | `FMRefiner` | Parallel Fiduccia-Mattheyses: localized search with gain caches. Supports sparse, dense, and hashing gain cache strategies. |
| **Two-way Flow** | `TwowayFlowRefiner` | Flow-based refinement using preflow-push max-flow. Finds minimum cuts between pairs of blocks. Supports sequential and parallel scheduling. |
| **JET** | `JetRefiner` | GPU-inspired refinement with simulated annealing (gain temperature) and integrated load balancing. |
| **Overload Balancer** | `OverloadBalancer` | Greedy per-block balancing: moves nodes from overloaded to underloaded blocks. |
| **Underload Balancer** | `UnderloadBalancer` | MultiQueue-based approach targeting minimum block weights. |
| **MtKaHyPar** | `MtKaHyParRefiner` | External adapter to the MtKaHyPar hypergraph partitioner. |

**Source:**
- `kaminpar-shm/refinement/refiner.h` (interface)
- `kaminpar-shm/refinement/multi_refiner.h` / `.cc` (pipeline orchestrator)
- `kaminpar-shm/refinement/lp/lp_refiner.h` / `.cc`
- `kaminpar-shm/refinement/fm/fm_refiner.h` / `.cc`
- `kaminpar-shm/refinement/flow/twoway_flow_refiner.h` / `.cc`
- `kaminpar-shm/refinement/jet/jet_refiner.h` / `.cc`
- `kaminpar-shm/refinement/balancer/overload_balancer.h` / `.cc`
- `kaminpar-shm/refinement/balancer/underload_balancer.h` / `.cc`
- `kaminpar-shm/refinement/gains/` (gain cache implementations)

### 5.2 Flow Refinement Details

The flow-based refiner is the most complex. It builds a flow network from the border region between two blocks, runs preflow-push max-flow (sequential or parallel), and uses the minimum cut to reassign nodes. Components:

- **Flow network construction:** `kaminpar-shm/refinement/flow/flow_network/`
- **Max-flow algorithms:** `kaminpar-shm/refinement/flow/max_flow/` (preflow-push, parallel variants)
- **Block scheduling:** `kaminpar-shm/refinement/flow/scheduler/`

---

## 6. Configuration and Presets

All algorithm parameters are stored in the `Context` struct hierarchy:

```
Context
├── PartitioningContext (mode: DEEP, VCYCLE, RB, KWAY)
├── PartitionContext (k, block weights, epsilon)
├── CoarseningContext
│   ├── ClusterCoarseningContext (LP parameters)
│   └── ContractionCoarseningContext
├── InitialPartitioningContext
│   ├── InitialCoarseningContext
│   └── InitialPoolPartitionerContext
└── RefinementContext
    ├── KwayFMRefinementContext
    ├── TwowayFlowRefinementContext
    ├── JetRefinementContext
    └── ...
```

**Source:** `include/kaminpar-shm/kaminpar.h` (context structs, starting ~line 100)

### Presets

Presets configure the full pipeline for different quality/speed tradeoffs:

| Preset | Description |
|--------|-------------|
| `default` | Fast, METIS-quality. LP coarsening + LP/FM refinement. |
| `fast` | Speed-optimized variant. |
| `eco` / `fm` | More FM refinement iterations for better quality. |
| `strong` / `flow` | Adds flow-based refinement for highest quality. |
| `terapart` | Memory-efficient for very large graphs. |
| `largek` | Optimized for k > 1024. |
| `jet` | Uses JET refinement instead of FM. |
| `vcycle` | Multiple V-cycles over the multilevel hierarchy. |

**Source:** `kaminpar-shm/presets.h` / `.cc`

---

## 7. Putting It All Together

The default pipeline (`DEEP` mode with `default` preset) works as follows:

1. **Read/construct graph** into CSR or compressed format.
2. **Coarsen** using `BasicClusterCoarsener` with label propagation clustering and buffered contraction. Build a hierarchy of ~10-20 levels until the graph has ~2000 nodes.
3. **Initial partition** the coarsest graph using the portfolio bipartitioner (BFS + GGG + Random, best-of-N). The initial partitioning itself runs a nested multilevel scheme.
4. **Uncoarsen** level by level:
   - Project partition to the finer graph.
   - Run label propagation refinement, then FM refinement.
   - Extend the partition (split blocks via recursive bipartitioning) to reach the target k.
5. **Output** the final partition as `partition[node] = block_id`.

All phases are parallelized using Intel TBB.

---

## 8. Key Files Quick Reference

| Component | File |
|-----------|------|
| Public API | `include/kaminpar-shm/kaminpar.h` |
| Main orchestration | `kaminpar-shm/kaminpar.cc` |
| Configuration structs | `include/kaminpar-shm/kaminpar.h` (~line 100+) |
| Presets | `kaminpar-shm/presets.cc` |
| Factory (creates all components) | `kaminpar-shm/factories.cc` |
| Deep multilevel pipeline | `kaminpar-shm/partitioning/deep/deep_multilevel.cc` |
| Coarsener interface | `kaminpar-shm/coarsening/coarsener.h` |
| LP clustering | `kaminpar-shm/coarsening/clustering/lp_clusterer.cc` |
| Graph contraction | `kaminpar-shm/coarsening/contraction/buffered_cluster_contraction.cc` |
| Initial bipartitioner pool | `kaminpar-shm/initial_partitioning/bipartitioning/initial_pool_bipartitioner.cc` |
| Refinement interface | `kaminpar-shm/refinement/refiner.h` |
| FM refinement | `kaminpar-shm/refinement/fm/fm_refiner.cc` |
| LP refinement | `kaminpar-shm/refinement/lp/lp_refiner.cc` |
| Flow refinement | `kaminpar-shm/refinement/flow/twoway_flow_refiner.cc` |
| CSR graph | `kaminpar-shm/datastructures/csr_graph.h` |
| Compressed graph | `kaminpar-shm/datastructures/compressed_graph.h` |
| Partitioned graph | `kaminpar-shm/datastructures/partitioned_graph.h` |
