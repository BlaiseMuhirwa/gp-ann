# Routing in GP-ANN: KMeans Tree vs HNSW Router

## Overview

Both routers solve the same problem: given a query point Q, rank the k shards by likelihood of containing Q's true nearest neighbors. They differ in data structure, search strategy, and how they select representative points.

## What Points Does the HNSW Router Use?

The HNSW router is **point-set agnostic** — it takes any set of representative points tagged with shard IDs. However, in this codebase there are three different sources of representative points:

### Source 1: KMeans Tree Centroids (main pipeline)

In the main evaluation flow (`routes.cpp:164-266`), the HNSW router reuses centroids from the KMeans tree:

```
1. Train KMeansTreeRouter on each shard's points
2. Run tree-based routing queries
3. ExtractPoints() — BFS all tree nodes, collect every centroid from every level
4. Build HNSWRouter on those extracted centroids
5. Run HNSW routing queries
```

`ExtractPoints()` (`kmeans_tree_router.cpp:174-201`) walks every tree node across all shards via BFS, concatenates their centroid vectors, and tags each centroid with its shard ID. So if you have 40 shards, each with a tree containing ~1250 centroids (budget=50,000 / 40), you'd get ~50,000 routing points total.

This means the KMeans tree is trained first, but **only as a way to generate good representative points**. The HNSW router doesn't use the tree structure at all.

### Source 2: Pyramid Partitioning Points

For the `Pyramid` method (`routes.cpp:287-298`), the HNSW index is built during partitioning time on 10,000 aggregate k-means centroids (`partitioning.cpp:275-291`). It's loaded from disk at query time — no KMeans tree involved.

### Source 3: OurPyramid Partitioning Points

For `OurPyramid` (`routes.cpp:300-311`), representative points come from hierarchical k-means coarsening (`partitioning.cpp:509-528`). Again loaded from disk with no tree dependency.

**Bottom line: the HNSWRouter class is fully independent. The tree dependency is just a convenient way to generate representatives in the default pipeline.**

## KMeans Tree Router: How It Works

### Training (`kmeans_tree_router.cpp:9-78`)

Builds one independent tree per shard:

```
For each shard b:
    roots[b] = BuildTree(shard_b_points, budget_for_shard_b)

BuildTree(points, budget):
    1. Sample min(num_centroids, budget) centroids via k-means
    2. Partition points into buckets by nearest centroid
    3. Separate buckets into "large" (> min_cluster_size) and "small" (leaves)
    4. Recurse on large buckets, splitting budget proportionally by bucket size
    5. Stop when budget exhausted or single centroid remains
```

The budget is distributed proportionally: `budget_for_shard = (shard_size / total_points) * total_budget`. Within each tree, it's further split proportionally across subtrees.

With default settings (num_centroids=64, min_cluster_size=250, budget=50,000, 40 shards), each shard gets ~1,250 centroid budget, producing trees of depth ~2-3.

### Query (`kmeans_tree_router.cpp:93-130`)

Uses a **global priority queue** across all shard trees:

```
Query(Q, budget):
    1. Push all k shard roots into a min-heap (by distance to Q)
    2. While budget > 0:
        a. Pop closest node from heap
        b. Compute distance from Q to all centroids in that node
        c. Update min_dist[shard] with closest centroid seen
        d. Push child nodes into heap (keyed by centroid distance)
        e. budget -= num_centroids_in_node
    3. Sort shards by min_dist, return ranked list
```

Key insight: the priority queue interleaves exploration across *all* shard trees. A shard whose top-level centroids are far from Q will be deprioritized, and its subtrees won't be explored. Budget naturally concentrates on promising shards.

### Frequency Query Variant (`kmeans_tree_router.cpp:132-172`)

Same traversal, but also maintains a `TopN` of the closest centroids seen globally (across all shards). After traversal, counts how many of the top-N centroids belong to each shard. The shard with highest frequency is probed first, then the rest are sorted by min_dist.

## HNSW Router: How It Works

### Training (`hnsw_router.h:46-48`)

Straightforward — insert all representative points into a single HNSW graph:

```
Train(routing_points):
    for each point i in routing_points:
        hnsw.addPoint(point_i, i)
```

Each point has an ID, and a separate `partition[]` array maps point_id → shard_id.

### Query (`hnsw_router.h:106-119`)

```
Query(Q, num_voting_neighbors):
    1. Search HNSW for num_voting_neighbors nearest representative points
    2. For each returned (distance, point_id):
        shard = partition[point_id]
        min_dist[shard] = min(min_dist[shard], distance)
        frequency[shard]++
    3. Return ShardPriorities with both min_dist and frequency arrays
```

Then one of four ranking strategies picks the probe order:

- **RoutingQuery**: Sort shards by min_dist (closest representative wins)
- **FrequencyQuery**: Shard with highest vote count goes first, rest sorted by min_dist
- **PyramidRoutingQuery**: Only probe shards that had *any* representative in the results
- **SPANNRoutingQuery(eps)**: Probe all shards within (1+eps) factor of the closest shard's distance

## Side-by-Side Comparison

| Aspect | KMeans Tree Router | HNSW Router |
|--------|-------------------|-------------|
| **Data structure** | k independent trees (one per shard) | 1 shared HNSW graph over all representative points |
| **Representative points** | Generates its own centroids via hierarchical k-means | Takes any point set (typically the tree's centroids, but could be anything) |
| **Search cost** | Controlled by `budget` (total distance computations) | Controlled by `num_voting_neighbors` and `ef_search` |
| **Shard awareness** | Knows which tree = which shard; explores per-shard trees hierarchically | Shard-agnostic during search; only maps points→shards after retrieval |
| **Budget allocation** | Dynamic — concentrates budget on promising shards via PQ | Uniform — HNSW search doesn't know about shards, explores based on graph proximity |
| **Scaling with k shards** | Must push k roots into PQ; budget spread thin at large k | HNSW search is O(log n) regardless of k, but needs more voting neighbors as k grows |
| **Ranking strategies** | min_dist or frequency | min_dist, frequency, pyramid (presence-only), SPANN (distance threshold) |
| **Training cost** | k-means per shard (parallelized) | HNSW construction on all representative points |

## At 2000 Shards

**Tree router**: 2000 roots pushed into PQ. With budget=50,000, each shard gets ~25 distance computations — barely enough for one level of its tree. Routing quality degrades because the PQ can't explore deeply enough into any shard's tree.

**HNSW router**: The HNSW graph itself scales fine (log n search). But with 2000 shards and num_voting_neighbors=500, you average 0.25 votes per shard — too few to reliably identify the best shard. You'd need num_voting_neighbors >> 2000, making routing expensive.

Both approaches were designed and tested at k=40. Scaling to k=2000 would require fundamental changes, most likely a hierarchical routing scheme.
