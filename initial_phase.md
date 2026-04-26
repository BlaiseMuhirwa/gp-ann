# Initial Phase: Fast, Inexact Graph Building

The entry point is `BuildApproximateNearestNeighborGraph` in `ApproximateKNNGraphBuilder` (`src/knn_graph.h:145`). It builds a rough approximate k-NN graph in two stages: recursive sketching to produce small buckets of nearby points, then brute-force within each bucket.

## How points get into buckets

Nearby points should end up in the same small bucket. Once they're in a small bucket, you can afford brute-force.

Think of it like a recursive tournament, implemented in `RecursivelySketch` (`src/knn_graph.h:52`):

1. **Pick random "leaders"** from the crowd — a tiny sample (0.5%) of all points. At the top level, 950 leaders are sampled; at deeper levels it's `FRACTION_LEADERS` (0.5%) of the current bucket size, capped at 1500.

2. **Every point walks to its closest leader(s).** Each point finds its nearest leaders via `ClosestLeaders` (`src/defs.cpp:87`), which brute-force scans over all leaders using a `TopN` max-heap (`src/topn.h`). At the first level, each point picks its 3 closest leaders (`FANOUT=3`), so it gets copied into 3 groups. This redundancy is important — if your true nearest neighbor happened to walk to a different leader than you, the fact that you joined 3 groups gives you more chances to land in the same group as them.

3. **Each group is still too big**, so you recurse — sample new leaders within each group, assign points to their closest leader (now just 1 since the groups are already somewhat local), and split again.

4. **Keep splitting until groups are small enough** (≤ `MAX_CLUSTER_SIZE` = 5000 points). These final small groups are your "buckets." Tiny clusters below `MIN_CLUSTER_SIZE` (50) are merged together to avoid isolated nodes.

5. **Repeat the whole thing 3 times** (`REPETITIONS=3`) with different random leader samples. Each repetition produces a different set of buckets, giving every point even more chances to land in a bucket with its true neighbors.

There is a safety valve for near-duplicate points: if recursion exceeds `MAX_DEPTH` (14) or a cluster isn't shrinking past `CONCERNING_DEPTH` (10), it falls back to random splitting instead of recursing forever.

## What happens inside buckets

Once all repetitions produce their buckets, `BruteForceBuckets` (`src/knn_graph.h:187`) processes them:

- `CrunchBucket` (`src/knn_graph.h:165`) computes every pairwise distance within each bucket — this is cheap because buckets are small (≤5000 points).
- Each point remembers its best k neighbors across all buckets it appeared in. Neighbors from different buckets are merged under a per-point spinlock (`src/spinlock.h`), deduplicated, and trimmed to the global top-k.
- The output is a plain adjacency list (`AdjGraph`).

## Post-processing

After the graph is built, `Symmetrize` (`src/knn_graph.h:272`) makes it undirected — if point A has B as a neighbor, B also gets A. This is required by the KaMinPar graph partitioner that consumes the graph in the next phase.

## Why "inexact" is fine

Some true nearest neighbors will inevitably end up in different buckets and never get compared. The graph will have some wrong edges and miss some real neighbors. But the paper's insight is that this doesn't matter — the graph only needs to be good enough for the partitioner to understand the rough neighborhood structure. It gets thrown away after partitioning anyway.
