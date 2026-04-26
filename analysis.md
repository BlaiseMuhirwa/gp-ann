# Scaling GP-ANN to 2000 UPMEM DPUs: Assessment

## Your intuition is right — but the story is nuanced

The core concern is well-founded: graph partitioning quality *does* degrade with more partitions. But the degradation isn't just about the optimizer struggling — it's more fundamental than that. Here's a systematic breakdown:

## 1. Partition Quality: The Edge-Cut Problem

With k=40 partitions of 1B points, the paper gets ~96% of a point's 10 neighbors in the same shard. At k=2000, this drops dramatically due to a geometric argument:

- Each point has ~10 true nearest neighbors in the k-NN graph fed to KaMinPar
- With k=40, a balanced shard holds 1/40 = 2.5% of all points, but the graph locality means neighbors cluster — so you keep ~96% in one shard
- With k=2000, a shard holds 1/2000 = 0.05% of points. Even with perfect partitioning, a point's 10 neighbors occupy a *tiny* neighborhood in the high-dimensional space. The probability that all 10 neighbors fit in a shard of 500K points (out of 1B) shrinks fast

The edge-cut ratio (fraction of edges crossing partition boundaries) typically scales roughly as O(k^α) for some α < 1 for well-structured graphs (mesh-like). For k-NN graphs on high-dimensional data, this is worse because the graph has less geometric locality than a 2D mesh. **Estimate: oracle recall@1-probe would drop from ~96% to perhaps 30-50% at k=2000**, meaning you'd need 5-15 probes to match the recall that 1-2 probes achieves at k=40.

## 2. KaMinPar Scalability

KaMinPar (`partitioning.cpp:172-199`) uses a multilevel scheme. It can technically handle k=2000, but:

- **Quality**: KaMinPar's recursive bisection / direct k-way approach has to make more granular decisions at k=2000. The solution space is vastly larger. The `epsilon=0.05` balance constraint becomes harder to satisfy simultaneously with low edge cuts
- **Runtime**: Multilevel partitioners generally scale as O(|E| * log(k)) or similar — going from k=40 to k=2000 is a ~5.6x increase in partitioning time, which is manageable
- The `strong` mode (`kaminpar::shm::create_strong_context()`) helps quality but adds runtime

## 3. Routing Becomes the Bottleneck

This is where the real problems surface. Both routers were designed for ~40 shards:

### KMeans Tree Router (`kmeans_tree_router.cpp`)
- Builds one tree per shard (`roots.resize(num_shards)` at line 11)
- With 2000 shards, you'd have 2000 trees. The priority queue in `Query()` (line 93-130) starts by pushing all 2000 roots, then explores based on budget
- Budget=50,000 spread across 2000 shards = ~25 distance computations per shard tree — very shallow exploration
- The routing itself becomes O(budget * log(2000)) due to the priority queue, which is fine, but the *quality* suffers because you can't explore deeply enough

### HNSW Router (`hnsw_router.h`)
- Builds one HNSW on representative points from all shards
- `Query()` (line 106-119) searches for `num_voting_neighbors` nearest representatives, then votes by shard
- With 2000 shards and, say, 500 voting neighbors, many shards get 0 votes — the routing becomes extremely noisy
- You'd need num_voting_neighbors >> 2000 to reliably distinguish among shards, which defeats the purpose

**Net effect**: Routing cost would need to grow substantially (perhaps 10-50x) to maintain quality, and at that point routing may dominate total query latency.

## 4. Per-Shard Search: Paradoxically Gets Easier but Less Useful

With 1B/2000 = 500K points per shard:
- HNSW on 500K points is very fast and cheap to build
- But the recall loss from routing errors dominates — searching a shard perfectly when it's the *wrong* shard doesn't help

## 5. UPMEM DPU-Specific Constraints

This is where the plan really breaks down:

### Memory
Each DPU has ~64MB MRAM.
- 500K points × 100 dims × 4 bytes = **200MB** — already 3x the MRAM
- Even at 128 dims (SIFT-1B): 500K × 128 × 1 byte (uint8) = 64MB — barely fits with zero room for the index
- You'd need to use PQ (product quantization) or store compressed vectors, which adds recall loss on top of the routing loss

### No hardware FPU
DPUs have integer ALUs only.
- All distance computations in `src/dist.h` use `float`
- You'd need fixed-point or integer-quantized distance functions
- This is solvable but adds engineering complexity and more approximation error

### HNSW on DPU
Building and querying HNSW requires pointer-chasing (random memory access), which is not what DPUs excel at. Brute-force scan might actually be faster on DPUs for 500K points since they favor streaming access patterns.

## 6. What Could Work Instead

If you want to use 2000 DPUs for ANN search, the GP-ANN approach as-is is a poor fit. But elements of it could be adapted:

### Option A: Hierarchical Partitioning
- Partition into ~40 groups of ~50 DPUs each
- First-level routing picks 2-3 groups (fast, same quality as paper)
- Within each group, route to 2-3 DPUs (much easier 50-way problem)
- Total probes: 4-9 DPUs out of 2000, still a 200-500x reduction
- Routing quality: much better since each level is a tractable partitioning problem

This would map naturally onto the `OurPyramidPartitioning` code at `partitioning.cpp:509-552`, which already does hierarchical k-means + graph partitioning on a coarsened graph.

### Option B: Replicated Routing + Brute-Force Shards
- Use the HNSW router on the host CPU (plenty of memory)
- DPUs only do brute-force scan on their shard (plays to DPU strengths: streaming computation)
- With PQ-compressed vectors, 500K points might fit in 64MB
- Host routes → sends query to N DPUs → DPUs scan in parallel → host merges

### Option C: Overlapping Partitions (OGP)
- The overlapping approach (`overlapping_partitioning.cpp`) assigns border points to multiple shards
- With 20% overlap, more neighbors are reachable per shard, reducing probes needed
- But at k=2000, even 20% overlap may not help enough, and it increases per-shard size

## Summary Table

| Aspect | k=40 (paper) | k=2000 (DPU) | Verdict |
|--------|-------------|---------------|---------|
| Oracle recall@1 | ~96% | ~30-50% | Fundamental degradation |
| Probes needed for 95% recall | 1-2 | 10-30+ | Routing cost explodes |
| KaMinPar runtime | Minutes | ~5x more | Manageable |
| Routing cost/query | ~5% of total | Could dominate | Architecture mismatch |
| Memory per shard | GBs | 64MB MRAM | Need quantization |
| HNSW per shard | Sweet spot | Overkill for 500K pts | Brute-force may be better |

## Bottom Line

The flat k=2000 graph partitioning would degrade badly. A **two-level hierarchical** approach (40 groups × 50 DPUs) is the most promising path — it keeps each level's partitioning in the regime where GP-ANN works well, and the code already has hierarchical building blocks (`OurPyramidPartitioning`, `HierarchicalKMeans`) that could be built upon.
