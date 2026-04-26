# KMeans Tree Router: Structure and Query Routing

## Tree Structure (one tree per shard)

Each shard gets its own hierarchical k-means tree built during `Train`.
Within each tree node, centroids are ordered so that centroids **with children come first**.
Centroid `i` has a child subtree only if `i < node.children.size()`.

```mermaid
graph TD
    subgraph GlobalPQ["Global Min-Priority Queue (shared across all shards)"]
        PQ["Pop closest node across ALL shards<br/>Push children back with their distance"]
    end

    subgraph Shard0["Shard 0 Tree (roots[0])"]
        R0["Root Node<br/>64 centroids: c0..c63"]
        R0 --> C0a["Child 0<br/>64 centroids"]
        R0 --> C0b["Child 1<br/>64 centroids"]
        R0 --> C0c["Child 2<br/>..."]
        R0 ~~~ L0["Centroids 12..63<br/>are LEAVES<br/>(no children)"]
        C0a --> C0a1["Child 0.0<br/>centroids"]
        C0a --> C0a2["Child 0.1<br/>centroids"]
        C0b --> C0b1["Child 1.0<br/>centroids"]
    end

    subgraph Shard1["Shard 1 Tree (roots[1])"]
        R1["Root Node<br/>64 centroids: c0..c63"]
        R1 --> C1a["Child 0<br/>64 centroids"]
        R1 --> C1b["Child 1<br/>64 centroids"]
        R1 ~~~ L1["Centroids 8..63<br/>are LEAVES<br/>(no children)"]
    end

    subgraph ShardK["Shard k-1 Tree (roots[k-1])"]
        RK["Root Node<br/>64 centroids"]
        RK --> CKa["..."]
    end

    PQ -.->|"explores"| R0
    PQ -.->|"explores"| R1
    PQ -.->|"explores"| RK

    style GlobalPQ fill:#e94560,stroke:#e94560,color:#fff
    style L0 fill:#555,stroke:#555,color:#ccc
    style L1 fill:#555,stroke:#555,color:#ccc
```

## Query Routing: Best-First Interleaved Traversal

The key idea: all shard trees are explored **simultaneously** through a single
shared priority queue, ordered by distance to query. The budget limits total
distance computations across all shards.

```mermaid
sequenceDiagram
    participant Q as Query Point
    participant PQ as Min-Priority Queue
    participant S0 as Shard 0 Tree
    participant S1 as Shard 1 Tree
    participant S2 as Shard 2 Tree
    participant MD as min_dist[shard]

    Note over PQ: Initialize: push all root nodes<br/>with dist = -infinity

    Q->>PQ: Pop closest entry<br/>(Shard 1 root, dist=-inf)
    PQ->>S1: Visit root node, compute dist(Q, c_i)<br/>for all 64 centroids. budget -= 64
    S1->>MD: Update min_dist[1] = min of all centroid dists
    S1->>PQ: Push children back:<br/>{dist=0.3, shard=1, child0}<br/>{dist=0.7, shard=1, child1}<br/>...

    Q->>PQ: Pop closest entry<br/>(Shard 0 root, dist=-inf)
    PQ->>S0: Visit root node, compute dist(Q, c_i)<br/>for all 64 centroids. budget -= 64
    S0->>MD: Update min_dist[0] = min of all centroid dists
    S0->>PQ: Push children of large clusters

    Q->>PQ: Pop closest entry<br/>(Shard 1 child0, dist=0.3)
    PQ->>S1: Visit child0 node, compute dist(Q, c_i)<br/>for its centroids. budget -= 64
    S1->>MD: Update min_dist[1] if closer centroid found
    S1->>PQ: Push grandchildren

    Note over PQ: Shard 1 gets explored deeper<br/>because its centroids are closer to Q

    Q->>PQ: Pop closest entry<br/>(Shard 0 child2, dist=0.5)
    PQ->>S0: Visit child2... budget -= 64

    Note over PQ: ... continues until budget = 0 ...

    Q->>MD: Rank shards by min_dist
    Note over MD: Result: [shard1, shard0, shard2, ...]<br/>Probe top-η shards
```

## How shards get ranked

After the budget is exhausted, `min_dist[shard]` holds the closest centroid
distance seen for each shard. Shards are sorted by this value — the shard
whose tree had a centroid closest to the query gets probed first.

```mermaid
flowchart LR
    subgraph AfterTraversal["After budget exhausted"]
        MD0["min_dist[0] = 0.42"]
        MD1["min_dist[1] = 0.15"]
        MD2["min_dist[2] = 0.89"]
        MD3["min_dist[3] = 0.31"]
    end

    AfterTraversal --> Sort["Sort shards<br/>by min_dist"]

    Sort --> Ranked["Probe order:<br/>1. Shard 1 (0.15)<br/>2. Shard 3 (0.31)<br/>3. Shard 0 (0.42)<br/>4. Shard 2 (0.89)"]

    Ranked --> Probe["Probe top η shards<br/>(typically η = 1-3)"]

    style Probe fill:#e94560,stroke:#e94560,color:#fff
```

## Why the interleaving matters

The budget is **shared across all shards**. A shard whose root-level centroids
are far from the query won't get much exploration — its children sit deep in the
priority queue and the budget runs out before reaching them. A shard with a close
root centroid gets explored deeply (multiple levels), producing a tighter
`min_dist` estimate. This naturally allocates more computation to promising shards.

### Code references

| Step | Function | Location |
|------|----------|----------|
| Build per-shard trees | `KMeansTreeRouter::Train` | `src/kmeans_tree_router.cpp:9` |
| Recursive tree building | `KMeansTreeRouter::TrainRecursive` | `src/kmeans_tree_router.cpp:39` |
| Query routing (distance) | `KMeansTreeRouter::Query` | `src/kmeans_tree_router.cpp:111` |
| Query routing (frequency) | `KMeansTreeRouter::FrequencyQuery` | `src/kmeans_tree_router.cpp:151` |
| Tree node definition | `KMeansTreeRouter::TreeNode` | `src/kmeans_tree_router.h:57` |
| Router options | `KMeansTreeRouterOptions` | `src/kmeans_tree_router.h:6` |
