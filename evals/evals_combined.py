import basalt
from pydantic import BaseModel, Field
import numpy as np
from sklearn.cluster import KMeans
import tqdm
import time
from abc import ABC, abstractmethod
import argparse
from typing import Literal, Union, Annotated, Iterable
import yaml
from collections import defaultdict, Counter
import pandas as pd
from collections import deque
import json
import os
import kaminpar


class IndexConfig(BaseModel):
    distance: str = "l2"
    max_nbrs: int
    ef_construction: int = 100
    ef_search: int = 100


class TrainedRouter(ABC):
    @abstractmethod
    def partitions(self) -> list[np.ndarray]:
        pass

    @abstractmethod
    def route(self, query: np.ndarray) -> list[int]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @staticmethod
    def load(path: str) -> "TrainedRouter":
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        router_type = config["type"]
        if router_type == "kmeans":
            return TrainedKMeansRouter.load(path)
        elif router_type == "hierarchical":
            return HierarchicalRouter.load(path)
        else:
            raise ValueError(f"Unknown router type: '{router_type}'")


class Router(ABC):
    @abstractmethod
    def train(self, dataset: np.ndarray) -> TrainedRouter:
        pass


def get_nearest_centroids(
    centroids: np.ndarray, queries: np.ndarray, n_probes: int
) -> np.ndarray:
    if queries.dtype != centroids.dtype:
        queries = queries.astype(centroids.dtype)

    # The squared norm can be optimized because (a - b)^2 = a^2 + b^2 - 2ab
    q_sq = (queries * queries).sum(axis=1, keepdims=True)  # (N, 1)
    c_sq = (centroids * centroids).sum(axis=1, keepdims=True).T  # (1, M)
    dists = q_sq + c_sq - 2 * np.dot(queries, centroids.T)

    if n_probes == 1:
        return np.argmin(dists, axis=1, keepdims=True)
    return np.argpartition(dists, n_probes, axis=1)[:, :n_probes]


class TrainedKMeansRouter(TrainedRouter):
    centroids: np.ndarray
    n_query_probes: int
    clusters: list[np.ndarray]  # list of data point ids per cluster

    def __init__(
        self, centroids: np.ndarray, n_query_probes: int, clusters: list[np.ndarray]
    ):
        self.centroids = centroids
        self.n_query_probes = n_query_probes
        self.clusters = clusters

    def partitions(self) -> list[np.ndarray]:
        return self.clusters

    def route(self, query: np.ndarray) -> list[int]:
        return list(
            get_nearest_centroids(
                self.centroids, query.reshape(1, -1), n_probes=self.n_query_probes
            )[0]
        )

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        config = {"type": "kmeans", "n_query_probes": self.n_query_probes}
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        np.save(os.path.join(path, "centroids.npy"), self.centroids)
        np.savez(os.path.join(path, "clusters.npz"), *self.clusters)

    @classmethod
    def load(cls, path: str) -> "TrainedKMeansRouter":
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        centroids = np.load(os.path.join(path, "centroids.npy"))
        clusters_data = np.load(os.path.join(path, "clusters.npz"))
        clusters = [
            clusters_data[k]
            for k in sorted(clusters_data.files, key=lambda x: int(x.split("_")[1]))
        ]
        return cls(
            centroids=centroids,
            n_query_probes=config["n_query_probes"],
            clusters=clusters,
        )


class GraphPartioningConfig(BaseModel):
    overlap_frac: float
    knn_index_config: IndexConfig
    knn_nbrs: int


class HierarchicalConfig(BaseModel):
    # The maximum number of points to sample across partitions
    budget: int
    # The max numbers of centroids for each level of KMeans
    max_level_size: int
    # The topk to select at query time to determine what to route to
    topk: int
    # The strategy to use to map the queried points to partitions. Options:
    #   freq - route to the most frequently partitions in the topk
    #   best - map to the partitions with the closest points in the topk
    #   sum - group points in topk by partition, sum (1 - dist) for all points, route by highest sum
    selection: str


class KMeansRouter(Router, BaseModel):
    type: Literal["kmeans"] = "kmeans"
    n_partitions: int
    n_insert_probes: int
    n_query_probes: int

    # this is used for the graph based overlap here, not actual graph partitioning
    overlap: GraphPartioningConfig | None = None

    hierarchical: HierarchicalConfig | None = None

    def train(self, dataset: np.ndarray, max_points: int = 1_000_000) -> TrainedRouter:
        kmeans = KMeans(n_clusters=self.n_partitions, n_init="auto", random_state=42)
        if len(dataset) <= max_points:
            kmeans.fit(dataset.astype(np.float32, copy=False))
        else:
            points = np.arange(len(dataset))
            np.random.shuffle(points)
            subsample = dataset[points[:max_points]]
            kmeans.fit(subsample.astype(np.float32, copy=False))

        centroids = kmeans.cluster_centers_

        batch_size = 10000
        assignments = [
            get_nearest_centroids(
                centroids,
                dataset[start : start + batch_size],
                n_probes=self.n_insert_probes,
            )
            for start in tqdm.trange(
                0, len(dataset), batch_size, desc="Assigning clusters"
            )
        ]

        assignments = np.vstack(assignments)

        partitions = [[] for _ in range(self.n_partitions)]
        for i, clusters in enumerate(assignments):
            for c in clusters:
                partitions[c].append(i)

        partitions = list(map(np.array, partitions))

        if self.overlap is not None:
            if self.n_insert_probes > 1:
                print(
                    "WARNING: overlap for kmeans router is not well defined if insert_probes > 1"
                )
            out_edges = _build_knn_graph(
                dataset=dataset,
                index_config=self.overlap.knn_index_config,
                knn_nbrs=self.overlap.knn_nbrs,
            )

            partitions = _add_partition_overlap(
                out_edges=out_edges,
                partitions=partitions,
                node_to_partition=assignments[:, 0],
                overlap_frac=self.overlap.overlap_frac,
            )

        if self.hierarchical:
            return HierarchicalRouter(
                dataset=dataset,
                partitions=partitions,
                budget=self.hierarchical.budget,
                max_level_size=self.hierarchical.max_level_size,
                topk=self.hierarchical.topk,
                n_query_probes=self.n_query_probes,
                selection=self.hierarchical.selection,
            )

        return TrainedKMeansRouter(
            centroids=centroids, n_query_probes=self.n_query_probes, clusters=partitions
        )


class HierarchicalRouter(TrainedRouter):
    def __init__(
        self,
        dataset: np.ndarray,
        partitions: list[np.ndarray],
        budget: int,
        max_level_size: int,
        topk: int,
        n_query_probes: int,
        selection: str,
    ):
        s = time.perf_counter()
        samples = HierarchicalRouter._find_samples(
            dataset,
            partitions=partitions,
            budget=budget,
            max_level_size=max_level_size,
        )
        e = time.perf_counter()

        sample_to_partition, samples = zip(*samples)

        print(f"Selected {len(samples)} hierarchical samples in {e-s:.3f}s")

        self._index = basalt.build_index(
            distance="l2",
            data=np.vstack(samples),
            max_nbrs=32,
            ef_construction=100,
            parallel=True,
            verbose=False,
        )

        self._sample_to_partition = sample_to_partition
        self._partitions = partitions
        self._topk = topk
        self._n_query_probes = n_query_probes
        self._selection = selection

    def partitions(self) -> list[np.ndarray]:
        return self._partitions

    def route(self, query: np.ndarray) -> list[int]:
        closest = self._index.query(
            query.astype(np.float32, copy=False), ef_search=100, topk=self._topk
        )

        per_partition = defaultdict(list)
        for i, s in closest:
            per_partition[self._sample_to_partition[i]].append(s)

        if self._selection == "freq":
            freq = sorted(
                [
                    (partition, len(scores))
                    for partition, scores in per_partition.items()
                ],
                key=lambda x: x[1],
                reverse=True,
            )

            return [p for p, _ in freq[: self._n_query_probes]]
        elif self._selection == "best":
            best = sorted(
                [
                    (partition, max(scores))
                    for partition, scores in per_partition.items()
                ],
                key=lambda x: x[1],
            )
            return [p for p, _ in best[: self._n_query_probes]]
        elif self._selection == "sum":
            sums = sorted(
                [
                    (partition, sum(1 - s for s in scores))
                    for partition, scores in per_partition.items()
                ],
                key=lambda x: x[1],
            )
            return [p for p, _ in sums[: self._n_query_probes]]
        else:
            raise ValueError(f"Invalid selection: '{self._selection}'")

    @staticmethod
    def _find_samples(
        dataset: np.ndarray,
        partitions: list[np.ndarray],
        budget: int,
        max_level_size: int,
    ):
        partition_budgets = HierarchicalRouter._distribute_budget(
            budget, list(map(len, partitions))
        )
        queue = deque(zip(range(len(partitions)), partitions, partition_budgets))

        samples = []
        while len(queue) > 0:
            partition, points, rem_budget = queue.popleft()

            n_clusters = min(rem_budget, max_level_size)

            sub_cluster = dataset[points].astype(np.float32, copy=False)

            if n_clusters > len(points):
                for s in sub_cluster:
                    samples.append((partition, s))
                continue

            kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
            assignments = kmeans.fit_predict(sub_cluster)

            clusters = [[] for _ in range(n_clusters)]
            for p, a in zip(points, assignments):
                clusters[a].append(p)

            budgets = HierarchicalRouter._distribute_budget(
                rem_budget, list(map(len, clusters))
            )

            for i, (cluster, cluster_budget) in enumerate(zip(clusters, budgets)):
                if cluster_budget > 1:
                    queue.append((partition, np.array(cluster), cluster_budget))
                else:
                    samples.append((partition, kmeans.cluster_centers_[i]))

        return samples

    @staticmethod
    def _distribute_budget(budget: int, cluster_sizes: list[int]) -> list[int]:
        assert budget >= len(cluster_sizes)

        total = sum(cluster_sizes)

        ideals = [
            size / total * (budget - len(cluster_sizes)) for size in cluster_sizes
        ]
        assigned = [int(i) + 1 for i in ideals]

        rem = budget - sum(assigned)
        assert (
            rem >= 0
        ), f"budget={budget}, cluster_sizes={cluster_sizes}, ideals={ideals}, assigned={assigned}"

        missing = [(i, ideal - a) for i, (ideal, a) in enumerate(zip(ideals, assigned))]
        missing.sort(key=lambda x: x[1], reverse=True)
        for i, _ in missing[:rem]:
            assigned[i] += 1

        return assigned

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        config = {
            "type": "hierarchical",
            "topk": self._topk,
            "n_query_probes": self._n_query_probes,
            "selection": self._selection,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        self._index.save(os.path.join(path, "index"))
        np.save(
            os.path.join(path, "sample_to_partition.npy"),
            np.array(self._sample_to_partition),
        )
        np.savez(os.path.join(path, "partitions.npz"), *self._partitions)

    @classmethod
    def load(cls, path: str) -> "HierarchicalRouter":
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        obj = object.__new__(cls)
        obj._index = basalt.load_index(
            os.path.join(path, "index"), backend="mem", distance="l2"
        )
        obj._sample_to_partition = list(
            np.load(os.path.join(path, "sample_to_partition.npy"))
        )
        partitions_data = np.load(os.path.join(path, "partitions.npz"))
        obj._partitions = [
            partitions_data[k]
            for k in sorted(partitions_data.files, key=lambda x: int(x.split("_")[1]))
        ]
        obj._topk = config["topk"]
        obj._n_query_probes = config["n_query_probes"]
        obj._selection = config["selection"]
        return obj


def _build_knn_graph(
    dataset: np.ndarray, index_config: IndexConfig, knn_nbrs: int
) -> list[list[int]]:
    print("Building index to get KNN graph")
    index = basalt.build_index(
        data=dataset,
        distance="l2",
        max_nbrs=index_config.max_nbrs,
        ef_construction=index_config.ef_construction,
        parallel=True,
        verbose=True,
        batch_size=1000,
    )

    out_edges = []
    batch_size = 1000
    for start in tqdm.trange(
        0, len(dataset), batch_size, desc="Building KNN graph edges"
    ):
        batch = dataset[start : start + batch_size]
        nbrs = index.query_batch(batch, topk=knn_nbrs, ef_search=index_config.ef_search)
        out_edges.extend([[x[0] for x in row] for row in nbrs])

    return out_edges


def _add_partition_overlap(
    out_edges: list[list[int]],
    partitions: list[Iterable[int]],
    node_to_partition: np.ndarray,
    overlap_frac: float,
) -> list[np.ndarray]:
    partitions = list(map(set, partitions))

    max_overlap = int(overlap_frac * len(node_to_partition) / len(partitions))
    partition_requests = defaultdict(list)
    for node, neighbors in enumerate(out_edges):
        node_partition = node_to_partition[node]
        most_freq = Counter(node_to_partition[nbr] for nbr in neighbors)
        for p, cnt in most_freq.items():
            if p != node_partition:
                partition_requests[p].append((node, cnt))

    for pid, requests in partition_requests.items():
        requests.sort(key=lambda x: x[1], reverse=True)
        added = 0
        for node, _ in requests:
            if added >= max_overlap:
                break
            partitions[pid].add(node)
            added += 1

    return [np.array(list(p)) for p in partitions]


class GraphPartitioningRouter(Router, BaseModel):
    type: Literal["graph-partitioning"] = "graph-partitioning"

    n_partitions: int
    n_query_probes: int
    graph_partitioning: GraphPartioningConfig
    hierarchical: HierarchicalConfig

    def train(self, dataset: np.ndarray):
        out_edges = _build_knn_graph(
            dataset=dataset,
            index_config=self.graph_partitioning.knn_index_config,
            knn_nbrs=self.graph_partitioning.knn_nbrs,
        )

        n = len(out_edges)

        # Pad rows with -1 so the array is rectangular (some nodes may have
        # fewer than knn_nbrs results if the index is small).
        out_edges_arr = np.full(
            (n, self.graph_partitioning.knn_nbrs), -1, dtype=np.int32
        )
        for i, nbrs in enumerate(out_edges):
            out_edges_arr[i, : len(nbrs)] = nbrs

        tmp_path = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), "kaminpar_graph.metis"
        )
        try:
            basalt.save_metis(out_edges_arr, tmp_path)
            graph = kaminpar.load_graph(tmp_path, kaminpar.GraphFileFormat.METIS)
        finally:
            os.unlink(tmp_path)

        print("Partitioning KNN graph")
        gp_s = time.perf_counter()
        # Partition using KaMinPar
        partitioner = kaminpar.KaMinPar(num_threads=1, ctx=kaminpar.default_context())
        partition = partitioner.compute_partition(graph, k=self.n_partitions, eps=0.03)
        gp_e = time.perf_counter()
        print(f"Parititioned graph in {gp_e-gp_s:.3f}s")

        # Group node IDs by partition
        partitions = [set() for _ in range(self.n_partitions)]
        for node_id in range(len(out_edges)):
            partitions[partition[node_id]].add(node_id)

        if self.graph_partitioning.overlap_frac > 0:
            partitions = _add_partition_overlap(
                out_edges=out_edges,
                partitions=partitions,
                node_to_partition=partition,
                overlap_frac=self.graph_partitioning.overlap_frac,
            )
        else:
            partitions = [np.array(list(p)) for p in partitions]

        return HierarchicalRouter(
            dataset=dataset,
            partitions=partitions,
            budget=self.hierarchical.budget,
            max_level_size=self.hierarchical.max_level_size,
            topk=self.hierarchical.topk,
            n_query_probes=self.n_query_probes,
            selection=self.hierarchical.selection,
        )


class ParitionedIndex:
    def __init__(
        self,
        router: TrainedRouter,
        partitions: list[np.ndarray],
        indexes: list,
        config: IndexConfig,
    ):
        self.router = router
        self.partitions = partitions
        self.indexes = indexes
        self.config = config

    @staticmethod
    def build(
        router: TrainedRouter,
        dataset: np.ndarray,
        config: IndexConfig,
    ):
        assert dataset.ndim == 2

        partitions = router.partitions()

        indexes = []
        for partition in tqdm.tqdm(partitions, desc="Building indexes"):
            data_partition = dataset[partition]

            indexes.append(
                basalt.build_index(
                    distance=config.distance,
                    data=data_partition,
                    max_nbrs=config.max_nbrs,
                    ef_construction=config.ef_construction,
                    parallel=True,
                    verbose=False,
                )
            )

        return ParitionedIndex(
            router=router, partitions=partitions, indexes=indexes, config=config
        )

    def query(self, query: np.ndarray, topk: int) -> list[tuple[int, float]]:
        partitions_to_query = self.router.route(query)

        results = []
        for partition in partitions_to_query:
            partial_results = self.indexes[partition].query(
                query=query, topk=topk, ef_search=self.config.ef_search
            )

            results.extend(
                (self.partitions[partition][i], s) for i, s in partial_results
            )

        results.sort(key=lambda x: x[1])

        topk_results = []
        seen = set()
        for i, s in results:
            if i not in seen:
                seen.add(i)
                topk_results.append((i, s))
            if len(topk_results) == topk:
                break

        return topk_results


def evaluate(
    index: ParitionedIndex,
    test: np.ndarray,
    groundtruths: np.ndarray,
    topk: int,
) -> dict:
    tp = 0
    fp = 0
    fn = 0
    total_time = 0

    for query, gtruths in zip(test, groundtruths):
        s = time.perf_counter()
        results = index.query(query, topk=topk)
        assert len(results) == topk
        e = time.perf_counter()
        total_time += e - s

        results = set(x[0] for x in results)
        gtruths = set(gtruths[:topk])

        tp += len(results.intersection(gtruths))
        fp += len(results.difference(gtruths))
        fn += len(gtruths.difference(results))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    avg_lat = total_time / len(test) * 1000

    return {
        "recall": recall,
        "precision": precision,
        "avg_latency_ms": avg_lat,
        "topk": topk,
    }


RouterType = Annotated[
    Union[KMeansRouter, GraphPartitioningRouter],
    Field(discriminator="type"),
]


class DatasetConfig(BaseModel):
    train: str
    test: str
    groundtruths: str


class ExperimentConfig(BaseModel):
    results: str
    dataset: DatasetConfig
    index: IndexConfig
    routers: dict[str, RouterType] = {}
    eval_at_k: list[int] = [10, 100]


def run_experiment(experiment: ExperimentConfig, existing_results: set[str]):
    train = np.load(experiment.dataset.train, mmap_mode="r")
    test = np.load(experiment.dataset.test)
    groundtruths = np.load(experiment.dataset.groundtruths)

    assert train.dtype in (np.float32, np.uint8)
    assert test.dtype in (np.float32, np.uint8)
    assert groundtruths.dtype in (np.int32, np.int64, np.uint32, np.uint64)

    results = []

    for router_name, router in experiment.routers.items():
        if router_name in existing_results:
            print(f"Router {router_name} has already run - skipping")
            continue

        trained_router = router.train(dataset=train)

        index = ParitionedIndex.build(
            router=trained_router,
            dataset=train,
            config=experiment.index,
        )

        index_sizes = list(map(len, trained_router.partitions()))

        metrics = [
            evaluate(index, test, groundtruths, topk=topk)
            for topk in experiment.eval_at_k
        ]

        results.append(
            {
                "router": router_name,
                "min_index_size_mb": min(index_sizes) / 1e6,
                "max_index_size_mb": max(index_sizes) / 1e6,
                "metrics": metrics,
            }
        )

    return results


def get_summary(res: dict) -> dict:
    return {
        "router": res["router"],
        **{f"r@{m['topk']}": m["recall"] for m in res["metrics"]},
        **{f"p@{m['topk']}": m["precision"] for m in res["metrics"]},
        "min_size_mb": res["min_index_size_mb"],
        "max_size_mb": res["max_index_size_mb"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = ExperimentConfig.model_validate(yaml.safe_load(f))

    if os.path.exists(config.results):
        with open(config.results, "r") as f:
            results = json.load(f)
    else:
        results = []

    results += run_experiment(config, existing_results={r["router"] for r in results})

    os.makedirs(os.path.dirname(config.results), exist_ok=True)
    with open(config.results, "w") as f:
        json.dump(results, f, indent=2)

    summary = pd.DataFrame([get_summary(res) for res in results])
    summary = summary.sort_values(by="r@10", ascending=False)
    print(summary.to_markdown(index=False))


if __name__ == "__main__":
    main()
