#!/usr/bin/env python3
"""Convert .npy files (bigann format) to GP-ANN binary formats.

Converts:
  - Base vectors (uint8 .npy) -> .u8bin (uint32 n, uint32 d, then n*d uint8 values)
  - Query vectors (uint8 .npy) -> .u8bin
  - Ground truth (uint32 .npy, IDs only) -> GP-ANN binary (IDs + squared L2 distances)

Usage:
    python3 convert_npy_to_gpann.py \
        --base-npy /scratch/brc7/bigann/train.npy \
        --queries-npy /scratch/brc7/bigann/queries.npy \
        --gt-npy /scratch/brc7/bigann/ground_truth_100m.npy \
        --output-dir /scratch/brc7/partitioning \
        --num-base 100000000
"""

import argparse
import struct
import time
import numpy as np


def convert_base_to_u8bin(npy_path, output_path, num_vectors):
    """Convert first num_vectors of a uint8 .npy to .u8bin format."""
    data = np.load(npy_path, mmap_mode="r")
    total_n, d = data.shape
    n = min(num_vectors, total_n)
    print(f"Converting base: {npy_path} -> {output_path}")
    print(f"  Total vectors: {total_n}, using first {n}, dim={d}, dtype={data.dtype}")

    t0 = time.time()
    with open(output_path, "wb") as f:
        f.write(struct.pack("<II", n, d))
        chunk_size = 1_000_000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = np.array(data[start:end])
            f.write(chunk.astype(np.uint8).tobytes())
            if (start // chunk_size) % 10 == 0:
                print(f"  Written {end}/{n} vectors ({100*end/n:.1f}%)")
    print(f"  Done in {time.time() - t0:.1f}s")


def convert_queries_to_u8bin(npy_path, output_path):
    """Convert query vectors from .npy to .u8bin format."""
    data = np.load(npy_path)
    n, d = data.shape
    print(f"Converting queries: {npy_path} -> {output_path}")
    print(f"  n={n}, d={d}, dtype={data.dtype}")

    with open(output_path, "wb") as f:
        f.write(struct.pack("<II", n, d))
        f.write(data.astype(np.uint8).tobytes())
    print("  Done")


def convert_ground_truth(gt_npy_path, queries_npy_path, base_npy_path,
                         output_path, num_base):
    """Convert ground truth .npy (IDs only) to GP-ANN format (IDs + distances).

    GP-ANN format: uint32 num_queries, uint32 num_neighbors,
                   then num_queries * num_neighbors uint32 IDs,
                   then num_queries * num_neighbors float32 distances.
    """
    gt_ids = np.load(gt_npy_path).astype(np.uint32)
    queries = np.load(queries_npy_path).astype(np.float32)
    base = np.load(base_npy_path, mmap_mode="r")

    num_queries, num_neighbors = gt_ids.shape
    print(f"Converting ground truth: {gt_npy_path} -> {output_path}")
    print(f"  num_queries={num_queries}, num_neighbors={num_neighbors}")

    # Verify all IDs are within the base set
    max_id = gt_ids.max()
    if max_id >= num_base:
        raise ValueError(
            f"Ground truth contains ID {max_id} >= num_base {num_base}. "
            "Ground truth may not match the base set size."
        )

    # Compute squared L2 distances (matching GP-ANN's sqr_l2_dist)
    t0 = time.time()
    distances = np.zeros((num_queries, num_neighbors), dtype=np.float32)
    for q in range(num_queries):
        query_vec = queries[q]  # already float32
        neighbor_ids = gt_ids[q]
        neighbor_vecs = np.array(base[neighbor_ids]).astype(np.float32)
        diff = neighbor_vecs - query_vec
        distances[q] = np.sum(diff ** 2, axis=1)
        if q % 2000 == 0 and q > 0:
            print(f"  Computed distances for {q}/{num_queries} queries")
    print(f"  Distance computation took {time.time() - t0:.1f}s")

    # Write in GP-ANN binary format
    with open(output_path, "wb") as f:
        f.write(struct.pack("<II", num_queries, num_neighbors))
        gt_ids.tofile(f)
        distances.tofile(f)
    print(f"  Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert .npy files to GP-ANN binary formats"
    )
    parser.add_argument("--base-npy", required=True, help="Path to train.npy")
    parser.add_argument("--queries-npy", required=True, help="Path to queries.npy")
    parser.add_argument("--gt-npy", required=True, help="Path to ground_truth .npy")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--num-base", type=int, default=100_000_000,
                        help="Number of base vectors to use (default: 100M)")
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    base_out = os.path.join(args.output_dir, "bigann_100m_base.u8bin")
    queries_out = os.path.join(args.output_dir, "bigann_100m_queries.u8bin")
    gt_out = os.path.join(args.output_dir, "bigann_100m_gt.bin")

    convert_base_to_u8bin(args.base_npy, base_out, args.num_base)
    convert_queries_to_u8bin(args.queries_npy, queries_out)
    convert_ground_truth(
        args.gt_npy, args.queries_npy, args.base_npy, gt_out, args.num_base
    )

    print("\nAll conversions complete.")
    print(f"  Base:    {base_out}")
    print(f"  Queries: {queries_out}")
    print(f"  GT:      {gt_out}")


if __name__ == "__main__":
    main()
