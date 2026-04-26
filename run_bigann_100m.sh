#!/bin/bash
# Experiment: BigANN 100M with GP partitioning + HNSW routing
# Measures how graph partitioning and routing cost scale with number of shards.
#
# Prerequisites:
#   - Pre-built executables in release_l2/ (Partition, QueryAttribution)
#   - BigANN .npy data in /scratch/brc7/bigann/
#   - Python 3 with numpy
#
# Usage:
#   bash run_bigann_100m.sh [--convert-only] [--skip-convert]

set -euo pipefail

# ---- Configuration ----
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${REPO_DIR}/release_l2"
DATA_DIR="/scratch/brc7/bigann"
OUTPUT_DIR="/scratch/brc7/partitioning"

BASE_NPY="${DATA_DIR}/train.npy"
QUERIES_NPY="${DATA_DIR}/queries.npy"
GT_NPY="${DATA_DIR}/ground_truth_100m.npy"

NUM_BASE=100000000
NUM_NEIGHBORS=10
PART_METHOD="GP"
PART_CONFIG="default"  # "default" or "strong"

BASE_BIN="${OUTPUT_DIR}/bigann_100m_base.u8bin"
QUERIES_BIN="${OUTPUT_DIR}/bigann_100m_queries.u8bin"
GT_BIN="${OUTPUT_DIR}/bigann_100m_gt.bin"

SHARD_COUNTS=(50 100 500 1000 2000)

# ---- Parse arguments ----
CONVERT_ONLY=false
SKIP_CONVERT=false
for arg in "$@"; do
    case $arg in
        --convert-only) CONVERT_ONLY=true ;;
        --skip-convert) SKIP_CONVERT=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---- Verify prerequisites ----
echo "=== BigANN 100M Experiment ==="
echo "Repository:  ${REPO_DIR}"
echo "Build dir:   ${BUILD_DIR}"
echo "Data dir:    ${DATA_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Shard counts: ${SHARD_COUNTS[*]}"
echo "Partitioning: ${PART_METHOD} (${PART_CONFIG})"
echo ""

for exe in Partition QueryAttribution; do
    if [ ! -x "${BUILD_DIR}/${exe}" ]; then
        echo "ERROR: ${BUILD_DIR}/${exe} not found. Build with cmake first."
        exit 1
    fi
done

for f in "${BASE_NPY}" "${QUERIES_NPY}" "${GT_NPY}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: ${f} not found."
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

# ---- Step 1: Convert .npy to GP-ANN binary format ----
if [ "${SKIP_CONVERT}" = false ]; then
    echo "=== Step 1: Converting .npy to GP-ANN format ==="
    if [ -f "${BASE_BIN}" ] && [ -f "${QUERIES_BIN}" ] && [ -f "${GT_BIN}" ]; then
        echo "Converted files already exist, skipping. Use --skip-convert=false to force."
    else
        python3 "${REPO_DIR}/convert_npy_to_gpann.py" \
            --base-npy "${BASE_NPY}" \
            --queries-npy "${QUERIES_NPY}" \
            --gt-npy "${GT_NPY}" \
            --output-dir "${OUTPUT_DIR}" \
            --num-base "${NUM_BASE}"
    fi
    echo ""
fi

if [ "${CONVERT_ONLY}" = true ]; then
    echo "Conversion complete (--convert-only). Exiting."
    exit 0
fi

# ---- Step 2 & 3: Partition + Query Attribution for each shard count ----
RESULTS_FILE="${OUTPUT_DIR}/experiment_summary.csv"
echo "num_shards,partition_method,partition_time_s,output_prefix" > "${RESULTS_FILE}"

for K in "${SHARD_COUNTS[@]}"; do
    echo "============================================================"
    echo "=== Shard count: K=${K} ==="
    echo "============================================================"

    PART_PREFIX="${OUTPUT_DIR}/bigann_100m.partition"
    PART_FILE="${PART_PREFIX}.k=${K}.${PART_METHOD}"
    OUTPUT_PREFIX="${OUTPUT_DIR}/bigann_100m.${PART_METHOD}.k=${K}"

    # -- Step 2: Partitioning --
    if [ -f "${PART_FILE}" ]; then
        echo "Partition file ${PART_FILE} already exists, skipping partitioning."
    else
        echo "--- Partitioning (K=${K}, method=${PART_METHOD}) ---"
        PART_START=$(date +%s)

        "${BUILD_DIR}/Partition" \
            "${BASE_BIN}" \
            "${PART_PREFIX}" \
            "${K}" \
            "${PART_METHOD}" \
            "${PART_CONFIG}" \
            2>&1 | tee "${OUTPUT_DIR}/partition_k${K}.log"

        PART_END=$(date +%s)
        PART_TIME=$((PART_END - PART_START))
        echo "Partitioning K=${K} took ${PART_TIME}s"
        echo "${K},${PART_METHOD},${PART_TIME},${OUTPUT_PREFIX}" >> "${RESULTS_FILE}"
    fi

    # -- Step 3: Query Attribution (routing + shard searches) --
    if [ -f "${OUTPUT_PREFIX}.routes" ] && [ -f "${OUTPUT_PREFIX}.searches" ]; then
        echo "Query attribution output already exists, skipping."
    else
        echo "--- Query Attribution (K=${K}) ---"

        "${BUILD_DIR}/QueryAttribution" \
            "${BASE_BIN}" \
            "${QUERIES_BIN}" \
            "${GT_BIN}" \
            "${NUM_NEIGHBORS}" \
            "${PART_FILE}" \
            "${OUTPUT_PREFIX}" \
            "${PART_METHOD}" \
            "${K}" \
            2>&1 | tee "${OUTPUT_DIR}/query_attribution_k${K}.log"
    fi

    echo ""
done

echo "============================================================"
echo "=== Experiment complete ==="
echo "Results summary: ${RESULTS_FILE}"
echo "Per-shard logs:  ${OUTPUT_DIR}/partition_k*.log"
echo "                 ${OUTPUT_DIR}/query_attribution_k*.log"
echo "============================================================"
