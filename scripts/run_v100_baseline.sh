#!/usr/bin/env bash
set -euo pipefail

# Execute the complete NNSLR baseline pipeline on one explicitly selected V100.
# This script never kills competing GPU processes and never deploys anything to
# Sunnypilot or a comma device.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${NNSLR_DATA_ROOT:-/srv/nnslr-data}"
GPU_INDEX="${NNSLR_GPU_INDEX:-0}"
RUN_ID="${1:-}"
DATASET_KIND="${NNSLR_DATASET_KIND:-gold}"
VENV="${NNSLR_GPU_VENV:-$ROOT_DIR/.venv-gpu}"

if [[ -z "$RUN_ID" ]]; then
  echo "usage: $0 RUN_ID" >&2
  echo "example: NNSLR_DATA_ROOT=/srv/nnslr-data NNSLR_GPU_INDEX=0 $0 baseline-001" >&2
  exit 2
fi

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: RUN_ID contains unsupported characters" >&2
  exit 2
fi

if [[ "$DATASET_KIND" != "gold" && "$DATASET_KIND" != "training-candidate" ]]; then
  echo "error: NNSLR_DATASET_KIND must be gold or training-candidate" >&2
  exit 2
fi

if [[ ! -x "$VENV/bin/nnslr" ]]; then
  echo "error: GPU environment missing: $VENV/bin/nnslr" >&2
  echo "run: bash scripts/setup_v100_training_env.sh" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi not found" >&2
  exit 2
fi

if ! nvidia-smi -i "$GPU_INDEX" >/dev/null 2>&1; then
  echo "error: invalid/unavailable GPU index: $GPU_INDEX" >&2
  exit 2
fi

BUSY="$(
  nvidia-smi -i "$GPU_INDEX"     --query-compute-apps=pid,process_name,used_gpu_memory     --format=csv,noheader,nounits 2>/dev/null     | sed '/^[[:space:]]*$/d' || true
)"
if [[ -n "$BUSY" && "${NNSLR_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "error: selected GPU already has compute processes; refusing to start training" >&2
  echo "$BUSY" >&2
  echo "Set NNSLR_ALLOW_BUSY_GPU=1 only after intentionally accepting contention." >&2
  exit 3
fi

RUN_ROOT="$DATA_ROOT/runs/$RUN_ID"
if [[ -e "$RUN_ROOT" ]]; then
  echo "error: run directory already exists: $RUN_ROOT" >&2
  exit 3
fi

DETECTOR_RUN="$RUN_ROOT/detector"
READER_RUN="$RUN_ROOT/reader"
EVAL_DIR="$RUN_ROOT/evaluation"
EXPORT_DIR="$RUN_ROOT/export"
BUNDLE_DIR="$RUN_ROOT/bundle"
mkdir -p "$EVAL_DIR" "$EXPORT_DIR"

export NNSLR_DATA_ROOT="$DATA_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

NN="$VENV/bin/nnslr"

echo "== NNSLR baseline pipeline =="
echo "run:          $RUN_ID"
echo "data root:    $DATA_ROOT"
echo "dataset kind: $DATASET_KIND"
echo "physical GPU: $GPU_INDEX"
echo

echo "== 1/10 Validate exact training populations (CPU-only) =="
"$NN" train --task detector --dataset-kind "$DATASET_KIND" --dry-run
"$NN" train --task reader --dataset-kind "$DATASET_KIND" --dry-run

echo "== 2/10 Validate held-out evaluation populations (CPU-only) =="
"$NN" evaluate --task detector --dataset-kind "$DATASET_KIND" --partition test --dry-run
"$NN" evaluate --task reader --dataset-kind "$DATASET_KIND" --partition test --dry-run

echo "== 3/10 Explicit CUDA smoke test =="
# CUDA_VISIBLE_DEVICES maps the selected physical GPU to logical cuda:0.
"$NN" env --gpu-smoke --gpu-device 0

echo "== 4/10 Train detector =="
"$NN" train   --task detector   --dataset-kind "$DATASET_KIND"   --device cuda   --output "$DETECTOR_RUN"

echo "== 5/10 Train reader =="
"$NN" train   --task reader   --dataset-kind "$DATASET_KIND"   --device cuda   --output "$READER_RUN"

echo "== 6/10 Evaluate held-out detector and reader =="
"$NN" evaluate   --task detector   --dataset-kind "$DATASET_KIND"   --partition test   --device cuda   --checkpoint "$DETECTOR_RUN/detector-best.pt"   --output "$EVAL_DIR/detector-test.json"
"$NN" evaluate   --task reader   --dataset-kind "$DATASET_KIND"   --partition test   --device cuda   --checkpoint "$READER_RUN/reader-best.pt"   --output "$EVAL_DIR/reader-test.json"

echo "== 7/10 Mine hard examples =="
"$NN" mine-hard-examples "$EVAL_DIR/detector-test.json" --output "$EVAL_DIR/detector-hard.jsonl"
"$NN" mine-hard-examples "$EVAL_DIR/reader-test.json" --output "$EVAL_DIR/reader-hard.jsonl"

echo "== 8/10 Export and parity-check ONNX =="
"$NN" export-onnx   --task detector   --checkpoint "$DETECTOR_RUN/detector-best.pt"   --output "$EXPORT_DIR/detector.onnx"
"$NN" export-onnx   --task reader   --checkpoint "$READER_RUN/reader-best.pt"   --output "$EXPORT_DIR/reader.onnx"

echo "== 9/10 Package immutable verified bundle =="
"$NN" package-model   --detector-onnx "$EXPORT_DIR/detector.onnx"   --detector-contract "$EXPORT_DIR/detector.onnx.contract.json"   --reader-onnx "$EXPORT_DIR/reader.onnx"   --reader-contract "$EXPORT_DIR/reader.onnx.contract.json"   --output "$BUNDLE_DIR"

echo "== 10/10 Verify final bundle =="
"$NN" verify-bundle "$BUNDLE_DIR"

echo
echo "Pipeline complete: $RUN_ROOT"
echo "Nothing was deployed to Sunnypilot or a comma device."
