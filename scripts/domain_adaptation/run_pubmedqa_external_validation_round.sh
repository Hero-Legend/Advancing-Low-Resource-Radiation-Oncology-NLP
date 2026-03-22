#!/usr/bin/env bash
set -euo pipefail

ROOT="/gz-fs/codex-workspace"
PYTHON_BIN="/usr/local/miniconda3/bin/python"
BASE_MODEL="/gz-fs/Bi-mamba/model_path/PubMedBert"
MAX_LEN=128
BATCH=8
MATCHED_STEPS=737
SEEDS=(7 13 21 42 87)

FOCUSED_JSONL="$ROOT/data/corpora/public_radonc_adaptation_focused.jsonl"
PUBMEDQA_TRAIN="$ROOT/research_data/processed/pubmedqa/train.jsonl"
PUBMEDQA_VAL="$ROOT/research_data/processed/pubmedqa/val.jsonl"
PUBMEDQA_TEST="$ROOT/research_data/processed/pubmedqa/test.jsonl"
PUBMEDQA_REPLAY_TRAIN="$ROOT/data/corpora/pubmedqa_replay_train_only.jsonl"

REPLAY10_JSONL="$ROOT/data/corpora/public_radonc_adaptation_focused_replay10_trainonly.jsonl"
REPLAY20_JSONL="$ROOT/data/corpora/public_radonc_adaptation_focused_replay20_trainonly.jsonl"
REPLAY10_OUT="$ROOT/results/domain_adaptation/pubmedbert_public_radonc_focused_replay10_trainonly_matched_steps"
REPLAY20_OUT="$ROOT/results/domain_adaptation/pubmedbert_public_radonc_focused_replay20_trainonly_matched_steps"

mkdir -p "$ROOT/results/domain_adaptation/external_validation"
cd "$ROOT"

echo "[1/8] Prepare PubMedQA train/val/test and train-only replay pool"
"$PYTHON_BIN" scripts/public_corpus/prepare_pubmedqa_external_assets.py \
  --train-jsonl "$PUBMEDQA_TRAIN" \
  --val-jsonl "$PUBMEDQA_VAL" \
  --test-jsonl "$PUBMEDQA_TEST" \
  --replay-train-jsonl "$PUBMEDQA_REPLAY_TRAIN" \
  --stats-json "$ROOT/results/domain_adaptation/external_validation/pubmedqa_split_stats.json"

echo "[2/8] Build replay10 train-only mixed corpus"
"$PYTHON_BIN" scripts/public_corpus/build_replay_regularized_corpus.py \
  --focused-jsonl "$FOCUSED_JSONL" \
  --replay-jsonl "$PUBMEDQA_REPLAY_TRAIN" \
  --tokenizer "$BASE_MODEL" \
  --replay-ratio 0.10 \
  --seed 42 \
  --output-jsonl "$REPLAY10_JSONL" \
  --output-stats-json "$ROOT/results/domain_adaptation/external_validation/pubmedqa_replay10_stats.json"

echo "[3/8] Build replay20 train-only mixed corpus"
"$PYTHON_BIN" scripts/public_corpus/build_replay_regularized_corpus.py \
  --focused-jsonl "$FOCUSED_JSONL" \
  --replay-jsonl "$PUBMEDQA_REPLAY_TRAIN" \
  --tokenizer "$BASE_MODEL" \
  --replay-ratio 0.20 \
  --seed 42 \
  --output-jsonl "$REPLAY20_JSONL" \
  --output-stats-json "$ROOT/results/domain_adaptation/external_validation/pubmedqa_replay20_stats.json"

echo "[4/8] Train replay10 train-only adaptation"
"$PYTHON_BIN" scripts/domain_adaptation/train_mlm_adaptation.py \
  --train-jsonl "$REPLAY10_JSONL" \
  --model-name "$BASE_MODEL" \
  --output-dir "$REPLAY10_OUT" \
  --max-seq-length "$MAX_LEN" \
  --learning-rate 5e-5 \
  --train-batch-size "$BATCH" \
  --max-steps "$MATCHED_STEPS" \
  --save-steps 500 \
  --logging-steps 50 \
  --seed 42

echo "[5/8] Train replay20 train-only adaptation"
"$PYTHON_BIN" scripts/domain_adaptation/train_mlm_adaptation.py \
  --train-jsonl "$REPLAY20_JSONL" \
  --model-name "$BASE_MODEL" \
  --output-dir "$REPLAY20_OUT" \
  --max-seq-length "$MAX_LEN" \
  --learning-rate 5e-5 \
  --train-batch-size "$BATCH" \
  --max-steps "$MATCHED_STEPS" \
  --save-steps 500 \
  --logging-steps 50 \
  --seed 42

echo "[6/8] PubMedQA external validation multiseed comparison"
"$PYTHON_BIN" scripts/rond/run_multiseed_experiments.py \
  --train "$PUBMEDQA_TRAIN" \
  --val "$PUBMEDQA_VAL" \
  --test "$PUBMEDQA_TEST" \
  --output-root results/rond/pubmedqa_external_validation_comparison \
  --configs-json configs/pubmedqa/pubmedqa_external_validation_comparison.json \
  --seeds "${SEEDS[@]}" \
  --cwd "$ROOT"

echo "[7/8] Summaries"
"$PYTHON_BIN" scripts/rond/summarize_multiseed_results.py \
  --results-root results/rond/pubmedqa_external_validation_comparison \
  --output-json results/rond/pubmedqa_external_validation_comparison/summary.json \
  --output-csv results/rond/pubmedqa_external_validation_comparison/summary.csv
"$PYTHON_BIN" scripts/rond/export_multiseed_assets.py \
  --summary-json results/rond/pubmedqa_external_validation_comparison/summary.json \
  --table-csv results/tables/pubmedqa_external_validation_comparison.csv \
  --per-seed-csv results/figures_data/pubmedqa_external_validation_comparison_per_seed.csv

echo "[8/8] Save run manifest"
cat > "$ROOT/results/rond/pubmedqa_external_validation_round_manifest.json" <<EOF
{
  "base_model": "$BASE_MODEL",
  "matched_steps": $MATCHED_STEPS,
  "pubmedqa_train": "$PUBMEDQA_TRAIN",
  "pubmedqa_val": "$PUBMEDQA_VAL",
  "pubmedqa_test": "$PUBMEDQA_TEST",
  "pubmedqa_replay_train": "$PUBMEDQA_REPLAY_TRAIN",
  "seeds": [7, 13, 21, 42, 87]
}
EOF

echo "Done."
