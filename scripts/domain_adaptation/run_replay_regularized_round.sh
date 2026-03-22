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
PUBMEDQA_JSONL="$ROOT/data/corpora/pubmedqa_replay.jsonl"
REPLAY10_JSONL="$ROOT/data/corpora/public_radonc_adaptation_focused_replay10.jsonl"
REPLAY20_JSONL="$ROOT/data/corpora/public_radonc_adaptation_focused_replay20.jsonl"

REPLAY10_OUT="$ROOT/results/domain_adaptation/pubmedbert_public_radonc_focused_replay10_matched_steps"
REPLAY20_OUT="$ROOT/results/domain_adaptation/pubmedbert_public_radonc_focused_replay20_matched_steps"
REPLAY_DIR="$ROOT/results/domain_adaptation/replay_regularized"

mkdir -p "$REPLAY_DIR"
cd "$ROOT"

echo "[1/10] Build PubMedQA replay corpus"
"$PYTHON_BIN" scripts/public_corpus/build_pubmedqa_replay_corpus.py \
  --output-jsonl "$PUBMEDQA_JSONL" \
  --min-chars 120

echo "[2/10] Build replay10 mixed corpus"
"$PYTHON_BIN" scripts/public_corpus/build_replay_regularized_corpus.py \
  --focused-jsonl "$FOCUSED_JSONL" \
  --replay-jsonl "$PUBMEDQA_JSONL" \
  --tokenizer "$BASE_MODEL" \
  --replay-ratio 0.10 \
  --seed 42 \
  --output-jsonl "$REPLAY10_JSONL" \
  --output-stats-json "$REPLAY_DIR/replay10_stats.json"

echo "[3/10] Build replay20 mixed corpus"
"$PYTHON_BIN" scripts/public_corpus/build_replay_regularized_corpus.py \
  --focused-jsonl "$FOCUSED_JSONL" \
  --replay-jsonl "$PUBMEDQA_JSONL" \
  --tokenizer "$BASE_MODEL" \
  --replay-ratio 0.20 \
  --seed 42 \
  --output-jsonl "$REPLAY20_JSONL" \
  --output-stats-json "$REPLAY_DIR/replay20_stats.json"

echo "[4/10] Train replay10 adaptation"
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

echo "[5/10] Train replay20 adaptation"
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

echo "[6/10] Text classification multiseed comparison"
"$PYTHON_BIN" scripts/rond/run_multiseed_experiments.py \
  --train research_data/processed/rond/text_classification/train.jsonl \
  --val research_data/processed/rond/text_classification/val.jsonl \
  --test research_data/processed/rond/text_classification/test.jsonl \
  --output-root results/rond/text_classification_replay_regularized_comparison \
  --configs-json configs/rond/text_classification_replay_regularized_comparison.json \
  --seeds "${SEEDS[@]}" \
  --cwd "$ROOT"

echo "[7/10] Logic reasoning multiseed comparison"
"$PYTHON_BIN" scripts/rond/run_multiseed_experiments.py \
  --train research_data/processed/rond/logic_reasoning/train.jsonl \
  --val research_data/processed/rond/logic_reasoning/val.jsonl \
  --test research_data/processed/rond/logic_reasoning/test.jsonl \
  --output-root results/rond/logic_reasoning_replay_regularized_comparison \
  --configs-json configs/rond/logic_reasoning_replay_regularized_comparison.json \
  --seeds "${SEEDS[@]}" \
  --cwd "$ROOT"

echo "[8/10] QA answer-selection multiseed comparison"
"$PYTHON_BIN" scripts/rond/run_multiseed_experiments.py \
  --train research_data/processed/rond/qa_answer_selection/train.jsonl \
  --val research_data/processed/rond/qa_answer_selection/val.jsonl \
  --test research_data/processed/rond/qa_answer_selection/test.jsonl \
  --output-root results/rond/qa_answer_selection_replay_regularized_comparison \
  --configs-json configs/rond/qa_answer_selection_replay_regularized_comparison.json \
  --seeds "${SEEDS[@]}" \
  --cwd "$ROOT"

echo "[9/10] Export assets"
"$PYTHON_BIN" scripts/rond/summarize_multiseed_results.py \
  --results-root results/rond/text_classification_replay_regularized_comparison \
  --output-json results/rond/text_classification_replay_regularized_comparison/summary.json \
  --output-csv results/rond/text_classification_replay_regularized_comparison/summary.csv
"$PYTHON_BIN" scripts/rond/summarize_multiseed_results.py \
  --results-root results/rond/logic_reasoning_replay_regularized_comparison \
  --output-json results/rond/logic_reasoning_replay_regularized_comparison/summary.json \
  --output-csv results/rond/logic_reasoning_replay_regularized_comparison/summary.csv
"$PYTHON_BIN" scripts/rond/summarize_multiseed_results.py \
  --results-root results/rond/qa_answer_selection_replay_regularized_comparison \
  --output-json results/rond/qa_answer_selection_replay_regularized_comparison/summary.json \
  --output-csv results/rond/qa_answer_selection_replay_regularized_comparison/summary.csv
"$PYTHON_BIN" scripts/rond/export_multiseed_assets.py \
  --summary-json results/rond/text_classification_replay_regularized_comparison/summary.json \
  --table-csv results/tables/rond_text_classification_replay_regularized_comparison.csv \
  --per-seed-csv results/figures_data/rond_text_classification_replay_regularized_comparison_per_seed.csv
"$PYTHON_BIN" scripts/rond/export_multiseed_assets.py \
  --summary-json results/rond/logic_reasoning_replay_regularized_comparison/summary.json \
  --table-csv results/tables/rond_logic_reasoning_replay_regularized_comparison.csv \
  --per-seed-csv results/figures_data/rond_logic_reasoning_replay_regularized_comparison_per_seed.csv
"$PYTHON_BIN" scripts/rond/export_multiseed_assets.py \
  --summary-json results/rond/qa_answer_selection_replay_regularized_comparison/summary.json \
  --table-csv results/tables/rond_qa_answer_selection_replay_regularized_comparison.csv \
  --per-seed-csv results/figures_data/rond_qa_answer_selection_replay_regularized_comparison_per_seed.csv

echo "[10/10] Save run manifest"
cat > "$ROOT/results/rond/replay_regularized_round_manifest.json" <<EOF
{
  "base_model": "$BASE_MODEL",
  "max_seq_length": $MAX_LEN,
  "train_batch_size": $BATCH,
  "matched_steps": $MATCHED_STEPS,
  "focused_jsonl": "$FOCUSED_JSONL",
  "pubmedqa_jsonl": "$PUBMEDQA_JSONL",
  "replay10_jsonl": "$REPLAY10_JSONL",
  "replay20_jsonl": "$REPLAY20_JSONL",
  "replay_ratios": [0.10, 0.20],
  "seeds": [7, 13, 21, 42, 87]
}
EOF

echo "Done."
