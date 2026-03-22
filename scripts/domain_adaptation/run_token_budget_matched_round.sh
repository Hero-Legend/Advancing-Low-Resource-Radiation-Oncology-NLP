#!/usr/bin/env bash
set -euo pipefail

ROOT="/gz-fs/codex-workspace"
PYTHON_BIN="/usr/local/miniconda3/bin/python"
BASE_MODEL="/gz-fs/Bi-mamba/model_path/PubMedBert"
MAX_LEN=128
BATCH=8
SEEDS=(7 13 21 42 87)

FULL_JSONL="$ROOT/data/corpora/public_radonc_adaptation.jsonl"
FOCUSED_JSONL="$ROOT/data/corpora/public_radonc_adaptation_focused.jsonl"

FULL_OUT="$ROOT/results/domain_adaptation/pubmedbert_public_radonc_full_matched_steps"
FOCUSED_OUT="$ROOT/results/domain_adaptation/pubmedbert_public_radonc_focused_matched_steps"
BUDGET_DIR="$ROOT/results/domain_adaptation/token_budget_matched_budget"

mkdir -p "$BUDGET_DIR"
cd "$ROOT"

echo "[1/10] Estimate focused adaptation budget"
"$PYTHON_BIN" scripts/domain_adaptation/estimate_mlm_steps.py \
  --train-jsonl "$FOCUSED_JSONL" \
  --model-name "$BASE_MODEL" \
  --max-seq-length "$MAX_LEN" \
  --train-batch-size "$BATCH" \
  --output-json "$BUDGET_DIR/focused_budget.json"

MATCHED_STEPS=$("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("/gz-fs/codex-workspace/results/domain_adaptation/token_budget_matched_budget/focused_budget.json").read_text(encoding="utf-8"))
print(payload["optimizer_steps_per_epoch"])
PY
)

echo "Matched optimizer steps: $MATCHED_STEPS"

echo "[2/10] Estimate full adaptation budget"
"$PYTHON_BIN" scripts/domain_adaptation/estimate_mlm_steps.py \
  --train-jsonl "$FULL_JSONL" \
  --model-name "$BASE_MODEL" \
  --max-seq-length "$MAX_LEN" \
  --train-batch-size "$BATCH" \
  --output-json "$BUDGET_DIR/full_budget.json"

echo "[3/10] Token-budget-matched full adaptation"
"$PYTHON_BIN" scripts/domain_adaptation/train_mlm_adaptation.py \
  --train-jsonl "$FULL_JSONL" \
  --model-name "$BASE_MODEL" \
  --output-dir "$FULL_OUT" \
  --max-seq-length "$MAX_LEN" \
  --learning-rate 5e-5 \
  --train-batch-size "$BATCH" \
  --max-steps "$MATCHED_STEPS" \
  --save-steps 500 \
  --logging-steps 50 \
  --seed 42

echo "[4/10] Token-budget-matched focused adaptation"
"$PYTHON_BIN" scripts/domain_adaptation/train_mlm_adaptation.py \
  --train-jsonl "$FOCUSED_JSONL" \
  --model-name "$BASE_MODEL" \
  --output-dir "$FOCUSED_OUT" \
  --max-seq-length "$MAX_LEN" \
  --learning-rate 5e-5 \
  --train-batch-size "$BATCH" \
  --max-steps "$MATCHED_STEPS" \
  --save-steps 500 \
  --logging-steps 50 \
  --seed 42

echo "[5/10] Text classification multiseed comparison"
"$PYTHON_BIN" scripts/rond/run_multiseed_experiments.py \
  --train research_data/processed/rond/text_classification/train.jsonl \
  --val research_data/processed/rond/text_classification/val.jsonl \
  --test research_data/processed/rond/text_classification/test.jsonl \
  --output-root results/rond/text_classification_token_budget_comparison \
  --configs-json configs/rond/text_classification_token_budget_comparison.json \
  --seeds "${SEEDS[@]}" \
  --cwd "$ROOT"

echo "[6/10] Logic reasoning multiseed comparison"
"$PYTHON_BIN" scripts/rond/run_multiseed_experiments.py \
  --train research_data/processed/rond/logic_reasoning/train.jsonl \
  --val research_data/processed/rond/logic_reasoning/val.jsonl \
  --test research_data/processed/rond/logic_reasoning/test.jsonl \
  --output-root results/rond/logic_reasoning_token_budget_comparison \
  --configs-json configs/rond/logic_reasoning_token_budget_comparison.json \
  --seeds "${SEEDS[@]}" \
  --cwd "$ROOT"

echo "[7/10] QA answer-selection multiseed comparison"
"$PYTHON_BIN" scripts/rond/run_multiseed_experiments.py \
  --train research_data/processed/rond/qa_answer_selection/train.jsonl \
  --val research_data/processed/rond/qa_answer_selection/val.jsonl \
  --test research_data/processed/rond/qa_answer_selection/test.jsonl \
  --output-root results/rond/qa_answer_selection_token_budget_comparison \
  --configs-json configs/rond/qa_answer_selection_token_budget_comparison.json \
  --seeds "${SEEDS[@]}" \
  --cwd "$ROOT"

echo "[8/10] Summaries"
"$PYTHON_BIN" scripts/rond/summarize_multiseed_results.py \
  --results-root results/rond/text_classification_token_budget_comparison \
  --output-json results/rond/text_classification_token_budget_comparison/summary.json \
  --output-csv results/rond/text_classification_token_budget_comparison/summary.csv
"$PYTHON_BIN" scripts/rond/summarize_multiseed_results.py \
  --results-root results/rond/logic_reasoning_token_budget_comparison \
  --output-json results/rond/logic_reasoning_token_budget_comparison/summary.json \
  --output-csv results/rond/logic_reasoning_token_budget_comparison/summary.csv
"$PYTHON_BIN" scripts/rond/summarize_multiseed_results.py \
  --results-root results/rond/qa_answer_selection_token_budget_comparison \
  --output-json results/rond/qa_answer_selection_token_budget_comparison/summary.json \
  --output-csv results/rond/qa_answer_selection_token_budget_comparison/summary.csv

echo "[9/10] Export assets"
"$PYTHON_BIN" scripts/rond/export_multiseed_assets.py \
  --summary-json results/rond/text_classification_token_budget_comparison/summary.json \
  --table-csv results/tables/rond_text_classification_token_budget_comparison.csv \
  --per-seed-csv results/figures_data/rond_text_classification_token_budget_comparison_per_seed.csv
"$PYTHON_BIN" scripts/rond/export_multiseed_assets.py \
  --summary-json results/rond/logic_reasoning_token_budget_comparison/summary.json \
  --table-csv results/tables/rond_logic_reasoning_token_budget_comparison.csv \
  --per-seed-csv results/figures_data/rond_logic_reasoning_token_budget_comparison_per_seed.csv
"$PYTHON_BIN" scripts/rond/export_multiseed_assets.py \
  --summary-json results/rond/qa_answer_selection_token_budget_comparison/summary.json \
  --table-csv results/tables/rond_qa_answer_selection_token_budget_comparison.csv \
  --per-seed-csv results/figures_data/rond_qa_answer_selection_token_budget_comparison_per_seed.csv

echo "[10/10] Save run manifest"
cat > "$ROOT/results/rond/token_budget_matched_round_manifest.json" <<EOF
{
  "base_model": "$BASE_MODEL",
  "max_seq_length": $MAX_LEN,
  "train_batch_size": $BATCH,
  "matched_steps": $MATCHED_STEPS,
  "full_jsonl": "$FULL_JSONL",
  "focused_jsonl": "$FOCUSED_JSONL",
  "seeds": [7, 13, 21, 42, 87]
}
EOF

echo "Done."
