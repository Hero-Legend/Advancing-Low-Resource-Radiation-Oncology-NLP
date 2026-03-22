#!/usr/bin/env bash
set -euo pipefail

ROOT="/gz-fs/codex-workspace"
PYTHON_BIN="/usr/local/miniconda3/bin/python"
SEEDS=(7 13 21 42 87)

cd "$ROOT"

echo "[1/6] Logic reasoning multiseed run"
"$PYTHON_BIN" scripts/rond/run_multiseed_experiments.py \
  --train research_data/processed/rond/logic_reasoning/train.jsonl \
  --val research_data/processed/rond/logic_reasoning/val.jsonl \
  --test research_data/processed/rond/logic_reasoning/test.jsonl \
  --output-root results/rond/logic_reasoning_full_corpus_comparison \
  --configs-json configs/rond/logic_reasoning_full_corpus_comparison.json \
  --seeds "${SEEDS[@]}" \
  --cwd "$ROOT"

echo "[2/6] Logic reasoning summary"
"$PYTHON_BIN" scripts/rond/summarize_multiseed_results.py \
  --results-root results/rond/logic_reasoning_full_corpus_comparison \
  --output-json results/rond/logic_reasoning_full_corpus_comparison/summary.json \
  --output-csv results/rond/logic_reasoning_full_corpus_comparison/summary.csv

echo "[3/6] Logic reasoning export assets"
"$PYTHON_BIN" scripts/rond/export_multiseed_assets.py \
  --summary-json results/rond/logic_reasoning_full_corpus_comparison/summary.json \
  --table-csv results/tables/rond_logic_reasoning_full_corpus_comparison.csv \
  --per-seed-csv results/figures_data/rond_logic_reasoning_full_corpus_comparison_per_seed.csv

echo "[4/6] QA answer-selection multiseed run"
"$PYTHON_BIN" scripts/rond/run_multiseed_experiments.py \
  --train research_data/processed/rond/qa_answer_selection/train.jsonl \
  --val research_data/processed/rond/qa_answer_selection/val.jsonl \
  --test research_data/processed/rond/qa_answer_selection/test.jsonl \
  --output-root results/rond/qa_answer_selection_full_corpus_comparison \
  --configs-json configs/rond/qa_answer_selection_full_corpus_comparison.json \
  --seeds "${SEEDS[@]}" \
  --cwd "$ROOT"

echo "[5/6] QA answer-selection summary"
"$PYTHON_BIN" scripts/rond/summarize_multiseed_results.py \
  --results-root results/rond/qa_answer_selection_full_corpus_comparison \
  --output-json results/rond/qa_answer_selection_full_corpus_comparison/summary.json \
  --output-csv results/rond/qa_answer_selection_full_corpus_comparison/summary.csv

echo "[6/6] QA answer-selection export assets"
"$PYTHON_BIN" scripts/rond/export_multiseed_assets.py \
  --summary-json results/rond/qa_answer_selection_full_corpus_comparison/summary.json \
  --table-csv results/tables/rond_qa_answer_selection_full_corpus_comparison.csv \
  --per-seed-csv results/figures_data/rond_qa_answer_selection_full_corpus_comparison_per_seed.csv

echo "Done."
