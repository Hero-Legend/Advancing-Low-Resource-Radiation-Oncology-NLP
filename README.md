# Advancing Low-Resource Radiation Oncology NLP

This repository contains the public release package for our paper:

**Advancing Low-Resource Radiation Oncology NLP: Keyword-Focused Domain Adaptation via Public Corpora**

The project studies how public-data-only adaptation behaves in a highly specialized radiation oncology NLP setting. Rather than assuming that narrowly filtered in-domain text is always better, we evaluate when focused adaptation helps, when it breaks, and how replay-regularized adaptation changes external generalization.

## What Is Included

- `scripts/`
  - public corpus construction
  - ROND preprocessing and downstream training
  - continued pretraining / replay-regularized adaptation
  - robustness analysis utilities
- `configs/`
  - experiment configuration files used for the paper
- `data/manifests/`
  - manually curated radiation-oncology keyword list
- `data/processed/rond/`
  - processed public ROND task splits used in the released experiments
- `results/tables/`
  - exported tables supporting the paper
- `paper/`
  - current manuscript source and compiled PDF snapshot

## Main Study Questions

We focus on four questions:

1. Can keyword-focused public domain adaptation help low-resource radiation oncology NLP?
2. How sensitive are the results to corpus construction choices such as sentence-window size?
3. Does the method remain competitive against other biomedical / clinical encoder baselines?
4. Can controlled replay regularization improve external generalization under a leakage-controlled setup?

## Core Findings

- On the original ROND text-classification anchor task, focused adaptation improved mean macro-F1 from `0.6360` to `0.6600`.
- Window-size effects were non-monotonic: larger context windows increased corpus size but did not consistently improve downstream performance.
- Family-level comparison showed that `PubMedBERT + focused adaptation` remained competitive against `BioLinkBERT` and `Bio_ClinicalBERT`.
- Robustness analyses showed that the anchor-task gains were conditional rather than uniformly dominant.
- In leakage-controlled external validation on PubMedQA, the strongest external signal came from `focused + replay10 (train-only)`, which outperformed the weighted PubMedBERT baseline.

## Repository Layout

```text
.
├── configs/
│   ├── pubmedqa/
│   └── rond/
├── data/
│   ├── manifests/
│   └── processed/
│       └── rond/
├── paper/
├── results/
│   └── tables/
└── scripts/
    ├── analysis/
    ├── domain_adaptation/
    ├── public_corpus/
    └── rond/
```

## Reproducibility Notes

- The adaptation pipeline is **public-data-only**.
- The released keyword lexicon is the exact manifest used for focused corpus construction.
- The repository includes processed public benchmark splits and summary result tables.
- Some large regenerated corpora from NCI PDQ / PMC and remote training artifacts are not checked in directly; they should be rebuilt from the released scripts.
- The external validation protocol on PubMedQA is leakage-controlled: replay text is restricted to the train split only.

## Suggested Reproduction Order

1. Prepare / inspect the processed ROND splits in `data/processed/rond/`.
2. Review the keyword manifest in `data/manifests/public_radonc_keywords.txt`.
3. Rebuild public corpora with the scripts in `scripts/public_corpus/`.
4. Run continued pretraining with `scripts/domain_adaptation/train_mlm_adaptation.py`.
5. Run downstream multi-seed evaluation using the configs in `configs/rond/`.
6. Recompute robustness summaries with the analysis scripts in `scripts/analysis/`.

## Key Result Tables

- `results/tables/paper1_table_main_results.csv`
- `results/tables/paper1_table_encoder_ablation.csv`
- `results/tables/paper1_encoder_family_comparison.csv`
- `results/tables/window_size_ablation_final_summary.csv`
- `results/tables/paper1_repeated_stratified_cv_summary.csv`
- `results/tables/pubmedqa_external_validation_comparison.csv`
- `results/tables/pubmedqa_external_validation_bootstrap_ci.csv`
- `results/tables/pubmedqa_external_validation_pairwise.csv`
- `results/tables/pubmedqa_external_validation_classwise.csv`

## Environment

The experiments were developed in a Python / PyTorch / Hugging Face workflow on remote GPU infrastructure. See `requirements.txt` for the main Python dependencies used by the released scripts.

## Data and Usage

All data included here are public or derived from public resources. If you reuse this repository, please keep the source attributions for ROND, NCI PDQ, PMC, and PubMedQA.
