# Advancing Low-Resource Radiation Oncology NLP

This repository contains the public reproducibility package for our radiation oncology NLP study:

**Advancing Low-Resource Radiation Oncology NLP: Keyword-Focused Domain Adaptation via Public Corpora**

The project studies how public-data-only adaptation behaves in a highly specialized radiation oncology NLP setting. Rather than assuming that narrowly filtered in-domain text is always better, we examine when focused adaptation helps, when it fails, and how replay-regularized adaptation affects external generalization.

## What Is Included

- `scripts/`
  - public corpus construction
  - ROND preprocessing and downstream training
  - continued pretraining / replay-regularized adaptation
  - robustness analysis utilities
- `configs/`
  - experiment configuration files used in the released study workflow
- `data/manifests/`
  - manually curated radiation-oncology keyword list
- `data/processed/rond/`
  - processed public ROND task splits used in the released experiments
- `results/`
  - placeholder structure for regenerated experiment outputs

## What Is Not Included

- manuscript snapshots
- paper-facing result tables
- remote checkpoints and large transient training artifacts

## Main Study Questions

We focus on four questions:

1. Can keyword-focused public domain adaptation help low-resource radiation oncology NLP?
2. How sensitive are the results to corpus construction choices such as sentence-window size?
3. Does the method remain competitive against other biomedical / clinical encoder baselines?
4. Can controlled replay regularization improve external generalization under a leakage-controlled setup?

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
├── results/
└── scripts/
    ├── analysis/
    ├── domain_adaptation/
    ├── public_corpus/
    └── rond/
```

## Reproducibility Notes

- The adaptation pipeline is **public-data-only**.
- The released keyword lexicon is the exact manifest used for focused corpus construction.
- The repository includes processed public benchmark splits needed to rerun the released experiments.
- Large regenerated corpora from NCI PDQ / PMC, paper-facing result exports, and remote training artifacts are intentionally not checked in directly.
- The external validation protocol on PubMedQA is leakage-controlled: replay text is restricted to the train split only.

## Suggested Reproduction Order

1. Prepare / inspect the processed ROND splits in `data/processed/rond/`.
2. Review the keyword manifest in `data/manifests/public_radonc_keywords.txt`.
3. Rebuild public corpora with the scripts in `scripts/public_corpus/`.
4. Run continued pretraining with `scripts/domain_adaptation/train_mlm_adaptation.py`.
5. Run downstream multi-seed evaluation using the configs in `configs/rond/`.
6. Recompute robustness summaries with the analysis scripts in `scripts/analysis/`.

## Environment

The experiments were developed in a Python / PyTorch / Hugging Face workflow on remote GPU infrastructure. See `requirements.txt` for the main Python dependencies used by the released scripts.

## Data and Usage

All data included here are public or derived from public resources. If you reuse this repository, please keep the source attributions for ROND, NCI PDQ, PMC, and PubMedQA.
