# Advancing Low-Resource Radiation Oncology NLP

![Study Type](https://img.shields.io/badge/Study-Reproducibility%20Package-2f6fed)
![Data](https://img.shields.io/badge/Data-Public%20Only-1f9d55)
![Domain](https://img.shields.io/badge/Domain-Radiation%20Oncology%20NLP-b45f06)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Transformers-7a3ff2)
![License](https://img.shields.io/badge/License-MIT-111111)

Public reproducibility package for the study:

**Advancing Low-Resource Radiation Oncology NLP: Keyword-Focused Domain Adaptation via Public Corpora**

## Overview

This repository releases the code, configurations, keyword manifest, and processed public benchmark assets used to study public-data-only adaptation for radiation oncology NLP.

The project asks a focused question: when does narrow, radiation-oncology-specific adaptation help low-resource clinical NLP, and when does it fail? Rather than treating in-domain adaptation as universally beneficial, the study examines corpus construction, robustness, and external generalization under privacy-preserving constraints.

## Scope

The released workflow covers:

- public corpus construction from NCI PDQ and PMC
- keyword-focused sentence-window extraction
- continued pretraining of biomedical encoders
- downstream evaluation on public ROND tasks
- robustness analyses, including repeated cross-validation and small-sample checks
- leakage-controlled external validation with PubMedQA

## Included

### Code

- `scripts/public_corpus/`
  - corpus building, filtering, and replay-corpus preparation
- `scripts/domain_adaptation/`
  - continued pretraining and adaptation-round orchestration
- `scripts/rond/`
  - ROND preprocessing, multi-seed runs, and downstream training
- `scripts/analysis/`
  - robustness summaries and post-hoc analysis helpers

### Configurations

- `configs/rond/`
  - released experiment configurations for ROND-based analyses
- `configs/pubmedqa/`
  - configurations for leakage-controlled external validation

### Public Data Assets

- `data/manifests/public_radonc_keywords.txt`
  - manually curated radiation-oncology keyword lexicon used for focused corpus construction
- `data/processed/rond/`
  - processed public ROND splits used by the released experiments

### Runtime Layout

- `results/`
  - placeholder directory for regenerated outputs

## Not Included

This repository does **not** ship:

- manuscript snapshots
- paper-facing result tables
- remote checkpoints
- transient server-side run logs
- large regenerated corpora mirrored from public upstream sources

Those artifacts are omitted to keep the public package clean, lightweight, and focused on reproducibility rather than paper packaging.

## Repository Structure

```text
.
+-- configs/
|   +-- pubmedqa/
|   \-- rond/
+-- data/
|   +-- manifests/
|   \-- processed/
|       \-- rond/
+-- results/
\-- scripts/
    +-- analysis/
    +-- domain_adaptation/
    +-- public_corpus/
    \-- rond/
```

## Quick Start

Recommended order:

1. Inspect the released ROND assets in `data/processed/rond/`.
2. Review the keyword manifest in `data/manifests/public_radonc_keywords.txt`.
3. Rebuild public corpora with `scripts/public_corpus/`.
4. Run continued pretraining with `scripts/domain_adaptation/train_mlm_adaptation.py`.
5. Launch downstream multi-seed experiments using `configs/rond/`.
6. Recompute robustness summaries with `scripts/analysis/`.
7. Recreate external validation with `configs/pubmedqa/`.

## Reproducibility Notes

- The adaptation pipeline is strictly **public-data-only**.
- The released keyword list is the exact lexicon used for focused corpus construction.
- The PubMedQA external validation protocol is leakage-controlled: replay text is restricted to the training split only.
- Large public corpora should be regenerated from the released scripts rather than redistributed directly here.

## Environment

The released scripts assume a Python-based machine learning workflow centered on:

- PyTorch
- Hugging Face Transformers
- Datasets
- scikit-learn

See `requirements.txt` for the dependency list used in the public package.

## License

This repository is released under the MIT License. See `LICENSE` for details.

## Data Sources

This repository uses public or public-derived resources only. Please retain the original attributions and usage terms of:

- ROND
- NCI PDQ
- PMC Open Access
- PubMedQA

## Citation

If you use this repository, please cite the corresponding paper and project metadata in [`CITATION.cff`](./CITATION.cff).

## Release Notes

- Public package goal: reproducibility rather than manuscript mirroring
- Paper-facing tables and manuscript snapshots are intentionally omitted
- See [`RELEASE_CHECKLIST.md`](./RELEASE_CHECKLIST.md) for the release-maintenance checklist
