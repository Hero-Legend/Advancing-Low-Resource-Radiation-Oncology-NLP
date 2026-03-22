# Data Notes

This repository ships only public or public-derived artifacts.

## Included

- `manifests/public_radonc_keywords.txt`
  - manually curated keyword manifest used for focused corpus construction
- `processed/rond/`
  - processed ROND benchmark task splits used in the released experiments

## Not Included

- large regenerated adaptation corpora downloaded from public upstream sources
- remote training checkpoints
- transient run logs from GPU servers

These omitted artifacts can be reconstructed with the scripts in `scripts/public_corpus/` and `scripts/domain_adaptation/`.

## Public Upstream Sources

- ROND
- NCI PDQ
- PMC Open Access
- PubMedQA

Please follow the original source terms when redistributing regenerated corpora.
