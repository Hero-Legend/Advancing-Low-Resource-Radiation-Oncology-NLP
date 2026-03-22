# Data Notes

This folder contains only public or public-derived artifacts that are lightweight enough to distribute directly.

## Included

### Keyword Manifest

- `manifests/public_radonc_keywords.txt`
  - manually curated radiation-oncology keyword list used for focused corpus construction

### Processed Benchmark Assets

- `processed/rond/`
  - processed public ROND splits used by the released experiments

## Not Included

The following artifacts are intentionally omitted:

- large regenerated adaptation corpora derived from public upstream sources
- remote checkpoints
- server-side transient logs
- manuscript-facing result exports

## Reconstruction

The omitted public corpora can be rebuilt with:

- `scripts/public_corpus/build_nci_pdq_corpus.py`
- `scripts/public_corpus/build_pmc_manifest.py`
- `scripts/public_corpus/build_keyword_focused_corpus.py`
- `scripts/public_corpus/compile_public_radonc_corpus.py`

## Upstream Public Sources

- ROND
- NCI PDQ
- PMC Open Access
- PubMedQA

Please follow the original source licenses and access terms when redistributing regenerated corpora.
