# Example data (5% stratified sample)

This directory is a **5% stratified random sample** of the full feature dataset. It exists only so the code runs out of the box.
**It is not the complete dataset, and metrics computed from it are not the results of a full run.**

## How it was sampled

- Stratified by `class`: each class contributes `round(n × 5%)` rows
- Fixed random seed `42`; rows copied verbatim
- **Column names and column count are identical to the original files** (no column altered, none reordered)

## Size

| Dataset | Original rows | Sampled rows |
|---|---|---|
| AMP family (`t5onnx_amp_*` / `Amp_5props_*`) | 6693 | 335 |
| biofilm family (`t5onnx_biof_*` / `biofilm_posneg_*`) | 1331 | 66 |

## Mapping to the full dataset

The files here **keep the original file names and subdirectory structure**, so swapping in the complete dataset lets you run at full scale directly:

```
<full dataset root>/
├── protT5xl/{t5onnx_amp_2merged.csv, t5onnx_biof_2merged.csv}
└── 2Feats/{Amp_5props_pogneg.csv, biofilm_posneg_comb.csv}
```

```bash
DATA_DIR=<full dataset root> python3 main.py --prefilter 600
```

## Raw sequences

The four files under `fasta/` are the upstream raw sequences behind the feature CSVs (see the top-level README for the mapping). They are **complete raw sequences, not sampled**. This pipeline does not read them; they are kept only as data provenance.
