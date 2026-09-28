# ABPclass — End-to-End Training Pipeline

`main.py` implements an end-to-end pipeline for **in-fold MRMD3.0 feature selection followed by a Transformer + RBF-SVM ensemble**, using ESM2, ProtT5-XL and physicochemical (ACC / CC / nmbroto) features.

In each outer 5-fold CV, feature selection and training use **only the training fold (80%)**, while the validation fold (20%) is evaluated independently. Results are reported as ACC / F1 / Sn / Sp / MCC / AUC, plus a pooled AUC computed over the concatenated folds.

## Pipeline

Inside each fold (training-fold labels only, no leakage):

1. **Pre-filter** — keep the top-`PREFILTER` features by `|Pearson corr with class|`
2. **MRMD3.0 ranking** — the official `run()` computes Euclidean / Cosine / Tanimoto / Person distances to produce three rankings, then takes the top-`K_USE` (1020) single features
3. **Second-level crosses** — build pairwise cross features from the top-200 single features, pre-filter the top-300 by `|Pearson corr|`, rank them with the official `run()`, and take the top-`N_USE` (30)
4. **Training** — a two-stage Transformer (AdamW + cross-entropy, then higher dropout + focal loss) plus an RBF-SVM; the two probability outputs are averaged before evaluation

## Layout

```
main.py                          # entry point
requirements.txt
LICENSE                          # MIT
README.md
data/                            # example data (see data/README.md)
├── fasta/                       # raw sequences (data provenance)
├── protT5xl/                    # feature CSVs, 5% sampled
└── 2Feats/
mrmd3_official/
└── feature_selection/           # official MRMD3.0 (only the files needed to run)
    ├── __init__.py
    └── MRMD.py
```

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on: macOS / Python 3.11 / torch 2.0.1 (CPU) / numpy 1.26.4 / pandas 2.3.3 / scikit-learn 1.7.2 / imbalanced-learn 0.14.1. **CPU only** — no GPU required.

## Running

```bash
# quick test (2 folds; the sampled data shipped in this repo is enough)
python3 main.py --test --prefilter 250

# full run (5 folds; needs the complete dataset)
python3 main.py --prefilter 600

# pick an MRMD variant
python3 main.py --variant cos
```

### Using the full dataset

`data/` here contains only a **5% sampled subset**. For a full run, point the environment variable `DATA_DIR` at the directory holding the complete feature CSVs (it must have the same subdirectory structure):

```bash
DATA_DIR=/path/to/full/features python3 main.py --prefilter 600
```

## Input data

Four CSVs are required (the first column of each is `class`). Each file supplies the features for one side — `AMP` or `biofilm`:

| Path | Side | Feature source |
|---|---|---|
| `2Feats/Amp_5props_pogneg.csv` | AMP | ESM2 + physicochemical (ACC / CC / nmbroto) |
| `2Feats/biofilm_posneg_comb.csv` | biofilm | ESM2 + physicochemical (ACC / CC / nmbroto) |
| `protT5xl/t5onnx_amp_2merged.csv` | AMP | ProtT5-XL |
| `protT5xl/t5onnx_biof_2merged.csv` | biofilm | ProtT5-XL |

Per side the blocks are concatenated in the order `ESM2 → ProtT5-XL → ACC → CC → nmbroto`, giving **2408** input features (ESM2 1280 + ProtT5-XL 1024 + ACC 18 + CC 6 + nmbroto 80).

## Raw sequences (FASTA)

`data/fasta/` holds the upstream raw sequences behind the feature CSVs, mapped to the classes of the two datasets:

| File | Sequences | Corresponds to |
|---|---|---|
| `amp_nr_40.fasta` | 1157 | AMP set, positive class |
| `nonamp_nr_40.fasta` | 5536 | AMP set, negative class |
| `biofilm_5Sets_delX_95.fasta` | 308 | biofilm set, positive class |
| `biofNeg_delX_95.fasta` | 1023 | biofilm set, negative class |

> This pipeline reads the feature CSVs directly and **does not parse FASTA**; the FASTA files are kept only as data provenance.

## Output

A run writes the following into the script directory:

- `main_results.csv` — aggregate metrics (ACC / F1 / Sn / Sp / MCC / AUC + confusion matrix)
- `main_results.npz` — per-fold scores / labels / metrics, for reproduction and fold-level analysis
- `main_fold_metrics.csv` — per-fold details

## Third-party code

`mrmd3_official/feature_selection/MRMD.py` is the **official MRMD 3.0 implementation**; copyright belongs to its original authors, so please cite their paper if you use it. Only the files actually called by this pipeline are included (`MRMD.py` plus an empty `__init__.py`); the official package's `util/`, `mrmr2/` and other feature-selection scripts are unrelated to this pipeline and are not included.

## Notes and limitations

- `data/` in this repository is a 5% stratified sample, so **metrics computed from it are not the results of a full run**. Obtain the complete dataset separately.
- Everything runs on CPU; a full 5-fold run takes a while.

## License

This project is released under the MIT License (see `LICENSE`). Third-party code under `mrmd3_official/` remains under its original authors' copyright and is not covered by this project's license.
