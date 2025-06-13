# TabPFN_MergerAcqusition

This repository contains a lightweight pipeline for predicting the outcome of US stock mergers and acquisitions. The code was developed as part of a master thesis at University College Dublin.

The pipeline trains several tuned baseline classifiers together with TabPFN, explains predictions via SHAP, and optionally runs decision curve analysis.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py \
  --input-csv /path/to/Model\ Training\ Data.csv \
  --output-cache cache_light_rq \
  --seed 42 \
  --smote-sample-size 2000 \
  --top-k 10 \
  --tabpfn-time 10
```

| Flag | Description | Default |
|------|-------------|---------|
| `--input-csv` | Path to input CSV | **required** |
| `--output-cache` | Directory for TabPFN cache | `cache_light_rq` |
| `--seed` | Random seed | `42` |
| `--smote-sample-size` | Maximum samples for SMOTE sub-sampling | `2000` |
| `--top-k` | Number of top features selected by RF importance | `10` |
| `--tabpfn-time` | Maximum seconds for TabPFN training | `10` |

## Pipeline Steps

1. **Load and preprocess data** – categorical attitude values are one-hot encoded and missing values are imputed with medians.
2. **SMOTE and feature pruning** – the training set is balanced with SMOTE and the top features are selected using a random forest.
3. **Model training** – logistic regression, random forest, gradient boosting, SVM and MLP are tuned with randomized search. TabPFN is trained with a user-defined time limit and cached for reuse.
4. **Evaluation** – models are evaluated on the held-out test set (AUC, F1, calibration metrics). SHAP explanations are computed for one test instance and decision curve analysis can be performed on the predictions.

Results and any cached TabPFN model are written to the directory given by `--output-cache`.

See `src/main.py` for the full implementation.
