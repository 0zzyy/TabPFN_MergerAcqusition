# TabPFN_MergerAcqusition
TabPFN tested for US stock asset merger and acquisition. Conducted as part of Master Thesis at University College Dublin.

# Light Pipeline for RQ1 & RQ2

This repository contains a lightweight, reproducible pipeline to:
1. Train and evaluate tuned baselines (LR, RF, GB, SVM, MLP)
2. Train and evaluate TabPFN
3. Generate SHAP explanations
4. (Optionally) run Decision Curve Analysis (DCA)

Designed to run on macOS M1 Air (8 GB RAM) as of 2025-06-04.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

##Usage
python src/main.py \
  --input-csv "/path/to/Model Training Data.csv" \
  --output-cache "cache_light_rq" \
  --seed 42 \
  --smote-sample-size 2000 \
  --top-k 10 \
  --tabpfn-time 10
