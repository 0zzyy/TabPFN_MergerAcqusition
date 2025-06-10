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

| Flag                  | Description                                       | Default          |
| --------------------- | ------------------------------------------------- | ---------------- |
| `--input-csv`         | Path to input CSV                                 | **Required**     |
| `--output-cache`      | Directory for TabPFN cache                        | `cache_light_rq` |
| `--seed`              | Random seed                                       | `42`             |
| `--smote-sample-size` | Max samples for SMOTE sub-sampling                | `2000`           |
| `--top-k`             | Number of top features to select by RF importance | `10`             |
| `--tabpfn-time`       | Max seconds for TabPFN training                   | `10`             |

##Outputs

---

### 5. `src/main.py`

```python
#!/usr/bin/env python3
"""
Light Pipeline for RQ1 & RQ2:
  • Tuned baselines (LR, RF, GB, SVM, MLP)
  • TabPFN (max_time)
  • SHAP for each model
  • (Optional) Decision Curve Analysis (DCA)

Designed for macOS M1 Air (8 GB RAM) as of 2025-06-04.
"""
import argparse
import logging
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import shap
from joblib import dump, load
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from tabpfn import TabPFNClassifier
from dcurves import dca

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
warnings_kwargs = {
    "category": ConvergenceWarning,
    "action": "ignore"
}
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(
        description="Train baselines + TabPFN + SHAP explanations"
    )
    p.add_argument("--input-csv", type=pathlib.Path, required=True)
    p.add_argument("--output-cache", type=pathlib.Path, default=pathlib.Path("cache_light_rq"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smote-sample-size", type=int, default=2000)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--tabpfn-time", type=int, default=10)
    return p.parse_args()

def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    y_true = np.asarray(y_true)
    y_prob = np.nan_to_num(np.clip(y_prob, 0, 1), nan=0.5)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob > lo) & (y_prob <= hi)
        prop = mask.mean()
        if prop > 0:
            acc  = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += abs(acc - conf) * prop
    return ece

def summarize_calibration(name, y_true, y_prob):
    brier = brier_score_loss(y_true, y_prob)
    ece   = compute_ece(y_true, y_prob)
    log.info(f"{name}: Brier={brier:.4f}, ECE={ece:.4f}")

def extract_shap_vals(vals):
    """Flatten SHAP output to 1D array for binary positive class."""
    if hasattr(vals, "values"):
        arr = vals.values
    else:
        arr = vals
    if isinstance(arr, list):
        arr = arr[1]
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:
        arr = arr[..., 1]
        if arr.shape[0] == 1:
            arr = arr[0]
    return arr

def main():
    args = parse_args()
    np.random.seed(args.seed)
    logging.captureWarnings(True)

    log.info("=== PIPELINE START ===")

    # Load
    if not args.input_csv.exists():
        log.error(f"File not found: {args.input_csv}")
        sys.exit(1)
    df = pd.read_csv(args.input_csv, index_col=0, low_memory=False)
    if "status" not in df:
        log.error("Target 'status' missing"); sys.exit(1)
    y = df["status"].astype(int)
    X = df.drop(columns=[c for c in [
        "status","target_id","acquirer_id",
        "announcement_date","exit_date","target_naics_code"
    ] if c in df])

    # One-hot attitude
    if "attitude" in X:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        att = enc.fit_transform(X[["attitude"]])
        cols = enc.get_feature_names_out(["attitude"])
        X = X.drop(columns=["attitude"])
        X = pd.concat([X, pd.DataFrame(att, columns=cols, index=X.index)], axis=1)

    # Fill
    for col in X.select_dtypes(include="number"):
        if X[col].isna().any():
            X[col].fillna(X[col].median(), inplace=True)
    X.fillna(0, inplace=True)
    log.info(f"Data: X={X.shape}, y={y.shape}")

    # Split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=args.seed
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=args.seed
    )
    log.info(f"Split: train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}")

    # SMOTE & feature prune
    sub = min(len(X_tr), args.smote_sample_size)
    X_sm, y_sm = train_test_split(
        X_tr, y_tr, train_size=sub, stratify=y_tr, random_state=args.seed
    ) if sub < len(X_tr) else (X_tr, y_tr)
    Xr, yr = SMOTE(random_state=args.seed).fit_resample(X_sm, y_sm)
    rf0 = RandomForestClassifier(n_estimators=50, random_state=args.seed, n_jobs=-1)
    rf0.fit(Xr, yr)
    top = pd.Series(rf0.feature_importances_, index=Xr.columns).nlargest(args.top_k).index
    log.info(f"Top{args.top_k} features: {list(top)}")

    X_tr_p, X_val_p, X_te_p = X_tr[top], X_val[top], X_te[top]
    Xr_p = pd.DataFrame(Xr, columns=Xr.columns)[top]

    # Scale
    scaler = StandardScaler().fit(Xr_p)
    Xr_s = scaler.transform(Xr_p)
    X_te_s = scaler.transform(X_te_p)

    # Models
    defs = {
        "LR": LogisticRegression(solver="liblinear", max_iter=500, random_state=args.seed),
        "RF": RandomForestClassifier(random_state=args.seed, n_jobs=-1),
        "GB": GradientBoostingClassifier(random_state=args.seed),
        "SVM": SVC(kernel="rbf", probability=True, random_state=args.seed),
        "MLP": MLPClassifier(max_iter=200, tol=1e-4, early_stopping=True, random_state=args.seed, hidden_layer_sizes=(50,)),
    }
    grids = {
        "LR": {"C":[0.1,1],"penalty":["l2"]},
        "RF": {"n_estimators":[50],"max_depth":[None,5]},
        "GB": {"n_estimators":[50],"max_depth":[3],"learning_rate":[0.1]},
        "SVM": {"C":[0.1,1],"gamma":["scale"]},
        "MLP":{"alpha":[1e-3],"learning_rate_init":[1e-3]}
    }

    perf, fitted = {}, {}
    for name, mdl in defs.items():
        log.info(f"Tuning {name}")
        try:
            search = RandomizedSearchCV(
                mdl, grids[name], n_iter=2,
                cv=StratifiedKFold(2, shuffle=True, random_state=args.seed),
                scoring="roc_auc", random_state=args.seed, n_jobs=1
            )
            search.fit(Xr_s, yr)
            m = search.best_estimator_; fitted[name]=m
            probs = m.predict_proba(X_te_s)[:,1]
            perf[name] = {
                "AUC": roc_auc_score(y_te, probs),
                "F1":  f1_score(y_te, (probs>0.5).astype(int))
            }
            log.info(f"{name} → AUC={perf[name]['AUC']:.4f}, F1={perf[name]['F1']:.4f}")
            summarize_calibration(name, y_te, probs)
        except Exception as e:
            log.error(f"{name} failed: {e}")
            fitted[name]=None; perf[name]={"AUC":np.nan,"F1":np.nan}

    # TabPFN
    args.output_cache.mkdir(exist_ok=True)
    pkl = args.output_cache / "tabpfn_rq.joblib"
    if pkl.exists():
        try:
            tp = load(pkl); log.info("Loaded cached TabPFN")
        except:
            tp = None
    else:
        tp = None

    if tp is None:
        try:
            tp = TabPFNClassifier(device="cpu", max_time=args.tabpfn_time)
            tp.fit(Xr_p.values.astype(np.float32), yr.values.astype(np.int64))
            dump(tp, pkl); log.info("Trained & cached TabPFN")
        except Exception as e:
            log.error(f"TabPFN failed: {e}")
            tp = None

    if tp:
        probs = tp.predict_proba(X_te_p.values.astype(np.float32))[:,1]
        perf["TabPFN"] = {
            "AUC": roc_auc_score(y_te, probs),
            "F1":  f1_score(y_te, (probs>0.5).astype(int))
        }
        log.info(f"TabPFN → AUC={perf['TabPFN']['AUC']:.4f}, F1={perf['TabPFN']['F1']:.4f}")
        summarize_calibration("TabPFN", y_te, probs)

    # SHAP explanations (one background + one test sample)
    bg = pd.DataFrame(Xr_p).sample(1, random_state=args.seed)
    ts = X_te_p.iloc[:1].reset_index(drop=True)
    shap_imps = {}
    for name, mdl in {**fitted, **({"TabPFN":tp} if tp else {})}.items():
        if mdl is None: continue
        try:
            log.info(f"Explaining {name}")
            expl = shap.Explainer(
                mdl.predict_proba if name!="RF" and name!="GB" else mdl, 
                bg
            )
            out = expl(ts)
            arr = extract_shap_vals(out)
            imp = pd.Series(np.abs(arr), index=ts.columns).nlargest(1)
            shap_imps[name] = imp
            log.info(f"  Top SHAP for {name}: {imp.index[0]} ({imp.iloc[0]:.4f})")
        except Exception as e:
            log.warning(f"{name} SHAP failed: {e}")

    # DCA (optional)
    all_df = ts.copy(); all_df["status"]=y_te.values[:1]
    for name, mdl in fitted.items():
        if mdl:
            try: all_df[name]=mdl.predict_proba(X_te_s)[:,1]
            except: pass
    if tp:
        try: all_df["TabPFN"]=tp.predict_proba(X_te_p.values.astype(np.float32))[:,1]
        except: pass

    try:
        dca(all_df, outcome="status", modelnames=[n for n in all_df.columns if n!="status"])
        log.info("DCA computed")
    except Exception:
        log.info("DCA skipped/failure")

    # Summary
    log.info("=== FINAL METRICS ===")
    for k,v in perf.items():
        log.info(f"{k:6s} → AUC={v['AUC']:.4f}, F1={v['F1']:.4f}")

    log.info(f"Total time: {(time.time()):.1f}s")
    log.info("=== COMPLETE ===")

if __name__ == "__main__":
    main()
