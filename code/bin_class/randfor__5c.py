#!/usr/bin/env python3
"""
Random Forest (with Optuna) for radiomics classification.

- Label y = 1 if "Time to First Progression (Days)" is non-NA
- Train set:  traincv_set__4_topk.xlsx
- Test set:   test_set__4_topk.xlsx
- Group-aware CV to avoid patient-level leakage
- Threshold selected on calibrated OOF probabilities
- Saves artifact to: radiomics_rf.joblib

Outputs:
  - calibration_oof_vs_test__rf.png
  - brier_bars__rf.png
  - roc_plot_test__rf.png
  - feature_index_name_map__rf.csv
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import optuna

# sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold as SGKFold
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    brier_score_loss, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier

import argparse

# =================== SETTINGS ===================
parser = argparse.ArgumentParser()
parser.add_argument("--topk", type=int, required=True, help="Top-k features to use")
args = parser.parse_args()

TOPK_FEATURES = args.topk

OUTPUT_DIR = f"{TOPK_FEATURES}_features/"
TRAIN_XLSX = OUTPUT_DIR + "traincv_set__4_topk.xlsx"
TEST_XLSX  = OUTPUT_DIR + "test_set__4_topk.xlsx"
ARTIFACT   = OUTPUT_DIR + "radiomics_rf.joblib"
CALIBRATE  = True

RANDOM_STATE = 42
N_SPLITS     = 5
N_TRIALS     = 350   # RF is fast, we can tune many trials

# RF hyperparameter ranges
N_EST_MIN, N_EST_MAX     = 200, 1200
MAX_DEPTH_MIN, MAX_DEPTH_MAX = 3, 32
MIN_SAMPLES_SPLIT_MIN, MIN_SAMPLES_SPLIT_MAX = 2, 20
MIN_SAMPLES_LEAF_MIN, MIN_SAMPLES_LEAF_MAX   = 1, 15
MAX_FEATURES_OPTIONS = ["sqrt", "log2", None]

# label & columns
PROG_COL = "Time to First Progression (Days)"
SUBJECT_COL_CANDIDATES = ["subject_id", "Patient_ID", "Subject_ID"]
EXCLUDE_COLS = {"subject_id", "Patient_ID", "Subject_ID", "timepoint",
                PROG_COL, "delta_days", "days_from_dx"}

DO_TRAIN = 1

# =================== HELPERS ===================
def get_subject_col(df):
    for c in SUBJECT_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("Could not find a subject column")

def get_xy(df):
    from pandas.api.types import is_numeric_dtype
    y = df[PROG_COL].notna().astype(int).values
    feat_cols = [c for c in df.columns
                 if c not in EXCLUDE_COLS and is_numeric_dtype(df[c])]
    X = df[feat_cols].copy()
    X.columns = X.columns.astype(str)
    return X, y, feat_cols

# ---------- Correlation Filter ----------
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.80, method="spearman"):
        self.threshold = threshold
        self.method = method
        self.keep_features_ = None

    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X).copy()
        med = pd.Series(np.nanmedian(Xdf.values, axis=0), index=Xdf.columns)
        Xdf = Xdf.fillna(med)
        valid = Xdf.notna().sum(axis=0) >= 2
        Xdf = Xdf.loc[:, valid]

        corr = Xdf.corr(method=self.method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = set()
        for col in upper.columns:
            if col in drop:
                continue
            high = upper.index[upper[col] >= self.threshold].tolist()
            drop.update(high)

        self.keep_features_ = [c for c in Xdf.columns if c not in drop]
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X)
        if self.keep_features_ is None:
            return Xdf.values
        return Xdf[self.keep_features_].values

# ---------- Pipeline ----------
def build_pipeline(params, corr_threshold):
    rf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("vt",  VarianceThreshold(1e-12)),
        ("corr", CorrelationFilter(threshold=corr_threshold, method="spearman")),
        ("clf", rf),
    ])

# ---------- OOF predictions ----------
def oof_cv_predict(X, y, groups, pipe, n_splits, seed):
    n = len(y)
    oof_proba = np.full(n, np.nan)
    aucs = []

    skf = SGKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y, groups):
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[va])[:, 1]
        oof_proba[va] = p
        if len(np.unique(y[va])) == 2:
            aucs.append(roc_auc_score(y[va], p))

    return oof_proba, roc_auc_score(y, oof_proba), aucs

# ---------- Platt calibration ----------
def fit_platt_on_oof(y_true, oof_proba):
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(oof_proba.reshape(-1,1), y_true)
    return lr

# ---------- Threshold ----------
def find_best_threshold(y, p):
    vals = np.unique(p)
    if len(vals) > 1000:
        thr_grid = np.quantile(p, np.linspace(0.01,0.99,199))
        thr_grid = np.unique(thr_grid)
    else:
        thr_grid = vals

    best_thr, best_err = 0.5, float("inf")
    best_stats = {}
    for t in thr_grid:
        pred = (p >= t).astype(int)
        tp = ((pred==1)&(y==1)).sum()
        tn = ((pred==0)&(y==0)).sum()
        fp = ((pred==1)&(y==0)).sum()
        fn = ((pred==0)&(y==1)).sum()
        err = fp + fn
        if err < best_err:
            best_err = err
            best_thr = t
            best_stats = {"tp": int(tp), "tn": int(tn),
                          "fp": int(fp), "fn": int(fn),
                          "errors": int(err),
                          "accuracy": (tp+tn)/len(y)}
    return float(best_thr), best_stats

# =================== TRAIN ===================
def train_rf():
    df = pd.read_excel(TRAIN_XLSX)
    subj = get_subject_col(df)
    df[subj] = df[subj].astype(str)

    X, y, feat_cols = get_xy(df)
    groups = df[subj].values

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators",
                                              N_EST_MIN, N_EST_MAX),
            "max_depth": trial.suggest_int("max_depth",
                                            MAX_DEPTH_MIN, MAX_DEPTH_MAX),
            "min_samples_split": trial.suggest_int("min_samples_split",
                                                   MIN_SAMPLES_SPLIT_MIN, MIN_SAMPLES_SPLIT_MAX),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf",
                                                  MIN_SAMPLES_LEAF_MIN, MIN_SAMPLES_LEAF_MAX),
            "max_features": trial.suggest_categorical(
                                "max_features", MAX_FEATURES_OPTIONS),
        }
        corr = trial.suggest_float("corr_threshold", 0.75, 0.95)

        pipe = build_pipeline(params, corr)
        oof_raw, oof_auc, _ = oof_cv_predict(X, y, groups,
                                             pipe, N_SPLITS, RANDOM_STATE)

        if CALIBRATE:
            platt = fit_platt_on_oof(y, oof_raw)
            oof_cal = platt.predict_proba(oof_raw.reshape(-1,1))[:,1]
            thr, stats = find_best_threshold(y, oof_cal)
        else:
            thr, stats = find_best_threshold(y, oof_raw)

        trial.set_user_attr("threshold", thr)
        trial.set_user_attr("errors_fp_fn", stats["errors"])
        return oof_auc

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        direction="maximize"
    )
    study.optimize(objective, n_trials=N_TRIALS,
                   n_jobs=-1, show_progress_bar=True)

    best = study.best_trial
    params = best.params
    corr = params.pop("corr_threshold")

    print("\n[optuna] Best AUC:", best.value)
    print("[optuna] Params:", params)

    # Recompute OOF to get final threshold & calibration
    best_pipe = build_pipeline(params, corr)
    oof_raw, oof_auc, _ = oof_cv_predict(X, y, groups,
                                         best_pipe, N_SPLITS, RANDOM_STATE)

    calibrator = None
    if CALIBRATE:
        calibrator = fit_platt_on_oof(y, oof_raw)
        oof_cal = calibrator.predict_proba(oof_raw.reshape(-1,1))[:,1]
        thr, stats = find_best_threshold(y, oof_cal)
        try:
            print("[calibration] Brier(calibrated) =",
                  brier_score_loss(y, oof_cal))
        except:
            pass
    else:
        thr, stats = find_best_threshold(y, oof_raw)

    print("[final OOF] AUC:", round(oof_auc,4))
    print("[final OOF] threshold:", thr)
    print("[final OOF] stats:", stats)

    # Fit full model
    best_pipe.fit(X, y)

    artifact = {
        "pipeline": best_pipe,
        "threshold": thr,
        "params": params,
        "feature_cols": feat_cols,
        "subject_col": subj,
        "train_oof_auc": float(oof_auc),
        "train_oof_threshold_stats": stats,
        "calibrate": CALIBRATE,
        "platt_calibrator": calibrator
    }
    joblib.dump(artifact, ARTIFACT)
    print(f"[saved] {ARTIFACT}")

# =================== TEST ===================
def test_rf():
    art = joblib.load(ARTIFACT)
    pipe = art["pipeline"]
    thr = art["threshold"]
    feat_cols = art["feature_cols"]
    calibrate = art["calibrate"]
    platt = art["platt_calibrator"]

    # ----- Load test -----
    df = pd.read_excel(TEST_XLSX)
    y = df[PROG_COL].notna().astype(int).values
    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan
    X = df[feat_cols].copy()

    # ----- Predict -----
    proba_raw = pipe.predict_proba(X)[:,1]
    proba = (platt.predict_proba(proba_raw.reshape(-1,1))[:,1]
             if calibrate and platt is not None else proba_raw)
    pred = (proba >= thr).astype(int)

    # ----- Train-CV for calibration curve -----
    df_tr = pd.read_excel(TRAIN_XLSX)
    y_tr = df_tr[PROG_COL].notna().astype(int).values
    for c in feat_cols:
        if c not in df_tr.columns:
            df_tr[c] = np.nan
    X_tr = df_tr[feat_cols].copy()
    proba_raw_tr = pipe.predict_proba(X_tr)[:,1]
    proba_tr = (platt.predict_proba(proba_raw_tr.reshape(-1,1))[:,1]
                if calibrate and platt else proba_raw_tr)

    # ----- Calibration curve -----
    fig, ax = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
    n_bins = 7
    fr1, mp1 = calibration_curve(y_tr, proba_tr, n_bins=n_bins, strategy="quantile")
    ax[0].plot(mp1,fr1,"o-",label="TrainCV")
    ax[0].plot([0,1],[0,1],"k--")
    ax[0].set_title("Calibration – Train-CV")
    ax[0].grid(True)

    fr2, mp2 = calibration_curve(y, proba, n_bins=n_bins, strategy="quantile")
    ax[1].plot(mp2,fr2,"o-",label="Test")
    ax[1].plot([0,1],[0,1],"k--")
    ax[1].set_title("Calibration – Test")
    ax[1].grid(True)

    plt.savefig("calibration_oof_vs_test__rf.png", dpi=150)
    plt.close()

    # ----- Brier -----
    brier_tr = brier_score_loss(y_tr, proba_tr)
    brier_te = brier_score_loss(y, proba)
    print(f"[brier] TrainCV={brier_tr:.4f} | Test={brier_te:.4f}")

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    bars = ax.bar(["TrainCV","Test"], [brier_tr,brier_te])
    ax.set_title("Brier Scores – RF")
    ax.set_ylabel("Brier")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
                f"{b.get_height():.3f}", ha="center")
    plt.tight_layout()
    plt.savefig("brier_bars__rf.png", dpi=150)
    plt.close()

    # ----- ROC -----
    fpr, tpr, _ = roc_curve(y, proba)
    auc_test = roc_auc_score(y, proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr,tpr,label=f"RF (AUC={auc_test:.3f})")
    plt.plot([0,1],[0,1],"k--")
    plt.grid(True)
    plt.legend()
    plt.title("ROC – Test (RF)")
    plt.tight_layout()
    plt.savefig("roc_plot_test__rf.png", dpi=300)
    plt.close()

    # ----- Confusion & report -----
    cm = confusion_matrix(y, pred)
    print("\n[test] Confusion Matrix:\n", cm)
    print(f"\n[test] ROC AUC: {auc_test}\n")
    print("\n[test] classification report:\n",
          classification_report(y, pred, digits=4))
    print("[test] first 10 probabilities:", np.round(proba[:10],4))

# ---------- Feature mapping ----------
def export_feature_mapping_numbers(csv_path=OUTPUT_DIR + "feature_index_name_map__rf.csv"):
    art = joblib.load(ARTIFACT)
    pipe = art["pipeline"]
    feat_cols = art["feature_cols"]

    imp = pipe.named_steps["imp"]
    vt = pipe.named_steps["vt"]
    corr = pipe.named_steps["corr"]

    vt_mask = vt.get_support(False)
    names_after_vt = np.array(feat_cols)[vt_mask]

    kept = corr.keep_features_
    kept_names = np.array([n for n in kept if n in names_after_vt])
    kept_idx = np.array([np.where(names_after_vt == n)[0][0]
                         for n in kept_names], dtype=int)

    orig_idx = [feat_cols.index(n) for n in kept_names]

    df_map = pd.DataFrame({
        "feature_id": np.arange(len(kept_names)),
        "feature_name": kept_names,
        "vt_index": kept_idx,
        "orig_index": orig_idx,
        "orig_col": [feat_cols[i] for i in orig_idx],
    })
    df_map.to_csv(csv_path, index=False)
    print(f"[mapping] saved {csv_path}")

# =================== entry ===================
if __name__ == "__main__":
    if DO_TRAIN:
        train_rf()
    test_rf()
    export_feature_mapping_numbers()

