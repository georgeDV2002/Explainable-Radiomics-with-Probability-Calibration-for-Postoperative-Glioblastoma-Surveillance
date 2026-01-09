#!/usr/bin/env python3
"""
Elastic Net Logistic (with Optuna + Platt calibration + SHAP) — MGMT.

- Label y = MGMT (0/1); rows with NaN in MGMT are dropped.
- Train set:  derivatives/traincv_set__4_topk.xlsx
- Test set:   derivatives/test_set__4_topk.xlsx
- CV: StratifiedGroupKFold by subject to avoid leakage
- Objective: maximize OOF ROC AUC
- Threshold: chosen on *calibrated* OOF predictions to minimize FP+FN
- Saves artifact to: mgmt_logreg_enet__5.joblib

Outputs (plots):
  - calibration_oof_vs_test__logreg__5.png
  - roc_plot_test__logreg__5.png
  - shap_beeswarm_logreg__test__5.png
  - shap_bar_logreg__test__5.png
  - shap_waterfall_repr_pos_logreg__test__5.png
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import joblib
import optuna
import matplotlib.pyplot as plt
import shap

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedGroupKFold as SGKFold

# =================== SETTINGS ===================
TRAIN_XLSX = "derivatives/traincv_set__4_topk.xlsx"
TEST_XLSX  = "derivatives/test_set__4_topk.xlsx"
ARTIFACT   = "mgmt_logreg_enet__5.joblib"

RANDOM_STATE = 42
N_SPLITS = 5
MAX_ITER = 10000
CALIBRATE = True

# Optuna space
C_MIN, C_MAX = 1e-3, 100.0
L1R_MIN, L1R_MAX = 0.0, 1.0
CORR_MIN, CORR_MAX = 0.75, 0.95
C_SEL_MIN, C_SEL_MAX = 1e-3, 10.0
KEEP_FRAC_MIN, KEEP_FRAC_MAX = 0.2, 0.6
KEEP_FRAC_DEFAULT = 0.25
N_TRIALS = 400

# Data config
LABEL_COL = "MGMT"
SUBJECT_COL_CANDIDATES = ["subject_id", "Patient_ID", "Subject_ID", "patient_id"]
EXTRA_EXCLUDE = {"dataset", "sample_id", "roi", "timepoint"}

DO_TRAIN = 1

# =================== HELPERS ===================
class ModelFractionSelector(BaseEstimator, TransformerMixin):
    """
    Fit a base estimator (e.g., L1-logreg) and keep the top keep_frac of features
    by absolute coefficient magnitude.
    """
    def __init__(self, base_estimator=None, keep_frac=0.45, random_state=42):
        self.base_estimator = base_estimator
        self.keep_frac = float(keep_frac)
        self.random_state = random_state
        self.est_ = None
        self.keep_idx_ = None
        self.mask_ = None

    def fit(self, X, y=None):
        if not (0 < self.keep_frac <= 1):
            raise ValueError("keep_frac must be in (0, 1].")
        est = self.base_estimator
        if est is None:
            est = LogisticRegression(
                penalty="l1", solver="saga", C=0.5, max_iter=10000,
                class_weight="balanced", random_state=self.random_state, n_jobs=-1
            )
        self.est_ = clone(est)
        self.est_.fit(X, y)

        # coef_ shape: (n_classes, n_features) for multi-class; reduce to mean |coef|
        coef = getattr(self.est_, "coef_", None)
        if coef is None:
            raise RuntimeError("Base estimator does not expose coef_.")
        coef_abs = np.mean(np.abs(coef), axis=0)
        n = coef_abs.shape[0]
        k = max(1, int(round(self.keep_frac * n)))

        rank = np.argsort(-coef_abs)  # descending
        self.keep_idx_ = np.sort(rank[:k])
        self.mask_ = np.zeros(n, dtype=bool)
        self.mask_[self.keep_idx_] = True
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_idx_] if self.keep_idx_ is not None else np.asarray(X)

    def get_support(self, indices=False):
        if self.mask_ is None:
            return None
        return self.keep_idx_ if indices else self.mask_


def get_subject_col(df: pd.DataFrame) -> str:
    for c in SUBJECT_COL_CANDIDATES:
        if c in df.columns: return c
    raise ValueError(f"Could not find a subject column. Tried: {SUBJECT_COL_CANDIDATES}")

def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    from pandas.api.types import is_numeric_dtype
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found")
    y = pd.to_numeric(df[LABEL_COL], errors="coerce")
    mask = y.isin([0, 1])
    df = df.loc[mask].copy()
    y = y.loc[mask].astype(int).values
    exclude = set(SUBJECT_COL_CANDIDATES) | EXTRA_EXCLUDE | {LABEL_COL}
    feat_cols = [c for c in df.columns if c not in exclude and is_numeric_dtype(df[c])]
    X = df[feat_cols].copy()
    X.columns = X.columns.astype(str)
    return X, y, feat_cols

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one feature from any pair whose |corr| >= threshold (fit on input only).
    Stores kept *names* when possible, but transform() also supports integer indices.
    """
    def __init__(self, threshold: float = 0.80, method: str = "spearman"):
        self.threshold = threshold
        self.method = method
        self.keep_features_: list[str] | list[int] | None = None

    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X).copy()
        # fill by column median (robust to NaNs)
        med = pd.Series(np.nanmedian(Xdf.values, axis=0), index=Xdf.columns)
        Xdf = Xdf.fillna(med)
        # keep columns with >=2 non-nans to compute corr
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

        # Prefer storing NAMES of the kept columns
        self.keep_features_ = [c for c in Xdf.columns if c not in drop]
        return self

    def transform(self, X):
        """Support both name- and index-based keepers to be robust to old artifacts."""
        Xdf = pd.DataFrame(X)
        if not self.keep_features_:
            return Xdf.values

        first = self.keep_features_[0]
        # If integers → select by position
        if isinstance(first, (int, np.integer)):
            return Xdf.iloc[:, self.keep_features_].values

        # Otherwise treat as names (strings); coerce columns to string to match
        Xdf.columns = Xdf.columns.astype(str)
        keep_as_str = [str(k) for k in self.keep_features_]
        missing = [k for k in keep_as_str if k not in Xdf.columns]
        if missing:
            try:
                idx = [int(k) for k in keep_as_str]
                return Xdf.iloc[:, idx].values
            except Exception:
                raise KeyError(f"CorrelationFilter: kept names not found in incoming columns: {missing}")
        return Xdf[keep_as_str].values

def build_pipeline(C: float, l1_ratio: float, corr_threshold: float, C_sel: float, keep_frac: float) -> Pipeline:
    return Pipeline([
        ("imp",  SimpleImputer(strategy="median")),
        ("vt",   VarianceThreshold(threshold=1e-8)),
        ("corr", CorrelationFilter(threshold=corr_threshold, method="spearman")),
        ("sc",   StandardScaler()),
        # model-based percentile selector
        ("sel",  ModelFractionSelector(
            base_estimator=LogisticRegression(
                penalty="l1", solver="saga", C=C_sel, max_iter=MAX_ITER,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            ),
            keep_frac=keep_frac, random_state=RANDOM_STATE
        )),
        ("clf",  LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=l1_ratio, C=C,
            max_iter=MAX_ITER, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])

def oof_cv_predict(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, pipe: Pipeline, n_splits: int, seed: int):
    n = len(y); oof = np.full(n, np.nan, dtype=float); aucs = []
    skf = SGKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y, groups):
        Xtr, Xva = X.iloc[tr], X.iloc[va]; ytr, yva = y[tr], y[va]
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xva)[:, 1]
        oof[va] = p
        if np.unique(yva).size == 2:
            aucs.append(roc_auc_score(yva, p))
    oof_auc = roc_auc_score(y, oof) if np.unique(y).size == 2 else np.nan
    return oof, oof_auc, aucs

def find_best_threshold(y_true: np.ndarray, proba: np.ndarray):
    vals = np.unique(proba)
    thr_grid = np.quantile(proba, np.linspace(0.01, 0.99, 199)) if vals.size > 1000 else vals
    thr_grid = np.unique(thr_grid)
    best_thr, best_err, best_stats = 0.5, float("inf"), {}
    n = len(y_true)
    for t in thr_grid:
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        err = fp + fn
        if err < best_err:
            best_err = err; best_thr = float(t)
            acc = (tp + tn) / max(1, n)
            best_stats = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "errors": err, "accuracy": acc}
    return best_thr, best_stats

def fit_platt_on_oof(y_true: np.ndarray, oof_proba: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(oof_proba.reshape(-1, 1), y_true.astype(int))
    return lr

# --- for SHAP (linear): transform to the model's final feature space (post-imp, VT, Corr, Scaler)
def _prep_final_features(pipe: Pipeline, feat_cols: list[str], df_sub: pd.DataFrame):
    imp  = pipe.named_steps["imp"]
    vt   = pipe.named_steps["vt"]
    corr = pipe.named_steps["corr"]
    sc   = pipe.named_steps["sc"]

    X0 = df_sub[feat_cols].copy()
    Xp = imp.transform(X0)
    Xv = vt.transform(Xp)

    vt_mask = vt.get_support(indices=False)
    names_after_vt = np.array(feat_cols)[vt_mask].astype(str)

    Xv_df = pd.DataFrame(Xv, columns=names_after_vt)
    X_corr = corr.transform(Xv_df)

    kept = corr.keep_features_ or []
    if len(kept) == 0:
        final_names = names_after_vt
    else:
        first = kept[0]
        if isinstance(first, (int, np.integer)):
            final_names = names_after_vt[np.array(kept, dtype=int)]
        else:
            kept_str = np.array([str(k) for k in kept])
            final_names = kept_str[np.isin(kept_str, names_after_vt)]

    Xs = sc.transform(X_corr)

    if "sel" in pipe.named_steps:
        sel = pipe.named_steps["sel"]
        Xs = sel.transform(Xs)
        mask = sel.get_support(indices=False)
        if mask is not None and mask.shape[0] == len(final_names):
            final_names = np.array(final_names)[mask]
        else:
            final_names = np.array([str(i) for i in range(Xs.shape[1])])

    if Xs.shape[1] != len(final_names):
        final_names = np.array([str(i) for i in range(Xs.shape[1])])

    Xf_df = pd.DataFrame(Xs, columns=np.array(final_names).astype(str))
    return Xf_df, np.array(final_names).astype(str)

# =================== TRAIN & TEST ===================
def train_logistic():
    np.random.seed(RANDOM_STATE)
    df = pd.read_excel(TRAIN_XLSX)
    subj_col = get_subject_col(df)
    df[subj_col] = df[subj_col].astype(str)

    X, y, feat_cols = get_xy(df)
    groups = df.loc[X.index, subj_col].values

    def objective(trial: optuna.trial.Trial):
        C    = trial.suggest_float("C", C_MIN, C_MAX, log=True)
        l1r  = trial.suggest_float("l1_ratio", L1R_MIN, L1R_MAX)
        corr = trial.suggest_float("corr_threshold", CORR_MIN, CORR_MAX)
        C_sel = trial.suggest_float("C_sel", C_SEL_MIN, C_SEL_MAX, log=True)
        kf   = trial.suggest_float("keep_frac", KEEP_FRAC_MIN, KEEP_FRAC_MAX)
    
        pipe = build_pipeline(C=C, l1_ratio=l1r, corr_threshold=corr, C_sel=C_sel, keep_frac=kf)
        oof, oof_auc, _ = oof_cv_predict(X, y, groups, pipe, N_SPLITS, RANDOM_STATE)
        thr_tmp, stats_tmp = find_best_threshold(y, oof)
        trial.set_user_attr("oof_auc", oof_auc)
        trial.set_user_attr("threshold_raw_oof_min_fpfn", thr_tmp)
        trial.set_user_attr("errors_fp_fn_raw_oof", stats_tmp.get("errors", None))
        return oof_auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1, show_progress_bar=True)

    best = study.best_trial
    best_params = {
        "C": best.params["C"],
        "l1_ratio": best.params["l1_ratio"],
        "corr_threshold": best.params["corr_threshold"],
        "C_sel": best.params["C_sel"],
        "keep_frac": best.params.get("keep_frac", KEEP_FRAC_DEFAULT),
    }
    print("\n[optuna] Best AUC:", best.value)
    print("[optuna] Best params:", best_params)

    best_pipe = build_pipeline(**best_params)

    oof_raw, oof_auc, fold_aucs = oof_cv_predict(X, y, groups, best_pipe, N_SPLITS, RANDOM_STATE)

    calibrator = None
    if CALIBRATE:
        calibrator = fit_platt_on_oof(y, oof_raw)
        oof_cal = calibrator.predict_proba(oof_raw.reshape(-1, 1))[:, 1]
        thr, stats = find_best_threshold(y, oof_cal)
        try:
            raw_brier = brier_score_loss(y, oof_raw)
            cal_brier = brier_score_loss(y, oof_cal)
            print(f"[calibration] Platt: Brier raw={raw_brier:.4f} -> cal={cal_brier:.4f}")
        except Exception:
            pass
    else:
        thr, stats = find_best_threshold(y, oof_raw)

    print("\n[final OOF] AUC:", round(oof_auc, 4), "| mean fold AUC:", round(float(np.mean(fold_aucs)), 4))
    print("[final OOF] threshold (chosen on calibrated OOF if CALIBRATE):", thr, "| stats:", stats)

    # train on all
    best_pipe.fit(X, y)

    try:
        # Build a mini-pipeline of preprocessors only to recover names before selection
        pre_only = Pipeline([
            ("imp",  best_pipe.named_steps["imp"]),
            ("vt",   best_pipe.named_steps["vt"]),
            ("corr", best_pipe.named_steps["corr"]),
            ("sc",   best_pipe.named_steps["sc"]),
        ])
        Xf_all_df, names_before_sel = _prep_final_features(
            pre_only, feat_cols, df
        )
        sel_mask = best_pipe.named_steps["sel"].get_support(indices=False)
        kept_names = np.array(names_before_sel)[sel_mask]
        pd.Series(kept_names, name="kept_feature").to_csv("kept_features__logregL1_frac.csv", index=False)
        print(f"[feature selection] kept {sel_mask.sum()} features → saved to kept_features__logregL1_frac.csv")
    except Exception as e:
        print(f"[feature selection] could not save kept names: {e}")

    artifact = {
        "pipeline": best_pipe,
        "threshold": thr,
        "params": best_params,
        "feature_cols": feat_cols,
        "subject_col": subj_col,
        "train_oof_auc": float(oof_auc),
        "train_oof_threshold_stats": stats,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "label_col": LABEL_COL,
        "calibrate": bool(CALIBRATE),
        "platt_calibrator": calibrator,
        "kept_feature_names": kept_names.tolist() if 'kept_names' in locals() else None,
        "keep_frac": best_params["keep_frac"],
    }
    joblib.dump(artifact, ARTIFACT)
    print(f"[saved] {ARTIFACT}")

def test_logistic():
    # load artifact
    art = joblib.load(ARTIFACT)
    pipe = art["pipeline"]
    thr  = float(art["threshold"])
    feat_cols = list(art["feature_cols"])
    calibrate = bool(art.get("calibrate", False))
    platt = art.get("platt_calibrator", None)

    # TEST
    df = pd.read_excel(TEST_XLSX)
    if LABEL_COL not in df.columns: raise ValueError(f"Label column '{LABEL_COL}' not found in test file")
    y = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(-1).astype(int).values
    mask_eval = np.isin(y, [0, 1])
    y = y[mask_eval]
    for c in feat_cols:
        if c not in df.columns: df[c] = np.nan
    X = df.loc[mask_eval, feat_cols].copy()

    # probs
    proba_raw = pipe.predict_proba(X)[:, 1]
    proba = (platt.predict_proba(proba_raw.reshape(-1, 1))[:, 1] if calibrate and platt is not None else proba_raw)
    pred = (proba >= thr).astype(int)

    # TRAIN (for calibration curves & SHAP background)
    df_tr = pd.read_excel(TRAIN_XLSX)
    y_tr = pd.to_numeric(df_tr[LABEL_COL], errors="coerce")
    mask_tr = y_tr.isin([0, 1]).values
    y_tr = y_tr.loc[mask_tr].astype(int).values
    for c in feat_cols:
        if c not in df_tr.columns: df_tr[c] = np.nan
    X_tr = df_tr.loc[mask_tr, feat_cols].copy()
    proba_raw_tr = pipe.predict_proba(X_tr)[:, 1]
    proba_tr = (platt.predict_proba(proba_raw_tr.reshape(-1, 1))[:, 1] if calibrate and platt is not None else proba_raw_tr)

    # ===== Calibration curves (raw + Platt) =====
    n_bins = 7
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    ax = axes[0]
    fr_raw_tr, mp_raw_tr = calibration_curve(y_tr, proba_raw_tr, n_bins=n_bins, strategy="quantile")
    ax.plot(mp_raw_tr, fr_raw_tr, "o-", label="Raw (TrainCV)")
    if calibrate and platt is not None:
        fr_cal_tr, mp_cal_tr = calibration_curve(y_tr, proba_tr, n_bins=n_bins, strategy="quantile")
        ax.plot(mp_cal_tr, fr_cal_tr, "o-", label="Platt (TrainCV)")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_title("Calibration — Train-CV"); ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.grid(True); ax.legend(loc="best")

    ax = axes[1]
    fr_raw_te, mp_raw_te = calibration_curve(y, proba_raw, n_bins=n_bins, strategy="quantile")
    ax.plot(mp_raw_te, fr_raw_te, "o-", label="Raw (Test)")
    if calibrate and platt is not None:
        fr_cal_te, mp_cal_te = calibration_curve(y, proba, n_bins=n_bins, strategy="quantile")
        ax.plot(mp_cal_te, fr_cal_te, "o-", label="Platt (Test)")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_title("Calibration — Test"); ax.set_xlabel("Mean predicted probability")
    ax.grid(True); ax.legend(loc="best")

    plt.suptitle("Calibration curves: Train-CV vs Test (MGMT, Logistic)")
    plt.tight_layout(); plt.savefig("calibration_oof_vs_test__logreg__5.png", dpi=150); plt.close()

    # ===== Brier (raw + cal) =====
    brier_raw_tr = brier_score_loss(y_tr, proba_raw_tr)
    brier_cal_tr = brier_score_loss(y_tr, proba_tr)
    brier_raw_te = brier_score_loss(y, proba_raw)
    brier_cal_te = brier_score_loss(y, proba)
    print(f"[brier] TrainCV raw={brier_raw_tr:.4f} | TrainCV cal={brier_cal_tr:.4f} | Test raw={brier_raw_te:.4f} | Test cal={brier_cal_te:.4f}")

    # ===== ROC (raw + cal) =====
    fpr_raw, tpr_raw, _ = roc_curve(y, proba_raw); auc_raw = roc_auc_score(y, proba_raw)
    has_cal = calibrate and (platt is not None)
    if has_cal:
        from sklearn.metrics import roc_curve as _roc_curve, roc_auc_score as _roc_auc_score
        fpr_cal, tpr_cal, _ = _roc_curve(y, proba); auc_cal = _roc_auc_score(y, proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_raw, tpr_raw, label=f"Raw (AUC={auc_raw:.3f})")
    if has_cal: plt.plot(fpr_cal, tpr_cal, label=f"Platt (AUC={auc_cal:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC — Test set (MGMT, Logistic)")
    plt.grid(True, linestyle="--", alpha=0.4); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig("roc_plot_test__logreg__5.png", dpi=350); plt.close()

    if np.unique(y).size == 2:
        auc_val = roc_auc_score(y, proba); print(f"[test] ROC AUC (calibrated): {auc_val:.4f}")
    else:
        print("[test] ROC AUC: N/A (single-class test)")

    # ===== Confusion + report (with calibrated threshold) =====
    cm = confusion_matrix(y, pred, labels=[0, 1])
    print("\n[test] Confusion Matrix (rows=true, cols=pred):"); print(cm)
    print("\n[test] classification report:\n" + classification_report(
        y, pred, labels=[0, 1],
        target_names=["Unmethylated (0)", "Methylated (1)"], zero_division=0, digits=4
    ))
    print("\n[test] first 10 calibrated probabilities:", np.round(proba[:10], 4))

    # ===== SHAP (LinearExplainer) on final feature space =====
    # prepare background (train) and eval (test) in final model space (after imp, VT, Corr, Scaler)
    Xf_tr_df, final_names = _prep_final_features(pipe, feat_cols, df_tr.loc[mask_tr])
    Xf_te_df, _           = _prep_final_features(pipe, feat_cols, df.loc[mask_eval])

    clf = pipe.named_steps["clf"]
    try:
        explainer = shap.LinearExplainer(clf, Xf_tr_df, feature_perturbation="interventional")
    except TypeError:
        explainer = shap.LinearExplainer(clf, Xf_tr_df)

    exp_te = explainer(Xf_te_df)

    shap.plots.beeswarm(exp_te, show=False, max_display=30)
    plt.tight_layout(); plt.savefig("shap_beeswarm_logreg__test__5.png", dpi=500); plt.close()

    shap.plots.bar(exp_te, show=False, max_display=30)
    plt.tight_layout(); plt.savefig("shap_bar_logreg__test__5.png", dpi=500); plt.close()

    pos_idxs = np.where(y == 1)[0]
    rep = pos_idxs[np.argmax(proba[pos_idxs])] if pos_idxs.size > 0 else int(np.argmax(proba))
    shap.plots.waterfall(exp_te[rep], show=False, max_display=20)
    plt.tight_layout(); plt.savefig("shap_waterfall_repr_pos_logreg__test__5.png", dpi=500); plt.close()
    print("[shap] saved SHAP plots (beeswarm, bar, waterfall).")

# ---- run ----
if __name__ == "__main__":
    if DO_TRAIN:
        train_logistic()
    test_logistic()

