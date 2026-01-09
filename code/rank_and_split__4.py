#!/usr/bin/env python3
"""
One-shot prep:
- Split dataset by patient into train-cv and test (no overlap).
- Compute feature importance on TRAIN-CV ONLY (same logic as your importance_ranking__4.py).
- Save importance CSV.
- Keep Top-K features and save pruned train/test XLSX with identical columns/orders.

Inputs:
  derivatives/radiomics_features_with_mgmt.xlsx

Outputs (written to derivatives/):
  traincv_set__4.xlsx
  test_set__4.xlsx
  importance_traincv__4.csv
  feature_list__4.csv
  traincv_set__4_topk.xlsx
  test_set__4_topk.xlsx
  radiomics_family_summary_top500.csv
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.feature_selection import VarianceThreshold
from pandas.api.types import is_numeric_dtype

# ============================== CONFIG ==============================
ROOT = Path(".")
DERIV = ROOT / "derivatives"

INPUT_XLSX = DERIV / "radiomics_features_with_mgmt.xlsx"   # <- from collect_dataset__2.py

# split outputs (raw)
OUTPUT_TRAIN_RAW = DERIV / "traincv_set__4.xlsx"
OUTPUT_TEST_RAW  = DERIV / "test_set__4.xlsx"

# importance + pruned outputs
OUTPUT_IMPORTANCE  = DERIV / "importance_traincv__4.csv"
OUTPUT_FEATURELIST = DERIV / "feature_list__4.csv"
OUTPUT_TRAIN_TOPK  = DERIV / "traincv_set__4_topk.xlsx"
OUTPUT_TEST_TOPK   = DERIV / "test_set__4_topk.xlsx"
OUTPUT_FAMILY_SUM  = DERIV / "radiomics_family_summary_top500.csv"

# split settings
RANDOM_SEED = 42
TEST_FRACTION = 0.05

# label + meta
LABEL_COL = "MGMT"  # <-- binary label (0/1), NaNs = unlabeled -> dropped before split/fit
SUBJECT_COL_CANDIDATES = ["subject_id", "Patient_ID", "Subject_ID", "patient_id"]
ALWAYS_KEEP_META = ["timepoint", "dataset", "sample_id", "roi"]  # keep if present

# importance (train-cv only)
N_SPLITS = 10
PERM_REPEATS = 100
CORR_THRESHOLD = 0.80
MAX_ITER = 10000
C_REG = 1.0
MIN_TEST_SIZE_FOR_PI = 30
RANDOM_STATE = 42
TOPK_PRINT = 30
TOPK_FAMILY = 500
TINY = 1e-9

# pruning
TOPK_FEATURES = 500
# ====================================================================


# ===================== helpers: split =====================
# ===================== Stratified-by-patient helpers =====================
def patient_majority_label(df: pd.DataFrame, subj_col: str, label_col: str) -> pd.Series:
    """Return a Series index=subject_id, value in {0,1}, computed by majority vote across the subject's rows.
       Ties are broken by the global class frequency (favor the more common class); if still tied, favor 1."""
    g = df.groupby(subj_col)[label_col]
    # counts per subject
    cnt1 = g.sum()                 # since label is 0/1, sum = #positives
    cnt  = g.size()
    cnt0 = cnt - cnt1
    # global freq for tie-break
    global_pos = float(df[label_col].mean())  # fraction in (0,1)
    # majority with tie-break
    maj = (cnt1 > cnt0).astype(int)
    ties = (cnt1 == cnt0)
    if ties.any():
        # favor whichever class is globally more common; if exactly 50/50, favor 1
        maj.loc[ties] = 1 if global_pos >= 0.5 else 0
    return maj

def choose_patients_stratified_by_rows(
    df: pd.DataFrame,
    subj_col: str,
    label_col: str,
    target_rows: int,
    rng: np.random.Generator,
    attempts: int = 5000,
) -> np.ndarray:
    """Pick subjects to reach ~target_rows (by rows), stratified by patient majority label.
       Greedy+random heuristic to match both row target and class ratio."""
    subj_label = patient_majority_label(df, subj_col, label_col)
    # per-subject row counts
    counts = df.groupby(subj_col).size()

    pos_subj = counts[subj_label == 1].index.to_numpy().astype(str)
    neg_subj = counts[subj_label == 0].index.to_numpy().astype(str)
    pos_cnts = counts.loc[pos_subj].to_numpy()
    neg_cnts = counts.loc[neg_subj].to_numpy()

    # global desired positive fraction by rows
    frac_pos_rows = (df[label_col] == 1).mean()

    best_sel, best_gap = None, float("inf")
    for _ in range(attempts):
        # shuffle each pool
        ip = rng.permutation(len(pos_subj))
        ineg = rng.permutation(len(neg_subj))
        pos_order = pos_subj[ip]; pos_c = pos_cnts[ip]
        neg_order = neg_subj[ineg]; neg_c = neg_cnts[ineg]

        i, j = 0, 0
        sel = []
        tot = 0
        pos_rows = 0

        while tot < target_rows and (i < len(pos_order) or j < len(neg_order)):
            err_pos = err_neg = float("inf")
            if i < len(pos_order):
                tot_pos = tot + int(pos_c[i])
                pos_rows_pos = pos_rows + int(pos_c[i])
                frac_pos_est = pos_rows_pos / max(1, tot_pos)
                err_pos = abs(frac_pos_est - frac_pos_rows) + max(0, (tot_pos - target_rows)) / max(1, target_rows)
            if j < len(neg_order):
                tot_neg = tot + int(neg_c[j])
                pos_rows_neg = pos_rows
                frac_pos_est = pos_rows_neg / max(1, tot_neg)
                err_neg = abs(frac_pos_est - frac_pos_rows) + max(0, (tot_neg - target_rows)) / max(1, target_rows)

            if err_pos < err_neg:
                sel.append(pos_order[i]); tot += int(pos_c[i]); pos_rows += int(pos_c[i]); i += 1
            else:
                sel.append(neg_order[j]); tot += int(neg_c[j]); j += 1

        gap = abs(tot - target_rows) + abs((pos_rows / max(1, tot)) - frac_pos_rows)
        if gap < best_gap:
            best_gap = gap
            best_sel = sel
            if best_gap == 0:
                break

    return np.array(best_sel if best_sel is not None else [], dtype=str)

def pick_subject_col(df: pd.DataFrame) -> str:
    for c in SUBJECT_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a subject column. Tried: {SUBJECT_COL_CANDIDATES}")

def class_counts(series: pd.Series) -> dict:
    y = pd.to_numeric(series, errors="coerce")
    y = y[y.isin([0, 1])].astype(int)
    d = y.value_counts().to_dict()
    return {1: d.get(1, 0), 0: d.get(0, 0)}

# ===================== Corr filter =====================
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one feature from any pair whose |corr| >= threshold (fit on train only)."""
    def __init__(self, threshold=0.95, method="spearman"):
        self.threshold = threshold
        self.method = method
        self.keep_features_: list[str] | None = None

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
        return Xdf[self.keep_features_].values

def perm_importance_kept(pipe, Xte, yte, kept_names, repeats=50, random_state=42):
    """Permutation importance ONLY for features that reached the classifier.
       Metric: delta log-loss (positive = worse when shuffled = important).
    """
    rng = np.random.default_rng(random_state)
    # 1) impute with original names
    Xi = pd.DataFrame(pipe.named_steps["imp"].transform(Xte), columns=Xte.columns)
    # 2) VT names
    vt = pipe.named_steps["vt"]
    vt_cols = Xi.columns[vt.get_support()]
    # 3) align kept names
    Xi_vt = Xi.loc[:, vt_cols]
    kept = [c for c in kept_names if c in Xi_vt.columns] if len(vt_cols) else []
    if not kept:
        return pd.Series(dtype=float)
    Xi_corr = Xi_vt.loc[:, kept]
    # 4) scale
    Zte = pipe.named_steps["sc"].transform(Xi_corr.values)
    # 5) baseline logloss
    clf = pipe.named_steps["clf"]
    p_base = clf.predict_proba(Zte)[:, 1]
    base = log_loss(yte, p_base, labels=[0, 1])
    # 6) permute one-by-one
    Z = Zte.copy()
    importances = np.zeros(Z.shape[1], dtype=float)
    for j in range(Z.shape[1]):
        drops = []
        for _ in range(repeats):
            saved = Z[:, j].copy()
            rng.shuffle(Z[:, j])
            p_perm = clf.predict_proba(Z)[:, 1]
            score = log_loss(yte, p_perm, labels=[0, 1])
            drops.append(score - base)
            Z[:, j] = saved
        importances[j] = float(np.mean(drops))
    return pd.Series(importances, index=[str(n) for n in kept])

# ===================== importance on train-cv =====================
def compute_importance_traincv(df_traincv: pd.DataFrame):
    df = df_traincv.copy()
    df.columns = df.columns.map(lambda c: str(c).strip())

    # label: MGMT (drop unlabeled)
    y = pd.to_numeric(df[LABEL_COL], errors="coerce")
    mask_labeled = y.isin([0, 1])
    df = df.loc[mask_labeled].copy()
    y = y.loc[mask_labeled].astype(int).values

    # features: numeric radiomics only (exclude IDs/clinical helpers)
    exclude = set(SUBJECT_COL_CANDIDATES) | {LABEL_COL, "dataset", "sample_id", "roi", "timepoint"}
    feat_cols = [c for c in df.columns if c not in exclude and is_numeric_dtype(df[c])]
    X = df[feat_cols].copy()
    X.columns = X.columns.astype(str)

    counts = pd.Series(y).value_counts(dropna=False).to_dict()
    min_class = min(counts.get(0, 0), counts.get(1, 0))
    n_splits = N_SPLITS
    if min_class < 1:
        raise RuntimeError(f"No negatives or positives under this label. Class counts: {counts}")
    if n_splits > min_class:
        print(f"[info] Reducing N_SPLITS from {n_splits} to {min_class} to avoid single-class folds.")
        n_splits = max(2, min_class)

    print("Sample-wise class counts (train-cv):", counts, "| samples:", len(y))
    print(f"Features used (train-cv): {len(feat_cols)}")

    pipe = Pipeline([
        ("imp",  SimpleImputer(strategy="median")),
        ("vt",   VarianceThreshold(threshold=1e-8)),
        ("corr", CorrelationFilter(threshold=CORR_THRESHOLD, method="spearman")),
        ("sc",   StandardScaler()),
        ("clf",  LogisticRegression(
            penalty="l1", solver="saga", C=C_REG, max_iter=MAX_ITER,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    aucs = []
    coef_list = []
    perm_list = []
    presence_counts = {}

    for fold_i, (tr, te) in enumerate(skf.split(X, y), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]

        pipe.fit(Xtr, ytr)

        # names after VT
        Xi = pd.DataFrame(pipe.named_steps["imp"].transform(Xtr), columns=Xtr.columns)
        vt = pipe.named_steps["vt"]
        vt_cols = Xi.columns[vt.get_support()]  # names after VT

        # what CorrFilter kept
        kept = pipe.named_steps["corr"].keep_features_
        if kept is None or len(kept) == 0:
            kept_names = []
        elif isinstance(kept[0], (int, np.integer)):
            kept_names = [vt_cols[k] for k in kept if 0 <= k < len(vt_cols)]
        else:
            kept_names = [k for k in kept if k in vt_cols]

        if not kept_names:
            continue

        # Coefs aligned to kept_names
        coefs = pd.Series(pipe.named_steps["clf"].coef_[0], index=[str(k) for k in kept_names])
        coef_list.append(coefs)

        for f in kept_names:
            presence_counts[f] = presence_counts.get(f, 0) + 1

        # AUC + PI if both classes present
        if np.unique(yte).size == 2:
            prob = pipe.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(yte, prob))

            if len(yte) >= MIN_TEST_SIZE_FOR_PI:
                pi_kept = perm_importance_kept(pipe, Xte, yte, kept_names, repeats=PERM_REPEATS, random_state=RANDOM_STATE)
                if len(pi_kept):
                    perm_list.append(pi_kept)

    if aucs:
        print(f"Mean CV AUC (train-cv): {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    else:
        print("Note: AUC skipped in all folds (likely single-class test sets).")

    print(f"[diag] folds with PI: {len(perm_list)} / {n_splits}")

    # Aggregate
    if not coef_list:
        raise RuntimeError("No folds produced coefficients; check data/label balance and feature pipeline.")

    coef_df = pd.concat(coef_list, axis=1)  # keep NaNs
    coef_df.columns = [f"fold{i+1}" for i in range(len(coef_list))]

    median_abs_coef = coef_df.abs().median(axis=1, skipna=True)

    n_folds = len(coef_list)
    presence_freq = pd.Series(
        {f: presence_counts.get(f, 0) / n_folds for f in coef_df.index},
        dtype=float
    )

    sel_numer = (coef_df.abs() > 0).sum(axis=1, skipna=True)
    sel_denom = coef_df.notna().sum(axis=1).replace(0, np.nan)
    coef_nonzero_given_present = (sel_numer / sel_denom).fillna(0.0)

    if len(perm_list):
        perm_df = pd.concat(perm_list, axis=1)
        perm_df.index = perm_df.index.astype(str)
        perm_df.columns = [f"fold{i+1}" for i in range(len(perm_list))]
        median_perm = perm_df.median(axis=1, skipna=True)
        pi_numer = (perm_df > TINY).sum(axis=1)
        pi_denom = perm_df.notna().sum(axis=1).replace(0, np.nan)
        pi_gt0_freq = (pi_numer / pi_denom).fillna(0.0)
    else:
        median_perm = pd.Series(0.0, index=median_abs_coef.index, dtype=float)
        pi_gt0_freq = pd.Series(0.0, index=median_abs_coef.index, dtype=float)

    all_feats = median_abs_coef.index.union(median_perm.index)
    imp = pd.DataFrame({
        "median_abs_coef":              median_abs_coef.reindex(all_feats).fillna(0.0),
        "median_perm_importance":       median_perm.reindex(all_feats).fillna(0.0),
        "presence_freq":                presence_freq.reindex(all_feats).fillna(0.0),
        "coef_nonzero_given_present":   coef_nonzero_given_present.reindex(all_feats).fillna(0.0),
        "pi_nonzero_freq":              pi_gt0_freq.reindex(all_feats).fillna(0.0),
    })

    rank_coef = imp["median_abs_coef"].rank(ascending=False, method="average")
    rank_perm = imp["median_perm_importance"].rank(ascending=False, method="average")
    imp["consensus_rank"] = (rank_coef + rank_perm)

    imp = imp.sort_values(["consensus_rank", "median_perm_importance", "median_abs_coef"],
                          ascending=[True, False, False])

    return imp, feat_cols

# ===================== main =====================
def main():
    DERIV.mkdir(parents=True, exist_ok=True)

    in_path = Path(INPUT_XLSX)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path.resolve()}")

    df_all = pd.read_excel(in_path)
    subj_col = pick_subject_col(df_all)
    df_all[subj_col] = df_all[subj_col].astype(str)

    # keep only labeled rows for splitting/training
    if LABEL_COL not in df_all.columns:
        raise KeyError(f"Label column '{LABEL_COL}' not in dataset.")
    y_all = pd.to_numeric(df_all[LABEL_COL], errors="coerce")
    df_all = df_all.loc[y_all.isin([0, 1])].copy()
    y_all = pd.to_numeric(df_all[LABEL_COL], errors="coerce").astype(int)

    # -------- split by patient (stratified on patient-level MGMT) --------
    rng = np.random.default_rng(RANDOM_SEED)
    
    if TEST_FRACTION is None or not (0 < TEST_FRACTION < 1):
        raise ValueError("Set TEST_FRACTION to a float in (0,1), e.g. 0.2 for 20%.")
    
    target_rows = int(round(TEST_FRACTION * len(df_all)))
    test_subjects = choose_patients_stratified_by_rows(
        df_all, subj_col=subj_col, label_col=LABEL_COL,
        target_rows=target_rows, rng=rng, attempts=5000
    )
    
    mask_test = df_all[subj_col].isin(set(test_subjects))
    df_test_raw = df_all.loc[mask_test].copy()
    df_train_raw = df_all.loc[~mask_test].copy()

    # diagnostics
    print(f"[split] Patients total: {df_all[subj_col].nunique()} "
          f"| train-cv: {df_train_raw[subj_col].nunique()} | test: {df_test_raw[subj_col].nunique()}")
    print(f"[split] Rows     total: {len(df_all)} | train-cv: {len(df_train_raw)} | test: {len(df_test_raw)}")
    print(f"[label] Train-CV class counts (1/0): {class_counts(df_train_raw[LABEL_COL])}")
    print(f"[label] Test     class counts (1/0): {class_counts(df_test_raw[LABEL_COL])}")
    print("[test patients] " + ", ".join(map(str, test_subjects)))

    # save raw splits
    df_train_raw.to_excel(OUTPUT_TRAIN_RAW, index=False)
    df_test_raw.to_excel(OUTPUT_TEST_RAW, index=False)
    print(f"[saved] {OUTPUT_TRAIN_RAW.resolve()}")
    print(f"[saved] {OUTPUT_TEST_RAW.resolve()}")

    # -------- importance on TRAIN-CV only --------
    imp, feat_cols_all = compute_importance_traincv(df_train_raw)

    # save importance
    imp.to_csv(OUTPUT_IMPORTANCE, index=True)
    print(f"[saved] {OUTPUT_IMPORTANCE.resolve()}")

    with pd.option_context("display.float_format", "{:.3e}".format):
        print("\nTop features (consensus) — first 30 rows:")
        print(imp.head(TOPK_PRINT))

    # family summary (TopK_FAMILY)
    topk_for_family = imp.head(TOPK_FAMILY).copy()
    def family_of(name: str) -> str:
        if isinstance(name, str) and "__" in name:
            return name.split("__", 1)[0]
        return "misc"
    families = topk_for_family.index.to_series().apply(family_of)
    family_summary = families.value_counts().rename_axis("family").reset_index(name=f"count_in_top{TOPK_FAMILY}")
    print(f"\nFamily summary in top-{TOPK_FAMILY}:")
    print(family_summary.to_string(index=False))
    family_summary.to_csv(OUTPUT_FAMILY_SUM, index=False)

    # -------- prune to Top-K (train-only ranking) --------
    top_feats = [f for f in imp.index.tolist() if f in df_all.columns][:TOPK_FEATURES]
    if len(top_feats) < TOPK_FEATURES:
        print(f"[warn] Only {len(top_feats)} of the requested Top-{TOPK_FEATURES} exist in data columns.")
    pd.Series(top_feats, name="feature").to_csv(OUTPUT_FEATURELIST, index=False)
    print(f"[saved] {OUTPUT_FEATURELIST.resolve()} ({len(top_feats)} features)")

    keep_meta = [c for c in ([subj_col] + ALWAYS_KEEP_META + [LABEL_COL]) if c in df_all.columns]
    cols_final = keep_meta + top_feats

    # ensure columns exist in both splits (missing → NaN)
    for c in cols_final:
        if c not in df_train_raw.columns:
            df_train_raw[c] = np.nan
        if c not in df_test_raw.columns:
            df_test_raw[c] = np.nan

    df_train_topk = df_train_raw.loc[:, cols_final]
    df_test_topk  = df_test_raw.loc[:, cols_final]

    # final saves
    df_train_topk.to_excel(OUTPUT_TRAIN_TOPK, index=False)
    df_test_topk.to_excel(OUTPUT_TEST_TOPK, index=False)
    print(f"[saved] {OUTPUT_TRAIN_TOPK.resolve()}")
    print(f"[saved] {OUTPUT_TEST_TOPK.resolve()}")

    # quick sanity
    assert set(df_train_topk[subj_col]).isdisjoint(set(df_test_topk[subj_col])), "Leak: patients overlap!"
    train_feats = [c for c in df_train_topk.columns if c not in keep_meta]
    test_feats  = [c for c in df_test_topk.columns  if c not in keep_meta]
    assert train_feats == test_feats, "Train/Test feature columns differ!"
    print("Sanity checks passed.")

if __name__ == "__main__":
    main()

