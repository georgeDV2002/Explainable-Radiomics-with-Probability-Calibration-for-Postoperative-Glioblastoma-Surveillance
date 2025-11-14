#!/usr/bin/env python3
"""
Compute per-sample statistics from the merged dataset:

Groups:
  - progressed: Time to First Progression (Days) >= 0
  - no_progression: Time to First Progression (Days) is NaN/empty

Metrics:
  - Age: mean and std (per group)
  - Sex counts: "Male", "Female"
  - Race counts: "White", "Black or African American", "Unknown ", "Asian"
  - WHO Grade counts (integer values)
  - IDH (bin): 1=mutant, 0=wildtype, else NaN (excluded from counts)
  - MGMT (0/1, else NaN)
  - Radiation Therapy: "Yes" (count only "Yes"; all else treated as NaN)
  - Name of Initial Chemo Therapy: "Temozolomide", "Lomustine" (others -> NaN)
  - Overall Survival (Death) (bin): 1=yes/deceased, 0=no/alive, else NaN

Outputs:
  - Prints a summary
  - Saves CSVs for each metric under ./stats_outputs/
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------- CONFIG ----------
INPUT_XLSX = "file1.xlsx"
OUT_DIR    = Path("stats_outputs")
TTFP_COL   = "Time to First Progression (Days)"

# Canonical target column names (as in your merged dataset)
COLS = {
    "age": "Age",
    "sex": "Sex",
    "race": "Race",
    "who": "WHO Grade",
    "idh": "IDH",
    "mgmt": "MGMT",
    "rt": "Radiation Therapy",
    "chemo": "Name of Initial Chemo Therapy",
    "osd": "Overall Survival (Death)",
}

# Allowed/counted labels (exact as requested)
SEX_LABELS  = ["Male", "Female"]
RACE_LABELS = ["White", "Black or African American", "Unknown ", "Asian"]
CHEMO_LABELS = ["Temozolomide", "Lomustine"]

# ---------- Normalization helpers ----------
def norm_str(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return s if s != "" else np.nan

def norm_yes_no_to_bin(x):
    """Return 1 for yes/true/1, 0 for no/false/0, else NaN."""
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "1"}: return 1
    if s in {"no", "n", "false", "0"}: return 0
    return np.nan

def norm_idh_to_bin(x):
    """IDH bin: 1 for mutant-like, 0 for wildtype-like, else NaN."""
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in {"mutant", "idh1 mutant", "idh mutant", "m", "pos", "positive", "1"}:
        return 1
    if s in {"wildtype", "wt", "idh wildtype", "idh1 wildtype", "neg", "negative", "0"}:
        return 0
    return np.nan

def norm_mgmt_to_bin(x):
    """
    MGMT: 1 = methylated, 0 = unmethylated, else NaN.
    Accepts {1,'1','yes','methylated','pos'} for 1 and {0,'0','no','unmethylated','neg'} for 0.
    """
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in {"1", "yes", "y", "true", "methylated", "pos", "positive"}:
        return 1
    if s in {"0", "no", "n", "false", "unmethylated", "neg", "negative"}:
        return 0
    # Some datasets use numeric 0/1 already:
    try:
        v = float(s)
        if v == 1.0: return 1
        if v == 0.0: return 0
    except Exception:
        pass
    return np.nan

def norm_radiation_yes(x):
    """Keep only 'Yes' (case-insensitive); else return NaN."""
    if pd.isna(x): return np.nan
    s = str(x).strip()
    return "Yes" if s.lower() in {"yes", "y", "true", "1"} else np.nan

def norm_chemo_label(x):
    """Keep only 'Temozolomide' or 'Lomustine' (case-insensitive); else NaN."""
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s == "temozolomide":
        return "Temozolomide"
    if s == "lomustine":
        return "Lomustine"
    return np.nan

def coerce_int_or_nan(x):
    try:
        v = int(float(x))
        return v
    except Exception:
        return np.nan

# ---------- Counting helper ----------
def two_group_counts(series, allowed_labels, mask_prog):
    """
    Count occurrences of given labels in two groups.
    Returns a DataFrame with index=labels, columns=['progressed','no_progression'].
    """
    s = series.copy()
    # Ensure only allowed labels are counted; others become NaN (excluded)
    s = s.where(s.isin(allowed_labels))
    progressed = s[mask_prog].value_counts().reindex(allowed_labels, fill_value=0)
    no_prog    = s[~mask_prog].value_counts().reindex(allowed_labels, fill_value=0)
    return pd.DataFrame({"progressed": progressed, "no_progression": no_prog})

def two_group_counts_bin(series_bin, mask_prog, labels=(1,0)):
    """
    For binary series (0/1 with NaN allowed), count per group.
    Returns DataFrame with rows for labels present (1 then 0 by default).
    """
    progressed = series_bin[mask_prog].value_counts().reindex(labels, fill_value=0)
    no_prog    = series_bin[~mask_prog].value_counts().reindex(labels, fill_value=0)
    idx_names = [str(x) for x in progressed.index]
    return pd.DataFrame({"progressed": progressed.values, "no_progression": no_prog.values}, index=idx_names)

# ---------- Main ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(INPUT_XLSX)
    df = df[df["WHO Grade"] == 4]  # Only keep GBM
    # Normalize key columns
    for k, col in COLS.items():
        if col in df.columns:
            if k in {"sex", "race", "rt", "chemo"}:
                df[col] = df[col].map(norm_str)
            elif k == "idh":
                df[col] = df[col].map(norm_idh_to_bin)   # becomes 0/1/NaN
            elif k == "mgmt":
                df[col] = df[col].map(norm_mgmt_to_bin)  # becomes 0/1/NaN
            elif k == "osd":
                df[col] = df[col].map(norm_yes_no_to_bin)  # becomes 0/1/NaN
            elif k == "who":
                df[col] = df[col].map(coerce_int_or_nan)
            elif k == "age":
                # keep numeric
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Post-normalization refinements
    # Radiation Therapy only counts "Yes"
    if COLS["rt"] in df.columns:
        df[COLS["rt"]] = df[COLS["rt"]].map(norm_radiation_yes)
    # Chemo only counts selected labels
    if COLS["chemo"] in df.columns:
        df[COLS["chemo"]] = df[COLS["chemo"]].map(norm_chemo_label)

    # Race: unify "Unknown"/"unknown" etc. to exactly "Unknown "
    if COLS["race"] in df.columns:
        def fix_unknown_space(x):
            if pd.isna(x): return np.nan
            s = str(x).strip()
            if s.lower() == "unknown":
                return "Unknown "
            return s
        df[COLS["race"]] = df[COLS["race"]].map(fix_unknown_space)

    # Build group masks
    ttfp = pd.to_numeric(df.get(TTFP_COL, pd.Series(index=df.index)), errors="coerce")
    mask_prog = ttfp >= 0  # progressed if >= 0 days

    # ---- Time to First Progression (Days): median (IQR), n (%) ----
    # Use only non-NA and >= 0 (i.e., progressed samples)
    ttfp_non_na = ttfp[ttfp.notna() & (ttfp >= 0)]
    n_ttfp = int(ttfp_non_na.count())
    pct_ttfp = (n_ttfp / len(df) * 100.0) if len(df) else 0.0
    ttfp_median = ttfp_non_na.median()
    ttfp_q1 = ttfp_non_na.quantile(0.25)
    ttfp_q3 = ttfp_non_na.quantile(0.75)

    print("--- Time to First Progression (Days) ---")
    print(f"median (IQR): {ttfp_median:.1f} ({ttfp_q1:.1f}–{ttfp_q3:.1f}), n (%): {n_ttfp} ({pct_ttfp:.1f}%)\n")

    # ---- Age (mean, std) per group ----
    age_col = COLS["age"]
    if age_col in df.columns:
        age_prog = df.loc[mask_prog, age_col]
        age_no   = df.loc[~mask_prog, age_col]
        age_summary = pd.DataFrame({
            "mean": [age_prog.mean(), age_no.mean()],
            "std":  [age_prog.std(ddof=1), age_no.std(ddof=1)],
            "count_non_na": [age_prog.notna().sum(), age_no.notna().sum()],
        }, index=["progressed", "no_progression"])
        age_summary.to_csv(OUT_DIR / "age_summary.csv")
    else:
        age_summary = None

    # ---- Age (overall mean, std) ----
    if age_col in df.columns:
        age_all = df[age_col].dropna()
        age_all_mean = age_all.mean()
        age_all_std  = age_all.std(ddof=1)
        print("--- Age (overall, all samples) ---")
        print(f"mean: {age_all_mean:.2f}, std: {age_all_std:.2f}, n={len(age_all)}\n")

    # ---- Sex counts ----
    sex_counts = two_group_counts(df[COLS["sex"]], SEX_LABELS, mask_prog) if COLS["sex"] in df.columns else None
    if sex_counts is not None:
        sex_counts.to_csv(OUT_DIR / "sex_counts.csv")

    # ---- Race counts ----
    race_counts = two_group_counts(df[COLS["race"]], RACE_LABELS, mask_prog) if COLS["race"] in df.columns else None
    if race_counts is not None:
        race_counts.to_csv(OUT_DIR / "race_counts.csv")

    # ---- WHO Grade counts (ints) ----
    if COLS["who"] in df.columns:
        who = df[COLS["who"]]
        labels_who = sorted([int(x) for x in pd.unique(who.dropna())])
        who_prog = who[mask_prog].value_counts().reindex(labels_who, fill_value=0)
        who_no   = who[~mask_prog].value_counts().reindex(labels_who, fill_value=0)
        who_counts = pd.DataFrame({"progressed": who_prog.values, "no_progression": who_no.values}, index=[str(x) for x in labels_who])
        who_counts.to_csv(OUT_DIR / "who_grade_counts.csv")
    else:
        who_counts = None

    # ---- IDH bin counts (1/0) ----
    idh_counts = two_group_counts_bin(df[COLS["idh"]], mask_prog) if COLS["idh"] in df.columns else None
    if idh_counts is not None:
        idh_counts.to_csv(OUT_DIR / "idh_bin_counts.csv")

    # ---- MGMT bin counts (1/0) ----
    mgmt_counts = two_group_counts_bin(df[COLS["mgmt"]], mask_prog) if COLS["mgmt"] in df.columns else None
    if mgmt_counts is not None:
        mgmt_counts.to_csv(OUT_DIR / "mgmt_bin_counts.csv")

    # ---- Radiation Therapy counts ("Yes" only) ----
    if COLS["rt"] in df.columns:
        rt = df[COLS["rt"]]
        rt_labels = ["Yes"]  # only count 'Yes'; NaN is ignored
        rt_counts = two_group_counts(rt, rt_labels, mask_prog)
        rt_counts.to_csv(OUT_DIR / "radiation_yes_counts.csv")
    else:
        rt_counts = None

    # ---- Chemo counts ("Temozolomide", "Lomustine") ----
    if COLS["chemo"] in df.columns:
        chemo_counts = two_group_counts(df[COLS["chemo"]], CHEMO_LABELS, mask_prog)
        chemo_counts.to_csv(OUT_DIR / "chemo_counts.csv")
    else:
        chemo_counts = None

    # ---- Overall Survival (Death) bin counts (1/0) ----
    osd_counts = two_group_counts_bin(df[COLS["osd"]], mask_prog) if COLS["osd"] in df.columns else None
    if osd_counts is not None:
        osd_counts.to_csv(OUT_DIR / "overall_survival_death_bin_counts.csv")

    # --------- Print summary ---------
    print(f"[info] Total samples: {len(df)}")
    print(f"[info] progressed (TTFP >= 0): {int(mask_prog.sum())}")
    print(f"[info] no_progression (TTFP NaN): {int((~mask_prog).sum())}\n")

    def maybe_print(title, dfobj):
        if dfobj is not None:
            print(f"--- {title} ---")
            print(dfobj, "\n")

    if age_summary is not None:
        print("--- Age (years) ---")
        print(age_summary, "\n")

    maybe_print("Sex counts", sex_counts)
    maybe_print("Race counts", race_counts)
    maybe_print("WHO Grade counts", who_counts)
    maybe_print("IDH (bin) counts (1=mutant, 0=wildtype)", idh_counts)
    maybe_print("MGMT (bin) counts (1=methylated, 0=unmethylated)", mgmt_counts)
    maybe_print("Radiation Therapy counts (Yes only)", rt_counts)
    maybe_print("Initial Chemo counts (Temozolomide/Lomustine)", chemo_counts)
    maybe_print("Overall Survival (Death) (bin) counts", osd_counts)

    print(f"[done] CSVs saved under: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()

