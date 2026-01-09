#!/usr/bin/env python3
"""
Make a Table 1-style summary from UCSF & UPENN clinical files.

Columns standardized to: Age, Sex, MGMT, IDH
Output columns: Total, UCSF, UPENN
Rows: N; Age mean (SD); Sex (Male/Female); MGMT methylated, n (%); IDH mutant, n (%)

"""

import numpy as np
import pandas as pd
import re

# Flexible capture with optional suffix (e.g., _11)
UCSF_CAP = re.compile(r"(UCSF[-_\s]?PDGM)[-_\s]?(\d{1,5})(?:[A-Za-z_]\d+)?", re.IGNORECASE)
UPENN_CAP = re.compile(r"(UPENN[-_\s]?GBM)[-_\s]?(\d{1,5})(?:[A-Za-z_]\d+)?", re.IGNORECASE)

def standardize_subject_id(s: str) -> str | None:
    """
    Find a UCSF/UPENN ID anywhere in the string, strip suffixes (e.g. _11),
    and normalize to:
      UCSF-PDGM-####   (4 digits)
      UPENN-GBM-#####  (5 digits)
    Returns None if no pattern found.
    """
    if s is None:
        return None
    s = str(s).strip()
    m = UCSF_CAP.search(s)
    if m:
        prefix = "UCSF-PDGM"
        num = m.group(2).zfill(4)
        return f"{prefix}-{num}"
    m = UPENN_CAP.search(s)
    if m:
        prefix = "UPENN-GBM"
        num = m.group(2).zfill(5)
        return f"{prefix}-{num}"
    return None

def detect_subject_id_column(df):
    """
    Scan columns; if any yields at least one standardized ID, return that Series
    as 'subject_id' (others are None). Otherwise return None.
    """
    for col in df.columns:
        s = df[col].apply(standardize_subject_id)
        if s.notna().any():
            s.name = "subject_id"
            return s
    return None

# ---------- mapping helpers ----------
def load_allowed_ids(train_path, test_path):
    """Return set of standardized subject_ids present in either split Excel."""
    df_train = pd.read_excel(train_path)
    df_test  = pd.read_excel(test_path)
    ids_train = df_train["subject_id"].astype(str).str.strip().map(standardize_subject_id)
    ids_test  = df_test["subject_id"].astype(str).str.strip().map(standardize_subject_id)
    allowed = set(ids_train.dropna()) | set(ids_test.dropna())
    return allowed

def _to_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def map_ucsf_mgmt(x):
    s = _to_str(x).lower()
    if s in ("", "indeterminate"):
        return np.nan
    if s.startswith("pos"):
        return 1
    if s.startswith("neg"):
        return 0
    return np.nan

def map_ucsf_idh(x):
    s = _to_str(x).lower()
    if s == "":
        return np.nan
    if "wild" in s:
        return 0
    # "mutated (nos)" -> 1, and per spec: any other non-empty value -> 1
    return 1

def map_upenn_mgmt(x):
    s = _to_str(x).lower()
    if s in ("", "not available", "indeterminate"):
        return np.nan
    if s.startswith("methylated"):
        return 1
    if s.startswith("unmethylated"):
        return 0
    return np.nan

def map_upenn_idh(x):
    s = _to_str(x).lower()
    if s in ("", "nos/nec"):
        return np.nan
    if s.startswith("wildtype"):
        return 0
    if s.startswith("mutated"):
        return 1
    return np.nan

def normalize_sex(x):
    s = _to_str(x).upper()
    if s in ("M", "F"):
        return s
    return np.nan

def fmt_mean_sd(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return "—"
    return f"{series.mean():.1f} ({series.std(ddof=1):.1f})"

def fmt_count_pct(count, denom):
    if denom is None or denom == 0:
        return "0 (0.0%)"
    return f"{int(count)} ({100.0*count/denom:.1f}%)"

# ---------- IO & standardization ----------
def load_ucsf(path):
    df = pd.read_excel(path)
    rename_map = {
        "Sex": "Sex",
        "Age at MRI": "Age",
        "MGMT status": "MGMT",
        "IDH": "IDH",
    }
    df = df.rename(columns=rename_map)

    # --- detect the WHO CNS Grade column (handles trailing spaces etc.) ---
    def _norm(c):  # normalize column names for matching
        return str(c).strip().replace("\xa0", " ").strip().lower()

    grade_col = None
    for c in df.columns:
        nc = _norm(c)
        if nc.startswith("who cns grade"):
            grade_col = c
            break

    if grade_col is not None:
        df = df.rename(columns={grade_col: "Grade"})
    else:
        df["Grade"] = np.nan

    keep_cols = list(rename_map.values()) + ["Grade"] + [c for c in df.columns if c not in list(rename_map.values()) + ["Grade"]]
    df = df[keep_cols].copy()

    # subject_id detection
    subj = detect_subject_id_column(df)
    if subj is None:
        raise RuntimeError("UCSF clinical file: could not find UCSF IDs (e.g., 'UCSF-PDGM-004' / 'UCSF-PDGM-0123').")
    df["subject_id"] = subj

    # standardize fields
    df["Sex"]   = df["Sex"].map(normalize_sex)
    df["Age"]   = pd.to_numeric(df["Age"], errors="coerce")
    df["MGMT"]  = df["MGMT"].map(map_ucsf_mgmt)
    df["IDH"]   = df["IDH"].map(map_ucsf_idh)

    # WHO CNS Grade -> numeric (2/3/4) when possible
    df["Grade"] = pd.to_numeric(df["Grade"], errors="coerce")

    df["Cohort"] = "UCSF"
    return df

def load_upenn(path):
    df = pd.read_excel(path)
    rename_map = {
        "Gender": "Sex",
        "Age_at_scan_years": "Age",
        "MGMT": "MGMT",
        "IDH1": "IDH",
    }
    df = df.rename(columns=rename_map)
    df = df[list(rename_map.values()) + [c for c in df.columns if c not in rename_map.values()]].copy()

    subj = detect_subject_id_column(df)
    if subj is None:
        raise RuntimeError("UPENN clinical file: could not find UPENN IDs (e.g., 'UPENN-GBM-00001_11' / 'UPENN-GBM-00125').")
    df["subject_id"] = subj

    df["Sex"]  = df["Sex"].map(normalize_sex)
    df["Age"]  = pd.to_numeric(df["Age"], errors="coerce")
    df["MGMT"] = df["MGMT"].map(map_upenn_mgmt)
    df["IDH"]  = df["IDH"].map(map_upenn_idh)
    df["Cohort"] = "UPENN"
    return df

# ---------- table builder ----------
def summarize_block(df):
    out = {}
    # N
    out["N"] = f"{len(df)}"

    # Age mean (SD)
    out["Age, mean (SD)"] = fmt_mean_sd(df["Age"])

    # Sex counts (% of known)
    sex_known = df["Sex"].dropna()
    denom_sex = len(sex_known)
    m = (sex_known == "M").sum()
    f = (sex_known == "F").sum()
    out["  Male"]   = fmt_count_pct(m, denom_sex)
    out["  Female"] = fmt_count_pct(f, denom_sex)

    # MGMT methylated (1) among known MGMT
    mgmt_known = df["MGMT"].dropna()
    denom_mgmt = len(mgmt_known)
    mgmt_pos = (mgmt_known == 1).sum()
    out["MGMT methylated, n (%)"] = fmt_count_pct(mgmt_pos, denom_mgmt)

    # IDH mutant (1) among known IDH
    idh_known = df["IDH"].dropna()
    denom_idh = len(idh_known)
    idh_mut = (idh_known == 1).sum()
    out["IDH mutant, n (%)"] = fmt_count_pct(idh_mut, denom_idh)

    return out

def build_table(df_all):
    # cohorts
    ucsf  = df_all[df_all["Cohort"] == "UCSF"]
    upenn = df_all[df_all["Cohort"] == "UPENN"]

    tot_stats  = summarize_block(df_all)
    ucsf_stats = summarize_block(ucsf)
    up_stats   = summarize_block(upenn)

    rows = [
        ("N",                         tot_stats["N"],                  ucsf_stats["N"],                  up_stats["N"]),
        ("Age, mean (SD)",            tot_stats["Age, mean (SD)"],     ucsf_stats["Age, mean (SD)"],     up_stats["Age, mean (SD)"]),
        ("Sex, n (%)",                "",                              "",                               ""),
        ("  Male",                    tot_stats["  Male"],             ucsf_stats["  Male"],             up_stats["  Male"]),
        ("  Female",                  tot_stats["  Female"],           ucsf_stats["  Female"],           up_stats["  Female"]),
        ("MGMT methylated, n (%)",    tot_stats["MGMT methylated, n (%)"], ucsf_stats["MGMT methylated, n (%)"], up_stats["MGMT methylated, n (%)"]),
        ("IDH mutant, n (%)",         tot_stats["IDH mutant, n (%)"],  ucsf_stats["IDH mutant, n (%)"],  up_stats["IDH mutant, n (%)"]),
    ]
    table = pd.DataFrame(rows, columns=["Variable", "Total", "UCSF", "UPENN"])
    return table

def main():
    allowed_ids = load_allowed_ids(
        "derivatives/traincv_set__4_topk.xlsx",
        "derivatives/test_set__4_topk.xlsx"
    )
    allowed_df = pd.DataFrame({"subject_id": sorted(allowed_ids)})
    
    df_ucsf  = load_ucsf("UCSF-PDGM-metadata_v2.xlsx")
    df_upenn = load_upenn("UPENN-GBM_clinical_info_v2.1.xlsx")
    
    df_ucsf  = (df_ucsf.merge(allowed_df, on="subject_id", how="inner")
                        .sort_values("subject_id")
                        .drop_duplicates(subset=["subject_id"], keep="first"))
    
    df_upenn = (df_upenn.merge(allowed_df, on="subject_id", how="inner")
                         .sort_values("subject_id")
                         .drop_duplicates(subset=["subject_id"], keep="first"))

    # --- UCSF WHO CNS Grade distribution (restricted to train/test subjects) ---
    grades = pd.to_numeric(df_ucsf["Grade"], errors="coerce")
    total_ucsf = len(df_ucsf)
    g2 = int((grades == 2).sum())
    g3 = int((grades == 3).sum())
    g4 = int((grades == 4).sum())
    g_unknown = int(grades.isna().sum())
    
    print("\nUCSF WHO CNS Grade (restricted to train/test):")
    print(f"  Grade 2: {g2} ({g2/total_ucsf:.1%})")
    print(f"  Grade 3: {g3} ({g3/total_ucsf:.1%})")
    print(f"  Grade 4: {g4} ({g4/total_ucsf:.1%})")
    print(f"  Unknown/other: {g_unknown} ({g_unknown/total_ucsf:.1%})")

    df_all = pd.concat([df_ucsf, df_upenn], ignore_index=True)
    table = build_table(df_all)

    print(f"[DEBUG] Unique allowed subjects: {len(allowed_df)}")
    print(f"[DEBUG] UCSF kept unique: {len(df_ucsf)}")
    print(f"[DEBUG] UPENN kept unique: {len(df_upenn)}")
    print(f"[DEBUG] Total kept unique: {len(df_ucsf) + len(df_upenn)}")

    kept_ids = set(df_ucsf["subject_id"]) | set(df_upenn["subject_id"])
    missing_from_clinical = sorted(set(allowed_ids) - kept_ids)
    extra_not_in_splits = sorted(kept_ids - set(allowed_ids))
    
    print(f"[DEBUG] Missing from clinical (in splits but not found): {len(missing_from_clinical)}")
    print(f"[DEBUG] Extra not in splits (should be 0): {len(extra_not_in_splits)}")

    print("\nTable 1: Descriptive statistics (restricted to train/test samples).\n")
    print(table.to_string(index=False))
    print()

if __name__ == "__main__":
    main()

