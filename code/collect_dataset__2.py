#!/usr/bin/env python3
"""
collect_dataset__2.py

Inputs (expected under ~/programs/radiomics2/):
- derivatives/radiomics_features.csv           # produced by the extraction pipeline
- UCSF-PDGM-metadata_v2.xlsx                   # UCSF clinical file, MGMT index column
- UPENN-GBM_clinical_info_v2.1.xlsx           # UPENN clinical file, MGMT string column

Output (under derivatives/):
- radiomics_features_with_mgmt.csv
- radiomics_features_with_mgmt.xlsx
- collect_dataset__2_report.txt               # simple merge/label summary

Rules
-----
UCSF:
  - Match radiomics subject_id/sample_id like 'UCSF-PDGM-XXXX' to Excel 'ID' like 'UCSF-PDGM-XXX'.
    Conversion: take the last 4 digits of radiomics ID, drop the first of those 4 (leading train 0),
    and format as 3 digits (zero-padded). Example: UCSF-PDGM-0108 -> UCSF-PDGM-108.
  - Excel column: 'MGMT index' (numeric). Make new 'MGMT' column as:
      NaN -> NaN, 0 -> 0, any integer > 0 -> 1

UPENN:
  - Match radiomics sample_id: 'UPENN-GBM-XXXXX_YY' to Excel 'ID' exactly.
  - Excel column: 'MGMT' (string). Map to new 'MGMT' as:
      'Methylated' -> 1, 'Unmethylated' -> 0, 'Not Available'/'Indeterminate'/missing/other -> NaN

The final dataset preserves all radiomics columns and adds a numeric 'MGMT' column (float with NaN).
"""
from __future__ import annotations

import sys
from pathlib import Path
import re
import math
import numpy as np
import pandas as pd

ROOT = Path("~/programs/radiomics2").expanduser()
DERIV = ROOT / "derivatives"

RAD_FEATS_CSV = DERIV / "radiomics_features.csv"
UCSF_XLSX     = ROOT / "UCSF-PDGM-metadata_v2.xlsx"
UPENN_XLSX    = ROOT / "UPENN-GBM_clinical_info_v2.1.xlsx"

OUT_CSV  = DERIV / "radiomics_features_with_mgmt.csv"
OUT_XLSX = DERIV / "radiomics_features_with_mgmt.xlsx"
REPORT   = DERIV / "collect_dataset__2_report.txt"

# ------------------------------ helpers ------------------------------

def ucsf_radiomics_id_to_excel_id(rid: str) -> str | None:
    """Convert 'UCSF-PDGM-XXXX' -> 'UCSF-PDGM-XXX' by dropping the first digit of the 4-digit suffix.
    Returns None if pattern doesn't match.
    """
    m = re.match(r"^UCSF-PDGM-(\d{4})$", rid)
    if not m:
        return None
    last4 = m.group(1)
    last3 = last4[1:]  # drop the first of the 4 (often a leading train '0')
    # ensure zero-padded to 3 digits
    try:
        last3_int = int(last3)
    except ValueError:
        return None
    return f"UCSF-PDGM-{last3_int:03d}"


def map_ucsf_mgmt(series: pd.Series) -> pd.Series:
    """MGMT index numeric -> NaN/0/1 as specified."""
    def _map(v):
        if pd.isna(v):
            return np.nan
        try:
            x = float(v)
        except Exception:
            return np.nan
        if math.isfinite(x) is False:
            return np.nan
        if x == 0:
            return 0.0
        return 1.0 if x > 0 else np.nan
    return series.map(_map)


def map_upenn_mgmt(series: pd.Series) -> pd.Series:
    """MGMT string -> NaN/0/1 with case-insensitive mapping."""
    def _map(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s == "methylated":
            return 1.0
        if s == "unmethylated":
            return 0.0
        if s in ("not available", "indeterminate"):
            return np.nan
        return np.nan
    return series.map(_map)


# ------------------------------ main ------------------------------

def main():
    DERIV.mkdir(parents=True, exist_ok=True)

    # Load radiomics
    if not RAD_FEATS_CSV.exists():
        print(f"[ERROR] Missing radiomics features: {RAD_FEATS_CSV}")
        sys.exit(1)
    rdx = pd.read_csv(RAD_FEATS_CSV)

    # Basic sanity
    needed_cols = {"dataset", "subject_id", "sample_id"}
    if not needed_cols.issubset(set(rdx.columns)):
        missing = needed_cols - set(rdx.columns)
        print(f"[ERROR] radiomics_features.csv missing columns: {sorted(missing)}")
        sys.exit(1)

    # Split views for keys
    rdx_ucsf = rdx[rdx["dataset"] == "UCSF"].copy()
    rdx_up   = rdx[rdx["dataset"] == "UPENN"].copy()

    # ---------------- UCSF ----------------
    # Build UCSF merge key from radiomics
    rdx_ucsf["UCSF_ID_merge"] = rdx_ucsf["subject_id"].map(ucsf_radiomics_id_to_excel_id)

    # Load UCSF excel
    if not UCSF_XLSX.exists():
        print(f"[WARN] UCSF metadata not found: {UCSF_XLSX} — UCSF MGMT will be NaN")
        ucsf_meta = pd.DataFrame(columns=["ID", "MGMT index"])  # empty
    else:
        ucsf_meta = pd.read_excel(UCSF_XLSX)

    # Normalize UCSF meta columns
    if "ID" not in ucsf_meta.columns:
        print("[WARN] UCSF metadata has no 'ID' column; UCSF MGMT will be NaN")
        ucsf_meta["ID"] = []
    if "MGMT index" not in ucsf_meta.columns:
        print("[WARN] UCSF metadata has no 'MGMT index' column; UCSF MGMT will be NaN")
        ucsf_meta["MGMT index"] = np.nan

    ucsf_meta["ID"] = ucsf_meta["ID"].astype(str).str.strip()
    ucsf_meta["MGMT"] = map_ucsf_mgmt(ucsf_meta["MGMT index"])  # new standardized MGMT

    # Merge UCSF
    rdx_ucsf = rdx_ucsf.merge(
        ucsf_meta[["ID", "MGMT"]], how="left", left_on="UCSF_ID_merge", right_on="ID"
    )
    rdx_ucsf.drop(columns=["ID"], inplace=True, errors="ignore")

    # ---------------- UPENN ----------------
    # Load UPENN excel
    if not UPENN_XLSX.exists():
        print(f"[WARN] UPENN clinical not found: {UPENN_XLSX} — UPENN MGMT will be NaN")
        up_meta = pd.DataFrame(columns=["ID", "MGMT"])  # empty
    else:
        up_meta = pd.read_excel(UPENN_XLSX)

    if "ID" not in up_meta.columns:
        print("[WARN] UPENN clinical has no 'ID' column; UPENN MGMT will be NaN")
        up_meta["ID"] = []
    if "MGMT" not in up_meta.columns:
        print("[WARN] UPENN clinical has no 'MGMT' column; UPENN MGMT will be NaN")
        up_meta["MGMT"] = np.nan

    up_meta["ID"] = up_meta["ID"].astype(str).str.strip()
    up_meta["MGMT"] = map_upenn_mgmt(up_meta["MGMT"])  # normalize to 0/1/NaN

    # Merge UPENN by sample_id exactly
    rdx_up = rdx_up.merge(up_meta[["ID", "MGMT"]], how="left", left_on="sample_id", right_on="ID")
    rdx_up.drop(columns=["ID"], inplace=True, errors="ignore")

    # ---------------- Combine back ----------------
    rdy = pd.concat([rdx_ucsf, rdx_up], ignore_index=True, sort=False)

    # Save
    rdy.to_csv(OUT_CSV, index=False)
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as w:
        rdy.to_excel(w, index=False, sheet_name="radiomics_with_mgmt")

    # Report
    n_all   = len(rdy)
    n_ucsf  = len(rdx_ucsf)
    n_upenn = len(rdx_up)
    n_mgmt  = int(rdy["MGMT"].notna().sum())
    n_miss  = n_all - n_mgmt

    # Any unmatched UCSF IDs for reference
    unmatched_ucsf = rdx_ucsf[rdx_ucsf["MGMT"].isna()][["subject_id", "UCSF_ID_merge"]]
    unmatched_up   = rdx_up[rdx_up["MGMT"].isna()][["sample_id"]]

    lines = [
        "collect_dataset__2 summary",
        f"Total rows: {n_all}",
        f"  UCSF rows: {n_ucsf}",
        f"  UPENN rows: {n_upenn}",
        f"MGMT present: {n_mgmt}",
        f"MGMT missing: {n_miss}",
        "",
        "Unmatched UCSF (first 20):",
    ]
    lines += [f"  {a} -> {b}" for a, b in unmatched_ucsf.head(20).itertuples(index=False)]
    lines += ["", "Unmatched UPENN (first 20):"]
    lines += [f"  {a}" for a in unmatched_up.head(20)["sample_id"].tolist()]

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("\nDone. Wrote:\n  -", OUT_CSV, "\n  -", OUT_XLSX, "\n  -", REPORT)


if __name__ == "__main__":
    main()

