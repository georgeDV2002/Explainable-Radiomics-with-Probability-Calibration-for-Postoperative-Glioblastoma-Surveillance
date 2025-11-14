import pandas as pd
import numpy as np
import re

radiomics = pd.read_excel("derivatives/radiomics_features.xlsx")
clinical  = pd.read_excel("MU-Glioma-Post_ClinicalData-July2025.xlsx", sheet_name=1)

# Keep GBM first
clinical = clinical[clinical["Primary Diagnosis"] == "GBM"].copy()

# Select only the timepoint day columns + keys
time_cols = [c for c in clinical.columns
             if c.startswith("Number of Days from Diagnosis to ") and "(Timepoint_" in c]
base_cols = ["Patient_ID", "Progression", "Time to First Progression (Days)"]
clinical = clinical[base_cols + time_cols].copy()

# Melt to long
cl_long = clinical.melt(
    id_vars=base_cols,
    value_vars=time_cols,
    var_name="day_col",
    value_name="days_from_dx"
)

cl_long["timepoint"] = cl_long["day_col"].str.extract(r"\((Timepoint_\d+)\)")

# Keep only valid, non-negative timepoint days (from diagnosis)
cl_long = cl_long[pd.notna(cl_long["days_from_dx"]) & (cl_long["days_from_dx"] >= 0)]

# --- map "Time to First Progression (Days)" per patient ---
# (so we can compute delta_days = days_from_dx - time_to_first_prog)
prog_map = clinical.set_index("Patient_ID")["Time to First Progression (Days)"]

# Merge radiomics with clinical timepoint days (left join keeps all radiomics rows)
merged = radiomics.merge(
    cl_long[["Patient_ID", "timepoint", "days_from_dx"]],
    left_on=["subject_id", "timepoint"],
    right_on=["Patient_ID", "timepoint"],
    how="left"
)

# Attach progression day per patient (same name as in clinical)
prog_map = clinical.set_index("Patient_ID")["Time to First Progression (Days)"]
merged["Time to First Progression (Days)"] = merged["Patient_ID"].map(prog_map)

# Compute delta_days; keep NaN if progression day is NaN
merged["delta_days"] = np.where(
    pd.notna(merged["Time to First Progression (Days)"]),
    merged["days_from_dx"] - merged["Time to First Progression (Days)"],
    np.nan
)
merged.loc[merged["delta_days"] < 0, "delta_days"] = np.nan

# Drop helper + reorder; keep the progression column
merged = merged.drop(columns=["Patient_ID"])

id_cols = ["subject_id", "timepoint"]
priority_cols = ["Time to First Progression (Days)", "delta_days"]
other_cols = [c for c in merged.columns if c not in id_cols + priority_cols]
merged = merged[id_cols + priority_cols + other_cols]

merged.to_excel("dataset__2.xlsx", index=False)



