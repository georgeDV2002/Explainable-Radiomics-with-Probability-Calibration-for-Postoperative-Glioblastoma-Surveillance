#!/usr/bin/env python3
"""
pipeline_inmemory_radiomics.py
Hardcoded, in-memory PyRadiomics pipeline (no argparse, no NIfTI saves).

What it does:
1) Scans MU-Glioma-Post for PatientID_XXXX/Timepoint_X pairs
2) Checks availability of required files (T1c/T1/T2/FLAIR + tumorMask)
3) Preprocesses in memory: optional N4, resample to 1mm isotropic, optional z-score within mask
4) Extracts PyRadiomics features per available modality

Requirements:
  pip install pyradiomics SimpleITK nibabel tqdm pandas numpy
"""

# ===================== HARD-CODED SETTINGS =====================
BASE_DIR    = "/home/georgedv/programs/MU-Glioma-Post"
OUTDIR      = "/home/georgedv/programs/radiomics1/derivatives"
PARAMS_YAML = "/home/georgedv/programs/radiomics1/radiomics_params.yaml"

APPLY_N4           = True     # set False to skip N4 bias correction
ZSCORE_WITHIN_MASK = True     # z-score each modality within the (core) mask after resampling
OUTPUT_SPACING_MM  = (1.0, 1.0, 1.0)  # isotropic spacing (mm)

# Filename token -> sequence label (used as feature prefix)
MOD_MAP = {
    "brain_t1c": "t1ce",
    "brain_t1n": "t1",
    "brain_t2w": "t2",
    "brain_t2f": "flair",
}
MASK_KEY = "tumorMask"
# ===============================================================

import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
from radiomics.featureextractor import RadiomicsFeatureExtractor
from nibabel.filebasedimages import ImageFileError

import logging
logging.getLogger("radiomics").setLevel(logging.ERROR)
logging.getLogger("pykwalify").setLevel(logging.ERROR)


# -------------------------- I/O + helpers --------------------------

def is_nonempty_file(p: Path) -> bool:
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False

def expected_files(patient_id: str, timepoint: str, base_dir: Path) -> Dict[str, Path]:
    tp_dir = base_dir / patient_id / timepoint
    stub = f"{patient_id}_{timepoint}_"
    files = {k: tp_dir / f"{stub}{k}.nii.gz" for k in list(MOD_MAP.keys()) + [MASK_KEY]}
    return files


def discover_timepoints(base_dir: Path) -> List[Tuple[str, str]]:
    """Find (patient_id, timepoint) pairs like PatientID_XXXX/Timepoint_X."""
    pairs = []
    for pdir in sorted(base_dir.glob("PatientID_*")):
        if not pdir.is_dir():
            continue
        for tpdir in sorted(pdir.glob("Timepoint_*")):
            if tpdir.is_dir():
                pairs.append((pdir.name, tpdir.name))
    return pairs


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# -------------------------- image conversions --------------------------
def load_nifti(path: Path) -> nib.Nifti1Image:
    return nib.load(str(path))


def to_sitk(nib_img: nib.Nifti1Image) -> sitk.Image:
    """Convert nibabel image to SimpleITK (with spacing set)."""
    data = np.asarray(nib_img.get_fdata(dtype=np.float32))
    sitk_img = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))  # nib (x,y,z) -> sitk (z,y,x)
    zooms = nib_img.header.get_zooms()[:3]
    sitk_img.SetSpacing((float(zooms[2]), float(zooms[1]), float(zooms[0])))
    return sitk_img


# -------------------------- preprocessing ops --------------------------
def n4_bias_correction(sitk_img: sitk.Image, mask: Optional[sitk.Image] = None) -> sitk.Image:
    img = sitk.Cast(sitk_img, sitk.sitkFloat32)
    if mask is None:
        mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(img, mask)
    return corrected


def resample_isotropic(img: sitk.Image, spacing=(1.0, 1.0, 1.0), is_label: bool = False) -> sitk.Image:
    """Resample to isotropic spacing with BSpline for images, NN for labels."""
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, spacing)
    ]
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interp)
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(img)


def zscore_within_mask(img: sitk.Image, mask: sitk.Image) -> sitk.Image:
    """Z-score normalization using voxels inside the mask; returns a new sitk.Image."""
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    msk = sitk.GetArrayFromImage(mask) > 0
    if msk.sum() == 0:
        return img
    vals = arr[msk]
    mu = float(vals.mean())
    sd = float(vals.std()) if float(vals.std()) > 1e-6 else 1.0
    arr_norm = (arr - mu) / sd
    out = sitk.GetImageFromArray(arr_norm)
    out.CopyInformation(img)
    return out


# -------------------------- feature extraction --------------------------
def extract_features_for_tp_inmemory(
    seq_imgs: Dict[str, sitk.Image],           # e.g., {"brain_t1c": sitk.Image, ...}
    params_path: Path,
    subject_id: str,
    timepoint: str,
    mask_img: sitk.Image,
    prefix: str,
) -> Dict[str, float]:
    """
    Extract PyRadiomics features from provided sitk images (already preprocessed) using mask_img.
    Returns a flat dict of features with keys like "{prefix}__t1ce__firstorder_Mean".
    """
    #extractor = RadiomicsFeatureExtractor(str(params_path))
    extractor = RadiomicsFeatureExtractor(str(params_path), enableCExtensions=True, verbose=False)
    extractor.settings["label"] = 1  # label 1 for binary masks (foreground)

    features = {"subject_id": subject_id, "timepoint": timepoint, "roi": prefix}

    for mod_key, seq_label in MOD_MAP.items():
        if mod_key not in seq_imgs:
            continue
        img_sitk = seq_imgs[mod_key]
        result = extractor.execute(img_sitk, mask_img)

        for k, v in result.items():
            if k.startswith("diagnostics_"):
                continue
            if isinstance(v, (int, float, np.floating)):
                features[f"{prefix}__{seq_label}__{k}"] = float(v)
            else:
                # Keep string values (rare) as-is
                features[f"{prefix}__{seq_label}__{k}"] = str(v)

    return features


# -------------------------- main --------------------------
def main():
    header_written = False
    header_cols = None

    def write_row(row):
        nonlocal header_written, header_cols
        if not header_written:
            # Freeze header from the first row (assumes same features for all subjects)
            base_cols = ["subject_id", "timepoint", "roi"]
            header_cols = base_cols + sorted([k for k in row.keys() if k not in base_cols])
            with open(feats_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header_cols)
                writer.writeheader()
            header_written = True
        with open(feats_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header_cols, extrasaction="ignore")
            writer.writerow(row)

    base = Path(BASE_DIR)
    outdir = Path(OUTDIR)
    ensure_dir(outdir)

    avail_csv = outdir / "availability_report.csv"
    feats_csv = outdir / "radiomics_features.csv"

    # Availability header
    with avail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["subject_id", "timepoint"] + list(MOD_MAP.keys()) + [MASK_KEY, "all_present"]
        writer.writerow(header)

    pairs = discover_timepoints(base)
    if not pairs:
        print(f"[WARN] No PatientID_*/Timepoint_* pairs found under {base}")
        return

    # Accumulate feature rows to write a single CSV with union of columns

    assert OUTPUT_SPACING_MM[0] == OUTPUT_SPACING_MM[1] == OUTPUT_SPACING_MM[2], \
        "Ring thickness assumes isotropic spacing."

    for patient_id, timepoint in tqdm(pairs, desc="Timepoints"):
        files = expected_files(patient_id, timepoint, base)

        # Availability
        flags = {k: int(is_nonempty_file(p)) for k, p in files.items()}
        all_present = int(all(flags.values()))
        with avail_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([patient_id, timepoint] + [flags[k] for k in MOD_MAP.keys()] + [flags[MASK_KEY], all_present])

        # --- Load and preprocess mask (core) ---
        # Need mask to proceed
        if not is_nonempty_file(files[MASK_KEY]):
            print(f"[WARN] Empty or missing mask: {files[MASK_KEY]} — skipping {patient_id} {timepoint}")
            continue
        try:
            mask_nib = load_nifti(files[MASK_KEY])
        except ImageFileError as e:
            print(f"[WARN] Failed to load mask {files[MASK_KEY]}: {e} — skipping {patient_id} {timepoint}")
            continue

        mask_sitk = to_sitk(mask_nib)
        mask_res  = resample_isotropic(mask_sitk, spacing=OUTPUT_SPACING_MM, is_label=True)
        mask_bin = sitk.Cast(mask_res > 0, sitk.sitkUInt8)   # binary 0/1 UInt8 mask
    
        # --- Preprocess each available modality in memory ---
        seq_imgs: Dict[str, sitk.Image] = {}
        for mod_key, seq_label in MOD_MAP.items():
            p = files[mod_key]
            if not is_nonempty_file(p):
                continue
            try:
                img_nib = load_nifti(p)
            except ImageFileError as e:
                print(f"[WARN] Failed to load {p}: {e} — skipping {mod_key}")
                continue
            img_sitk = to_sitk(img_nib)

            if APPLY_N4:
                img_sitk = n4_bias_correction(img_sitk, mask=mask_bin)

            img_res = resample_isotropic(img_sitk, spacing=OUTPUT_SPACING_MM, is_label=False)

            if ZSCORE_WITHIN_MASK:
                img_res = zscore_within_mask(img_res, mask_bin)

            seq_imgs[mod_key] = img_res

        # If no modalities are available, skip
        if not seq_imgs:
            continue

        # CORE
        try:
            core_feats = extract_features_for_tp_inmemory(
                seq_imgs=seq_imgs,
                params_path=Path(PARAMS_YAML),
                subject_id=patient_id,
                timepoint=timepoint,
                mask_img=mask_bin,
                prefix="core",
            )
            write_row(core_feats)
        except Exception as e:
            print(f"[ERROR] CORE radiomics failed for {patient_id} {timepoint}: {e}")

    print(f"\nDone.\nAvailability: {avail_csv}\nRadiomics:    {feats_csv}\n")


if __name__ == "__main__":
    main()
