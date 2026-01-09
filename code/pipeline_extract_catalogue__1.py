#!/usr/bin/env python3
"""
Radiomics extraction for UCSF (PDGM) and UPENN (GBM) datasets — single, combined pipeline.

What it does
============
1) Scans the two datasets and builds a single catalogue of *samples*:
   - UCSF samples:     UCSF-PDGM-XXXX  (single time, no suffix)
   - UPENN samples:    UPENN-GBM-XXXXX_YY (YY ∈ {1,2}; treat each _YY as a distinct sample)
2) Validates availability of required files per dataset.
3) Preprocesses *in memory*:
   - Optional N4 bias correction (skipped for ADC/ASL/DWI)
   - Resample to 1.0 mm isotropic **only if** not already isotropic (within tolerance)
   - Resample all modalities exactly onto the mask grid (size/spacing/origin/direction)
   - Z-score within mask (foreground voxels only)
4) Extracts PyRadiomics features using the provided YAML settings (all features enabled there).
5) Outputs two CSVs under OUTDIR (union of columns across datasets):
   - availability_report.csv
   - radiomics_features.csv

Requirements
===========
  pip install pyradiomics SimpleITK nibabel tqdm pandas numpy

Notes
=====
- Foreground label is any positive value in the mask (>0).
- Processing is serial and robust to missing modalities/masks (skips sample if mask missing/empty).
- Uses C-extensions in PyRadiomics when available.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
from radiomics.featureextractor import RadiomicsFeatureExtractor
from nibabel.filebasedimages import ImageFileError

import logging
logging.getLogger("radiomics").setLevel(logging.ERROR)
logging.getLogger("pykwalify").setLevel(logging.ERROR)

# ===================== HARD-CODED SETTINGS =====================
# Working dir and parameters
OUTDIR       = Path("~/programs/radiomics2/derivatives").expanduser()
PARAMS_YAML  = Path("~/programs/radiomics2/radiomics_params.yaml").expanduser()

# Dataset roots
UCSF_BASE    = Path("~/programs/UCSF/UCSF-PDGM-v3").expanduser()
UPENN_IMG    = Path("~/programs/UPENN/UPENN-GBM-NIfTI/UPENN-GBM/NIfTI-files/images_structural").expanduser()
UPENN_SEGM_M = Path("~/programs/UPENN/UPENN-GBM-NIfTI/UPENN-GBM/NIfTI-files/images_segm").expanduser()
UPENN_SEGM_A = Path("~/programs/UPENN/UPENN-GBM-NIfTI/UPENN-GBM/NIfTI-files/automated_segm").expanduser()

# Toggle preprocessing
APPLY_N4_DEFAULT       = True          # skipped for ADC/ASL/DWI
ZSCORE_WITHIN_MASK     = True
TARGET_SPACING         = (1.0, 1.0, 1.0)
ISOTROPY_TOLERANCE_MM  = 1e-3          # tolerance to consider spacing isotropic and equal to 1.0

# Modality maps (filename token -> feature prefix) per dataset
UCSF_MODALITIES = {
    "ADC":   "adc",
    "ASL":   "asl",
    "DWI":   "dwi",
    "FLAIR": "flair",
    "T1":    "t1",
    "T1c":   "t1ce",
    "T2":    "t2",
}

UPENN_MODALITIES = {
    "FLAIR": "flair",
    "T1":    "t1",
    "T1GD":  "t1ce",  # map to T1c
    "T2":    "t2",
}
# ===============================================================

# -------------------------- helpers --------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def is_nonempty_file(p: Path) -> bool:
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False

# -------------------------- discovery --------------------------
def discover_ucsf_samples(ucsf_base: Path) -> List[Tuple[str, Dict[str, Path]]]:
    """
    Return list of (sample_id, file_dict) for UCSF.
    Directory: UCSF-PDGM-v3 / UCSF-PDGM-XXXX_nifti /
      Files: UCSF-PDGM-XXXX_{ADC,ASL,DWI,FLAIR,T1,T1c,T2}.nii.gz
             UCSF-PDGM-XXXX_tumor_segmentation.nii.gz
    sample_id == subject_id == UCSF-PDGM-XXXX
    """
    samples: List[Tuple[str, Dict[str, Path]]] = []
    for d in sorted(ucsf_base.glob("UCSF-PDGM-*_nifti")):
        if not d.is_dir():
            continue
        subject_id = d.name.split("_nifti")[0]  # UCSF-PDGM-XXXX
        sample_id = subject_id
        files: Dict[str, Path] = {}
        # modalities
        for token in UCSF_MODALITIES.keys():
            files[token] = d / f"{subject_id}_{token}.nii.gz"
        # mask
        files["MASK"] = d / f"{subject_id}_tumor_segmentation.nii.gz"
        samples.append((sample_id, files))
    return samples

def discover_upenn_samples(upenn_img: Path) -> List[Tuple[str, Dict[str, Path]]]:
    """
    Return list of (sample_id, file_dict) for UPENN.
    Directory: images_structural / UPENN-GBM-XXXXX_YY /
      Files: UPENN-GBM-XXXXX_YY_{FLAIR,T1,T1GD,T2}.nii.gz
      Masks: prefer manual images_segm/UPENN-GBM-XXXXX_YY_segm.nii.gz,
             else automated_segm/UPENN-GBM-XXXXX_YY_automated_approx_segm.nii.gz
    sample_id == UPENN-GBM-XXXXX_YY
    """
    samples: List[Tuple[str, Dict[str, Path]]] = []
    for d in sorted(upenn_img.glob("UPENN-GBM-*_*")):
        if not d.is_dir():
            continue
        sample_id = d.name  # UPENN-GBM-XXXXX_YY
        files: Dict[str, Path] = {}
        # modalities
        for token in UPENN_MODALITIES.keys():
            files[token] = d / f"{sample_id}_{token}.nii.gz"
        # masks (prefer manual)
        manual = UPENN_SEGM_M / f"{sample_id}_segm.nii.gz"
        auto   = UPENN_SEGM_A / f"{sample_id}_automated_approx_segm.nii.gz"
        files["MASK_MANUAL"] = manual
        files["MASK_AUTO"]   = auto
        samples.append((sample_id, files))
    return samples

# -------------------------- I/O conversions --------------------------
def load_nifti(path: Path) -> nib.Nifti1Image:
    return nib.load(str(path))

def to_sitk(nib_img: nib.Nifti1Image) -> sitk.Image:
    data = np.asarray(nib_img.get_fdata(dtype=np.float32))
    sitk_img = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))  # nib (x,y,z) -> sitk (z,y,x)
    zooms = nib_img.header.get_zooms()[:3]
    sitk_img.SetSpacing((float(zooms[2]), float(zooms[1]), float(zooms[0])))
    return sitk_img

# -------------------------- preprocessing --------------------------
def is_isotropic_1mm(img: sitk.Image, tol: float = ISOTROPY_TOLERANCE_MM) -> bool:
    sx, sy, sz = img.GetSpacing()
    return (
        abs(sx - 1.0) <= tol and abs(sy - 1.0) <= tol and abs(sz - 1.0) <= tol
        and abs(sx - sy) <= tol and abs(sy - sz) <= tol
    )

def n4_bias_correction(sitk_img: sitk.Image, mask: Optional[sitk.Image] = None) -> sitk.Image:
    img = sitk.Cast(sitk_img, sitk.sitkFloat32)
    if mask is None:
        mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(img, mask)
    return corrected

def _resample(img: sitk.Image, spacing=(1.0, 1.0, 1.0), size=None, origin=None, direction=None, is_label: bool = False) -> sitk.Image:
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    if size is None:
        size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, spacing)]
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interp)
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(size)
    resampler.SetOutputDirection(direction if direction is not None else img.GetDirection())
    resampler.SetOutputOrigin(origin if origin is not None else img.GetOrigin())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(img)

def resample_isotropic(img: sitk.Image, spacing=(1.0, 1.0, 1.0), is_label: bool = False) -> sitk.Image:
    return _resample(img, spacing=spacing, is_label=is_label)

def resample_to_ref(img: sitk.Image, ref: sitk.Image, is_label: bool = False) -> sitk.Image:
    """Resample img exactly onto ref's grid (size/spacing/origin/direction)."""
    return _resample(
        img,
        spacing=ref.GetSpacing(),
        size=list(ref.GetSize()),
        origin=ref.GetOrigin(),
        direction=ref.GetDirection(),
        is_label=is_label,
    )

def zscore_within_mask(img: sitk.Image, mask: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    msk = sitk.GetArrayFromImage(mask) > 0  # accept any positive label values
    if msk.sum() == 0:
        return img
    vals = arr[msk]
    mu = float(vals.mean())
    sd = float(vals.std()) if float(vals.std()) > 1e-6 else 1.0
    arr_norm = (arr - mu) / sd
    arr_norm[~np.isfinite(arr_norm)] = 0.0  # guard against NaNs/Infs
    out = sitk.GetImageFromArray(arr_norm)
    out.CopyInformation(img)
    return out

def count_mask_voxels(mask: sitk.Image) -> int:
    arr = sitk.GetArrayFromImage(mask)
    return int((arr > 0).sum())

# -------------------------- feature extraction --------------------------
def extract_features(
    seq_imgs: Dict[str, sitk.Image],
    params_path: Path,
    dataset: str,
    subject_id: str,
    sample_id: str,
    mask_img: sitk.Image,
) -> Dict[str, float | str]:
    """Run PyRadiomics for provided preprocessed images and mask; returns flat dict.
    Keys are like "t1ce__firstorder_Mean".
    """
    extractor = RadiomicsFeatureExtractor(str(params_path), enableCExtensions=True, verbose=False)
    extractor.settings["label"] = 1

    features: Dict[str, float | str] = {
        "dataset": dataset,
        "subject_id": subject_id,
        "sample_id": sample_id,
        "roi": "core",
    }

    for token, img_sitk in seq_imgs.items():
        # Map token to feature prefix per dataset
        if dataset == "UCSF":
            prefix = UCSF_MODALITIES[token]
        else:  # UPENN
            prefix = UPENN_MODALITIES[token]

        result = extractor.execute(img_sitk, mask_img)
        for k, v in result.items():
            if k.startswith("diagnostics_"):
                continue
            key = f"{prefix}__{k}"
            if isinstance(v, (int, float, np.floating)):
                features[key] = float(v)
            else:
                features[key] = str(v)

    return features

# -------------------------- per-sample processing --------------------------
def process_ucsf_sample(sample_id: str, files: Dict[str, Path], write_avail_row, write_feat_row):
    dataset = "UCSF"
    subject_id = sample_id  # same for UCSF

    # Mask
    if not is_nonempty_file(files["MASK"]):
        print(f"[WARN] UCSF: missing mask for {sample_id} — skipping sample")
        write_avail_row(dataset, subject_id, sample_id, files, all_present=0)
        return

    try:
        mask_nib = load_nifti(files["MASK"])
        # sanity check dims
        if len(mask_nib.shape) < 3:
            raise ValueError(f"Mask has too few dimensions: shape={mask_nib.shape}")
    except Exception as e:
        print(f"[WARN] UCSF: failed to load/validate mask for {sample_id}: {e} — skipping sample")
        write_avail_row("UCSF", subject_id, sample_id, files, all_present=0)
        return

    mask_sitk = to_sitk(mask_nib)
    # Resample mask only if needed
    mask_res  = mask_sitk if is_isotropic_1mm(mask_sitk) else resample_isotropic(mask_sitk, spacing=TARGET_SPACING, is_label=True)
    mask_bin = sitk.Cast(mask_res > 0, sitk.sitkUInt8)
    vox = count_mask_voxels(mask_bin)
    if vox == 0:
        print(f"[WARN] UCSF: empty mask after resampling for {subject_id} — skipping sample")
        write_avail_row("UCSF", subject_id, sample_id, files, all_present=0)
        return

    # Modalities
    seq_imgs: Dict[str, sitk.Image] = {}
    present_flags: Dict[str, int] = {}

    for token in UCSF_MODALITIES.keys():
        p = files[token]
        ok = is_nonempty_file(p)
        present_flags[token] = int(ok)
        if not ok:
            continue
        try:
            img_nib = load_nifti(p)
        except ImageFileError as e:
            print(f"[WARN] UCSF: failed to load {p.name} for {sample_id}: {e}")
            continue
        img_sitk = to_sitk(img_nib)

        # N4 (skip for ADC/ASL/DWI)
        do_n4 = APPLY_N4_DEFAULT and (token not in {"ADC", "ASL", "DWI"})
        if do_n4:
            img_sitk = n4_bias_correction(img_sitk, mask=mask_bin)

        # Resample exactly onto mask grid
        img_res = img_sitk if (
            is_isotropic_1mm(img_sitk)
            and img_sitk.GetSpacing()   == mask_res.GetSpacing()
            and img_sitk.GetDirection() == mask_res.GetDirection()
            and img_sitk.GetOrigin()    == mask_res.GetOrigin()
        ) else resample_to_ref(img_sitk, ref=mask_res, is_label=False)

        # Z-score within mask
        if ZSCORE_WITHIN_MASK:
            img_res = zscore_within_mask(img_res, mask_bin)

        seq_imgs[token] = img_res

    # Availability row
    all_present = int(is_nonempty_file(files["MASK"])) and int(any(present_flags.values()))
    write_avail_row(dataset, subject_id, sample_id, files, all_present=all_present, present_flags=present_flags)

    if not seq_imgs:
        return

    # Features
    try:
        feats = extract_features(seq_imgs, PARAMS_YAML, dataset, subject_id, sample_id, mask_bin)
        write_feat_row(feats)
    except Exception as e:
        print(f"[ERROR] UCSF radiomics failed for {sample_id}: {e}")

def process_upenn_sample(sample_id: str, files: Dict[str, Path], write_avail_row, write_feat_row):
    dataset = "UPENN"
    subject_id = sample_id.split("_")[0]  # UPENN-GBM-XXXXX

    # Choose mask (manual preferred)
    mask_path = files.get("MASK_MANUAL") if is_nonempty_file(files.get("MASK_MANUAL", Path())) else files.get("MASK_AUTO")
    if not (mask_path and is_nonempty_file(mask_path)):
        print(f"[WARN] UPENN: missing mask for {sample_id} — skipping sample")
        write_avail_row(dataset, subject_id, sample_id, files, all_present=0)
        return

    try:
        mask_nib = load_nifti(mask_path)
        if len(mask_nib.shape) < 3:
            raise ValueError(f"Mask has too few dimensions: shape={mask_nib.shape}")
    except Exception as e:
        print(f"[WARN] UPENN: failed to load/validate mask for {sample_id}: {e} — skipping sample")
        write_avail_row(dataset, subject_id, sample_id, files, all_present=0)
        return

    mask_sitk = to_sitk(mask_nib)
    mask_res = mask_sitk if is_isotropic_1mm(mask_sitk) else resample_isotropic(mask_sitk, spacing=TARGET_SPACING, is_label=True)
    mask_bin = sitk.Cast(mask_res > 0, sitk.sitkUInt8)
    vox = count_mask_voxels(mask_bin)
    if vox == 0:
        print(f"[WARN] UPENN: empty mask after resampling for {sample_id} — skipping sample")
        write_avail_row("UPENN", subject_id, sample_id, files, all_present=0)
        return

    # Modalities
    seq_imgs: Dict[str, sitk.Image] = {}
    present_flags: Dict[str, int] = {}

    for token in UPENN_MODALITIES.keys():
        p = files[token]
        ok = is_nonempty_file(p)
        present_flags[token] = int(ok)
        if not ok:
            continue
        try:
            img_nib = load_nifti(p)
        except ImageFileError as e:
            print(f"[WARN] UPENN: failed to load {p.name} for {sample_id}: {e}")
            continue
        img_sitk = to_sitk(img_nib)

        # N4 (applies for all UPENN modalities here)
        if APPLY_N4_DEFAULT:
            img_sitk = n4_bias_correction(img_sitk, mask=mask_bin)

        # Resample exactly onto mask grid
        img_res = resample_to_ref(img_sitk, ref=mask_res, is_label=False)

        # Z-score within mask
        if ZSCORE_WITHIN_MASK:
            img_res = zscore_within_mask(img_res, mask_bin)

        seq_imgs[token] = img_res

    # Availability row
    all_present = int(bool(mask_path)) and int(any(present_flags.values()))
    write_avail_row(dataset, subject_id, sample_id, files, all_present=all_present, present_flags=present_flags)

    if not seq_imgs:
        return

    # Features
    try:
        feats = extract_features(seq_imgs, PARAMS_YAML, dataset, subject_id, sample_id, mask_bin)
        write_feat_row(feats)
    except Exception as e:
        print(f"[ERROR] UPENN radiomics failed for {sample_id}: {e}")

# -------------------------- CSV writers (union header) --------------------------
def make_writers(outdir: Path):
    ensure_dir(outdir)
    avail_csv = outdir / "availability_report.csv"
    feats_csv = outdir / "radiomics_features.csv"

    # availability header: dataset, subject_id, sample_id, per-modality flags, mask flags, all_present
    avail_header = [
        "dataset", "subject_id", "sample_id",
        # UCSF modalities
        *[f"UCSF_{k}" for k in UCSF_MODALITIES.keys()],
        # UPENN modalities
        *[f"UPENN_{k}" for k in UPENN_MODALITIES.keys()],
        "MASK_UCSF",
        "MASK_UPENN_MANUAL",
        "MASK_UPENN_AUTO",
        "all_present",
    ]
    with avail_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(avail_header)

    header_written = False
    header_cols: Optional[List[str]] = None

    def write_avail_row(dataset: str, subject_id: str, sample_id: str, files: Dict[str, Path], all_present: int, present_flags: Optional[Dict[str, int]] = None):
        row = {
            "dataset": dataset,
            "subject_id": subject_id,
            "sample_id": sample_id,
            # init zeros
            **{f"UCSF_{k}": 0 for k in UCSF_MODALITIES.keys()},
            **{f"UPENN_{k}": 0 for k in UPENN_MODALITIES.keys()},
            "MASK_UCSF": 0,
            "MASK_UPENN_MANUAL": 0,
            "MASK_UPENN_AUTO": 0,
            "all_present": int(all_present),
        }
        if dataset == "UCSF":
            for k in UCSF_MODALITIES.keys():
                row[f"UCSF_{k}"] = int(is_nonempty_file(files.get(k, Path())))
            row["MASK_UCSF"] = int(is_nonempty_file(files.get("MASK", Path())))
        else:
            for k in UPENN_MODALITIES.keys():
                row[f"UPENN_{k}"] = int(is_nonempty_file(files.get(k, Path())))
            row["MASK_UPENN_MANUAL"] = int(is_nonempty_file(files.get("MASK_MANUAL", Path())))
            row["MASK_UPENN_AUTO"]   = int(is_nonempty_file(files.get("MASK_AUTO", Path())))
        with avail_csv.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=avail_header).writerow(row)

    def write_feat_row(row: Dict[str, float | str]):
        nonlocal header_written, header_cols
        if not header_written:
            base_cols = ["dataset", "subject_id", "sample_id", "roi"]
            header_cols = base_cols + sorted([k for k in row.keys() if k not in base_cols])
            with feats_csv.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=header_cols).writeheader()
            header_written = True
        with feats_csv.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=header_cols, extrasaction="ignore").writerow(row)

    return write_avail_row, write_feat_row, avail_csv, feats_csv

# -------------------------- main --------------------------
def main():
    write_avail_row, write_feat_row, avail_csv, feats_csv = make_writers(OUTDIR)

    # Discover samples
    ucsf_samples  = discover_ucsf_samples(UCSF_BASE)
    upenn_samples = discover_upenn_samples(UPENN_IMG)

    if not ucsf_samples and not upenn_samples:
        print(f"[WARN] No samples discovered. Check dataset roots.\n  UCSF:  {UCSF_BASE}\n  UPENN: {UPENN_IMG}")
        return

    # Process UCSF
    for sample_id, files in tqdm(ucsf_samples, desc="UCSF samples"):
        process_ucsf_sample(sample_id, files, write_avail_row, write_feat_row)

    # Process UPENN
    for sample_id, files in tqdm(upenn_samples, desc="UPENN samples"):
        process_upenn_sample(sample_id, files, write_avail_row, write_feat_row)

    print(f"\nDone.\nAvailability: {avail_csv}\nRadiomics:    {feats_csv}\n")

if __name__ == "__main__":
    main()

