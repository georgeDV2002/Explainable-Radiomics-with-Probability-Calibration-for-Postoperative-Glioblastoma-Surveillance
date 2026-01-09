# Explainable, Modality-Adaptive Radiomics for MGMT Methylation Prediction in High-Grade Glioma

This repository contains the official code accompanying the article:

> Rafail C. Christodoulou, Georgios Vamvouras, Rafael Pitsillos,
> Elena E. Solomou, Michalis F. Georgiou.
> **Explainable, modality-adaptive radiomics for MGMT methylation prediction in high-grade glioma: a decision-curve analysis study**.
> *Frontiers in Oncology* (Neuro-Oncology and Neurosurgical Oncology), 2026.
> DOI: 10.3389/fonc.2025.1731258

### Contribution Statement
All code in this repository, including data preprocessing, feature extraction,
calibration, training pipelines, explainability, and evaluation scripts, was
entirely designed and implemented by **Georgios Vamvouras**.

## Other Coauthors (Manuscript Only)
The individuals listed as coauthors in any related manuscript contributed to
conceptual discussion and manuscript writing but **did not write or review** any
of the code in this repository.

The code implements the complete radiomics & machine learning workflow used in the paper, from multi-modal MRI preprocessing and radiomics feature extraction to MGMT label harmonization, model training, explainability, and decision-curve analysis.

---

## 1. Repository Contents

**Top-level files:**

* `README.md` — main documentation
* `LICENCE.txt` — license for code usage
* `AUTHORS.md` — authorship and contribution credits
* `code/` — all Python scripts for extraction, merging, splitting, ranking

---

## 2. Scope of This Code

This repository supports:

1. **Reproduction of the radiomics MGMT prediction experiments** described in the paper.
2. **Transparent implementation** of modality-adaptive radiomics with clinical integration.
3. **Extension** to new cohorts, radiomics settings, and explainability modules.

Focus areas:

* High-grade glioma (HGG)
* MGMT promoter methylation prediction
* Multi-modal MRI radiomics
* Explainability and interpretation
* Decision-curve analysis (DCA)

---

## 3. Data Requirements

MRI datasets are **not included**, as they are controlled-access. You must:

1. Acquire UCSF and/or UPENN datasets following proper agreements.
2. Organize files to match expected folder structures.
3. Update paths inside scripts to your local environment.

See the paper’s *Materials and Methods* for full dataset specifications.

---

## 4. Environment & Dependencies

Python **3.10+** recommended.

Required packages include:

* numpy, pandas, scipy
* SimpleITK, nibabel
* pyradiomics
* scikit-learn
* tqdm
* openpyxl

Install minimal requirements:

```
pip install numpy pandas scipy scikit-learn SimpleITK nibabel pyradiomics tqdm openpyxl
```

---

## 5. Workflow Overview

### 1) Radiomics Extraction (`pipeline_extract_catalogue__1.py`)

* Scans UCSF + UPENN structures
* Performs in-memory preprocessing (N4, resampling, z-score normalization)
* Extracts PyRadiomics features
* Saves:

  * `availability_report.csv`
  * `radiomics_features.csv`

### 2) Clinical Integration (`collect_dataset__2.py`)

* Merges radiomics with clinical spreadsheets
* Harmonizes MGMT labels across datasets
* Exports:

  * `radiomics_features_with_mgmt.xlsx`
  * `collect_dataset__2_report.txt`

### 3) Patient-Aware Split & Feature Ranking (`rank_and_split__4.py`)

* Splits by patient (no leakage)
* Performs stratification on MGMT
* Conducts L1-logistic and permutation importance
* Creates Top-K pruned datasets
* Outputs:

  * `traincv_set__4.xlsx`
  * `test_set__4.xlsx`
  * `importance_traincv__4.csv`
  * `traincv_set__4_topk.xlsx`
  * `test_set__4_topk.xlsx`

### 4) Clinical Summary (`stats__3.py`)

* Standardizes IDs, MGMT, IDH, age, sex
* Produces Table-1 style summary
* Prints WHO grade distribution (UCSF)

---

## 6. Reproducibility Notes

* Fixed random seeds
* Strict patient-level separation
* Deterministic preprocessing (mask-driven geometry alignment)
* Test set untouched during feature ranking

---

## 7. How to Cite

If you use this code, please cite:

> Rafail C. Christodoulou, Georgios Vamvouras, Rafael Pitsillos,
> Elena E. Solomou, Michalis F. Georgiou.
> **Explainable, modality-adaptive radiomics for MGMT methylation prediction in high-grade glioma: a decision-curve analysis study**.
> *Frontiers in Oncology*, 2026.
> DOI: 10.3389/fonc.2025.1731258

---

## 8. License & Contributions

* Distributed under terms described in `LICENCE.txt`.
* Authors and contributors listed in `AUTHORS.md`.

Contributions are welcome via pull requests and issues, subject to clinical data rest

