Explainable Radiomics with Probability Calibration for Postoperative Glioblastoma Surveillance
---------------------------------------------------------------------------------------------

This repository contains the full radiomics pipeline used in the study:
"Explainable Radiomics with Probability Calibration for Postoperative Glioblastoma Surveillance" (2025)
by Christodoulou RC, Vamvouras G., et al.

The code performs:
- MRI preprocessing (N4 correction, resampling, z-score within mask)
- Radiomics extraction using PyRadiomics (original, LoG, and 3D wavelets)
- Dataset construction combining clinical and radiomics features
- Feature ranking (variance filtering, correlation reduction, L1-logistic ranking)
- Patient-aware train/validation/test splitting to avoid leakage
- Model training: Logistic Regression, LightGBM, Random Forest, SVM
- Probability calibration using Platt scaling
- Correlation redundancy analysis (clustered heatmaps, block plots, violin plots)
- Clinical cohort statistics

---------------------------------------------------------------------------------------------
Repository Structure
---------------------------------------------------------------------------------------------
pipeline_extract_catalogue__1.py      : Preprocessing + PyRadiomics extraction
collect_dataset__2.py                 : Merge radiomics with clinical metadata
derivatives/                          : Output directory for radiomics and availability tables
bin_class/rank_and_split__4.py        : Feature ranking + patient-aware splitting
bin_class/logreg__5a.py               : Logistic Regression classifier
bin_class/lightGBM__5b.py             : LightGBM classifier
bin_class/randfor__5c.py              : Random Forest classifier
bin_class/svc__5d.py                  : SVM classifier
corr_plot__4a.py                      : Clustered correlation heatmaps
corr_plot__4b.py                      : Block correlation plots by feature family
violin_plot__4c.py                    : Family-level violin plots
violin_plot__4d.py                    : Redundancy tail plots (95th percentile and |r|>0.8)
stats__3.py                           : Clinical cohort statistics
radiomics_params.yaml                 : PyRadiomics configuration file

---------------------------------------------------------------------------------------------
Installation
---------------------------------------------------------------------------------------------
This project was developed and tested with Python 3.11.8.
Using a different Python version may lead to slightly different results.

1. Clone the repository and enter the folder:
   git clone https://github.com/georgeDV2002/Explainable-Radiomics-with-Probability-Calibration-for-Postoperative-Glioblastoma-Surveillance.git
   cd Explainable-Radiomics-with-Probability-Calibration-for-Postoperative-Glioblastoma-Surveillance


2. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate        (Windows: venv\Scripts\activate)

3. Install Python dependencies:
   pip install -r requirements.txt

---------------------------------------------------------------------------------------------
Usage
---------------------------------------------------------------------------------------------
1) Radiomics extraction:
   python pipeline_extract_catalogue__1.py

2) Create combined dataset:
   python collect_dataset__2.py

3) Feature ranking and splits:
   cd bin_class
   python rank_and_split__4.py --topk 256

4) Model training:
   python logreg__5a.py
   python lightGBM__5b.py
   python randfor__5c.py
   python svc__5d.py

5) Correlation redundancy analysis:
   python ../corr_plot__4a.py
   python ../corr_plot__4b.py
   python ../violin_plot__4c.py
   python ../violin_plot__4d.py

6) Clinical statistics:
   python stats__3.py

---------------------------------------------------------------------------------------------
Dataset
---------------------------------------------------------------------------------------------
The code requires the MU-Glioma-Post dataset from The Cancer Imaging Archive (TCIA):
DOI: 10.7937/7K9K-3C83
MRI volumes and segmentation masks must be downloaded separately.

---------------------------------------------------------------------------------------------
Citation
---------------------------------------------------------------------------------------------
If you use this code in your research, please cite:

Christodoulou RC, Vamvouras G., et al.
"Explainable Radiomics with Probability Calibration for Postoperative Glioblastoma Surveillance", 2025.

---------------------------------------------------------------------------------------------
License
---------------------------------------------------------------------------------------------
This project is released under the BSD 3-Clause License, which requires attribution.
