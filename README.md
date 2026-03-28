# Quantifying Neuroanatomical Atrophy
### A PCA-SVM Pipeline for Multi-Stage Alzheimer's Classification

This project builds a machine learning pipeline to classify Alzheimer's Disease into four stages using MRI brain scans from the OASIS-1 dataset. The core idea is to use PCA to reduce each brain scan into a compact set of mathematical features, then train an SVM to classify those features into one of four dementia stages: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented.

The goal is to show that accurate classification is possible using a resource-efficient approach, without deep learning, and with only a few hundred samples.

---

## Project Plan and Progress

| Phase | Description | Status |
|---|---|---|
| Stage 0 | Preprocessing script to select and clean MRI slices | Done |
| Phase 1 | Environment setup, metadata loading, class distribution | Done |
| Phase 2 | MRI image loading, flattening to feature vectors | Done |
| Phase 3 | PCA dimensionality reduction, eigenbrain visualisation | Done |
| Phase 4 | Train/test splitting, Moderate class augmentation, class weights | Done |
| Phase 5 | SVM training (Linear + RBF) with GridSearchCV | Not started |
| Phase 6 | Evaluation: classification report, confusion matrix, ROC curves | Not started |
| Final | Experiment summary and configuration export | Not started |

---

## Key Design Decisions

**One slice per subject, not per scan.** The dataset has thousands of MRI slices per person. Using multiple slices from the same person would let the model recognise the person rather than the disease. We select one axial slice per subject (targeting index 130, or the closest available) to keep the science valid.

**201 subjects had no CDR score in the metadata.** CDR (Clinical Dementia Rating) is the label source. Rather than drop those subjects and lose nearly half the dataset, we infer their label from the folder they are stored in, since the dataset author sorted subjects into folders based on CDR. The CSV value takes precedence when available.

**The Moderate class only has 2 real subjects.** This is a known limitation of the OASIS-1 dataset. To make the pipeline viable, we apply two rounds of augmentation:
- In `preprocess.py`: pixel-space augmentation (flip + rotate) brings the class to 8 samples
- In Phase 4: PCA-space augmentation (Gaussian noise + scaling) on the training set only brings it to 30 training samples

**Augmentation is done after splitting, not before.** This prevents data leakage. The test set is never touched and contains only original samples.

**5-fold cross-validation instead of 10.** With 30 Moderate training samples, 10-fold CV would give roughly 3 Moderate samples per fold, which is too few for stable SVM training. 5-fold gives around 6 per fold, which is the minimum viable amount.

**1 Moderate sample ends up in the test set.** With only 8 total Moderate samples at split time, stratified 80/20 gives 1 test sample for that class. Per-class metrics for Moderate in Phase 6 will be noted with a caveat. The confusion matrix is the more meaningful diagnostic for that class.

---

## How to Replicate

### Prerequisites

- Python 3.10 or higher (developed on 3.12.4)
- Windows 10/11

```bash
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python tqdm openpyxl
```

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/shrutirout/alzheimers-pca-svm.git
cd alzheimers-pca-svm
```

---

### Step 2: Download the Dataset

Download from Kaggle:

> https://www.kaggle.com/datasets/yiweilu2033/well-documented-alzheimers-dataset

The zip has a nested structure where each class folder contains another folder with the same name. You need to collapse it so the PNG files sit directly inside each class folder:

```
MildDemented/
    OAS1_0028_MR1_1.nii_slice_130.png
    ...
ModerateDemented/
    OAS1_0308_MR1_1.nii_slice_130.png
    ...
```

Also keep the metadata Excel file (`oasis_cross-sectional-*.xlsx`) alongside the folders.

---

### Step 3: Place the Dataset

Put the four class folders and the Excel file inside `Data/`:

```
alzheimers-pca-svm/
    Data/
        MildDemented/
        ModerateDemented/
        NonDemented/
        VeryMildDemented/
        oasis_cross-sectional-5708aa0a98d82080 (1).xlsx
        processed/        (already in repo, do not delete)
```

---

### Step 4: Run the Preprocessing Script

```bash
python src/preprocess.py
```

This selects one representative slice per subject, resizes to 128x128, normalises pixel values, applies augmentation to the Moderate class, and saves everything to `Data/processed/`.

Expected output:
```
Done: 442 samples saved, 0 subjects skipped.

Non-Demented          336
Very Mild Demented     70
Mild Demented          28
Moderate Demented       8
```

---

### Step 5: Run the Notebooks in Order

Launch Jupyter from the project root:

```bash
jupyter notebook
```

Run each notebook top to bottom in this order:

| Notebook | What it produces |
|---|---|
| `phase1_environment_setup.ipynb` | `class_distribution.png`, `metadata_processed.csv` |
| `phase2_mri_preprocessing.ipynb` | `X.npy`, `y.npy`, `sample_slices.png` |
| `phase3_pca_reduction.ipynb` | `X_pca.npy`, `pca_model.pkl`, variance curve, eigenbrains |
| `phase4_splitting_augmentation.ipynb` | `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `class_weights.json` |

---

## What is Already in the Repository

You do not need to run anything to view these. They are already committed:

- `outputs/plots/` : class distribution, sample slices, PCA variance curve, eigenbrains, train distribution
- `outputs/pca_components/` : eigenbrain_1, eigenbrain_2, eigenbrain_3
- `outputs/metrics/class_weights.json`
- `Data/processed/refined_images/` : 442 preprocessed PNGs (one per subject)
- `Data/processed/mapping.csv` and `metadata_processed.csv`
- `Data/processed/X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`

---

## Common Issues

**ModuleNotFoundError**: run `pip install <missing_module>` and retry.

**Images not loading**: confirm PNGs are directly inside each class folder, not inside a subfolder.

**Excel file not found**: make sure the `.xlsx` file is inside `Data/` with its full original filename.

**Jupyter cannot find notebooks**: always launch `jupyter notebook` from the project root, not from inside the `notebooks/` folder.
