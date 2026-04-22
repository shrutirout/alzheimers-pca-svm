# Quantifying Neuroanatomical Atrophy
### A PCA-SVM Pipeline for Multi-Stage Alzheimer's Classification

This project builds a machine learning pipeline to classify Alzheimer's Disease into four stages using MRI brain scans from the OASIS-1 dataset. Each scan is reduced from 16,384 raw pixel values to 233 principal components using PCA, and a Support Vector Machine is trained on those components to classify patients into one of four dementia stages: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented.

The approach is deliberately resource-efficient , no deep learning, no GPU, and only a few hundred samples. The full pipeline runs on a standard laptop CPU.

---

## Pipeline Overview

| Phase | Description | Status |
|---|---|---|
| Stage 0 | Preprocessing script , slice selection, normalisation, pixel-space augmentation | Done |
| Phase 1 | Environment setup, metadata loading, class distribution | Done |
| Phase 2 | MRI image loading, grayscale normalisation, flattening to feature vectors | Done |
| Phase 3 | StandardScaler + PCA dimensionality reduction, eigenbrain visualisation | Done |
| Phase 4 | Stratified train/test split, PCA-space augmentation, class weight computation | Done |
| Phase 5 | SVM training , Linear and RBF kernels with GridSearchCV (5-fold CV) | Done |
| Phase 6 | Evaluation , classification report, confusion matrix, ROC curves, PCA projection | Done |
| Final | Experiment summary and configuration export | Done |

---

## Key Design Decisions

**One slice per subject.** The dataset contains up to 180 axial slices per person. Using multiple slices causes subject leakage , the model learns to recognise the person's anatomy rather than the disease. We select one slice per subject, targeting index 130 (mid-axial plane, captures the hippocampus), or the closest available with no range restriction.

**CDR label inference for 201 subjects.** The OASIS-1 metadata Excel file has CDR = None for 201 of 436 subjects. Dropping them would leave only 235 samples , unworkable for a 4-class problem. Instead, we infer labels from the folder the dataset author placed them in, since the author sorted all subjects by CDR. The CSV value takes precedence; folder is fallback only.

**Two rounds of augmentation for minority classes.** The Moderate class has only 2 real subjects in OASIS-1, and Mild has 28.
- `preprocess.py`: pixel-space augmentation (horizontal flip + ±3° rotation) brings Moderate from 2 to 8 samples
- Phase 4: PCA-space augmentation (Gaussian noise + ×0.98/×1.02 scaling) on the training set only brings Moderate to 30 and Mild to 65 training samples

**Augmentation only after splitting.** The test set is locked before any augmentation. It contains only original, unmodified samples , ensuring a clean, honest evaluation.

**5-fold cross-validation.** With 30 Moderate training samples, 10-fold CV gives roughly 3 per fold , too few for stable SVM optimisation. 5-fold gives approximately 6 per fold, which is the minimum viable count.

**Macro F1 as the primary metric.** Accuracy is misleading on this dataset , predicting Non-Demented for every sample yields 76% accuracy while being clinically useless. Macro F1 weights all four classes equally and is the honest measure of performance.

---

## Results

Both models were evaluated on a clean, unaugmented test set of 89 samples.

| Model | Accuracy | Macro F1 | Best Params |
|---|---|---|---|
| Linear SVM | 0.69 | 0.38 | C = 0.1 |
| RBF SVM | 0.76 | 0.52 | C = 100, gamma = scale |

**RBF SVM is the better model.** It correctly identified 96% of Non-Demented cases and 14% of Very Mild cases. Mild Demented recall is 0 on both models , a known dataset limitation. With only 28 real Mild subjects against 336 Non-Demented, the SVM cannot learn a reliable boundary regardless of augmentation or class weighting. This is reported honestly rather than obscured.

The Moderate class has 1 test sample. Its per-class metrics are not statistically meaningful and should be interpreted from the confusion matrix, not the classification report.

Our RBF result of 76% is consistent with and marginally above the best published PCA+SVM result on comparable OASIS data (74.21% with linear SVM, reported in literature). Papers claiming 90%+ use 14× more data and deep learning.

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

The zip has a nested folder structure. Collapse it so PNG files sit directly inside each class folder:

```
Data/
    MildDemented/
        OAS1_0028_MR1_1.nii_slice_130.png
        ...
    ModerateDemented/
    NonDemented/
    VeryMildDemented/
    oasis_cross-sectional-*.xlsx
    processed/        ← already in repo, do not delete
```

---

### Step 3: Run the Preprocessing Script

```bash
python src/preprocess.py
```

Selects one slice per subject, resizes to 128×128, normalises pixel values, applies pixel-space augmentation to the Moderate class, and saves everything to `Data/processed/`.

Expected output:
```
Done: 442 samples saved, 0 subjects skipped.

Non-Demented          336
Very Mild Demented     70
Mild Demented          28
Moderate Demented       8
```

---

### Step 4: Run the Notebooks in Order

Launch Jupyter from the project root:

```bash
jupyter notebook
```

| Notebook | What it produces |
|---|---|
| `phase1_environment_setup.ipynb` | `class_distribution.png`, `metadata_processed.csv` |
| `phase2_mri_preprocessing.ipynb` | `X.npy`, `y.npy`, `sample_slices.png` |
| `phase3_pca_reduction.ipynb` | `X_pca.npy`, `pca_model.pkl`, variance curve, eigenbrains |
| `phase4_splitting_augmentation.ipynb` | `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `class_weights.json` |
| `phase5_svm_training.ipynb` | `svm_linear.pkl`, `svm_rbf.pkl`, `cv_scores.csv` |
| `phase6_evaluation.ipynb` | `classification_report.txt`, `confusion_matrix.png`, `roc_curves.png`, `pca_projection.png`, `component_importance.png` |

---

## What is Already in the Repository

The following outputs are pre-committed and do not require re-running anything to view:

- `outputs/plots/` , class distribution, sample slices, PCA variance curve, eigenbrains, train distribution, confusion matrix, ROC curves, PCA projection, component importance
- `outputs/pca_components/` , eigenbrain_1, eigenbrain_2, eigenbrain_3
- `outputs/metrics/` , `class_weights.json`, `cv_scores.csv`, `classification_report.txt`
- `reports/` , `experiment_summary.csv`, `configuration.json`
- `Data/processed/refined_images/` , 442 preprocessed PNGs, one per subject
- `Data/processed/` , `mapping.csv`, `metadata_processed.csv`, `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`

Note: `X.npy`, `X_pca.npy`, `y.npy`, and trained model `.pkl` files are excluded from the repo , they are regenerated by running the notebooks.

---

## Common Issues

**ModuleNotFoundError** , run `pip install <missing_module>` and retry.

**Images not loading** , confirm PNGs are directly inside each class folder, not inside a nested subfolder.

**Excel file not found** , the `.xlsx` file must be inside `Data/` with its full original filename.

**Jupyter cannot find notebooks** , always launch `jupyter notebook` from the project root, not from inside the `notebooks/` folder.
