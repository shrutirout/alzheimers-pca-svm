# Quantifying Neuroanatomical Atrophy
## A PCA-SVM Pipeline for Multi-Stage Alzheimer's Classification

This repository contains a complete machine learning pipeline that classifies Alzheimer's Disease into four stages : **Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented** : using MRI brain scans from the OASIS-1 dataset. The approach uses PCA for dimensionality reduction and SVM for classification.

---

## Prerequisites

- **Python 3.10 or higher** (developed on Python 3.12.4)
- **Windows 10/11**
- ~5 GB free disk space (for the dataset and processed outputs)

Install all required libraries in one command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python tqdm openpyxl
```

---

## Step 1 : Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/shrutirout/alzheimers-pca-svm.git
cd alzheimers-pca-svm
```

---

## Step 2 : Download the Dataset

Download the dataset from:

> **[https://www.kaggle.com/datasets/yiweilu2033/well-documented-alzheimers-dataset]**

The downloaded zip contains a nested folder structure like this:

```
MildDemented/
    MildDemented/        ← files are inside a second folder with the same name
        OAS1_0028_MR1_1.nii_slice_130.png
        ...
ModerateDemented/
    ModerateDemented/
        ...
NonDemented/
    NonDemented/
        ...
VeryMildDemented/
    VeryMildDemented/
        ...
```

**You must collapse the nested folders.** Move the inner folder's contents up one level so the structure becomes flat : the PNG files should be directly inside each class folder, not inside a subfolder. After collapsing, it should look like:

```
MildDemented/
    OAS1_0028_MR1_1.nii_slice_130.png
    OAS1_0028_MR1_2.nii_slice_130.png
    ...
ModerateDemented/
    OAS1_0308_MR1_1.nii_slice_130.png
    ...
NonDemented/
    OAS1_0001_MR1_1.nii_slice_130.png
    ...
VeryMildDemented/
    OAS1_0002_MR1_1.nii_slice_130.png
    ...
```

The dataset also contains a metadata Excel file (`oasis_cross-sectional-*.xlsx`). Keep this inside the `Data/` folder as well.

---

## Step 3 : Place the Dataset

Place the four class folders and the Excel file inside the `Data/` folder in the cloned repository:

```
alzheimers-pca-svm/
    Data/
        MildDemented/          ← paste here
        ModerateDemented/      ← paste here
        NonDemented/           ← paste here
        VeryMildDemented/      ← paste here
        oasis_cross-sectional-5708aa0a98d82080 (1).xlsx   ← paste here
        processed/             ← will be populated by the scripts automatically
    notebooks/
    outputs/
    src/
    ...
```

> Note: The `Data/` folder holds both the raw dataset and the processed outputs (inside `Data/processed/`). All scripts write their outputs there automatically.

---

## Step 4: Run the Preprocessing Script

Open a terminal in the project root folder and run:

```bash
python src/preprocess.py
```

This script will:
- Read the metadata Excel file to get CDR scores for each subject
- Select one representative axial slice per subject (targeting slice index 130)
- Resize each slice to 128×128, normalise pixel values to [0, 1]
- Perform minority augmentation on the Moderate class (flip + rotate)
- Save all refined images to `Data/processed/refined_images/`
- Save `Data/processed/mapping.csv` : a table linking each image to its subject ID, CDR score, and label

**Expected output at the end:**

```
Done: 442 samples saved, 0 subjects skipped.

Label distribution:
label
Non-Demented          336
Very Mild Demented     70
Mild Demented          28
Moderate Demented       8
```

If any subjects are skipped, check `Data/processed/skipped_subjects.txt` for details.

---

## Step 5: Run the Notebooks in Order

Open Jupyter Notebook or JupyterLab from the project root:

```bash
jupyter notebook
```

Then run each notebook **from top to bottom**, in this exact order:

---

### Phase 1: `notebooks/phase1_environment_setup.ipynb`

**What it does:**
- Creates all output directories
- Loads `mapping.csv` and builds the metadata dataframe
- Saves a class distribution bar chart

**Expected outputs:**
- `outputs/plots/class_distribution.png`
- `Data/processed/metadata_processed.csv`

---

### Phase 2: `notebooks/phase2_mri_preprocessing.ipynb`

**What it does:**
- Reads all 442 refined PNGs from `Data/processed/refined_images/`
- Flattens each 128×128 image into a 16,384-element feature vector
- Builds the feature matrix X and label vector y

**Expected outputs:**
- `Data/processed/X.npy` : shape (442, 16384)
- `Data/processed/y.npy` : shape (442,)
- `outputs/plots/sample_slices.png`

---

### Phase 3: `notebooks/phase3_pca_reduction.ipynb`

**What it does:**
- Standardises X using StandardScaler
- Applies PCA to capture 95% of variance (reduces 16,384 features → 233 components)
- Visualises the top 3 principal components as "eigenbrains"

**Expected outputs:**
- `Data/processed/X_pca.npy` : shape (442, 233)
- `outputs/models/pca_model.pkl`
- `outputs/plots/pca_variance_curve.png`
- `outputs/pca_components/eigenbrain_1.png`, `eigenbrain_2.png`, `eigenbrain_3.png`

---

### Phase 4: `notebooks/phase4_splitting_augmentation.ipynb`

**What it does:**
- Performs a stratified 80/20 train/test split
- Augments the Moderate class in the **training set only** (in PCA space) from ~7 → 30 samples using Gaussian noise and feature scaling : this prevents data leakage
- Computes class weights to handle class imbalance during SVM training
- Plots the final training and test distributions

**Expected outputs:**
- `Data/processed/X_train.npy` : shape (376, 233)
- `Data/processed/X_test.npy` : shape (89, 233)
- `Data/processed/y_train.npy`, `Data/processed/y_test.npy`
- `outputs/metrics/class_weights.json`
- `outputs/plots/train_distribution.png`

**Expected training distribution (post-augmentation):**

| Class | Train | Test |
|---|---|---|
| Non-Demented | 268 | 68 |
| Very Mild Demented | 56 | 14 |
| Mild Demented | 22 | 6 |
| Moderate Demented | 30 | 1 |

---

## Full Directory Structure (after all steps complete)

```
alzheimers-pca-svm/
│
├── Data/                               ← raw dataset (you add this, not in repo)
│   ├── MildDemented/
│   ├── ModerateDemented/
│   ├── NonDemented/
│   ├── VeryMildDemented/
│   ├── oasis_cross-sectional-*.xlsx
│   └── processed/                      ← generated by scripts
│       ├── refined_images/
│       ├── mapping.csv
│       ├── metadata_processed.csv
│       ├── X.npy, y.npy
│       ├── X_pca.npy
│       ├── X_train.npy, X_test.npy
│       └── y_train.npy, y_test.npy
│
├── notebooks/
│   ├── phase1_environment_setup.ipynb
│   ├── phase2_mri_preprocessing.ipynb
│   ├── phase3_pca_reduction.ipynb
│   └── phase4_splitting_augmentation.ipynb
│
├── outputs/
│   ├── plots/
│   ├── models/
│   ├── metrics/
│   └── pca_components/
│
├── src/
│   └── preprocess.py
└── README.md
```

---

## Common Issues

**`ModuleNotFoundError`**: run `pip install <missing_module>` and retry.

**`cv2.imread` returns None / images not loading** : confirm the PNG files are directly inside the class folders (not nested in a subfolder). See Step 2.

**Excel file not found** : ensure the `.xlsx` file is inside `Data/` with its original filename intact.

**Jupyter can't find the notebooks** : launch `jupyter notebook` from the project root, not from inside the `notebooks/` folder.
