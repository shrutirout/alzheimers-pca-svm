# Alzheimer's MRI Preprocessing Script
# Filtering OASIS-1 dataset to one representative slice per subject

import os
import re
import sys
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
REFINED_DIR = os.path.join(BASE_DIR, "data", "processed", "refined_images")
MAPPING_OUT = os.path.join(BASE_DIR, "data", "processed", "mapping.csv")
SKIPPED_OUT = os.path.join(BASE_DIR, "data", "processed", "skipped_subjects.txt")

# Configuration
TARGET_SLICE = 130
FALLBACK_RANGE = (125, 135)
IMG_SIZE = (128, 128)

CDR_TO_LABEL = {
    0.0: "Non-Demented",
    0.5: "Very Mild Demented",
    1.0: "Mild Demented",
    2.0: "Moderate Demented",
}

FOLDER_TO_CDR = {
    "NonDemented":      0.0,
    "VeryMildDemented": 0.5,
    "MildDemented":     1.0,
    "ModerateDemented": 2.0,
}

os.makedirs(REFINED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MAPPING_OUT), exist_ok=True)

# Loading OASIS metadata 
xlsx_files = glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
if not xlsx_files:
    sys.exit("ERROR: No xlsx file found in Data directory.")

df_meta = pd.read_excel(xlsx_files[0])[["ID", "CDR"]].set_index("ID")
print(f"Loaded metadata: {len(df_meta)} subjects")

# Building subject index 
# Maps subject_id -> {folder, slices: {slice_num: filepath}}
subject_index = {}

for cls_folder in FOLDER_TO_CDR:
    folder_path = os.path.join(DATA_DIR, cls_folder)
    if not os.path.isdir(folder_path):
        print(f"WARNING: Folder not found: {cls_folder}")
        continue
    for fname in os.listdir(folder_path):
        if not fname.endswith(".png"):
            continue
        m = re.match(r"(OAS1_\d+_MR\d+)_\d+\.nii_slice_(\d+)\.png", fname)
        if not m:
            continue
        subj_id = m.group(1)
        slice_num = int(m.group(2))
        if subj_id not in subject_index:
            subject_index[subj_id] = {"folder": cls_folder, "slices": {}}
        subject_index[subj_id]["slices"][slice_num] = os.path.join(folder_path, fname)

print(f"Indexed {len(subject_index)} unique subjects across all class folders")

# Processing subjects 
records = []
skipped = []

def save_image(arr, name):
    out_path = os.path.join(REFINED_DIR, name)
    cv2.imwrite(out_path, (arr * 255).astype(np.uint8))
    return out_path

for subj_id, info in tqdm(subject_index.items(), desc="Processing subjects"):
    cls_folder = info["folder"]
    slices = info["slices"]

    # Getting CDR: from CSV first, fallback to folder label
    raw_cdr = df_meta.loc[subj_id, "CDR"] if subj_id in df_meta.index else None
    if raw_cdr is None or (isinstance(raw_cdr, float) and np.isnan(raw_cdr)):
        cdr = FOLDER_TO_CDR[cls_folder]
    else:
        cdr = float(raw_cdr)
    label = CDR_TO_LABEL[cdr]

    # Selecting representative slice: target 130, fallback to closest available
    if TARGET_SLICE in slices:
        img_path = slices[TARGET_SLICE]
    else:
        img_path = slices[min(slices, key=lambda s: abs(s - TARGET_SLICE))]

    # Loading, resizing, normalizing
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        skipped.append(subj_id)
        continue
    img = cv2.resize(img, IMG_SIZE).astype(np.float32) / 255.0

    # Saving original slice
    out_path = save_image(img, f"{subj_id}.png")
    records.append({"file_path": out_path, "subject_id": subj_id,
                    "cdr_score": cdr, "label": label})

    # Augmenting Moderate Demented class only
    # FLAG: Only 2 Moderate subjects exist in this dataset, yielding ~8 total
    # Moderate samples after augmentation instead of the ~40 estimated in prompt3.md.
    # This will be revisited during SVM training and evaluation phases.
    if label == "Moderate Demented":
        # Horizontal flip
        aug = cv2.flip(img, 1)
        out_path = save_image(aug, f"{subj_id}_aug_flip.png")
        records.append({"file_path": out_path, "subject_id": subj_id,
                        "cdr_score": cdr, "label": label})

        h, w = img.shape
        for angle, suffix in [(3, "rot_p3"), (-3, "rot_n3")]:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            aug = cv2.warpAffine(img, M, (w, h))
            out_path = save_image(aug, f"{subj_id}_aug_{suffix}.png")
            records.append({"file_path": out_path, "subject_id": subj_id,
                            "cdr_score": cdr, "label": label})

# Saving outputs 
df_mapping = pd.DataFrame(records)
df_mapping.to_csv(MAPPING_OUT, index=False)

with open(SKIPPED_OUT, "w") as f:
    f.write("\n".join(skipped))

print(f"\nDone: {len(records)} samples saved, {len(skipped)} subjects skipped.")
print("\nLabel distribution:")
print(df_mapping["label"].value_counts().to_string())
print(f"\nMapping saved to: {MAPPING_OUT}")
print(f"Skipped log saved to: {SKIPPED_OUT}")
