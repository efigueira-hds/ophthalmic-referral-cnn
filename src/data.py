"""Image loading, patient/label derivation, report join, split and results.

Extracted from the original monolith notebook: id_paciente was duplicated
(EDA cell and split cell), the CSV join used raw Windows backslash paths, and
the patient-grouped split lived inline. No behavioural change versus the
original inline code -- same values, now importable and shared.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from sklearn.preprocessing import label_binarize

CLASS_FOLDERS = [
    (0, "cataract"),
    (1, "diabetic_retinopathy"),
    (2, "normal"),
]
CLASS_NAMES = ["Cataract", "Diabetic Retinopathy", "Normal"]


def id_paciente(nome):
    """Patient id from filename: '1115_left.jpg' / '1115_right.jpg' -> '1115'."""
    base = nome.rsplit(".", 1)[0]
    base = re.sub(r'_(left|right)$', '', base, flags=re.I)
    return base


def load_images_from_folder(folder, label):
    """Load every image in `folder`, padded to 224x224, tagged with `label`."""
    images = []
    labels = []
    nomes = []
  
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert("RGB")

        
        img = ImageOps.pad(img, (224, 224), color=(0, 0, 0))

        images.append(np.array(img))
        labels.append(label)
        nomes.append(filename)
    return images, labels, nomes


def get_class_folders(data_dir):
    """label -> (class_name, folder_path), the three ocular/ subfolders."""
    data_dir = Path(data_dir)
    return {
        label: (name, str(data_dir / "ocular" / name))
        for label, name in CLASS_FOLDERS
    }


def load_dataset(data_dir):
    """Load all three classes, label derived from the subfolder.

    Same concatenation order the monolith used (cataract, diabetic_retinopathy,
    normal), which is what makes groups/X_original alignment structural.
    Returns X_original, y, nomes_ficheiros, labels_ficheiros, folders.
    """
    folders = get_class_folders(data_dir)

    all_images, all_labels, all_files = [], [], []
    counts_by_label = []
    for label, (_name, folder) in folders.items():
        images, labels, files = load_images_from_folder(folder, label)
        all_images += images
        all_labels += labels
        all_files += files
        counts_by_label.append((label, len(files)))

    X_original = np.array(all_images)
    y = np.array(all_labels)
    nomes_ficheiros = np.array(all_files)
    labels_ficheiros = np.concatenate(
        [np.full(count, label) for label, count in counts_by_label]
    )

    return X_original, y, nomes_ficheiros, labels_ficheiros, folders


def build_groups(nomes_ficheiros):
    """Patient id per file, built from the SAME filename sequence that
    produced X_original -- keeps groups aligned by construction rather than
    by re-listing folders (which could reorder independently)."""
    return np.array([id_paciente(f) for f in nomes_ficheiros])


def split_patient_grouped(X, y, groups, seed, n_splits_test=5, n_splits_val=4):
    """Two-stage patient-grouped, class-stratified split (~60/20/20).

    A patient's images never end up split across train/val/test. All
    returned indices are ABSOLUTE positions into X/y/groups, so callers in
    any notebook can slice X_original directly (X_original[train_idx],
    etc.) without an intermediate trainval-relative step.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sgkf.split(X, y, groups))

    groups_trainval = groups[trainval_idx]
    sgkf2 = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)
    train_rel_idx, val_rel_idx = next(
        sgkf2.split(X[trainval_idx], y[trainval_idx], groups_trainval)
    )

    train_idx = trainval_idx[train_rel_idx]
    val_idx = trainval_idx[val_rel_idx]

    return {
        "trainval_idx": trainval_idx,
        "test_idx": test_idx,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


def save_split_indices(path, split):
    np.savez(
        path,
        trainval_idx=split["trainval_idx"],
        test_idx=split["test_idx"],
        train_idx=split["train_idx"],
        val_idx=split["val_idx"],
    )


def load_split_indices(path):
    npz = np.load(path)
    return {k: npz[k] for k in ("trainval_idx", "test_idx", "train_idx", "val_idx")}


def slug(nome):
    """Model name -> stable filename slug, e.g. for artifacts/preds/{slug}.npy."""
    s = nome.lower()
    for ch in " ()-/":
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def load_reports(csv_path):
    """Load ocular_reports.csv, normalising Windows-style paths to forward
    slashes -- the CSV was produced on Windows and image<->report joins
    would silently fail on the Colab Linux runtime otherwise."""
    df = pd.read_csv(csv_path)
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)
    df["filename"] = df["image_path"].apply(lambda p: p.split("/")[-1])
    return df


def join_images_reports(nomes_ficheiros, df_reports):
    """Align reports to the image loading order.

    Asserts the join is complete in both directions: no image left without a
    report, and no report left over that doesn't match an image.
    """
    df_aligned = pd.DataFrame({"filename": list(nomes_ficheiros)})
    df_aligned = df_aligned.merge(df_reports, on="filename", how="left")

    assert len(df_aligned) == len(nomes_ficheiros), \
        f"aligned reports ({len(df_aligned)}) != images ({len(nomes_ficheiros)})"

    missing_reports = df_aligned["report"].isna().sum()
    assert missing_reports == 0, \
        f"{missing_reports} images have no matching report after the join"

    image_filenames = set(nomes_ficheiros)
    report_filenames = set(df_reports["filename"])
    orphan_reports = report_filenames - image_filenames
    assert not orphan_reports, \
        f"{len(orphan_reports)} report rows match no image: {sorted(orphan_reports)[:5]}"

    return df_aligned


def compute_metrics_row(nome, proba_test, y_test, class_names):
    """Test-set metrics for one model -- one row of the comparative table."""
    y_pred = np.argmax(proba_test, axis=1)
    y_bin = label_binarize(y_test, classes=list(range(len(class_names))))

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    model_auc = roc_auc_score(y_bin, proba_test, average="macro", multi_class="ovr")
    recalls = recall_score(y_test, y_pred, average=None)

    row = {"Modelo": nome, "Accuracy": acc, "F1-macro": f1, "AUC-ROC macro": model_auc}
    for cls_name, rec in zip(class_names, recalls):
        row[f"Recall {cls_name}"] = rec
    return row


def append_results_row(row, results_path):
    """Read-modify-write artifacts/results.csv: upsert this model's row.

    02/03/04 run as separate Colab sessions, so the comparative table can
    only be assembled incrementally on disk -- there is no in-memory
    registry that survives across notebooks.
    """
    results_path = Path(results_path)
    if results_path.exists():
        df = pd.read_csv(results_path)
        df = df[df["Modelo"] != row["Modelo"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(results_path, index=False)
    return df
