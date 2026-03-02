import json
from zipfile import ZipFile

import numpy as np

# Metadata
# ----------------------------------------------------------------------------------------------------


def read_lp_dataset_meta(path):
    with ZipFile(path) as z:
        with z.open("dataset.json", "r") as j:
            data = json.load(j)

    labels = [x[1] for x in data["labels"]]
    labelsets = data["labelsets"]
    labelspace = data["labelspace"]

    return {"labels": labels, "labelsets": labelsets, "labelspace": labelspace}


def read_br_dataset_meta(path):
    raise NotImplementedError


# Convert
# ----------------------------------------------------------------------------------------------------


def mhe2labs(mhe, labelspace):
    mhe = np.asarray(mhe)
    labelspace = np.asarray(labelspace)

    is_1d = mhe.ndim == 1
    if is_1d:
        mhe = mhe[None, :]

    out = [[] for _ in range(mhe.shape[0])]
    rows, cols = np.where(mhe == 1)

    for r, c in zip(rows, cols):
        out[r].append(
            labelspace[c].item() if hasattr(labelspace[c], "item") else labelspace[c]
        )

    return out[0] if is_1d else out


def lp_to_bl(classes, label_space, class2labelset):
    """
    Convert Label Powerset classes to Binary Labelset

    Args:
        classes (list(int)): List of class id (e.g. 0, 1, ...)
        label_space (list): List of all possible labels (e.g. "Smiling", "Happy", ...)
        class2labelset (dict): Mapping from class id to their corresponding label sets
    """
    # Prepare vector of 0
    Y = np.zeros((len(classes), len(label_space)))

    # Map label to idx in label_space
    label2idx = {lab: i for i, lab in enumerate(label_space)}

    # Fill with ones in the right spots
    for i, c in enumerate(classes):
        indices = [label2idx[label] for label in class2labelset[c]]
        Y[i, indices] = 1

    return Y
