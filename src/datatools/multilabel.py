import json
from zipfile import ZipFile

import numpy as np
import torch

# --------------------------------------------------------------------------------
# Conversions


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


# --------------------------------------------------------------------------------
# Metadata


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


# --------------------------------------------------------------------------------
# Batch processing


def br_batch_processing(x, y):
    x = x.to(torch.float32) / 255.0
    return x, y


def lp_batch_processing(x, y):
    x = x.to(torch.float32) / 255.0
    if y is None:
        return x
    else:
        y = torch.argmax(y, dim=1)  # convert from one-hot to class index
        return x, y


BATCH_PROCESSING = {
    "br": br_batch_processing,
    "lp": lp_batch_processing,
}


# --------------------------------------------------------------------------------
# Prediction processing


def br_pred_processing(outputs):
    preds = torch.sigmoid(outputs.logits) > 0.5
    return preds


def lp_pred_processing(outputs):
    _, preds = torch.max(outputs.logits, 1)
    return preds


PRED_PROCESSING = {
    "br": br_pred_processing,
    "lp": lp_pred_processing,
}
