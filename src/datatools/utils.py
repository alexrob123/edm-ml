import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

# Dataset utils
# ----------------------------------------------------------------------------------------------------


def extract_dataset_name(path):
    path = Path(path).expanduser()

    if path.is_dir():
        name = path.name
    elif path.suffix == ".zip":
        if "dataset" in path.name or "generated_images" in path.name:
            name = path.parent.name
        else:
            name = path.stem
    else:
        raise ValueError(f"Unsupported data path: {path}")

    return name.replace("-", "_")


def zip_meta(path, fname="dataset.json"):
    with ZipFile(path) as z:
        with z.open(fname, "r") as j:
            data = json.load(j)
    return data


def zip_images(path, label=None, n=None):
    with ZipFile(path) as z:
        with z.open("dataset.json", "r") as j:
            data = json.load(j)

    if label is not None:
        imgs = [x[0] for x in data["labels"] if x[1] == label]
        imgs = imgs[:n] if n is not None else imgs
    else:
        imgs = [x[0] for x in data["labels"]]
        imgs = imgs[:n] if n is not None else imgs

    images = []
    for img in imgs:
        with z.open(img) as f:
            images.append(f.read())

    return images


def zip_labels(path):
    with ZipFile(path) as z:
        with z.open("dataset.json", "r") as j:
            data = json.load(j)

    labels = [x[1] for x in data["labels"]]
    return labels


# Json utils
# ----------------------------------------------------------------------------------------------------


def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]

    else:
        return obj
