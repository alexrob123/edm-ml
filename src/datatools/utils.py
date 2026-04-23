import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

# Dataset utils
# ----------------------------------------------------------------------------------------------------


def extract_dataset_name(path, depth=1, suffix=None):
    path = Path(path).expanduser()

    if path.is_dir():
        name = path.name
    elif (
        path.suffix == ".zip"
        or path.suffix == ".json"
        or (suffix is not None and path.suffix == suffix)
    ):
        flag_dataset = "dataset" in path.name or "generated_images" in path.name
        flag_metrics = "metrics" in path.name or "eval" in path.name
        if flag_dataset or flag_metrics:
            parents = path.parents[:depth]
            name = "-".join(p.name for p in parents[::-1])
        else:
            name = path.stem
    else:
        raise ValueError(f"Unsupported data path: {path}")

    # return name.replace("-", "_")
    return name.lower()


def resolve_output(target, subdir=None, fname=None):
    target = Path(target).expanduser()

    # Treat paths without suffix as directories
    if target.suffix == "":
        outpath = target
        if subdir is not None:
            outpath = outpath / subdir

        outpath.mkdir(parents=True, exist_ok=True)

        if fname is not None:
            outpath = outpath / fname

        return outpath.as_posix()

    # File case
    if fname is None:
        raise ValueError("If target is a file, a default filename must be provided")

    if target.suffix != Path(fname).suffix:
        raise ValueError("Provided target extension does not match default extension")

    target.parent.mkdir(parents=True, exist_ok=True)

    return target.as_posix()


# ----------------------------------------------------------------------------------------------------


def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def read_dataset_meta(path):
    path = Path(path).expanduser()

    if path.is_dir():
        with open(Path(path) / "dataset.json", "r") as j:
            data = json.load(j)
    elif path.suffix == ".zip":
        with ZipFile(path) as z:
            with z.open("dataset.json", "r") as j:
                data = json.load(j)
    elif path.suffix == ".json":
        with open(path, "r") as j:
            data = json.load(j)
    else:
        raise ValueError(f"Unsupported data path: {path}")

    labels = [x[1] for x in data["labels"]]
    if isinstance(labels[0], (list, tuple)):
        labels = [tuple(int(v) for v in label) for label in labels]

    result = {"labels": labels}
    for key in ("labelsets", "labelspace"):
        if (value := data.get(key)) is not None:
            result[key] = value

    return result


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


# Hashing
# ----------------------------------------------------------------------------------------------------


def generate_hash(attrs: list[str]) -> str:
    """
    Generate a name and short hash from a list of attribute names.
    Deterministic: same list → same hash.
    """
    name = "_".join([s.replace("_", "") for s in sorted(attrs)])
    hash = hashlib.sha1(name.encode()).hexdigest()[:8]
    return hash


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
