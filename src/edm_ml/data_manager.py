"""
Utilities for building non-overlapping subdatasets from dataset and attribute selection.
Utilities for building, reading and checking subdatasets metadata.

This module provides helper functions to:
- Read attribute data
- Create subsets of attribute combinations from attribute selection
- Generate deterministic hashes for attribute combinations
- Maintain metadata.json files for subsets derived from attribute selection

Tree
    data_dir
    ├── selection_1_hash_dir
    │   ├── combination_1_hash_dir
    │   │   ├── 000003.jpg
    │   │   └── 000007.jpg
    │   ├── combination_2_hash_dir
    │   │   ├── 000001.jpg
    │   │   └── 000002.jpg
    │   └── metadata.json
    ├── selection_2_hash_dir
    ...

Metadata
    ```json
        {
            "hash": "f3b2a1c9",
            "selection":  ["Smile", "Male", "Eyeglasses"],
            "combinations": {
                "a1b2c3d4": {                       /
                    "attrs": ["Smile", "Male"],
                    "count": 120,
                    "images": ["000001.jpg", "000005.jpg", ...]
                },
                "d4e5f6g7": {
                    "attrs": ["Smile"],
                    "count": 340,
                    "images": [...]
                }
            }
        }
    ```
"""

import csv
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def read_celeba_labels(path: Path):
    path = Path(path).expanduser()
    logger.info(f"Reading labels from {path}")

    with open(path) as csvf:
        data = list(csv.reader(csvf, delimiter=" ", skipinitialspace=True))

    label_names = data[1]
    data = data[1 + 1 :]

    img_fnames = [row[0] for row in data]
    img_fnames = ["img_align_celeba/" + img_fname for img_fname in img_fnames]

    labels = [row[1:] for row in data]
    labels_int = [list(map(int, i)) for i in labels]

    return label_names, img_fnames, torch.tensor(labels_int)


############
# METADATA #
############


def generate_hash(attrs: list[str]) -> str:
    """
    Generate a name and short hash from a list of attribute names.
    Deterministic: same list → same hash.
    """
    name = "_".join([s.replace("_", "") for s in sorted(attrs)])
    hash = hashlib.sha1(name.encode()).hexdigest()[:8]
    return hash


def build_metadata(
    dir: Path,
    sel_attrs: list[str],
    comb_attrs: list[str],
    images: list[str],
):
    logger.info("Building metada...")

    sel_hash = generate_hash(sel_attrs)
    comb_hash = generate_hash(comb_attrs)

    meta_file = Path(dir).expanduser() / sel_hash / "metadata.json"

    if meta_file.exists():
        with open(meta_file, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "hash": sel_hash,
            "selection": sel_attrs,
            "combinations": {},
        }

    metadata["combinations"][comb_hash] = {
        "attrs": comb_attrs,
        "count": len(images),
        "images": images,
    }

    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info("... done")


def read_metadata(dir, sel_attrs):
    dir = Path(dir).expanduser() / generate_hash(sel_attrs)
    meta_file = dir / "metadata.json"

    if meta_file.exists():
        with open(meta_file, "r") as f:
            metadata = json.load(f)
        return metadata
    else:
        raise FileNotFoundError(f"No metadata in directory {dir}")


def check_metada(dir, sel_attrs):
    logger.info("Checking metada...")

    sel_dir = Path(dir).expanduser() / generate_hash(sel_attrs)

    logger.info(f"\t Directory {sel_dir}")
    logger.info(f"\t Selection {sel_attrs}")

    metadata = read_metadata(dir, sel_attrs)
    combinations = metadata["combinations"]

    for hash in combinations.keys():
        attrs = combinations[hash]["attrs"]
        count = combinations[hash]["count"]
        imgs = combinations[hash]["images"]

        if not count == len(imgs):
            logger.warning("\t Corrupted metadata, len(imgs) \\ne count)")
            logger.info(f"\t\t Combi. dir {hash}")
            logger.info(f"\t\t Combi. attrs {attrs}")

        combin_dir = sel_dir / generate_hash(attrs)
        file_imgs = [f.name for f in combin_dir.iterdir() if f.is_file()]
        meta_imgs = imgs

        if not len(file_imgs) == len(meta_imgs):
            logger.warning("\t Unaligned file count and metadata")
            logger.info(f"\t\t File count: {len(file_imgs)}")
            logger.info(f"\t\t Meta count: {len(meta_imgs)}")

        file_imgs = set(file_imgs)
        meta_imgs = set(imgs)

        extra_file = file_imgs - meta_imgs
        extra_meta = meta_imgs - file_imgs

        if extra_meta or extra_file:
            logger.warning("\t Unaligned files and metadata")
            logger.info(f"\t\t {len(extra_file)} extra file: {sorted(extra_file)[:5]}")
            logger.info(f"\t\t {len(extra_meta)} extra meta: {sorted(extra_meta)[:5]}")

    logger.info("... done")


###########
# SUBSETS #
###########


def build_subset(
    sel_attrs: List[str],
    comb_attrs: List[str],
    names: List[str],
    imgs: List[str],
    data: torch.Tensor,
    img_dir: str,
    out_dir: str,
):
    logger.info(f"Building subset: {sorted(comb_attrs)} of {sorted(sel_attrs)}...")

    select_hash = generate_hash(sel_attrs)
    combin_hash = generate_hash(comb_attrs)

    src_dir = Path(img_dir).expanduser()
    dst_dir = Path(out_dir).expanduser() / select_hash / combin_hash
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\t Source directory: {src_dir}")
    logger.info(f"\t Target directory: {dst_dir}")

    # filter in labels in combination + in selection
    # filter out labels not in combination + in selection
    name_to_id = {attr: i for (i, attr) in enumerate(names)}
    indices_in = [name_to_id[attr] for attr in sel_attrs if attr in comb_attrs]
    indices_out = [name_to_id[attr] for attr in sel_attrs if attr not in comb_attrs]

    data = (data + 1) / 2  # -1/1 -> 0/1
    mask_in = data[:, indices_in].all(dim=1)
    mask_out = (data[:, indices_out] == 0).all(dim=1)
    mask = mask_in & mask_out
    expected_imgs = [img for i, img in enumerate(imgs) if mask[i]]

    # consistency check
    missing_imgs = check_subset(dst_dir, expected_imgs)
    if not missing_imgs:
        logger.info(f"\t Folder consistent ({len(expected_imgs)} images).")
    else:
        logger.info(f"\t Missing {len(missing_imgs)} images. Copying...")

    # copy missing images
    for img in tqdm(missing_imgs, desc="Copying data"):
        src_path = src_dir / img
        dst_path = dst_dir / img
        shutil.copy(src_path, dst_path)

    # consistency check
    missing_after = check_subset(dst_dir, expected_imgs)
    if missing_after:
        raise RuntimeError(
            f"Subset still inconsistent after copy "
            f"({len(missing_after)} images missing)"
        )
    else:
        logger.info(f"\t Subset created successfully ({len(expected_imgs)} images).")

    logger.info("... done")

    # metadata
    build_metadata(out_dir, sel_attrs, comb_attrs, expected_imgs)
    check_metada(out_dir, sel_attrs)


def check_subset(dir, expected_imgs, img_exts={".jpg", ".png"}):
    """
    Returns:
        missing_imgs (list[str]) : images that should exist but don't
    Raises:
        RuntimeError if extra images are found
    """
    dir = Path(dir).expanduser()

    existing_imgs = {
        f.name for f in dir.iterdir() if f.is_file() and f.suffix.lower() in img_exts
    }

    expected_imgs = set(expected_imgs)
    missing = expected_imgs - existing_imgs
    extra = existing_imgs - expected_imgs

    if extra:
        raise RuntimeError(
            f"Extra images found in {dir.name}: "
            f"{sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}"
            "Bruteforce solution: empty folder and rerun."
        )

    return sorted(missing)
