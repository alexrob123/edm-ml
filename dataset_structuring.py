import argparse
import csv
import hashlib
import itertools
import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


# HELPER FUNCTIONS #


def generate_hash(attrs: list[str]) -> str:
    """
    Generate a name and short hash from a list of attribute names.
    Deterministic: same list → same hash.
    """
    name = "_".join([s.replace("_", "") for s in sorted(attrs)])
    hash = hashlib.sha1(name.encode()).hexdigest()[:8]
    return hash


# READ DATA #
# Supported datasets:
# - CelebA


def read_celeba_labels(path: Path):
    path = Path(path).expanduser()

    with open(path) as csvf:
        data = list(csv.reader(csvf, delimiter=" ", skipinitialspace=True))

    label_names = data[1]
    data = data[1 + 1 :]

    img_fnames = [row[0] for row in data]
    img_fnames = ["img_align_celeba/" + img_fname for img_fname in img_fnames]

    labels = [row[1:] for row in data]
    labels_int = [list(map(int, i)) for i in labels]

    return label_names, img_fnames, torch.tensor(labels_int)


# STRUCTURE DATA #
# Supported structuring:
# - Binaray Relevance
# - Label Powerset


def structure_data_for_br(
    dataset,
    label_imgs,
    label_data,
    label_names,
    label_space,
    dst_dir,
):
    """Copy data zip and add a dataset.json description of labels"""

    logger.info("Structuring dataset for MLC with Binary Relevance...")

    assert str(dataset).split(".")[-1] == "zip"

    # Target directory and file.
    label_space = sorted(label_space)
    label_space_h = generate_hash(label_space)

    src_dataset = dataset
    dst_dataset = dst_dir / (label_space_h + ".zip")

    if dst_dataset.exists():
        logger.warning(f"The target {dst_dataset} already exists. Not proceeding.")
        return

    name2id = {label: i for (i, label) in enumerate(label_names)}
    ls_id = [name2id[label] for label in label_space]
    ls_data = label_data[:, ls_id]
    ls_data = (ls_data + 1) / 2

    labels = [[i for i in range(len(label_space)) if row[i] == 1] for row in ls_data]

    # Build metadata.
    metadata = dict()
    metadata["labels"] = [[img, label] for img, label in zip(label_imgs, labels)]
    metadata["labelspace"] = label_space
    metadata["labelspace_h"] = label_space_h

    # Copy zip and add metadata.
    with tempfile.TemporaryDirectory() as tmp:
        metadata_path = Path(tmp) / "dataset.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        shutil.copyfile(src_dataset, dst_dataset)
        # subprocess.run(["zip", "-u", dst_dataset, "dataset.json"], cwd=tmp, check=True)

        with zipfile.ZipFile(
            dst_dataset,
            mode="a",
            compression=zipfile.ZIP_DEFLATED,
        ) as zf:
            zf.write(metadata_path, arcname="dataset.json")

    logger.info("...done!")


def structure_data_for_lp(
    dataset,
    label_imgs,
    label_data,
    label_names,
    label_space,
    dst_dir,
):
    """Build subset folders per labelset"""

    logger.info("Structuring dataset for MLC with Label Powersets")

    # Target directory and file.
    label_space = sorted(label_space)
    label_space_h = generate_hash(label_space)

    src_dataset = dataset
    dst_dataset = dst_dir / (label_space_h + ".zip")

    if dst_dataset.exists():
        logger.warning(f"The target {dst_dataset} already exists. Not proceeding.")
        return

    # Iterate over labelsets and fill img2lab map.
    name2id = {name: i for (i, name) in enumerate(label_names)}

    labelsets = []
    labelset_id = 0

    img2labelset = dict()

    data = (label_data + 1) / 2  # -1/1 -> 0/1

    for k in range(0, len(label_space) + 1):
        for c in itertools.combinations(range(len(label_space)), k):
            labelset = [label_space[i] for i in c]

            idx_in = [name2id[lab] for lab in label_space if lab in labelset]
            idx_out = [name2id[lab] for lab in label_space if lab not in labelset]

            mask_in = data[:, idx_in].all(dim=1)
            mask_out = (data[:, idx_out] == 0).all(dim=1)
            mask = mask_in & mask_out

            idxs = mask.nonzero(as_tuple=True)[0].tolist()
            for i in idxs:
                img2labelset[label_imgs[i]] = labelset_id

            labelsets.append(labelset)
            labelset_id += 1

    assert len(img2labelset) == len(label_imgs)

    # Build metadata.
    metadata = dict()
    metadata["labels"] = [[img, lab] for img, lab in sorted(img2labelset.items())]
    metadata["labelsets"] = labelsets
    metadata["labelspace"] = label_space
    metadata["labelspace_h"] = label_space_h

    # Copy zip and add metadata.
    with tempfile.TemporaryDirectory() as tmp:
        metadata_path = Path(tmp) / "dataset.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        shutil.copyfile(src_dataset, dst_dataset)
        # subprocess.run(["zip", "-u", dst_dataset, "dataset.json"], cwd=tmp, check=True)

        with zipfile.ZipFile(
            dst_dataset,
            mode="a",
            compression=zipfile.ZIP_DEFLATED,
        ) as zf:
            zf.write(metadata_path, arcname="dataset.json")

    logger.info("...done!")


# MAIN #


def main(args):
    logger.info("### STRUCTURING DATASET ###")
    logger.info("Structuration before feeding to EDM dataset_tool.py")

    # Source.
    data_dir = Path(args.data_dir).expanduser()
    dataset = data_dir / args.dataset
    labels = data_dir / args.labels

    logger.info(f"Reading from {data_dir}")
    logger.info(f"\timg from: {dataset}")
    logger.info(f"\tlab from: {labels}")

    # Destination.
    dir_names = {
        "br": data_dir / "ml-br",
        "lp": data_dir / "ml-lp",
    }
    dst_dir = Path(dir_names[args.method])
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing to {dst_dir}")

    # Read attributes.
    # FIX: add management for different datasets
    lab_names, img_fnames, lab_data = read_celeba_labels(labels)
    lab_space = args.labelspace

    logger.info(f"Labels: \n{lab_names}\n")
    logger.info(f"Selected labelspace: \n{lab_space}\n")

    for lab in lab_space:
        if lab not in lab_names:
            raise ValueError(f"Unexpected label: {lab}")

    # Structure dataset.
    structure_data = {
        "br": structure_data_for_br,
        "lp": structure_data_for_lp,
    }
    structure_data[args.method](
        dataset,
        img_fnames,
        lab_data,
        lab_names,
        lab_space,
        dst_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare dataset for selected learning task "
        + "before feeding to EDM dataset_tool.py",
    )

    parser.add_argument(
        "--data-dir",
        "-dd",
        type=str,
        default="~/data/CelebA/",
        help="Directory for reading data.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="img_align_celeba.zip",
        help="Dataset, directory or zip.",
    )
    parser.add_argument(
        "--labels",
        "-l",
        type=str,
        default="list_attr_celeba.txt",
        help="Name of label file inside data directory.",
    )
    parser.add_argument(
        "--labelspace",
        "-ls",
        type=str,
        nargs="+",
        default=[  # FIX: default should be all
            "Bangs",
            "Eyeglasses",
            "Male",
            "Smiling",
        ],
        help="Labels to consider, as a space-separated list.",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        required=True,
        choices=["br", "lp"],
        help="Multi-Label Learning method. Impacts the structuring of the dataset.",
    )

    args = parser.parse_args()

    main(args)
