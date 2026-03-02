import csv
import itertools
import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

import click
import numpy as np

from datatools.utils import generate_hash, make_json_serializable

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# READ DATA #
# Supported datasets:
# - CelebA
# --------------------------------------------------------------------------------


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

    return label_names, img_fnames, np.asarray(labels_int, dtype=np.long)


READ_DATA = {
    "celeba": read_celeba_labels,
}

# --------------------------------------------------------------------------------
# STRUCTURE DATA
# Supported structuring:
# - Binaray Relevance
# - Label Powerset
# --------------------------------------------------------------------------------


def structure_data(src, dst, metadata):
    metadata = make_json_serializable(metadata)

    # Copy zip and add metadata.
    with tempfile.TemporaryDirectory() as tmp:
        metadata_path = Path(tmp) / "dataset.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        shutil.copyfile(src, dst)

        with zipfile.ZipFile(
            dst,
            mode="a",
            compression=zipfile.ZIP_DEFLATED,
        ) as zf:
            zf.write(metadata_path, arcname="dataset.json")


def structure_data_for_br(
    dataset,
    label_imgs,
    label_data,
    label_names,
    labelspace,
    dst_dir,
):
    """Copy data zip and add a dataset.json description of labels"""

    logger.info("Structuring dataset for MLC with Binary Relevance...")
    assert Path(dataset).suffix == ".zip", f"Dir mngmt not implem. {dataset}"

    # Target directory and file.
    labelspace = sorted(labelspace)
    labelspace_h = generate_hash(labelspace)

    src_dataset = dataset
    dst_dataset = Path(dst_dir) / f"BR-{labelspace_h}" / "dataset_raw.zip"

    if dst_dataset.exists():
        logger.warning(f"The target {dst_dataset} already exists. Not proceeding.")
        return
    else:
        dst_dataset.parent.mkdir(parents=True, exist_ok=True)

    # Handle labels.
    name2id = {label: i for (i, label) in enumerate(label_names)}
    labelspace_ids = [name2id[label] for label in labelspace]

    label_data = np.asarray(label_data, dtype=np.long)
    labelspace_data = label_data[:, labelspace_ids]
    labelspace_data = (labelspace_data + 1) / 2  # -1/1 -> 0/1

    # Build metadata.
    metadata = dict()
    metadata["labels"] = [[img, lab] for img, lab in zip(label_imgs, labelspace_data)]
    metadata["labelspace"] = labelspace
    metadata["labelspace_hash"] = labelspace_h

    # Build dataset.
    structure_data(src_dataset, dst_dataset, metadata)


def structure_data_for_lp(
    dataset,
    label_imgs,
    label_data,
    label_names,
    labelspace,
    dst_dir,
):
    """Build subset folders per labelset"""

    logger.info("Structuring dataset for MLC with Label Powersets")

    # Target directory and file.
    labelspace = sorted(labelspace)
    labelspace_h = generate_hash(labelspace)

    src_dataset = dataset
    dst_dataset = Path(dst_dir) / f"test-LP-{labelspace_h}" / "dataset_raw.zip"

    if dst_dataset.exists():
        logger.warning(f"The target {dst_dataset} already exists. Not proceeding.")
        return
    else:
        dst_dataset.parent.mkdir(parents=True, exist_ok=True)

    # Iterate over labelsets and fill img2lab map.
    name2id = {name: i for (i, name) in enumerate(label_names)}

    labelsets = []
    labelset_id = 0

    img2labelset = dict()

    label_data = np.asarray(label_data, dtype=np.long)
    data = (label_data + 1) / 2  # -1/1 -> 0/1

    for k in range(0, len(labelspace) + 1):
        for c in itertools.combinations(range(len(labelspace)), k):
            labelset = [labelspace[i] for i in c]

            idx_in = [name2id[lab] for lab in labelspace if lab in labelset]
            idx_out = [name2id[lab] for lab in labelspace if lab not in labelset]

            mask_in = data[:, idx_in].all(axis=1)
            mask_out = (data[:, idx_out] == 0).all(axis=1)
            mask = mask_in & mask_out

            idxs = mask.nonzero()[0].tolist()
            for i in idxs:
                img2labelset[label_imgs[i]] = labelset_id

            labelsets.append(labelset)
            labelset_id += 1

    assert len(img2labelset) == len(label_imgs)

    # Build metadata.
    metadata = dict()
    metadata["labels"] = [[img, lab] for img, lab in sorted(img2labelset.items())]
    metadata["labelsets"] = labelsets
    metadata["labelspace"] = labelspace
    metadata["labelspace_hash"] = labelspace_h

    # Build dataset.
    structure_data(src_dataset, dst_dataset, metadata)


STRUCTURE_DATA = {
    "br": structure_data_for_br,
    "lp": structure_data_for_lp,
}

################################################################################
################################################################################
################################################################################


@click.command()
@click.option(
    "--data",
    "-d",
    type=click.Path(file_okay=True, readable=True, resolve_path=True),
    metavar="DIR|ZIP",
    required=True,
    help="Dataset directory or zip.",
)
@click.option(
    "--labels",
    "-l",
    type=click.Path(file_okay=True, readable=True, resolve_path=True),
    metavar="FILE",
    required=False,
    help="Name of label file inside data directory.",
)
@click.option(
    "--selection",
    "-s",
    type=str,
    multiple=True,
    required=False,
    help="Labels to consider. Repeat option, e.g. -s Bangs -s Male.",
)
@click.option(
    "--method",
    "-m",
    type=click.Choice(["br", "lp"]),
    required=True,
    help="Multi-Label Learning method. Impacts the structuring of the dataset.",
)
@click.option(
    "--outdir",
    "-o",
    type=click.Path(file_okay=False, writable=True, resolve_path=True),
    required=True,
    help="Directory for writing structured dataset.",
)

# main

def main(data, labels, selection, method, outdir):
    logger.info("Structure dataset")
    logger.info("----------------------------------------")
    logger.info("Structuration before feeding to EDM dataset_tool.py")

    logger.info(f"Reading img from: {data}")
    logger.info(f"Reading lab from: {labels}")

    # Read attributes.
    if "celeba" in str(data).lower():
        label_names, img_fnames, label_data = READ_DATA["celeba"](labels)

    logger.info(f"Labels in dataset: \n{label_names}\n")

    # Handle labelspace.
    labelspace = selection
    if labelspace is None:
        labelspace = label_names
    else:
        for label in labelspace:
            if label not in label_names:
                raise ValueError(f"Unexpected label: {label}")
    labelspace = sorted(labelspace)

    logger.info(f"Selected labelspace: \n{labelspace}\n")

    # Structure dataset.
    STRUCTURE_DATA[method](
        data,
        img_fnames,
        label_data,
        label_names,
        labelspace,
        outdir,
    )

    logger.info("...done!")


if __name__ == "__main__":
    main()
