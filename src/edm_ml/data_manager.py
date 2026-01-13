import csv
import logging
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def read_attr(path: Path):
    path = Path(path).expanduser()
    logger.info(f"Reading attributes from {path}")

    with open(path) as csv_file:
        data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

    headers = data[1]
    data = data[1 + 1 :]

    indices = [row[0] for row in data]
    data = [row[1:] for row in data]
    data_int = [list(map(int, i)) for i in data]

    return headers, indices, torch.tensor(data_int)


def check_subset_consistency(dest_dir, expected_imgs, img_exts={".jpg", ".png"}):
    """
    Returns:
        missing_imgs (list[str]) : images that should exist but don't
    Raises:
        RuntimeError if extra images are found
    """
    dest_dir = Path(dest_dir)

    existing_imgs = {
        f.name
        for f in dest_dir.iterdir()
        if f.is_file() and f.suffix.lower() in img_exts
    }

    expected_imgs = set(expected_imgs)

    missing = expected_imgs - existing_imgs
    extra = existing_imgs - expected_imgs

    if extra:
        raise RuntimeError(
            f"Extra images found in {dest_dir.name}: "
            f"{sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}"
            "Bruteforce solution: empty folder and rerun."
        )

    return sorted(missing)


def create_subset(selection, names, imgs, data, img_dir, out_dir):
    img_dir = Path(img_dir).expanduser()
    out_dir = Path(out_dir).expanduser()

    dest_dir_name = "_".join([s.replace("_", "") for s in sorted(selection)])
    dest_dir = out_dir / dest_dir_name
    logger.info(f"Subset: {dest_dir_name}")

    name_to_id = {attr: i for (i, attr) in enumerate(names)}
    indices = [name_to_id[attr] for attr in selection]

    data = (data + 1) / 2  # -1/1 -> 0/1
    mask = data[:, indices].all(dim=1)
    expected_imgs = [img for i, img in enumerate(imgs) if mask[i]]

    dest_dir.mkdir(parents=True, exist_ok=True)

    # initial consistency check
    missing_imgs = check_subset_consistency(dest_dir, expected_imgs)
    if not missing_imgs:
        logger.info(f"\tFolder consistent ({len(expected_imgs)} images).")
        return dest_dir_name, None
    else:
        logger.info(f"\tMissing {len(missing_imgs)} images. Repairing...")

    # copy only missing images
    for img in tqdm(missing_imgs, desc="Copying missing data"):
        src_path = img_dir / img
        dst_path = dest_dir / img
        shutil.copy(src_path, dst_path)

    # final consistency check
    missing_after = check_subset_consistency(dest_dir, expected_imgs)
    if missing_after:
        raise RuntimeError(
            f"Subset '{dest_dir_name}' still inconsistent after copy "
            f"({len(missing_after)} images missing)"
        )
    else:
        logger.info(f"\tSubset repaired successfully ({len(expected_imgs)} images).")

    return dest_dir_name, expected_imgs


def build_metadata(out_dir, count_file="subset_counts.txt", img_file="subset_imgs.txt"):
    out_dir = Path(out_dir).expanduser()
    logger.info(f"Building metadata for {out_dir}")

    counts = {}
    images = {}
    for folder in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        logger.info(f"\tProcessing {folder}")
        img_list = sorted(f.name for f in folder.iterdir() if f.is_file())
        counts[folder.name] = len(img_list)
        images[folder.name] = img_list

    count_path = out_dir / count_file
    img_path = out_dir / img_file

    logger.info(f"Writing file {count_path}")
    tmp_count = count_path.with_suffix(".tmp")
    with open(tmp_count, "w") as f:
        for folder in sorted(counts):
            f.write(f"{folder} {counts[folder]}\n")
    tmp_count.replace(count_path)

    logger.info(f"Writing file {img_path}")
    tmp_img = img_path.with_suffix(".tmp")
    with open(tmp_img, "w") as f:
        for folder in sorted(images):
            f.write(folder + " " + " ".join(images[folder]) + "\n")
    tmp_img.replace(img_path)


def load_counts(dir, file):
    path = Path(dir).expanduser() / file
    counts = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                folder, count = line.strip().split()
                counts[folder] = int(count)
    return counts


def load_imgs(dir, file):
    path = Path(dir).expanduser() / file
    images = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                images[parts[0]] = parts[1:]
    return images
