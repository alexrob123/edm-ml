import io
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import torch
from PIL import Image

logger = logging.getLogger(__name__)

COLOR, COLORS = "k", ["firebrick", "navy"]
SUPP_COLOR, SUPP_COLORS = "k", ["r", "b"]
FILL_COLOR, FILL_COLORS = "k", ["r", "b"]
LINESTYLE_P, LINESTYLE_Q = "-", "--"

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.left"] = True
plt.rcParams["axes.spines.bottom"] = True
plt.rcParams["axes.grid"] = False
plt.rcParams["grid.alpha"] = 0.2
plt.rcParams["font.size"] = 16
plt.rcParams["legend.framealpha"] = 0.0
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["xaxis.labellocation"] = "center"
plt.rcParams["yaxis.labellocation"] = "center"
plt.rcParams["legend.fontsize"] = "x-small"
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern Roman"],
    }
)


def plot_4x4(imgs, labels=None, save_path=None):
    """
    Args:
        imgs: A list of path to img files, numpy arrays, or a torch.Tensor
        labels: Optional list of labels
    """

    ### handle torch.Tensor ###

    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu()

        if imgs.ndim != 4:
            raise ValueError("Tensor imgs must have shape (B, C, H, W) or (B, H, W, C)")

        # Convert (B, C, H, W) → (B, H, W, C)
        if imgs.shape[1] in {1, 3}:
            imgs = imgs.permute(0, 2, 3, 1)

        imgs = imgs.numpy()

        # Normalize to [0, 1] if needed
        if imgs.max() > 1.0:
            imgs = imgs / 255.0

    ### plot ###

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")

        if i >= len(imgs):
            continue

        img = imgs[i]

        if isinstance(img, (str, Path)):
            img = Image.open(img)
            ax.imshow(img)
        else:
            ax.imshow(img)

        # Set title only if labels are provided
        if labels is not None and i < len(labels):
            label = labels[i]
            if isinstance(label, Iterable) and not isinstance(label, (str)):
                ax.set_title(" / ".join(label))
            else:
                ax.set_title(label)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_4x4_from_dir(dir, json_file="dataset.json", save_path=None):
    dir = Path(dir).expanduser()

    # --- Load dataset.json ---
    with open(dir / json_file) as f:
        dataset = json.load(f)

    data = dataset["labels"]  # [(img_path, label), ...]

    # --- Find one example per label ---
    labels_all = [x[1] for x in data]
    num_labels = max(labels_all)

    indices = []
    for i in range(num_labels + 1):
        indices.append(labels_all.index(i))

    # --- Load images from disk ---
    imgs = []
    labels = list(range(num_labels + 1))

    for i in indices:
        img_path = dir / data[i][0]
        img = Image.open(img_path).convert("RGB")
        imgs.append(img)

    # --- Plot ---
    plot_4x4(imgs, labels, save_path=save_path)


def plot_4x4_from_zip(path, json_file="dataset.json", save_path=None):
    path = Path(path).expanduser()

    # --- Load dataset.json ---
    with ZipFile(path) as zf:
        with zf.open(json_file) as f:
            dataset = json.load(f)

    data = dataset["labels"]  # [(img_path, label), ...]

    # --- Find one example per label ---
    labels_all = [x[1] for x in data]
    num_labels = max(labels_all)

    indices = []
    for i in range(num_labels + 1):
        indices.append(labels_all.index(i))

    # --- Load images from zip ---
    imgs = []
    labels = list(range(num_labels + 1))

    with ZipFile(path) as zf:
        for i in indices:
            img_path = data[i][0]  # path INSIDE the zip
            with zf.open(img_path) as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
            imgs.append(img)

    # --- Plot ---
    plot_4x4(imgs, labels, save_path=save_path)


def plot_4x4_from_celeba_zip(dir, file, imgs, save_path=None):
    path = Path(dir).expanduser() / file

    pil_imgs = []

    with ZipFile(path) as zf:
        if imgs is None:
            imgs_ = [
                name
                for name in zf.namelist()
                if name.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            imgs = imgs_[:16]
        else:
            if imgs[0].split("/")[0] != "img_align_celeba":
                imgs = ["img_align_celeba/" + img_ for img_ in imgs]

        print(imgs)
        print(imgs[:16])

        for img in imgs:
            with zf.open(img) as f:
                img_ = Image.open(io.BytesIO(f.read())).convert("RGB")
            pil_imgs.append(img_)

    plot_4x4(pil_imgs, labels=None, save_path=save_path)
