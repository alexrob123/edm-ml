import logging
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
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


def plot_4x4(imgs, labels, save_dir=None):
    """
    Args:
        imgs: A list of path to img files.
        labels: A list of labels
    """
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for ax, img, label in zip(axes.flatten(), imgs[:16], labels[:16]):
        img = Image.open(img)
        ax.imshow(img)
        ax.axis("off")
        if isinstance(label, Iterable) and not isinstance(label, (str)):
            ax.set_title(" / ".join(label))
        else:
            ax.set_title(label)

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / "partition_mosaic.png"
        fig.savefig(save_file, dpi=200, bbox_inches="tight")
        logger.info(f"Saved figure to {save_file}")
    else:
        plt.show()
