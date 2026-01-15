import argparse
import logging
from pathlib import Path

from edm_ml.data_manager import read_metadata
from edm_ml.monitor import set_logging
from edm_ml.plotter import plot_4x4

logger = logging.getLogger(__name__)


def main(args):
    logger.info("### PLOTTER ###")

    dir = Path(args.data_dir).expanduser()
    attrs = args.attrs
    out_dir = Path(args.out_dir).expanduser()

    logger.info(f"Reading from {dir}")
    logger.info(f"Selection: {attrs}")
    logger.info(f"Saving plots in {out_dir}")

    metadata = read_metadata(dir, attrs)
    subdir = dir / metadata["hash"]

    count, imgs, labels = 0, [], []
    for k, v in metadata["combinations"].items():
        count += v["count"]
        imgs.append(dir / metadata["hash"] / k / v["images"][0])
        labels.append(v["attrs"])

    logger.info(f"Folder: {subdir}")
    logger.info(f"# combinations: {len(metadata['combinations'])}")
    logger.info(f"# images in metadata: {count}")

    plot_4x4(imgs, labels, save_dir=out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create folder tree with combinated-attribute subsets of CelebA64"
    )

    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="~/data/CelebA/AlignedCropped/JPG/_NonOverlappingClasses",
        help="Directory for reading data.",
    )
    parser.add_argument(
        "--attrs",
        "-a",
        type=str,
        nargs="+",
        default=[
            "Bangs",
            "Eyeglasses",
            "Male",
            "Smiling",
        ],
        help="Attributes to consider, as a space-separated list",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default="./output",
        help="Name of directory for storing plots",
    )

    args = parser.parse_args()

    set_logging()
    main(args)
