import argparse
import itertools
import logging
from pathlib import Path

from edm_ml.data_manager import build_subset, read_attr
from edm_ml.monitor import set_logging

logger = logging.getLogger(__name__)


def main(args):
    data_dir = Path(args.data_dir).expanduser()
    img_dir = data_dir / args.img_dir
    att_file = data_dir / args.attr_file
    logger.info(f"Reading from {data_dir}")
    logger.info(f"\timg from: {img_dir}")
    logger.info(f"\tatt from: {att_file}")

    out_dir = data_dir / args.out_dir
    logger.info(f"Writing to {out_dir}")

    # Read attributes
    attr_names, attr_imgs, attr_data = read_attr(att_file)
    sel_attrs = args.attrs
    logger.info(f"Avail. attributes: \n{attr_names}\n")
    logger.info(f"Selec. attributes: \n{sel_attrs}\n")
    for a in sel_attrs:
        if a not in attr_names:
            raise ValueError(f"Unexpected attribute: {a}")

    # Build subset folders
    logger.info(f"Creating subsets of {img_dir} \n\t in {out_dir}")
    for k in range(0, len(args.attrs) + 1):
        for c in itertools.combinations(range(len(args.attrs)), k):
            comb_attrs = [sel_attrs[i] for i in c]

            build_subset(
                sel_attrs,
                comb_attrs,
                attr_names,
                attr_imgs,
                attr_data,
                img_dir=img_dir,
                out_dir=out_dir,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create folder tree with combinated-attribute subsets of CelebA64"
    )

    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="~/data/CelebA/AlignedCropped/JPG",
        help="Directory for reading data.",
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default="img_align_celeba",
        help="Name of image directory inside data directory.",
    )
    parser.add_argument(
        "--attr-file",
        type=str,
        default="list_attr_celeba.txt",
        help="Name of attr file inside data directory.",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default="_NonOverlappingClasses",
        help="Name of directory for storing subset data folders inside data directory",
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

    args = parser.parse_args()

    set_logging()
    main(args)
