import argparse
import logging
from pathlib import Path

import gdown

logger = logging.getLogger(__name__)


def download_aligned_jpg(dir):
    logger.info("CelebA images, aligned & cropped, JPG format.")

    # Prepare folder
    base_dir = Path(dir).expanduser()
    out_dir = base_dir / "CelebA" / "AlignedCropped" / "JPG"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\tPreparing folder {out_dir}")

    # Check already downloaded and download
    file = out_dir / "img_align_celeba.zip"
    if file.exists():
        logger.info(f"\tFile already exists: {file}, skipping download.")
    else:
        logger.info(f"\tDownloading file to {out_dir}...")
        gdown.download(
            "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684",
            output=str(file),
            quiet=False,
        )


def download_aligned_png(dir):
    logger.info("CelebA images, aligned & cropped, PNG format.")

    # Prepare folder
    base_dir = Path(dir).expanduser()
    out_dir = base_dir / "CelebA" / "AlignedCropped" / "PNG"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\tPreparing folder {out_dir}")

    # Check existing 7z parts and download
    files = sorted(
        f.name
        for f in out_dir.iterdir()
        if f.name.startswith("img_align_celeba_png.7z")
    )
    if len(files) == 16:
        logger.info("\tAll 16 PNG parts already downloaded, skipping download.")
    else:
        logger.info(f"\tDownloading PNG parts to {out_dir}...")
        gdown.download_folder(
            "https://drive.google.com/drive/folders/0B7EVK8r0v71pbWNEUjJKdDQ3dGc?resourcekey=0-B5NA6Xcog-KfbFaNG5rUuQ",
            output=str(out_dir),
            quiet=False,
            use_cookies=False,
        )

    # List downloaded files
    files = sorted(
        f.name
        for f in out_dir.iterdir()
        if f.name.startswith("img_align_celeba_png.7z")
    )
    logger.info(f"\tPNG 7z parts downloaded: {files}")

    expected_parts = [f"img_align_celeba_png.7z.{i:03d}" for i in range(1, 17)]
    missing = [p for p in expected_parts if not (out_dir / p).exists()]
    if missing:
        logger.warning(f"Missing PNG parts: {missing}")


def download_att(dir):
    logger.info("CelebA attributes")

    base_dir = Path(dir).expanduser()
    file = base_dir / "CelebA" / "list_attr_celeba.txt"

    if file.exists():
        logger.info(f"\tAttr file already exists: {file}, skipping download.")
    else:
        logger.info(f"\tDownloading CelebA attributes to {file} ...")
        gdown.download(
            "https://drive.google.com/uc?id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",
            output=str(file),
            quiet=False,
        )


def main(args):
    # Download images
    if args.aligned:
        if args.extension in ["jpg", "jpeg", "JPG", "JPEG"]:
            download_aligned_jpg(args.dir)
        elif args.extension in ["png", "PNG"]:
            download_aligned_png(args.dir)
        else:
            raise ValueError(f"Extension {args.extension} not supported.")
    else:
        raise NotImplementedError("URL for original CelebA download are not defined.")

    # Download attributes
    download_att(args.dir)


# nohup uv run download.py --dir ~/data --aligned 1 --extension png > download.log 2>&1 &
# tail -f download.log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download CelebA aligned and cropped dataset."
    )

    parser.add_argument(
        "--dir",
        type=str,
        default="~/data",
        help="Data directory for download.",
    )
    parser.add_argument(
        "--aligned",
        type=int,
        choices=[0, 1],
        default=1,
        help="Flag for data version.",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Data extension for download",
    )

    args = parser.parse_args()

    main(args)
