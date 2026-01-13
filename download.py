import argparse
import os

import gdown


def main(args):
    data_dir = os.path.expanduser(args.dir)
    os.makedirs(data_dir, exist_ok=True)

    dataset_dir = os.path.join(data_dir, "CelebA64")
    os.makedirs(dataset_dir, exist_ok=True)

    img_file = os.path.join(dataset_dir, "img_align_celeba.zip")
    if os.path.exists(img_file):
        print(f"Img file already exists: {img_file}, skipping download.")
    else:
        print(f"Downloading CelebA images to {img_file} ...")
        gdown.download(
            "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684",
            img_file,
            quiet=False,
        )

    attr_file = os.path.join(dataset_dir, "list_attr_celeba.zip")
    if os.path.exists(attr_file):
        print(f"Attr file already exists: {attr_file}, skipping download.")
    else:
        print(f"Downloading CelebA attributes to {attr_file} ...")
        gdown.download(
            "https://drive.google.com/uc?id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",
            attr_file,
            quiet=False,
        )


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
    args = parser.parse_args()

    main(args)
