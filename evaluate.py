import json
import os
import pickle
import time
from pathlib import Path

import click
import numpy as np
import scipy
import torch

import dnnlib
from evaluation.tpr import PCA, compute_top_pr
from torch_utils import distributed as dist
from training import classifier
from training.dataset import ImageFolderDataset

FLAG_INCEPTION_REF_PATH = "inception-ref"
FLAG_DINO_REF_PATH = "dino-ref"
INCEPTION_STATS_FILE = "inception_stats.npy"
DINO_STATS_FILE = "dino_stats.npy"


####################################################################################################
# Inception features and stats
####################################################################################################


def compute_inception_stats(
    image_path,
    num_expected=None,
    seed=0,
    max_batch_size=64,
    num_workers=3,
    prefetch_factor=2,
    device=torch.device("cuda"),
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load Inception-v3 model.
    # PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    dist.print0("Loading Inception-v3 model...")

    detector_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048

    with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = pickle.load(f).to(device)

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')

    dataset = ImageFolderDataset(
        path=image_path,
        max_size=num_expected,
        random_seed=seed,
        use_labels=True,
    )
    num_labels = dataset.label_dim
    dist.print0(f"Number of labels in the dataset: {num_labels}")

    if num_expected is not None and len(dataset) < num_expected:
        raise click.ClickException(
            f"Found {len(dataset)} images, but expected at least {num_expected}"
        )
    if len(dataset) < 2:
        raise click.ClickException(
            f"Found {len(dataset)} images, but need at least 2 to compute statistics"
        )

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = (
        (len(dataset) - 1) // (max_batch_size * dist.get_world_size()) + 1
    ) * dist.get_world_size()
    all_batches = torch.arange(len(dataset)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=rank_batches,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    # Accumulate statistics.
    # Index 0 for the overall statistics.
    dist.print0(f"Calculating statistics for {len(dataset)} images...")

    mu = [
        torch.zeros([feature_dim], dtype=torch.float64, device=device)
        for _ in range(num_labels + 1)
    ]
    sigma = [
        torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
        for _ in range(num_labels + 1)
    ]
    list_features = [[] for _ in range(num_labels + 1)]

    t0 = time.time()
    for k, (images, _labels) in enumerate(data_loader):
        if k == 100:
            dist.print0(
                f"Estimated time to finish:"
                f" {(time.time() - t0) / k * (len(data_loader) - k) / 60:.2f} min"
            )

        torch.distributed.barrier()

        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])

        # Overall statistics.
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu[0] += features.sum(0)
        sigma[0] += features.T @ features
        list_features[0].append(features.cpu())

        # Label-wise statistics.
        # Labels are either a single label or a one-hot encoded vector
        if _labels.ndim == 1:
            _labels = _labels.unsqueeze(1)
        _labels = torch.argmax(_labels, dim=1)

        for label in range(1, num_labels + 1):
            idx = _labels == label - 1
            if idx.sum() == 0:
                continue
            mu[label] += features[idx].sum(0)
            sigma[label] += features[idx].T @ features[idx]
            list_features[label].append(features[idx].cpu())

    for label in range(num_labels + 1):
        list_features[label] = torch.cat(list_features[label], dim=0)

    gathered_features = [
        [torch.zeros_like(list_features[i]) for _ in range(dist.get_world_size())]
        for i in range(num_labels + 1)
    ]

    # Calculate grand totals.
    for label in range(num_labels + 1):
        torch.distributed.all_reduce(mu[label])
        torch.distributed.all_reduce(sigma[label])
        torch.distributed.all_gather_object(
            gathered_features[label], list_features[label]
        )

        gathered_features[label] = torch.cat(gathered_features[label], dim=0)
        mu[label] /= len(gathered_features[label])
        sigma[label] -= mu[label].ger(mu[label]) * len(gathered_features[label])
        sigma[label] /= len(gathered_features[label]) - 1

    return (
        [x.numpy() for x in gathered_features],
        [x.cpu().numpy() for x in mu],
        [x.cpu().numpy() for x in sigma],
    )


####################################################################################################
# Fréchet Inception Distance
####################################################################################################


def compute_fid(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))


####################################################################################################
# DINO features and stats
####################################################################################################


def compute_dino_stats(
    image_path,
    num_expected=None,
    seed=0,
    max_batch_size=64,
    num_workers=3,
    prefetch_factor=2,
    device=torch.device("cuda"),
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load Dino v2 model.
    dist.print0("Loading DINO v2 model...")
    detector_net = classifier.FeatureExtractor(url=image_path).to(device)
    feature_dim = 768
    detector_kwargs = {}

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')

    dataset = ImageFolderDataset(
        path=image_path,
        max_size=num_expected,
        random_seed=seed,
        use_labels=True,
    )
    num_labels = dataset.label_dim

    if num_expected is not None and len(dataset) < num_expected:
        raise click.ClickException(
            f"Found {len(dataset)} images, but expected at least {num_expected}"
        )
    if len(dataset) < 2:
        raise click.ClickException(
            f"Found {len(dataset)} images, but need at least 2 to compute statistics"
        )

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = (
        (len(dataset) - 1) // (max_batch_size * dist.get_world_size()) + 1
    ) * dist.get_world_size()
    all_batches = torch.arange(len(dataset)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=rank_batches,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    # Accumulate statistics.
    # Index 0 for the overall statistics.
    dist.print0(f"Calculating statistics for {len(dataset)} images...")

    mu = [
        torch.zeros([feature_dim], dtype=torch.float64, device=device)
        for _ in range(num_labels + 1)
    ]
    sigma = [
        torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
        for _ in range(num_labels + 1)
    ]
    list_features = [[] for _ in range(num_labels + 1)]

    t0 = time.time()
    for k, (images, _labels) in enumerate(data_loader):
        if k == 100:
            dist.print0(
                f"Estimated time to finish:"
                f" {(time.time() - t0) / k * (len(data_loader) - k) / 60:.2f} minutes"
            )

        torch.distributed.barrier()

        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])

        with torch.no_grad():
            features = detector_net(images.to(device), **detector_kwargs).to(
                torch.float64
            )
        mu[0] += features.sum(0)
        sigma[0] += features.T @ features
        list_features[0].append(features.cpu())

        # Label-wise statistics.
        # Labels are either a single label or a one-hot encoded vector
        if _labels.ndim == 1:
            _labels = _labels.unsqueeze(1)
        _labels = torch.argmax(_labels, dim=1)

        for label in range(1, num_labels + 1):
            idx = _labels == label - 1
            if idx.sum() == 0:
                continue
            mu[label] += features[idx].sum(0)
            sigma[label] += features[idx].T @ features[idx]
            list_features[label].append(features[idx].cpu())

    for label in range(num_labels + 1):
        list_features[label] = torch.cat(list_features[label], dim=0)

    gathered_features = [
        [torch.zeros_like(list_features[i]) for _ in range(dist.get_world_size())]
        for i in range(num_labels + 1)
    ]

    # Calculate grand totals.
    for label in range(num_labels + 1):
        torch.distributed.all_reduce(mu[label])
        torch.distributed.all_reduce(sigma[label])
        torch.distributed.all_gather_object(
            gathered_features[label], list_features[label]
        )

        gathered_features[label] = torch.cat(gathered_features[label], dim=0)
        mu[label] /= len(gathered_features[label])
        sigma[label] -= mu[label].ger(mu[label]) * len(gathered_features[label])
        sigma[label] /= len(gathered_features[label]) - 1

    return (
        [x.numpy() for x in gathered_features],
        [x.cpu().numpy() for x in mu],
        [x.cpu().numpy() for x in sigma],
    )


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

# Handle for error in label alignement between generated and referecence images
# THIS IS A TEMPORARY FIX, SHOULD BE REMOVED ONCE THE ISSUE IS SOLVED
# dino model has been trained with good labels: dino-labels, hence predicts dino-labels
# gen model has been trained with bad ref labels: gen-labels, hence gen images are labeled with gen-labels
# therefore when evaluating dino on gen data,
# -> pred labels are in dino-labels
# -> "true" labels are in gen-labels
# => we map gen-labels (true labels) to dino-labels (pred labels)
# ----------------------------------------------------------------------------------------------------

APPLY_LABEL_MAPPING = True

LABEL_MAPPING = {
    0: 7,
    1: 1,
    2: 13,
    3: 14,
    4: 10,
    5: 4,
    6: 3,
    7: 8,
    8: 15,
    9: 2,
    10: 6,
    11: 5,
    12: 9,
    13: 12,
    14: 0,
    15: 11,
}


@click.group()
def main():
    """
    Computes Inception features and statistics.
    Evaluate data with metrics.

    Example:

    \b
    # Compute Inception reference features and statistics
    python evaluation.py inception-ref --data-path dataset/dataset.zip   

    \b
    # Compute Inception features and statistics for generated images 
    python evaluation.py inception-stats --img-path dataset/model/generated_images.zip --batch 128

    \b
    # Compute DINOv2 features and statistics for generated images 
    python evaluation.py dino-stats --img-path dataset/model/generated_images.zip --batch 128

    \b
    # Evaluate 
    python evaluation.py eval \
        --img-path dataset/model/generated_images.zip \
        --metrics fid toppr toppr-dino \
        --inception-ref dataset/inception-ref/inception_ref.npz \
        --dino-ref dataset/dino-ref/dino_ref.npz \
        --batch 128
    """


####################################################################################################
# Inception reference
####################################################################################################


@main.command()
@click.option(
    "--data-path",
    "data_path",
    metavar="PATH|ZIP",
    type=str,
    required=True,
    help="Path to the dataset",
)
@click.option(
    "--dst-dir",
    "dst_dir",
    metavar="PATH|ZIP",
    type=str,
    help="Directory for saving Inception features and statistics.",
)
@click.option(
    "--batch",
    "batch_size",
    metavar="INT",
    type=click.IntRange(min=1),
    default=128,
    show_default=True,
    help="Batch size",
)
def inception_ref(data_path, dst_dir, batch_size):
    """Calculate dataset reference Inception features and statistics."""

    click.echo("Computing reference Inception features and stats.")

    data_path = Path(data_path).expanduser()
    data_dir = data_path.parent
    dataset_name = data_path.name.split(".")[0]
    if dataset_name == "dataset":
        dataset_name = data_dir.name

    if dst_dir is None:
        dst_dir = data_dir / f"{dataset_name}-{FLAG_INCEPTION_REF_PATH}"
    else:
        dst_dir = Path(dst_dir).expanduser()
    dst_dir.mkdir(parents=True, exist_ok=True)

    data_path = str(data_path)
    dst_dir = str(dst_dir)

    torch.multiprocessing.set_start_method("spawn")
    dist.init()

    inception_feats, inception_mus, inception_sigmas = compute_inception_stats(
        image_path=data_path,
        max_batch_size=batch_size,
    )

    if dist.get_rank() == 0:
        for label in range(len(inception_mus)):
            # (Index 0 is for overall statistics)
            name = f"{dataset_name}-{label - 1}" if label > 0 else dataset_name
            dist.print0(f"Saving dataset ref Inception stats for {name}...")

            if not os.path.dirname(dst_dir):
                os.makedirs(dst_dir, parents=True, exist_ok=True)

            np.savez(
                os.path.join(dst_dir, f"{name}.npz"),
                inception_feats=inception_feats[label],
                inception_mus=inception_mus[label],
                inception_sigmas=inception_sigmas[label],
            )
            dist.print0(f"Saved dataset ref Inception stats in {dst_dir}/{name}.npz")

    torch.distributed.barrier()
    dist.print0("Done.")


####################################################################################################
# DINO reference
####################################################################################################


@main.command()
@click.option(
    "--data-path",
    "data_path",
    metavar="PATH|ZIP",
    type=str,
    required=True,
    help="Path to the dataset",
)
@click.option(
    "--dst-dir",
    "dst_dir",
    metavar="PATH|ZIP",
    type=str,
    help="Directory for saving Inception features and statistics.",
)
@click.option(
    "--batch",
    "batch_size",
    metavar="INT",
    type=click.IntRange(min=1),
    default=128,
    show_default=True,
    help="Batch size",
)
def dino_ref(data_path, dst_dir, batch_size):
    """Calculate dataset reference DINO features and statistics."""

    click.echo("Computing reference DINO features and stats.")

    data_path = Path(data_path).expanduser()
    data_dir = data_path.parent
    dataset_name = data_path.name.split(".")[0]
    if dataset_name == "dataset":
        dataset_name = data_dir.name

    if dst_dir is None:
        dst_dir = data_dir / f"{dataset_name}-{FLAG_DINO_REF_PATH}"
    else:
        dst_dir = Path(dst_dir).expanduser()
    dst_dir.mkdir(parents=True, exist_ok=True)

    data_path = str(data_path)
    dst_dir = str(dst_dir)

    torch.multiprocessing.set_start_method("spawn")
    dist.init()

    dino_feats, dino_mus, dino_sigmas = compute_dino_stats(
        image_path=data_path,
        max_batch_size=batch_size,
    )

    if dist.get_rank() == 0:
        for label in range(len(dino_mus)):
            # (Index 0 is for overall statistics)
            name = f"{dataset_name}-{label - 1}" if label > 0 else dataset_name
            dist.print0(f"Saving dataset ref DINO stats for {name}...")

            if not os.path.dirname(dst_dir):
                os.makedirs(dst_dir, parents=True, exist_ok=True)

            np.savez(
                os.path.join(dst_dir, f"{name}.npz"),
                dino_feats=dino_feats[label],
                dino_mus=dino_mus[label],
                dino_sigmas=dino_sigmas[label],
            )
            dist.print0(f"Saved dataset ref DINO stats in {dst_dir}/{name}.npz")

    torch.distributed.barrier()
    dist.print0("Done.")


####################################################################################################
# Inception stats
####################################################################################################


@main.command()
@click.option(
    "--img-path",
    "img_path",
    metavar="PATH|ZIP",
    type=str,
    required=True,
    help="Path to generated images",
)
# @click.option(
#     "--num",
#     "num_expected",
#     metavar="INT",
#     type=click.IntRange(min=2),
#     default=50000,
#     show_default=True,
#     help="Number of images to use",
# )
@click.option(
    "--seed",
    metavar="INT",
    type=int,
    default=0,
    show_default=True,
    help="Random seed for selecting the images",
)
@click.option(
    "--batch",
    "batch_size",
    metavar="INT",
    type=click.IntRange(min=1),
    default=64,
    show_default=True,
    help="Batch size",
)
@click.option(
    "--reset-inception",
    is_flag=True,
    default=False,
    show_default=True,
    help="Reset cached features (if previously calculated with different settings)",
)
def inception_stats(img_path, seed, batch_size, reset_inception):
    """Compute and cache Inception features and statistics."""

    click.echo(f"Computing inception feats and stats for: {img_path}...")

    torch.multiprocessing.set_start_method("spawn")
    dist.init()

    torch.manual_seed(seed)
    np.random.seed(seed)

    out_path = os.path.dirname(img_path)
    inception_file = os.path.join(out_path, INCEPTION_STATS_FILE)

    if os.path.exists(inception_file) and not reset_inception:
        dist.print0(f"Using cached Inception stats from {inception_file}")
        return

    inception_feats, inception_mus, inception_sigmas = compute_inception_stats(
        image_path=img_path,
        seed=seed,
        max_batch_size=batch_size,
    )

    if dist.get_rank() == 0:
        np.save(
            inception_file,
            {
                "inception_feats": inception_feats,
                "inception_mus": inception_mus,
                "inception_sigmas": inception_sigmas,
            },
        )
        dist.print0(f"Saved Inception stats to {inception_file}")

    torch.distributed.barrier()


####################################################################################################
# DINO stats
####################################################################################################


@main.command()
@click.option(
    "--img-path",
    "img_path",
    metavar="PATH|ZIP",
    type=str,
    required=True,
    help="Path to generated images",
)
# @click.option(
#     "--num",
#     "num_expected",
#     metavar="INT",
#     type=click.IntRange(min=2),
#     default=50000,
#     show_default=True,
#     help="Number of images to use",
# )
@click.option(
    "--seed",
    metavar="INT",
    type=int,
    default=0,
    show_default=True,
    help="Random seed for selecting the images",
)
@click.option(
    "--batch",
    "batch_size",
    metavar="INT",
    type=click.IntRange(min=1),
    default=64,
    show_default=True,
    help="Batch size",
)
@click.option(
    "--reset",
    is_flag=True,
    default=False,
    show_default=True,
    help="Reset cached features (if previously calculated with different settings)",
)
def dino_stats(img_path, seed, batch_size, reset):
    """Compute and cache DINO features and statistics."""

    click.echo(f"Computing DINO feats and stats for: {img_path}...")

    torch.multiprocessing.set_start_method("spawn")
    dist.init()

    torch.manual_seed(seed)
    np.random.seed(seed)

    out_path = os.path.dirname(img_path)
    dino_file = os.path.join(out_path, DINO_STATS_FILE)

    if os.path.exists(dino_file) and not reset:
        dist.print0(f"Using cached DINO stats from {dino_file}")
        return

    dino_feats, dino_mus, dino_sigmas = compute_dino_stats(
        image_path=img_path,
        seed=seed,
        max_batch_size=batch_size,
    )

    if dist.get_rank() == 0:
        np.save(
            dino_file,
            {
                "dino_feats": dino_feats,
                "dino_mus": dino_mus,
                "dino_sigmas": dino_sigmas,
            },
        )
        dist.print0(f"Saved DINO stats to {dino_file}")

    torch.distributed.barrier()


####################################################################################################
# Evaluation
####################################################################################################


@main.command()
@click.option(
    "--img-path",
    "img_path",
    metavar="PATH|ZIP",
    type=str,
    required=True,
    help="Path to generated images",
)
@click.option(
    "--metrics",
    multiple=True,
    default=("fid", "toppr", "toppr-dino"),
    help="Metrics to compute (e.g. --metrics fid --metrics toppr --metrics toppr-dino)",
)
@click.option(
    "--inception-ref",
    "inception_ref",
    metavar="NPZ|URL",
    type=str,
    required=False,
    help="Dataset reference Inception statistics (required for FID and PRDC)",
)
@click.option(
    "--dino-ref",
    "dino_ref",
    metavar="NPZ|URL",
    type=str,
    required=False,
    help="Dataset reference DINO statistics (required for PRCD-DINO)",
)
@click.option(
    "--seed",
    metavar="INT",
    type=int,
    default=0,
    show_default=True,
    help="Random seed for selecting the images",
)
# @click.option(
#     "--batch",
#     "batch_size",
#     metavar="INT",
#     type=click.IntRange(min=1),
#     default=64,
#     show_default=True,
#     help="Batch size",
# )
# Options for PCA
@click.option(
    "--pca",
    is_flag=True,
    default=True,
    show_default=True,
    help="Apply PCA to the features before calculating PRDC",
)
@click.option(
    "--pca-dim",
    metavar="INT",
    type=click.IntRange(min=1),
    default=64,
    show_default=True,
    help="Dimensionality to reduce to with PCA",
)
@click.option(
    "--whiten",
    is_flag=True,
    default=False,
    show_default=True,
    help="Apply whitening to the features before calculating PRDC (only if --pca is also passed)",
)
# Option for TopP&R
@click.option(
    "--tpr-alpha",
    help="Significance level alpha for Top P/R confidence bands (higher = less strict)",
    metavar="FLOAT",
    type=float,
    default=0.2,
    show_default=True,
)
@click.option(
    "--tpr-randproj/--no-tpr-randproj",
    help="Enable/disable 32D random projection inside Top P/R",
    default=True,
    show_default=True,
)
@click.option(
    "--tpr-n",
    help="Number of features to sample per evaluation (per label)",
    metavar="INT",
    type=click.IntRange(min=100),
    default=5000,
    show_default=True,
)
@click.option(
    "--tpr-reps",
    help="Number of random resamples (averaged) for Top P/R",
    metavar="INT",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
)
@click.option(
    "--tpr-l2norm/--no-tpr-l2norm",
    help="L2-normalize features before KDE for Top P/R",
    default=False,
    show_default=True,
)
def eval(
    img_path,
    metrics,
    inception_ref,
    dino_ref,
    seed,
    # Options for PCA
    pca,
    pca_dim,
    whiten,
    # Option for TopP&R
    tpr_alpha,
    tpr_randproj,
    tpr_n,
    tpr_reps,
    tpr_l2norm,
):
    """Evaluate metrics for a given set of generated images."""

    # Checking arguments.
    metrics = [m.lower().replace("_", "-") for m in metrics]
    click.echo(f"Metrics requested: {metrics}")

    if (set(metrics) & {"fid", "toppr"}) and inception_ref is None:
        raise click.UsageError(
            "--inception-ref is required when 'fid' or 'toppr' is included in --metrics"
        )
    if (set(metrics) & {"toppr-dino"}) and dino_ref is None:
        raise click.UsageError(
            "--dino-ref is required when 'toppr-dino' is included in --metrics"
        )

    # Checking files.
    data_path = Path(img_path).parent
    inception_stats = data_path / INCEPTION_STATS_FILE
    dino_stats = data_path / DINO_STATS_FILE

    if set(metrics) & {"fid", "toppr"}:
        if not os.path.exists(inception_ref):
            raise FileNotFoundError(f"Missing {inception_ref}. Run `inception-ref`.")
        elif not os.path.exists(inception_stats):
            raise FileNotFoundError(
                f"Missing {inception_stats}. Run `inception-stats`."
            )

    if set(metrics) & {"toppr-dino"}:
        if not os.path.exists(dino_ref):
            raise FileNotFoundError(f"Missing {dino_ref}. Run `dino-ref`.")
        if not os.path.exists(dino_stats):
            raise FileNotFoundError(f"Missing {dino_stats}. Run `dino-stats`.")

    # Distribute.
    # FIX: no need to distribute the evaluation across multiple processes, since the
    # metrics are not calculated on a per-image basis. However, we still need to init
    # the distributed environment for loading the features and stats (which are saved in a distributed manner).
    torch.multiprocessing.set_start_method("spawn")
    dist.init()

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load Inception ref.
    if set(metrics) & {"fid", "toppr"}:
        dist.print0(
            "\n"
            "Loading Inception reference...\n"
            "------------------------------------------------------------"
        )

        label = 0
        ref_files = [inception_ref]
        while os.path.exists(inception_ref.split(".")[0] + f"-{label}.npz"):
            ref_files.append(inception_ref.split(".")[0] + f"-{label}.npz")
            label += 1

        inception_refs = []
        for ref_file in ref_files:
            dist.print0(ref_file)

            if dist.get_rank() == 0:
                with dnnlib.util.open_url(ref_file) as f:
                    inception_refs.append(dict(np.load(f)))
                # dist.print0(f"Ref file keys: {list(inception_refs[-1].keys())}")
                dist.print0(f"Features: {len(inception_refs[-1]['inception_feats'])}")
    else:
        inception_refs = None
        dist.print0("Inception reference statistics not loaded.")

    # Load DINO ref.
    if set(metrics) & {"toppr-dino"}:
        dist.print0(
            "\n"
            "Loading DINO reference...\n"
            "------------------------------------------------------------"
        )

        label = 0
        ref_files = [dino_ref]
        while os.path.exists(dino_ref.split(".")[0] + f"-{label}.npz"):
            ref_files.append(dino_ref.split(".")[0] + f"-{label}.npz")
            label += 1

        dino_refs = []
        for ref_file in ref_files:
            dist.print0(f"Using dataset reference statistics from: {ref_file}")

            if dist.get_rank() == 0:
                with dnnlib.util.open_url(ref_file) as f:
                    dino_refs.append(dict(np.load(f)))
                # dist.print0(f"Ref file keys: {list(dino_refs[-1].keys())}")
                dist.print0(len(dino_refs[-1]["dino_feats"]))
    else:
        dino_refs = None
        dist.print0("DINO reference statistics not loaded.")

    # Load Inception stats.
    if set(metrics) & {"fid", "toppr"}:
        dist.print0(
            "\n"
            "Loading Inception statistics...\n"
            "------------------------------------------------------------"
        )

        if dist.get_rank() == 0:
            with dnnlib.util.open_url(str(inception_stats)) as f:
                inception_data = dict(np.load(f, allow_pickle=True).item())
                inception_feats = inception_data["inception_feats"]
                inception_mus = inception_data["inception_mus"]
                inception_sigmas = inception_data["inception_sigmas"]
    else:
        inception_feats, inception_mus, inception_sigmas = None, None, None
        dist.print0("Inception statistics not loaded.")

    # Load DINO stats.
    if set(metrics) & {"toppr-dino"}:
        dist.print0(
            "\n"
            "Loading DINO statistics...\n"
            "------------------------------------------------------------"
        )

        if dist.get_rank() == 0:
            with dnnlib.util.open_url(str(dino_stats)) as f:
                dino_data = dict(np.load(f, allow_pickle=True).item())
                dino_feats = dino_data["dino_feats"]
                # dino_mus = dino_data["dino_mus"]
                # dino_sigmas = dino_data["dino_sigmas"]
    else:
        dino_feats = None
        dist.print0("DINO statistics not loaded.")

    # Prepare computations.
    if inception_feats is not None:
        num_labels = len(inception_feats) - 1
    elif dino_feats is not None:
        num_labels = len(dino_feats) - 1
    else:
        assert len(inception_feats) == len(dino_feats), (
            "Inception and DINO lengths must match."
        )
        num_labels = len(dino_feats) - 1

    data = {"overall": {}}
    data.update({i: {} for i in range(num_labels)})

    def metric_key(i):
        return "overall" if i == 0 else (i - 1)

    # Compute FID.
    if "fid" in metrics:
        dist.print0(
            "\n"
            "Computing FID...\n"
            "------------------------------------------------------------"
        )

        if dist.get_rank() == 0:
            for i in range(1 + num_labels):
                metric_name = "Overall" if i == 0 else f"Label {i}"
                # dist.print0(f"Computing {metric_name} FID on {img_path}.")

                fid = compute_fid(
                    inception_mus[i],
                    inception_sigmas[i],
                    inception_refs[i]["inception_mus"],
                    inception_refs[i]["inception_sigmas"],
                )

                dist.print0(f"{metric_name} FID: {fid:.2f}")

                key = metric_key(i)
                data[key].update({"fid": fid})

    # Compute PRDC with TopP&R.
    if "toppr" in metrics:
        dist.print0(
            "\n"
            "Computing PRDC with TopP&R...\n"
            "------------------------------------------------------------"
        )
        dist.print0(
            f"Settings \n"
            f"\t alpha={tpr_alpha} \n"
            f"\t randproj={tpr_randproj} \n"
            f"\t l2norm={tpr_l2norm} \n"
            f"\t n={tpr_n} (out of 20k) \n"
            f"\t frepeats={tpr_reps} \n"
        )

        num_feats_refs = [ref["inception_feats"].shape[0] for ref in inception_refs]

        if APPLY_LABEL_MAPPING:
            dist.print0("Applying label mapping for INCEPTION features...")
            num_feats_stats = []
            for i in range(len(inception_feats)):
                if i == 0:
                    num_feats_stats.append(inception_feats[0].shape[0])  # overall
                else:
                    # i corresponds to "Label i" => label index is i-1 in mapping space
                    num_feats_stats.append(
                        inception_feats[LABEL_MAPPING[i - 1] + 1].shape[0]
                    )
        else:
            num_feats_stats = [f.shape[0] for f in inception_feats]

        # num_feats_stats = [f.shape[0] for f in inception_feats]
        num_features = [
            min(n1, n2) for (n1, n2) in zip(num_feats_refs, num_feats_stats)
        ]
        # features = [f[:20000] for f in inception_feats]

        if dist.get_rank() == 0:
            for i in range(1 + num_labels):
                metric_name = "Overall" if i == 0 else f"Label {i}"
                dist.print0(f"{metric_name} PRDC on {img_path}")

                # Inception TopPR
                Ps_inc, Rs_inc = [], []
                # Ds_inc, Cs_inc = [], []
                pool_size = min(20_000, num_features[i])
                use_n = min(tpr_n, pool_size)
                dist.print0(f"\t using {use_n} / {pool_size} features")

                num_test = tpr_reps
                dist.print0(f"\t averaging result over {num_test} runs")

                for _ in range(num_test):
                    index = np.random.permutation(pool_size)[:use_n]
                    feats_ref = inception_refs[i]["inception_feats"][index]
                    # feats = inception_feats[i][index]

                    if APPLY_LABEL_MAPPING:
                        dist.print0("Applying label mapping for INCEPTION features...")
                        if i != 0:
                            feats = inception_feats[LABEL_MAPPING[i - 1] + 1][index]
                        else:
                            feats = inception_feats[0][index]
                    else:
                        feats = inception_feats[i][index]

                    print(
                        f"feats_ref shape: {feats_ref.shape}, feats shape: {feats.shape}, mean feats_ref: {np.mean(feats_ref)}, mean feats: {np.mean(feats)}"
                    )

                    if pca:
                        feats_ref, feats = PCA(
                            feats_ref,
                            feats,
                            pca_dim=pca_dim,
                            whiten=whiten,
                        )
                    print(
                        f"feats_ref shape: {feats_ref.shape}, feats shape: {feats.shape}, mean feats_ref: {np.mean(feats_ref)}, mean feats: {np.mean(feats)}"
                    )

                    P_inc, R_inc = compute_top_pr(
                        real_features=feats_ref,
                        fake_features=feats,
                        alpha=tpr_alpha,
                        kernel="cosine",
                        random_proj=tpr_randproj,
                        f1_score=False,
                        l2norm=tpr_l2norm,
                    )
                    # D_inc, C_inc = 0, 0

                    Ps_inc.append(P_inc)
                    Rs_inc.append(R_inc)
                    # Ds_inc.append(D_inc)
                    # Cs_inc.append(C_inc)

                P_inc = np.mean(Ps_inc)
                R_inc = np.mean(Rs_inc)
                # D_inc = np.mean(Ds_inc)
                # C_inc = np.mean(Cs_inc)
                P_inc_std = np.std(Ps_inc)
                R_inc_std = np.std(Rs_inc)
                # D_inc_std = np.std(Ds_inc)
                # C_inc_std = np.std(Cs_inc)

                dist.print0(
                    f"Results: \n"
                    f"\t P_inc: {P_inc:.4f} pm {P_inc_std:.4f} \n"
                    f"\t R_inc: {R_inc:.4f} pm {R_inc_std:.4f} \n"
                    # f"\t D_inc: {D_inc:.4f} pm {D_inc_std:.4f} \n"
                    # f"\t C_inc: {C_inc:.4f} pm {C_inc_std:.4f} \n"
                )

                key = metric_key(i)
                data[key].update(
                    {
                        "P_inc": P_inc,
                        "R_inc": R_inc,
                        # "D_inc": D_inc,
                        # "C_inc": C_inc,
                        "P_inc_std": P_inc_std,
                        "R_inc_std": R_inc_std,
                        # "D_inc_std": D_inc_std,
                        # "C_inc_std": C_inc_std,
                        "num_features": num_features[i],
                    }
                )

    # Compute PRDC with TopP&R DINO.
    if "toppr-dino" in metrics:
        dist.print0(
            "\n"
            "Computing PRDC with TopP&R DINO...\n"
            "------------------------------------------------------------"
        )
        dist.print0(
            f"Settings \n"
            f"\t alpha={tpr_alpha} \n"
            f"\t randproj={tpr_randproj} \n"
            f"\t l2norm={tpr_l2norm} \n"
            f"\t n={tpr_n} (out of 20k) \n"
            f"\t frepeats={tpr_reps} \n"
        )

        num_feats_refs = [ref["dino_feats"].shape[0] for ref in dino_refs]

        if APPLY_LABEL_MAPPING:
            dist.print0("Applying label mapping for DINO features...")
            num_feats_stats = []
            for i in range(len(dino_feats)):
                if i == 0:
                    num_feats_stats.append(dino_feats[0].shape[0])  # overall
                else:
                    # i corresponds to "Label i" => label index is i-1 in mapping space
                    num_feats_stats.append(
                        dino_feats[LABEL_MAPPING[i - 1] + 1].shape[0]
                    )
        else:
            num_feats_stats = [f.shape[0] for f in dino_feats]

        num_features = [
            min(n1, n2) for (n1, n2) in zip(num_feats_refs, num_feats_stats)
        ]
        # num_features = [f.shape[0] for f in dino_feats]
        # features = [f[:20000] for f in dino_feats]

        if dist.get_rank() == 0:
            for i in range(1 + num_labels):
                metric_name = "Overall" if i == 0 else f"Label {i}"
                dist.print0(f"Computing {metric_name} PRDC on {img_path} with TopP&R.")

                # DINO TopPR
                Ps_dino, Rs_dino = [], []
                # Ds_dino, Cs_dino = [], []
                # pool_size = min(20_000, features[i].shape[0])
                pool_size = min(20_000, num_features[i])
                use_n = min(tpr_n, pool_size)
                dist.print0(f"\t using {use_n} / {pool_size}) features")

                num_test = tpr_reps
                dist.print0(f"\t averaging result over {num_test} runs")

                for _ in range(num_test):
                    index = np.random.permutation(pool_size)[:use_n]
                    feats_ref = dino_refs[i]["dino_feats"][index]
                    # feats = features[i][index]

                    if APPLY_LABEL_MAPPING:
                        dist.print0("Applying label mapping for DINO features...")
                        if i != 0:
                            feats = dino_feats[LABEL_MAPPING[i - 1] + 1][index]
                        else:
                            feats = dino_feats[0][index]
                    else:
                        feats = dino_feats[i][index]

                    print(
                        f"feats_ref shape: {feats_ref.shape}, feats shape: {feats.shape}, mean feats_ref: {np.mean(feats_ref)}, mean feats: {np.mean(feats)}"
                    )

                    APPLY_PCA_TO_DINO = False

                    # if pca:
                    if APPLY_PCA_TO_DINO:
                        feats_ref, feats = PCA(
                            feats_ref,
                            feats,
                            pca_dim=pca_dim,
                            whiten=whiten,
                        )

                    print(
                        f"feats_ref shape: {feats_ref.shape}, feats shape: {feats.shape}, mean feats_ref: {np.mean(feats_ref)}, mean feats: {np.mean(feats)}"
                    )

                    P_dino, R_dino = compute_top_pr(
                        real_features=feats_ref,
                        fake_features=feats,
                        alpha=tpr_alpha,
                        kernel="cosine",
                        random_proj=tpr_randproj,
                        f1_score=False,
                        l2norm=tpr_l2norm,
                    )
                    # D_dino, C_dino = 0, 0

                    Ps_dino.append(P_dino)
                    Rs_dino.append(R_dino)
                    # Ds_dino.append(D_dino)
                    # Cs_dino.append(C_dino)

                P_dino = np.mean(Ps_dino)
                R_dino = np.mean(Rs_dino)
                # D_dino = np.mean(Ds_dino)
                # C_dino = np.mean(Cs_dino)
                P_dino_std = np.std(Ps_dino)
                R_dino_std = np.std(Rs_dino)
                # D_dino_std = np.std(Ds_dino)
                # C_dino_std = np.std(Cs_dino)

                dist.print0(
                    f"Results: \n"
                    f"\t P_dino: {P_dino:.4f} pm {P_dino_std:.4f} \n"
                    f"\t R_dino: {R_dino:.4f} pm {R_dino_std:.4f} \n"
                    # f"\t D_dino: {D_dino:.4f} pm {D_dino_std:.4f} \n"
                    # f"\t C_dino: {C_dino:.4f} pm {C_dino_std:.4f} \n"
                )

                key = metric_key(i)
                data[key].update(
                    {
                        "P_dino": P_dino,
                        "R_dino": R_dino,
                        # "D_dino": D_dino,
                        # "C_dino": C_dino,
                        "P_dino_std": P_dino_std,
                        "R_dino_std": R_dino_std,
                        # "D_dino_std": D_dino_std,
                        # "C_dino_std": C_dino_std,
                    }
                )

    if dist.get_rank() == 0:
        with open(os.path.join(data_path, "evaluation.jsonl"), "w") as file:
            json.dump(data, file, indent=4)

    torch.distributed.barrier()


####################################################################################################
####################################################################################################
####################################################################################################


if __name__ == "__main__":
    main()
