import json
import os
import pickle
import time
from pathlib import Path

import click
import gudhi
import numpy as np
import scipy
import sklearn
import torch
from scipy.spatial import distance
from tqdm import tqdm

import dnnlib
from torch_utils import distributed as dist
from training import classifier
from training.dataset import ImageFolderDataset

FLAG_INCEPTION_REF_PATH = "inception-ref"
FLAG_DINO_REF_PATH = "dino-ref"
INCEPTION_FILE = "inception_stats.npy"
DINO_FILE = "dino_stats.npy"


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
# Top Precision&Recall
####################################################################################################


def compact_KDE(data, position, h, kernel="cosine"):
    # compact kernel options = {"epanechinikov", "cosine"}
    p_hat = np.array([])
    dist = sklearn.metrics.pairwise.euclidean_distances(position, data)

    # Epanechinikov kernel
    if kernel == "epanechinikov":
        for iloop in range(len(dist)):
            sample_score = dist[iloop][np.where(dist[iloop] ** 2 <= (h**2))]
            p_hat = np.append(
                p_hat,
                (1 / len(data))
                * ((3 / (4 * h)) ** len(data[0]))
                * ((len(sample_score)) - np.sum(sample_score / (h**2))),
            )
        return p_hat

    # Cosine kernel
    elif kernel == "cosine":
        for iloop in range(len(dist)):
            sample_score = dist[iloop][np.where(dist[iloop] ** 2 <= (h**2))]
            p_hat = np.append(
                p_hat,
                (1 / len(data))
                * ((np.pi / (4 * h)) ** len(data[0]))
                * np.sum(np.cos((np.pi / 2) * (sample_score / h))),
            )
        return p_hat


def confband_est(data, h, alpha=0.1, kernel="cosine", p_val=True, repeat=10):
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    positions = data

    # p_hat
    p_hat = compact_KDE(data, positions, h, kernel=kernel)

    # p_tilde
    theta_star = np.array([])
    for iloop in range(repeat):
        data_bs = data[
            np.random.choice(np.arange(len(data)), size=len(data), replace=True)
        ]
        p_tilde = compact_KDE(data_bs, positions, h, kernel=kernel)

        # theta
        theta_star = np.append(
            theta_star, np.sqrt(len(data)) * np.max(np.abs(p_hat - p_tilde))
        )

    # q_alpha
    if len(theta_star) == 0:
        q_alpha = 0
    else:
        q_alpha = np.quantile(theta_star, 1 - alpha)

    # confidence band
    if p_val:
        return q_alpha / np.sqrt(len(data)), p_hat
    else:
        return q_alpha / np.sqrt(len(data))


def set_grid(data):
    import numpy as np

    # find min max
    dim = len(data[0])
    mins = np.array([])
    maxs = np.array([])
    for dims in range(dim):
        mins = np.append(mins, min(data[:, dims]))
        maxs = np.append(maxs, max(data[:, dims]))

    # set grid
    # 2 dimensional data
    if len(mins) == 2:
        xval = np.linspace(mins[0], maxs[0], 1000)
        yval = np.linspace(mins[1], maxs[1], 1000)
        positions = np.array([[u, v] for u in xval for v in yval])
    # 3 dimensional data
    elif len(mins) == 3:
        xval = np.linspace(mins[0], maxs[0], 100)
        yval = np.linspace(mins[1], maxs[1], 100)
        zval = np.linspace(mins[2], maxs[2], 100)
        positions = np.array([[u, v, k] for u in xval for v in yval for k in zval])

    return positions


def bandwidth_est(
    data,
    bandwidth_list=[],
    confidence_band=False,
    kernel="cosine",
    alpha=0.1,
    Plot=False,
):
    # non-compact kernel list = {'gaussian','exponential'} | compact kernel list = {'tophat','epanechnikov','linear','cosine'}
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    # estimate bandwidth candidates with "balloon estimator" (variable-bandwidth estimation)
    if len(bandwidth_list) == 0:
        dist = distance.cdist(data, data, metric="euclidean")
        dist = dist[:-50,]
        for iloop in range(len(dist)):
            if iloop == 0:
                balloon_est = np.array(sorted(dist[iloop, (iloop + 1) :])[50 - 1])
            else:
                balloon_est = np.append(
                    balloon_est, sorted(dist[iloop, (iloop + 1) :])[50 - 1]
                )
        balloon_est = sorted(balloon_est)
        bandwidth_list = balloon_est[
            int(len(balloon_est) * 0.05) - 1
        ]  # top 5% estimated bandwidth
        bandwidth_list = np.append(
            bandwidth_list, balloon_est[int(len(balloon_est) * 0.2) - 1]
        )  # top 20% estimated bandwidth
        bandwidth_list = np.append(
            bandwidth_list, balloon_est[int(len(balloon_est) * 0.35) - 1]
        )  # top 35% estimated bandwidth
        bandwidth_list = np.append(
            bandwidth_list, balloon_est[int(len(balloon_est) * 0.5) - 1]
        )  # median estimated bandwidth
        bandwidth_list = np.append(
            bandwidth_list, balloon_est[int(len(balloon_est) * 0.65) - 1]
        )  # top 65% estimated bandwidth
        bandwidth_list = np.append(
            bandwidth_list, balloon_est[int(len(balloon_est) * 0.8) - 1]
        )  # top 80% estimated bandwidth
        bandwidth_list = np.append(
            bandwidth_list, balloon_est[int(len(balloon_est) * 0.95) - 1]
        )  # top 95% estimated bandwidth

    # estimate bandwidth
    n_h0 = np.array([])
    s_h0 = np.array([])
    cn_list = np.array([])
    for h in tqdm(bandwidth_list):
        # confidence band & p_hat
        cn = confband_est(data, h, alpha=alpha, kernel=kernel, p_val=False)
        cn_list = np.append(cn_list, cn)

        grid = set_grid(data)
        # filtration
        p_hat = compact_KDE(data, grid, h, kernel=kernel)
        PD = gudhi.CubicalComplex(
            dimensions=[
                round(len(grid) ** (1 / grid.shape[1])),
                round(len(grid) ** (1 / grid.shape[1])),
            ],
            top_dimensional_cells=-p_hat,
        ).persistence()

        # measure life length of all homology
        l_h0 = np.array([])
        for iloop in range(len(PD)):
            if PD[iloop][0] == 0:
                if np.abs(PD[iloop][1][1] - PD[iloop][1][0]) != float("inf"):
                    l_h0 = np.append(l_h0, np.abs(PD[iloop][1][1] - PD[iloop][1][0]))

        # N(h)
        n_h0 = np.append(n_h0, sum(l_h0 > cn) + 1)

        # S(h)
        S_h0 = l_h0 - cn
        s_h0 = np.append(s_h0, sum(list(filter(lambda S_h0: S_h0 > 0, S_h0))))
        print(
            "bandwidth: ",
            h,
            ", N_0(h): ",
            n_h0[-1],
            ", S_0(h): ",
            s_h0[-1],
            ", cn: ",
            cn,
        )

    try:
        if sum(s_h0 == max(s_h0)) == 1:
            if confidence_band:
                return (
                    bandwidth_list[s_h0.tolist().index(max(s_h0))],
                    cn_list[s_h0.tolist().index(max(s_h0))],
                )
            else:
                return bandwidth_list[s_h0.tolist().index(max(s_h0))]
        else:
            return bandwidth_list[0]
    except Exception as e:
        print(e)
        raise SystemExit


def compute_top_pr(
    *,
    real_features,
    fake_features,
    alpha=0.1,
    kernel="cosine",
    random_proj=True,
    f1_score=True,
    l2norm=False,
):
    """
    Computing Top Precision and Recall
        Args:
            real_features (n, d): input real features
            fake_features (n, d): input fake features
            alpha (float): significance level alpha in confidence band estimation (default=0.1)
            kernel (str): kernel for KDE                                          (default='cosine')
            random_proj (bool): If true, it will add linear layer from Pytorch library for random projection. (default=True)
                                However, If the dimension of the feature is lower than 32, even though random_proj is True, random projection will not be turned on.
            f1_score (bool): If True, it caculates f1 score for getting a 1-score evaluation (default=True)
        Returns:
            evaluation score (dict): fidelity, diversity and (opitionally f1 score.)

    """

    # --- helpers for robustness ---
    def _safe_bandwidth(data_np, h_candidate, floor_frac=1e-3, eps=1e-8):
        # Ensure strictly positive bandwidth. Use a small fraction of the median pairwise distance as a floor.
        if not isinstance(data_np, np.ndarray):
            data_np = np.asarray(data_np)
        if data_np.ndim != 2 or data_np.shape[0] < 2:
            return max(float(h_candidate), eps)
        dmat = distance.cdist(data_np, data_np, metric="euclidean")
        # take strictly positive distances only
        pos = dmat[dmat > 0]
        med = float(np.median(pos)) if pos.size > 0 else 1.0
        floor = max(eps, floor_frac * med)
        return float(max(h_candidate, floor))

    # match data format for random projection
    if not torch.is_tensor(real_features):
        real_features = torch.tensor(real_features, dtype=torch.float32)
    if not torch.is_tensor(fake_features):
        fake_features = torch.tensor(fake_features, dtype=torch.float32)

    # random projection
    if (random_proj) and (real_features.size()[1] > 32):
        projection = torch.nn.Linear(real_features.size()[1], 32, bias=False).eval()
        torch.manual_seed(99)
        torch.nn.init.xavier_normal_(projection.weight)
        for param in projection.parameters():
            param.requires_grad_(False)
        real_features = projection(real_features)
        fake_features = projection(fake_features)

    # to numpy
    real_features = real_features.detach().cpu().numpy()
    fake_features = fake_features.detach().cpu().numpy()

    # Optional L2 normalization
    if l2norm:
        real_features = real_features / (
            np.linalg.norm(real_features, axis=1, keepdims=True) + 1e-12
        )
        fake_features = fake_features / (
            np.linalg.norm(fake_features, axis=1, keepdims=True) + 1e-12
        )

    # use bandwidth estimator to calculate Top P&R
    if len(real_features[0]) <= 3:
        bandwidth_r, c_r = bandwidth_est(
            real_features, bandwidth_list=[], confidence_band=True, alpha=alpha
        )
        bandwidth_f, c_g = bandwidth_est(
            fake_features, bandwidth_list=[], confidence_band=True, alpha=alpha
        )
        bandwidth_r = _safe_bandwidth(real_features, bandwidth_r)
        bandwidth_f = _safe_bandwidth(fake_features, bandwidth_f)
        c_r, score_rr = confband_est(
            data=real_features, h=bandwidth_r, alpha=alpha, kernel=kernel, p_val=True
        )
        c_g, score_gg = confband_est(
            data=fake_features, h=bandwidth_f, alpha=alpha, kernel=kernel, p_val=True
        )
    else:
        # Robust balloon estimator for bandwidths in high dimension
        n_r, d_r = real_features.shape
        n_f, d_f = fake_features.shape

        k_r = max(1, min(d_r * 5, n_r - 1))
        dmat_r = distance.cdist(real_features, real_features, metric="euclidean")
        balloon_est = []
        for i in range(n_r):
            row = np.delete(dmat_r[i], i)  # drop self-distance
            if row.size == 0:
                continue
            row.sort()
            idx = min(k_r - 1, row.size - 1)
            balloon_est.append(row[idx])
        if len(balloon_est) == 0:
            # fallback: small fraction of median distance
            pos = dmat_r[dmat_r > 0]
            med = float(np.median(pos)) if pos.size > 0 else 1.0
            bandwidth_r = 1e-3 * med
        else:
            balloon_est = np.sort(np.asarray(balloon_est))
            bandwidth_r = balloon_est[len(balloon_est) // 2]  # median

        k_f = max(1, min(d_f * 5, n_f - 1))
        dmat_f = distance.cdist(fake_features, fake_features, metric="euclidean")
        balloon_est = []
        for i in range(n_f):
            row = np.delete(dmat_f[i], i)
            if row.size == 0:
                continue
            row.sort()
            idx = min(k_f - 1, row.size - 1)
            balloon_est.append(row[idx])
        if len(balloon_est) == 0:
            pos = dmat_f[dmat_f > 0]
            med = float(np.median(pos)) if pos.size > 0 else 1.0
            bandwidth_f = 1e-3 * med
        else:
            balloon_est = np.sort(np.asarray(balloon_est))
            bandwidth_f = balloon_est[len(balloon_est) // 2]

        # enforce strictly positive, sane bandwidths
        bandwidth_r = _safe_bandwidth(real_features, bandwidth_r)
        bandwidth_f = _safe_bandwidth(fake_features, bandwidth_f)

        # estimation of confidence band and manifold
        c_r, score_rr = confband_est(
            data=real_features, h=bandwidth_r, alpha=alpha, kernel=kernel, p_val=True
        )
        c_g, score_gg = confband_est(
            data=fake_features, h=bandwidth_f, alpha=alpha, kernel=kernel, p_val=True
        )

    # Replace NaNs (can arise with degenerate bandwidths) by zeros so comparisons work
    if np.isnan(score_rr).any():
        score_rr = np.nan_to_num(score_rr, nan=0.0)
    if np.isnan(score_gg).any():
        score_gg = np.nan_to_num(score_gg, nan=0.0)

    # count significant real & fake samples
    num_real = np.sum(score_rr > c_r)
    num_fake = np.sum(score_gg > c_g)

    # count significant fake samples on real manifold
    score_rg = compact_KDE(fake_features, real_features, bandwidth_f, kernel=kernel)
    inter_r = np.sum((score_rr > c_r) * (score_rg > c_g))

    # count significant real samples on fake manifold
    score_gr = compact_KDE(real_features, fake_features, bandwidth_r, kernel=kernel)
    inter_g = np.sum((score_gg > c_g) * (score_gr > c_r))

    # Avoid divide-by-zero; if no significant samples, precision/recall are 0.0
    t_precision = (inter_g / num_fake) if num_fake > 0 else 0.0
    t_recall = (inter_r / num_real) if num_real > 0 else 0.0

    # top f1-score
    if f1_score:
        if t_precision > 0.0001 and t_recall > 0.0001:
            F1_score = 2 / ((1 / t_precision) + (1 / t_recall))
        else:
            F1_score = 0
        return dict(fidelity=t_precision, diversity=t_recall, Top_F1=F1_score)
    else:
        return t_precision, t_recall


def PCA(features_ref, features, pca_dim=100, whiten=False):
    """Concatenate features and apply PCA, return the splitted features"""
    all_features = np.concatenate([features_ref, features], axis=0)
    all_features_mean = np.mean(all_features, axis=0, keepdims=True)
    all_features_centered = all_features - all_features_mean
    cov = np.cov(all_features_centered, rowvar=False)
    U, S, Vt = np.linalg.svd(cov)
    W = U[:, :pca_dim]
    if whiten:
        W = W / np.sqrt(S[:pca_dim] + 1e-5)
    all_features_pca = np.dot(all_features_centered, W)
    features_ref_pca = all_features_pca[: features_ref.shape[0]]
    features_pca = all_features_pca[features_ref.shape[0] :]
    return features_ref_pca, features_pca


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################


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
        --metrics fid toppr toppr_dino \
        --inception-ref dataset/inception-ref/inception_ref.npz    
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
    inception_file = os.path.join(out_path, INCEPTION_FILE)

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
    dino_file = os.path.join(out_path, DINO_FILE)

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
    default=("fid", "toppr"),
    help="Metrics to compute (e.g. --metrics fid --metrics toppr)",
)
@click.option(
    "--inception-ref",
    "inception_ref_file",
    metavar="NPZ|URL",
    type=str,
    required=False,
    help="Dataset reference Inception statistics (required for FID)",
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
    inception_ref_file,
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
    metrics = [m.lower() for m in metrics]
    click.echo(f"Metrics requested: {metrics}")

    if "fid" in metrics and inception_ref_file is None:
        raise click.UsageError(
            "--inception-ref is required when 'fid' is included in --metrics"
        )

    # Checking files.
    out_path = os.path.dirname(img_path)
    inception_file = os.path.join(out_path, INCEPTION_FILE)
    dino_file = os.path.join(out_path, DINO_FILE)

    if "fid" in metrics and not os.path.exists(inception_ref_file):
        raise FileNotFoundError(f"Missing ref {inception_ref_file}. Run `ref` first.")

    if not os.path.exists(inception_file):
        raise FileNotFoundError(f"Missing stats {inception_file}. Run `stats` first.")

    if not os.path.exists(dino_file):
        raise FileNotFoundError(f"Missing stats {dino_file}. Run `dino-stats` first.")

    # Distribute.
    torch.multiprocessing.set_start_method("spawn")
    dist.init()

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Loading Inception ref.
    if set(metrics) & {"fid", "toppr"}:
        dist.print0("Loading Inception reference...")

        label = 0
        ref_files = [inception_ref_file]
        while os.path.exists(inception_ref_file.split(".")[0] + f"-{label}.npz"):
            ref_files.append(inception_ref_file.split(".")[0] + f"-{label}.npz")
            label += 1

        inception_refs = []
        for ref_file in ref_files:
            dist.print0(f"Using dataset reference statistics from: {ref_file}")

            if dist.get_rank() == 0:
                with dnnlib.util.open_url(ref_file) as f:
                    inception_refs.append(dict(np.load(f)))
                dist.print0(f"Ref file keys: {list(inception_refs[-1].keys())}")
                dist.print0(len(inception_refs[-1]["inception_feats"]))

    # Load Inception stats.
    if set(metrics) & {"fid", "toppr"}:
        dist.print0("Loading Inception statistics...")

        if dist.get_rank() == 0:
            with dnnlib.util.open_url(inception_file) as f:
                feature_settings = dict(np.load(f, allow_pickle=True).item())
                inception_feats = feature_settings["inception_feats"]
                inception_mus = feature_settings["inception_mus"]
                inception_sigmas = feature_settings["inception_sigmas"]

    # Load DINO stats.
    if set(metrics) & {"toppr-dino"}:
        dist.print0("Loading DINO statistics...")

        if dist.get_rank() == 0:
            with dnnlib.util.open_url(dino_file) as f:
                feature_settings = dict(np.load(f, allow_pickle=True).item())
                dino_feats = feature_settings["dino_feats"]
                dino_mus = feature_settings["dino_mus"]
                dino_sigmas = feature_settings["dino_sigmas"]

    # Prepare computations.
    num_labels = len(inception_mus) - 1
    data = {"overall": {}}
    data.update({i: {} for i in range(num_labels)})

    def metric_key(i):
        return "overall" if i == 0 else (i - 1)

    # Compute FID.
    if "fid" in metrics:
        dist.print0("Computing FID...")

        if dist.get_rank() == 0:
            for i in range(1 + num_labels):
                metric_name = "Overall" if i == 0 else f"Label {i}"
                dist.print0(f"Computing {metric_name} FID on {img_path}.")

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
        dist.print0("Computing PRDC with TopP&R...")
        dist.print0(
            f"Settings — "
            f"alpha={tpr_alpha}, randproj={tpr_randproj}, l2norm={tpr_l2norm}, n={tpr_n} (out of 20k), repeats={tpr_reps}"
        )

        num_features = [f.shape[0] for f in inception_feats]
        features = [f[:20000] for f in inception_feats]

        if dist.get_rank() == 0:
            for i in range(1 + num_labels):
                metric_name = "Overall" if i == 0 else f"Label {i}"
                dist.print0(f"Computing {metric_name} PRDC on {img_path} with TopP&R.")

                # Inception TopPR
                Ps, Rs, Ds, Cs = [], [], [], []
                pool_size = min(20_000, features[i].shape[0])
                use_n = min(tpr_n, pool_size)

                num_test = tpr_reps

                for _ in range(num_test):
                    index = np.random.permutation(pool_size)[:use_n]
                    feats_ref = inception_refs[i]["inception_feats"][index]
                    feats = features[i][index]

                    if pca:
                        feats_ref, feats = PCA(
                            feats_ref,
                            feats,
                            pca_dim=pca_dim,
                            whiten=whiten,
                        )
                    Pinc, Rinc = compute_top_pr(
                        real_features=feats_ref,
                        fake_features=feats,
                        alpha=tpr_alpha,
                        kernel="cosine",
                        random_proj=tpr_randproj,
                        f1_score=False,
                        l2norm=tpr_l2norm,
                    )

                    Dinc, Cinc = 0, 0
                    Ps.append(Pinc)
                    Rs.append(Rinc)
                    Ds.append(Dinc)
                    Cs.append(Cinc)

                Pinc = np.mean(Ps)
                Rinc = np.mean(Rs)
                Dinc = np.mean(Ds)
                Cinc = np.mean(Cs)
                Pinc_std = np.std(Ps)
                Rinc_std = np.std(Rs)
                Dinc_std = np.std(Ds)
                Cinc_std = np.std(Cs)

                dist.print0(f"{metric_name} TopP&R results:")
                dist.print0(
                    f"P: {Pinc:.4f} pm {Pinc_std:.4f}, "
                    f"R: {Rinc:.4f} pm {Rinc_std:.4f}, "
                    f"D: {Dinc:.4f} pm {Dinc_std:.4f}, "
                    f"C: {Cinc:.4f} pm {Cinc_std:.4f}"
                )

                key = metric_key(i)
                data[key].update(
                    {
                        "P": Pinc,
                        "R": Rinc,
                        "D": Dinc,
                        "C": Cinc,
                        "P_std": Pinc_std,
                        "R_std": Rinc_std,
                        "D_std": Dinc_std,
                        "C_std": Cinc_std,
                        "num_features": num_features[i],
                    }
                )
    # Compute PRDC with TopP&R DINO.
    if "toppr-dino" in metrics:
        dist.print0("Computing PRDC with TopP&R DINO...")
        dist.print0(
            f"Settings — "
            f"alpha={tpr_alpha}, randproj={tpr_randproj}, l2norm={tpr_l2norm}, n={tpr_n} (out of 20k), repeats={tpr_reps}"
        )

        num_features = [f.shape[0] for f in dino_feats]
        features = [f[:20000] for f in dino_feats]

        if dist.get_rank() == 0:
            for i in range(1 + num_labels):
                metric_name = "Overall" if i == 0 else f"Label {i}"
                dist.print0(f"Computing {metric_name} PRDC on {img_path} with TopP&R.")

                # Inception TopPR
                Ps_dino, Rs_dino, Ds_dino, Cs_dino = [], [], [], []
                pool_size = min(20_000, features[i].shape[0])
                use_n = min(tpr_n, pool_size)

                num_test = tpr_reps

                # FIX: continue to rename dino

                for _ in range(num_test):
                    index = np.random.permutation(pool_size)[:use_n]
                    feats_ref = inception_refs[i]["inception_feats"][index]
                    feats = features[i][index]

                    if pca:
                        feats_ref, feats = PCA(
                            feats_ref,
                            feats,
                            pca_dim=pca_dim,
                            whiten=whiten,
                        )
                    Pinc, Rinc = compute_top_pr(
                        real_features=feats_ref,
                        fake_features=feats,
                        alpha=tpr_alpha,
                        kernel="cosine",
                        random_proj=tpr_randproj,
                        f1_score=False,
                        l2norm=tpr_l2norm,
                    )

                    Dinc, Cinc = 0, 0
                    Ps.append(Pinc)
                    Rs.append(Rinc)
                    Ds.append(Dinc)
                    Cs.append(Cinc)

                Pinc = np.mean(Ps)
                Rinc = np.mean(Rs)
                Dinc = np.mean(Ds)
                Cinc = np.mean(Cs)
                Pinc_std = np.std(Ps)
                Rinc_std = np.std(Rs)
                Dinc_std = np.std(Ds)
                Cinc_std = np.std(Cs)

                dist.print0(f"{metric_name} TopP&R results:")
                dist.print0(
                    f"P: {Pinc:.4f} pm {Pinc_std:.4f}, "
                    f"R: {Rinc:.4f} pm {Rinc_std:.4f}, "
                    f"D: {Dinc:.4f} pm {Dinc_std:.4f}, "
                    f"C: {Cinc:.4f} pm {Cinc_std:.4f}"
                )

                key = metric_key(i)
                data[key].update(
                    {
                        "P": Pinc,
                        "R": Rinc,
                        "D": Dinc,
                        "C": Cinc,
                        "P_std": Pinc_std,
                        "R_std": Rinc_std,
                        "D_std": Dinc_std,
                        "C_std": Cinc_std,
                        "num_features": num_features[i],
                    }
                )

    with open(os.path.join(out_path, "evaluation.jsonl"), "w") as file:
        json.dump(data, file, indent=4)

    torch.distributed.barrier()


####################################################################################################
####################################################################################################
####################################################################################################


if __name__ == "__main__":
    main()
