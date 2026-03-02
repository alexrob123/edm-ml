import gudhi
import numpy as np
import sklearn
import torch
from scipy.spatial import distance
from tqdm import tqdm

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
    print("computing top pr")
    print(real_features.shape, fake_features.shape)

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
        print("Using bandwidth estimator for low-dimensional features...")

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
        print("Using robust balloon estimator for high-dimensional features...")
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
        print("Warning: NaN values found in score_rr, replacing with zeros.")
        score_rr = np.nan_to_num(score_rr, nan=0.0)
    if np.isnan(score_gg).any():
        print("Warning: NaN values found in score_gg, replacing with zeros.")
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
        print(
            f"Bandwidth (real): {bandwidth_r:.4f}, Bandwidth (fake): {bandwidth_f:.4f}"
        )
        print(f"Num real: {num_real}, Num fake: {num_fake}")
        print(f"Top Precision: {t_precision:.4f}, Top Recall: {t_recall:.4f}")
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
