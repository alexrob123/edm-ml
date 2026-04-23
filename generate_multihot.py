# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import io
import json
import math
import os
import pickle
import re
import time
import zipfile

import click
import numpy as np
import PIL.Image
import torch

import dnnlib
from datatools.multilabel import all_bvec, bvec2int
from torch_utils import distributed as dist
from training.generator import StackedRandomGenerator, edm_sampler, seed_batch

# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# Parse a hyphen- or comma-separated list of floats like '20-80' or '30,10,60'.
# Returns a list of floats.


def parse_float_list(s):
    if s is None:
        return None
    if isinstance(s, (list, tuple, np.ndarray)):
        return [float(x) for x in s]
    parts = re.split(r"[-,]", s.strip())
    return [float(p) for p in parts if p != ""]


def parse_multihot_list(s, expected_dim=None):
    if isinstance(s, list):
        s = "".join(str(x) for x in s)
    s = s.replace(" ", "").replace(",", "").replace("-", "").replace("_", "")
    vals = [int(x) for x in s]
    if expected_dim is not None and len(vals) != expected_dim:
        raise click.ClickException(f"Expected {expected_dim} entries, got {len(vals)}.")
    if any(v not in (0, 1) for v in vals):
        raise click.ClickException("Multi-hot values must be 0 or 1.")
    return vals


# ----------------------------------------------------------------------------


def process_target_prior(
    target_prior_str, is_cond_model, net_label_dim, multihot, device
):
    """
    If multihot, the target_prior corresponds to the distribution of the 2^label_dim possible multi-hot vectors.
    """
    if target_prior_str is None:
        return None

    tp = torch.tensor(
        parse_float_list(target_prior_str),
        device=device,
        dtype=torch.float32,
    )

    if (tp <= 0).any():
        raise click.ClickException("All target-prior entries must be positive.")

    # Normalize
    target_prior = tp / tp.sum()
    dist.print0(f"Using target prior (normalized): {target_prior.tolist()}")

    # For conditional models, ensure prior length matches label_dim
    # if "is_cond_model" in locals() # useless check since we always define is_cond_model
    if is_cond_model and net_label_dim > 0:
        target_len = 2**net_label_dim if multihot else net_label_dim
        if target_prior.numel() != target_len:
            raise click.ClickException(
                f"Target prior length: {target_prior.numel()} not fit for model label_dim: {target_len} "
                f"(multihot={multihot})."
            )

    return target_prior


# ----------------------------------------------------------------------------


def sample_labels(
    net_label_dim,
    is_cond_model,
    multihot,
    target_prior,
    class_idx,
    max_batch_size,
    rnd,
    device,
):
    """
    If model has null label_dim return None.
    Otherwise return a (B, label_dim) tensor of labels sampled according to the target_prior or uniformly.
    Finally if class_idx is set, overwrite the sampled labels with class_idx.
    """

    labels = None

    # If model accepts labels, define labels
    if net_label_dim:
        if is_cond_model:
            if (target_prior is not None) and (class_idx is None):
                cdf = torch.cumsum(target_prior, dim=0)
                u = rnd.rand([max_batch_size], device=device)

                if multihot:
                    labelset_dim = 2 ** int(net_label_dim)
                    idx = torch.searchsorted(cdf, u, right=False).clamp(
                        max=labelset_dim - 1
                    )
                    labels = all_bvec(net_label_dim, device=device)[idx]
                else:
                    idx = torch.searchsorted(cdf, u, right=False).clamp(
                        max=int(net_label_dim) - 1
                    )
                    labels = torch.eye(net_label_dim, device=device)[idx]

        # Either model is uncond, target prior is None or class_idx is set: sample uniformly
        if labels is None:
            if multihot:
                labels = rnd.randint(
                    2,
                    size=[max_batch_size, net_label_dim],
                    device=device,
                )
            else:
                labels = torch.eye(net_label_dim, device=device)[
                    rnd.randint(net_label_dim, size=[max_batch_size], device=device)
                ]

        # If class_idx is set, overwrite labels with class_idx
        if class_idx is not None:
            if multihot:
                labels[:, class_idx] = 1
            else:
                labels[:, :] = 0
                labels[:, class_idx] = 1

    return labels


def sample_class_labels(
    net_label_dim,
    is_cond_model,
    multihot,
    target_prior,
    class_idx,
    max_batch_size,
    rnd,
    device,
):

    if net_label_dim:
        if is_cond_model:
            # Target prior and no class_idx
            if (target_prior is not None) and (class_idx is None):
                cdf = torch.cumsum(target_prior, dim=0)
                u = rnd.rand([max_batch_size], device=device)
                idx = torch.searchsorted(cdf, u, right=False).clamp(
                    max=int(net_label_dim) - 1
                )
                class_labels = torch.eye(net_label_dim, device=device)[idx]
            # No target prior or class_idx
            else:
                if multihot:
                    class_labels = rnd.randint(
                        2,
                        size=[max_batch_size, net_label_dim],
                        device=device,
                    )
                else:
                    class_labels = torch.eye(net_label_dim, device=device)[
                        rnd.randint(net_label_dim, size=[max_batch_size], device=device)
                    ]
        # If model is unconditional, keep uniform placeholder if present
        else:
            if multihot:
                class_labels = rnd.randint(
                    2,
                    size=[max_batch_size, net_label_dim],
                    device=device,
                )
            else:
                class_labels = torch.eye(net_label_dim, device=device)[
                    rnd.randint(net_label_dim, size=[max_batch_size], device=device)
                ]

    # If net has no label_dim, set class_labels to None
    else:
        class_labels = None

    if class_idx is not None and class_labels is not None:
        if multihot:
            class_labels[:, class_idx] = 1
        else:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

    return class_labels


# ----------------------------------------------------------------------------


def _ceil_div(a, b):
    return math.ceil(float(a) / float(max(b, 1e-12)))


def _format_counts(cnts: torch.Tensor, max_items: int = 20) -> str:
    try:
        arr = cnts.detach().to("cpu", non_blocking=True).tolist()
    except Exception:
        arr = [int(x) for x in cnts.view(-1).tolist()]
    K = len(arr)
    if K <= max_items:
        return "[" + ",".join(str(int(x)) for x in arr) + "]"
    head = ",".join(str(int(x)) for x in arr[: max_items // 2])
    tail = ",".join(str(int(x)) for x in arr[-(max_items // 2) :])
    # return "[" + head + ", …, " + tail + "] (K={K})"
    return f"[{head}, …, {tail}] (K={K})"


def compute_local_targets(prior, curr_counts, per_rank_total, per_rank_min):
    """Return (T, tgt_per_class) for this rank so that:
    - sum(tgt_per_class) = T
    - tgt_per_class[k] >= per_rank_min
    - tgt_per_class[k] ~ prior[k] * T (rounded up)
    - T is the smallest integer >= per_rank_total such that tgt_per_class[k] >= curr_counts[k] for all k
    """
    K = int(curr_counts.numel())
    p = prior[:K].to(curr_counts.device, dtype=torch.float32)
    p = p / p.sum()
    # Lower bound on T so that we can reach current counts and per-class min without deletions
    lb_from_curr = max(
        _ceil_div(int(curr_counts[k].item()), float(p[k].item())) if p[k] > 0 else 0
        for k in range(K)
    )
    lb_from_min = (
        max(
            _ceil_div(int(per_rank_min), float(p[k].item())) if p[k] > 0 else 0
            for k in range(K)
        )
        if per_rank_min > 0
        else 0
    )
    T = max(int(per_rank_total), int(lb_from_curr), int(lb_from_min))
    while True:
        tgt = torch.ceil(p * float(T)).to(curr_counts.device, dtype=torch.long)
        if per_rank_min > 0:
            tgt = torch.maximum(tgt, torch.full_like(tgt, int(per_rank_min)))
        if bool((tgt >= curr_counts).all().item()):
            break
        T += 1
    return T, tgt


def can_end(
    world_size,
    num_samples,
    min_per_class,
    gen_per_class,
    total_generated,
    target_prior,
):
    ws = world_size

    # Per-rank targets
    per_rank_total = (int(num_samples) + ws - 1) // ws
    per_rank_min = (int(min_per_class) + ws - 1) // ws if int(min_per_class) > 0 else 0

    # If no prior: stop when per-rank total reached and per-class minimums met
    if target_prior is None:
        cond_total = total_generated >= per_rank_total
        cond_min = True
        if gen_per_class.numel() > 0 and per_rank_min > 0:
            cond_min = bool((gen_per_class >= per_rank_min).all().item())
        return cond_total and cond_min

    # With a prior: compute per-class per-rank targets proportionally, respecting per-class minimum
    num_classes = int(gen_per_class.numel())
    prior = target_prior[:num_classes].to(gen_per_class.device, dtype=torch.float32)
    prior = prior / prior.sum()
    per_class_targets = torch.ceil(prior * float(per_rank_total)).to(
        gen_per_class.device, dtype=torch.long
    )
    if per_rank_min > 0:
        per_class_targets = torch.maximum(
            per_class_targets, torch.full_like(per_class_targets, per_rank_min)
        )
    cond_total = total_generated >= per_rank_total
    cond_quota = bool((gen_per_class.to(torch.long) >= per_class_targets).all().item())
    return cond_total and cond_quota


# ----------------------------------------------------------------------------


@click.command()
@click.option("--network", "network_pkl",           help="Network pickle filename", metavar="PATH|URL",                         type=str, required=True)  # fmt: skip
# Storage options
@click.option("--outdir",                           help="Where to save the output images", metavar="DIR",                      type=str)  # fmt: skip
@click.option("--subdirs",                          help="Create subdirectory for every 1000 samples",                          is_flag=True, default=True)  # fmt: skip
@click.option("--no-zip",                           help="Compress the output directory",                                       is_flag=True, default=False)  # fmt: skip
# Number of sample and distribution
@click.option("--num-samples",                      help="Number of samples", metavar="INT",                                    type=click.IntRange(min=0), default=50000, show_default=True)  # fmt: skip
@click.option("--min-per-class", "min_per_class",   help="Ensure at least samples per class across all ranks", metavar="INT",   type=click.IntRange(min=0), default=20000, show_default=True)  # fmt: skip
@click.option("--target-prior", "target_prior_str", help="Target class prior as '20-80' or '30-10-60' (will be renormalized).", type=str, default=None)  # fmt: skip
@click.option("--class", "class_idx",               help="Class label  [default: random]", metavar="INT",                       type=click.IntRange(min=0), default=None)  # fmt: skip
@click.option("--multihot",                         help="Allow multiple hot labels per sample (for conditional models only)",  is_flag=True, default=False)  # fmt: skip
# Sampler options
@click.option("--steps", "num_steps",               help="Number of sampling steps", metavar="INT",                             type=click.IntRange(min=1), default=18, show_default=True)  # fmt: skip
@click.option("--sigma_min",                        help="Lowest noise level  [default: varies]", metavar="FLOAT",              type=click.FloatRange(min=0.01, min_open=True))  # fmt: skip
@click.option("--sigma_max",                        help="Highest noise level  [default: varies]", metavar="FLOAT",             type=click.FloatRange(min=0, min_open=True))  # fmt: skip
@click.option("--rho",                              help="Time step exponent", metavar="FLOAT",                                 type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)  # fmt: skip
@click.option("--S_churn", "S_churn",               help="Stochasticity strength", metavar="FLOAT",                             type=click.FloatRange(min=0), default=0, show_default=True)  # fmt: skip
@click.option("--S_min", "S_min",                   help="Stoch. min noise level", metavar="FLOAT",                             type=click.FloatRange(min=0), default=0, show_default=True)  # fmt: skip
@click.option("--S_max", "S_max",                   help="Stoch. max noise level", metavar="FLOAT",                             type=click.FloatRange(min=0), default="inf", show_default=True)  # fmt: skip
@click.option("--S_noise", "S_noise",               help="Stoch. noise inflation", metavar="FLOAT",                             type=float, default=1, show_default=True)  # fmt: skip
@click.option("--batch", "max_batch_size",          help="Maximum batch size", metavar="INT",                                   type=click.IntRange(min=1), default=128, show_default=True)  # fmt: skip
@click.option("--clf-uncond",                       help="Infer class labels from classifier for unconditional generation",     is_flag=True, default=False)  # fmt: skip
def main(
    network_pkl,
    outdir,
    subdirs,
    num_samples,
    class_idx,
    max_batch_size,
    target_prior_str,
    min_per_class,
    multihot=False,
    device=torch.device("cuda"),
    no_zip=False,
    clf_uncond=False,
    **sampler_kwargs,
):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    # Generate 50k images using 2 GPUs using model $model
    torchrun --standalone --nproc_per_node=2  generate.py --num_samples=50000 \\
                --network=training-runs/$model --subdirs --w_boost 1.0

    Notes on Multi-hot sampling and target_prior:
    - In multihot mode, target_prior corresponds to the distribution of the labelsets (2** label_dim).
    - To sample a specific labelset, use --target-prior with a one-hot vector corresponding to desired labelset 
    - To sample from a specific label, use --class with desired label (other labels are random)
    """
    dist.init()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    dist.print0("Distributed setting initialized:")
    dist.print0(f"  world size: {world_size}")
    dist.print0(f"  rank: {rank}")

    # Show parameters
    dist.print0("Generating with the following constraints:")
    dist.print0(f"  multihot: {multihot}")
    dist.print0(f"  num_samples: {num_samples}")
    dist.print0(f"  class_idx: {class_idx}")
    dist.print0(f"  target_prior: {target_prior_str}")
    dist.print0(f"  min_per_class: {min_per_class}")

    dist.print0("Saving samples with the following constraints:")
    dist.print0(f"  subdirs: {subdirs}")
    dist.print0(f"  no_zip: {no_zip}")

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)["ema"].to(device)

    # Load classifier for labeling generated images (for unconditional models or when label_dim == 0).
    if clf_uncond:
        classifer_kwargs = dnnlib.EasyDict(
            class_name="training.classifier.Classifier", url=network_pkl
        )
        classif = dnnlib.util.construct_class_by_name(**classifer_kwargs).to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # --- Determine if the model is conditional or unconditional ---
    model_name = os.path.basename(network_pkl)
    name_cond = "-cond-" in model_name
    name_uncond = "-uncond-" in model_name
    if name_cond != name_uncond:
        is_cond_model = name_cond
    else:
        # Fallback to architecture attribute
        is_cond_model = bool(getattr(net, "label_dim", 0))
    dist.print0(
        f"Model type detected: {'cond' if is_cond_model else 'uncond'}"
        f" (from '{model_name}')"
    )

    net_label_dim = int(getattr(net, "label_dim", 0))

    # Process target prior.
    # if mulihot, target_prior corresponds to the distribution of the labelsets (2** label_dim)
    target_prior = process_target_prior(
        target_prior_str,
        is_cond_model,
        net_label_dim,
        multihot,
        device,
    )

    # --- Rejection sampling setup (optional) ---

    # File management.
    zip_filename = f"generated_images_worker_{dist.get_rank()}.zip"
    dataset_json = {"labels": []}
    if no_zip:
        dist.print0(f'Generating {num_samples} images to "{outdir}"...')
    else:
        print(
            f'Generating {num_samples} images to "{outdir}" '
            f'and saving them to "{zip_filename}"...'
        )
        zip_path = os.path.join(outdir, zip_filename)
        os.makedirs(outdir, exist_ok=True)

    # One-shot startup plan; targets per rank
    per_rank_total = (int(num_samples) + world_size - 1) // world_size
    per_rank_min = (
        (int(min_per_class) + world_size - 1) // world_size
        if int(min_per_class) > 0
        else 0
    )
    local_target_T = None
    local_target_per_class = None

    try:
        dev_name = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        dev_name = "cpu"
    if dist.get_rank() == 0:
        print(
            f"Plan: world_size={world_size}, "
            f"per-rank total≈{per_rank_total}, "
            f"per-class min≈{per_rank_min}, "
            f"device0='{dev_name}'"
        )
    print(
        f"[rank {rank}] "
        f"target_total≈{per_rank_total}, "
        f"per-class_min≈{per_rank_min}, "
        f"device='{dev_name}'"
    )

    regulation_started = False

    # Per-rank generation counters
    # in the multihot case with target prior, we keep track of labelsets
    if net_label_dim > 0 and target_prior is not None:
        gen_per_class = torch.zeros_like(target_prior, dtype=torch.long)
    else:
        gen_per_class = torch.zeros(net_label_dim, device=device, dtype=torch.long)
    total_generated = 0
    id = 0
    local_done = False

    # --- Loop over batches ---

    # Logging controls (reduced chatter)
    LOG_SECS = 5.0  # at most every 30s
    LOG_ITERS = 20  # or every 200 iters
    last_log_time = time.time()
    iter_idx = 0

    # Prepare a list to hold image data and their filenames
    image_data_list = []

    # While loop until we have accepted enough samples for this rank
    while True:
        if local_done:
            done_tensor = torch.tensor(1, device=device, dtype=torch.int32)
            torch.distributed.all_reduce(done_tensor, op=torch.distributed.ReduceOp.SUM)
            if int(done_tensor.item()) == int(world_size):
                break
            time.sleep(0.05)
            continue

        batch_seeds = seed_batch(max_batch_size, id, dist.get_rank())
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [max_batch_size, net.img_channels, net.img_resolution, net.img_resolution],
            device=device,
        )

        # Determine labels.
        # labels are either None or in hot encoding format with shape (B, label_dim)
        labels = sample_labels(
            net_label_dim,
            is_cond_model,
            multihot,
            target_prior,
            class_idx,
            max_batch_size,
            rnd,
            device,
        )

        # Generate images.
        sampler_kwargs = {
            key: value for key, value in sampler_kwargs.items() if value is not None
        }
        images = edm_sampler(
            net,
            latents,
            labels,
            randn_like=rnd.randn_like,
            **sampler_kwargs,
        )

        # Save images.
        images_np = (
            (images * 127.5 + 128)
            .clip(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )

        # Save labels.
        if labels is None or labels.numel() == 0:
            # if no labels provided by the net, classify generated images (unconditional models)
            if clf_uncond and multihot:
                # FIX
                raise NotImplementedError(
                    "Classification of generated images for unconditional multihot models is not implemented yet."
                )
            elif clf_uncond:
                logits = classif((images.clip(-1, 1) + 1) / 2).logits
                labels = logits.argmax(dim=1, keepdim=True)
        else:
            if not multihot:
                labels = labels.argmax(dim=1, keepdim=True)

        # labels shape is now (B, 1) for single-label or (B, label_dim) for multi-label

        # Lazy-init per-class counters for unconditional models (or when label_dim == 0)
        if gen_per_class.numel() == 0:
            if target_prior is not None:
                K = int(target_prior.numel())
            else:
                # Infer K from observed labels in this batch
                if multihot:
                    K = labels.size(1) if labels.numel() > 0 else 0
                else:
                    K = int(labels.max().item()) + 1 if labels.numel() > 0 else 0
            if K > 0:
                gen_per_class = torch.zeros(K, device=device, dtype=torch.long)

        # Phase switch for unconditional models: after warmup to per-rank total, compute local per-class targets
        # if no target_prior, regulation will not happen and we will just generate until per-rank total is reached
        # if no target_prior, local_target_per_class will remain None forever
        # if target_prior is set, if mulithot, local_target_per_class will correspond to the target counts of the labelsets, hence have labelset dim
        # we can say that if mulithot local_target_per_class has dim labelset_dim, else has dim label_dim
        if (
            (not regulation_started)
            and (not is_cond_model)
            and (total_generated >= per_rank_total)
            and (target_prior is not None)
            and (gen_per_class.numel() > 0)
        ):
            local_target_T, local_target_per_class = compute_local_targets(
                target_prior,
                gen_per_class,
                per_rank_total,
                per_rank_min,
            )
            regulation_started = True
            print(
                f"[Rank] {rank} - [Regulation] Starting with "
                f"T={local_target_T}, "
                f"per-rank min={per_rank_min}"
            )

        # Deterministic admission after regulation starts: keep only classes that still need quota
        if (
            regulation_started
            and local_target_per_class is not None
            and gen_per_class.numel() == local_target_per_class.numel()
        ):
            remaining = (local_target_per_class - gen_per_class).clamp_min(0)

            if labels.size(1) > 1:  # multihot
                labels_1d = bvec2int(labels).view(-1).to(torch.long)
            else:
                labels_1d = labels.view(-1).to(torch.long)

            keep_mask = torch.zeros(labels_1d.shape[0], device=device, dtype=torch.bool)

            take_local = torch.zeros_like(remaining)
            for i in range(labels_1d.shape[0]):
                c = int(labels_1d[i].item())
                if 0 <= c < remaining.numel() and take_local[c] < remaining[c]:
                    keep_mask[i] = True
                    take_local[c] += 1

            # Apply mask (drop over-represented classes)
            if keep_mask.sum().item() < labels_1d.shape[0]:
                images_np = images_np[keep_mask.cpu().numpy()]
                labels = labels[keep_mask]
                batch_seeds = batch_seeds[keep_mask.cpu()]

        # Update per-class counters from the kept labels of this batch
        if labels is not None and labels.numel() > 0 and gen_per_class.numel() > 0:
            if multihot:
                if labels.size(1) == gen_per_class.numel():
                    # tracking labels
                    lbl = torch.as_tensor(labels, device=device, dtype=torch.long)
                    binc = lbl.sum(dim=0)
                else:
                    # tracking labelsets
                    lbl = bvec2int(labels).view(-1).to(torch.long)
                    binc = torch.bincount(lbl, minlength=gen_per_class.numel())
            else:
                lbl = torch.as_tensor(labels.view(-1), device=device, dtype=torch.long)
                binc = torch.bincount(lbl, minlength=gen_per_class.numel())

            gen_per_class[: binc.numel()] += binc.to(torch.long)

        # If under regulation, stop when local per-class targets are reached
        if regulation_started and local_target_per_class is not None:
            if bool((gen_per_class >= local_target_per_class).all().item()):
                local_done = True

        # Convert labels to numpy for saving
        labels_np = labels.cpu().numpy() if multihot else labels.view(-1).cpu().numpy()

        # If nothing accepted this round, continue to try again
        if len(labels_np) == 0:
            pass

        # Ensure seeds are plain ints for formatting
        if torch.is_tensor(batch_seeds):
            batch_seeds_iter = [int(s) for s in batch_seeds.tolist()]
        else:
            batch_seeds_iter = batch_seeds

        # Save images and update dataset
        for seed, image_np, label in zip(batch_seeds_iter, images_np, labels_np):
            seed_int = int(seed)

            label_str = (
                "".join([str(int(x)) for x in label]) if multihot else str(int(label))
            )

            if no_zip or seed_int < 100:
                image_dir = (
                    os.path.join(
                        outdir, f"{seed_int - seed_int % 1000:06d}", f"{label_str}"
                    )
                    if subdirs
                    else outdir
                )
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f"{seed_int:06d}.png")
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], "L").save(image_path)
                else:
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)

            if not no_zip:
                if image_np.shape[2] == 1:
                    img = PIL.Image.fromarray(image_np[:, :, 0], "L")
                else:
                    img = PIL.Image.fromarray(image_np, "RGB")
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                image_filename = f"{seed_int:06d}.png"
                image_data_list.append((image_filename, img_bytes.getvalue()))
                label_json = label.astype(int).tolist() if multihot else int(label)
                dataset_json["labels"].append([f"{seed_int:06d}.png", label_json])

            total_generated += 1

        # Stop early if no prior or before regulation kicks in
        if (target_prior is None) or (not regulation_started):
            if can_end(
                world_size,
                num_samples,
                min_per_class,
                gen_per_class,
                total_generated,
                target_prior,
            ):
                local_done = True

        # Optional: throttled progress print (rank 0 only)
        iter_idx += 1
        now = time.time()

        log_sec_multiplier = 1 if rank == 0 else 100
        log_iters_multiplier = 1 if rank == 0 else 10
        should_log = (now - last_log_time) >= (LOG_SECS * log_sec_multiplier) or (
            iter_idx % (LOG_ITERS * log_iters_multiplier) == 0
        )
        if should_log:
            # if (rank == 0) and (
            #     (now - last_log_time) >= LOG_SECS or (iter_idx % LOG_ITERS == 0)
            # ):
            last_log_time = now
            try:
                mem_used = (
                    torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]
                ) / 1e9
                mem_tot = torch.cuda.mem_get_info()[1] / 1e9
                mem_str = f"mem {mem_used:.2f}/{mem_tot:.2f} GB"
            except Exception:
                mem_str = "mem N/A"
            per_class_min_now = (
                int(gen_per_class.min().item()) if gen_per_class.numel() else 0
            )
            extra = ""
            if (
                regulation_started
                and local_target_per_class is not None
                and gen_per_class.numel() == local_target_per_class.numel()
            ):
                rem_min = int(
                    (local_target_per_class - gen_per_class).clamp_min(0).min().item()
                )
                extra = f" | reg on, rem-min {rem_min}"
            per_class_counts_str = _format_counts(gen_per_class)
            print(
                f"[Rank {rank}] generated {total_generated} "
                f"| per-class min {per_class_min_now} "
                f"| counts {per_class_counts_str} "
                f"| {mem_str}{extra}"
            )

        done_tensor = torch.tensor(
            1 if local_done else 0,
            device=device,
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(done_tensor, op=torch.distributed.ReduceOp.SUM)
        if int(done_tensor.item()) == int(world_size):
            break

        id += 1
    print(f"[rank {rank}] accepted {total_generated} images")
    # Done.

    if not no_zip:
        # Sort the images by filename
        image_data_list.sort(key=lambda x: x[0])

        # Open a ZIP file to collect all sorted images
        with zipfile.ZipFile(zip_path, "w") as myzip:
            for filename, data in image_data_list:
                # Write sorted image data to the ZIP file
                myzip.writestr(filename, data)
            myzip.writestr("dataset.json", json.dumps(dataset_json))

    torch.distributed.barrier()
    dataset_json_all = {"labels": []}
    if dist.get_rank() == 0:
        all_zip_files = [
            os.path.join(outdir, f)
            for f in os.listdir(outdir)
            if f.startswith("generated_images_worker") and f.endswith(".zip")
        ]

        # Merge all ZIP files
        final_zip_path = os.path.join(outdir, "generated_images.zip")
        with zipfile.ZipFile(final_zip_path, "w") as final_zip:
            for zip_file in all_zip_files:
                with zipfile.ZipFile(zip_file, "r") as zfile:
                    for file_name in zfile.namelist():
                        # Extract file data from the zip
                        with zfile.open(file_name) as file:
                            file_data = file.read()
                        if file_name == "dataset.json":
                            dataset_json_all["labels"].extend(
                                json.loads(file_data)["labels"]
                            )
                        else:
                            # Write file to the final zip, maintaining directory structure if necessary
                            final_zip.writestr(file_name, file_data)
            final_zip.writestr("dataset.json", json.dumps(dataset_json_all))

        # Optionally, delete individual worker ZIP files after merging
        for zip_file in all_zip_files:
            os.remove(zip_file)
    dist.print0("Done.")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
