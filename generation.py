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


# ----------------------------------------------------------------------------


@click.command()
@click.option(
    "--network",
    "network_pkl",
    help="Network pickle filename",
    metavar="PATH|URL",
    type=str,
    required=True,
)
@click.option(
    "--outdir",
    help="Where to save the output images",
    metavar="DIR",
    type=str,
)
@click.option(
    "--num-samples",
    help="Number of samples",
    metavar="INT",
    type=click.IntRange(min=0),
    default=50000,
    show_default=True,
)
@click.option(
    "--subdirs",
    help="Create subdirectory for every 1000 samples",
    default=True,
    is_flag=True,
)
@click.option(
    "--class",
    "class_idx",
    help="Class label  [default: random]",
    metavar="INT",
    type=click.IntRange(min=0),
    default=None,
)
@click.option(
    "--batch",
    "max_batch_size",
    help="Maximum batch size",
    metavar="INT",
    type=click.IntRange(min=1),
    default=128,
    show_default=True,
)
@click.option(
    "--target-prior",
    "target_prior_str",
    help="Target class prior as '20-80' or '30-10-60' (will be renormalized).",
    type=str,
    default=None,
)
@click.option(
    "--no_zip", help="Compress the output directory", default=False, is_flag=True
)
@click.option(
    "--steps",
    "num_steps",
    help="Number of sampling steps",
    metavar="INT",
    type=click.IntRange(min=1),
    default=18,
    show_default=True,
)
@click.option(
    "--sigma_min",
    help="Lowest noise level  [default: varies]",
    metavar="FLOAT",
    type=click.FloatRange(min=0.01, min_open=True),
)
@click.option(
    "--sigma_max",
    help="Highest noise level  [default: varies]",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--rho",
    help="Time step exponent",
    metavar="FLOAT",
    type=click.FloatRange(min=0, min_open=True),
    default=7,
    show_default=True,
)
@click.option(
    "--S_churn",
    "S_churn",
    help="Stochasticity strength",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
    default=0,
    show_default=True,
)
@click.option(
    "--S_min",
    "S_min",
    help="Stoch. min noise level",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
    default=0,
    show_default=True,
)
@click.option(
    "--S_max",
    "S_max",
    help="Stoch. max noise level",
    metavar="FLOAT",
    type=click.FloatRange(min=0),
    default="inf",
    show_default=True,
)
@click.option(
    "--S_noise",
    "S_noise",
    help="Stoch. noise inflation",
    metavar="FLOAT",
    type=float,
    default=1,
    show_default=True,
)
# --- Minimum per-class option ---
@click.option(
    "--min-per-class",
    "min_per_class",
    help="Ensure at least this many samples per class across all ranks (0 to disable)",
    metavar="INT",
    type=click.IntRange(min=0),
    default=20000,
    show_default=True,
)
def main(
    network_pkl,
    outdir,
    subdirs,
    num_samples,
    class_idx,
    max_batch_size,
    target_prior_str,
    min_per_class,
    device=torch.device("cuda"),
    no_zip=False,
    **sampler_kwargs,
):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    # Generate 50k images using 2 GPUs using model $model
    torchrun --standalone --nproc_per_node=2  generate.py --num_samples=50000 \\
                --network=training-runs/$model --subdirs --w_boost 1.0
    """
    dist.init()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)["ema"].to(device)

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

    # --- Rejection sampling setup (optional) ---
    target_prior = None
    if target_prior_str is not None:
        tp = torch.tensor(
            parse_float_list(target_prior_str), device=device, dtype=torch.float32
        )
        if (tp <= 0).any():
            raise click.ClickException("All target-prior entries must be positive.")
        target_prior = tp / tp.sum()
        dist.print0(f"Using target prior (normalized): {target_prior.tolist()}")
        # For conditional models, ensure prior length matches label_dim
        if (
            "is_cond_model" in locals()
            and is_cond_model
            and int(getattr(net, "label_dim", 0)) > 0
        ):
            if target_prior.numel() != int(net.label_dim):
                raise click.ClickException(
                    f"Target prior length ({target_prior.numel()}) must equal model label_dim ({int(net.label_dim)})."
                )

    def can_end(
        num_samples, min_per_class, gen_per_class, total_generated, target_prior
    ):
        ws = world_size
        # Per-rank targets
        per_rank_total = (int(num_samples) + ws - 1) // ws
        per_rank_min = (
            (int(min_per_class) + ws - 1) // ws if int(min_per_class) > 0 else 0
        )

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
        cond_quota = bool(
            (gen_per_class.to(torch.long) >= per_class_targets).all().item()
        )
        return cond_total and cond_quota

    def _ceil_div(a, b):
        return math.ceil(float(a) / float(max(b, 1e-12)))

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
        return "[" + head + ", …, " + tail + "] (K={K})"

    # Loop over batches.
    zip_filename = f"generated_images_worker_{dist.get_rank()}.zip"
    dataset_json = {"labels": []}
    if no_zip:
        dist.print0(f'Generating {num_samples} images to "{outdir}"...')
    else:
        print(
            f'Generating {num_samples} images to "{outdir}" and saving them to "{zip_filename}"...'
        )
        zip_path = os.path.join(outdir, zip_filename)
        os.makedirs(outdir, exist_ok=True)

    # Prepare a list to hold image data and their filenames
    image_data_list = []

    # --- Logging controls (reduced chatter) ---
    LOG_SECS = 5.0  # at most every 30s
    LOG_ITERS = 20  # or every 200 iters
    last_log_time = time.time()
    iter_idx = 0

    # Per-rank generation counters
    gen_per_class = torch.zeros(
        int(getattr(net, "label_dim", 0)), device=device, dtype=torch.long
    )
    total_generated = 0
    id = 0

    regulation_started = False
    per_rank_total = (int(num_samples) + world_size - 1) // world_size
    per_rank_min = (
        (int(min_per_class) + world_size - 1) // world_size
        if int(min_per_class) > 0
        else 0
    )
    local_target_T = None
    local_target_per_class = None

    # One-shot startup plan
    try:
        dev_name = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        dev_name = "cpu"
    if dist.get_rank() == 0:
        print(
            f"Plan: world_size={world_size}, per-rank total≈{per_rank_total}, per-class min≈{per_rank_min}, device0='{dev_name}'"
        )
    print(
        f"[rank {rank}] target_total≈{per_rank_total}, per-class_min≈{per_rank_min}, device='{dev_name}'"
    )

    # While loop until we have accepted enough samples for this rank
    while True:
        batch_seeds = seed_batch(max_batch_size, id, dist.get_rank())
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [max_batch_size, net.img_channels, net.img_resolution, net.img_resolution],
            device=device,
        )
        class_labels = None
        if net.label_dim:
            if is_cond_model:
                # If a target prior is provided and not forcing a single class, sample labels from it
                if (target_prior is not None) and (class_idx is None):
                    cdf = torch.cumsum(target_prior, dim=0)
                    u = rnd.rand([max_batch_size], device=device)
                    idx = torch.searchsorted(cdf, u, right=False).clamp(
                        max=int(net.label_dim) - 1
                    )
                    class_labels = torch.eye(net.label_dim, device=device)[idx]
                else:
                    class_labels = torch.eye(net.label_dim, device=device)[
                        rnd.randint(net.label_dim, size=[max_batch_size], device=device)
                    ]
            else:
                # Unconditional model: sampler ignores labels anyway; keep uniform placeholder if present
                class_labels = torch.eye(net.label_dim, device=device)[
                    rnd.randint(net.label_dim, size=[max_batch_size], device=device)
                ]
        if class_idx is not None and class_labels is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {
            key: value for key, value in sampler_kwargs.items() if value is not None
        }
        images = edm_sampler(
            net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs
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
        if class_labels is not None:
            class_labels = class_labels.argmax(dim=1, keepdim=True)
        else:
            class_labels = classif((images.clip(-1, 1) + 1) / 2).logits.argmax(
                dim=1, keepdim=True
            )

        # Ensure we have the class dimension

        if class_labels is None or class_labels.numel() == 0:
            # If no labels provided by the net, classify generated images (unconditional models)
            class_labels = classif((images.clip(-1, 1) + 1) / 2).logits.argmax(
                dim=1, keepdim=True
            )

        # Lazy-init per-class counters for unconditional models (or when label_dim == 0)
        if gen_per_class.numel() == 0:
            if target_prior is not None:
                K = int(target_prior.numel())
            else:
                # Infer K from observed labels in this batch
                K = (
                    int(class_labels.max().item()) + 1
                    if class_labels.numel() > 0
                    else 0
                )
            if K > 0:
                gen_per_class = torch.zeros(K, device=device, dtype=torch.long)

        # Phase switch for unconditional models: after warmup to per-rank total, compute local per-class targets
        if (
            (not regulation_started)
            and (not is_cond_model)
            and (total_generated >= per_rank_total)
            and (target_prior is not None)
            and (gen_per_class.numel() > 0)
        ):
            local_target_T, local_target_per_class = compute_local_targets(
                target_prior, gen_per_class, per_rank_total, per_rank_min
            )
            regulation_started = True
            print(
                f"[Rank] {rank} - [Regulation] Starting with T={local_target_T}, per-rank min={per_rank_min}"
            )

        # Deterministic admission after regulation starts: keep only classes that still need quota
        if (
            regulation_started
            and local_target_per_class is not None
            and gen_per_class.numel() == local_target_per_class.numel()
        ):
            remaining = (local_target_per_class - gen_per_class).clamp_min(0)
            labels_1d = class_labels.view(-1).to(torch.long)
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
                class_labels = class_labels[keep_mask]
                batch_seeds = batch_seeds[keep_mask.cpu()]

        # Update per-class counters from the kept labels of this batch
        if (
            class_labels is not None
            and class_labels.numel() > 0
            and gen_per_class.numel() > 0
        ):
            lbl = torch.as_tensor(
                class_labels.view(-1), device=device, dtype=torch.long
            )
            binc = torch.bincount(lbl, minlength=gen_per_class.numel())
            gen_per_class[: binc.numel()] += binc.to(torch.long)

        # If under regulation, stop when local per-class targets are reached
        if regulation_started and local_target_per_class is not None:
            if bool((gen_per_class >= local_target_per_class).all().item()):
                break

        # Convert labels to numpy for saving
        class_labels_np = class_labels.view(-1).cpu().numpy()

        # If nothing accepted this round, continue to try again
        if len(class_labels_np) == 0:
            id += 1
            continue

        # Ensure seeds are plain ints for formatting
        if torch.is_tensor(batch_seeds):
            batch_seeds_iter = [int(s) for s in batch_seeds.tolist()]
        else:
            batch_seeds_iter = batch_seeds

        # Save images and update dataset
        for seed, image_np, label in zip(batch_seeds_iter, images_np, class_labels_np):
            seed_int = int(seed)
            label_int = int(label)
            if no_zip or seed_int < 100:
                image_dir = (
                    os.path.join(
                        outdir, f"{seed_int - seed_int % 1000:06d}", f"{label_int}"
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
                dataset_json["labels"].append([f"{seed_int:06d}.png", label_int])
            total_generated += 1

        # Stop early if no prior or before regulation kicks in
        if (target_prior is None) or (not regulation_started):
            if can_end(
                num_samples, min_per_class, gen_per_class, total_generated, target_prior
            ):
                break

        # Optional: throttled progress print (rank 0 only)
        iter_idx += 1
        now = time.time()
        if (rank == 0) and (
            (now - last_log_time) >= LOG_SECS or (iter_idx % LOG_ITERS == 0)
        ):
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
                f"[Rank 0] generated {total_generated} | per-class min {per_class_min_now} | counts {per_class_counts_str} | {mem_str}{extra}"
            )

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
