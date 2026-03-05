# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import copy
import json
import os
import pickle
import time

import numpy as np
import psutil
import torch

import dnnlib
from torch_utils import distributed as dist
from torch_utils import misc, training_stats

# ----------------------------------------------------------------------------


def training_loop(
    run_dir=".",  # Output directory.
    dataset_kwargs={},  # Options for training set.
    data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
    network_kwargs={},  # Options for model and preconditioning.
    loss_kwargs={},  # Options for loss function.
    optimizer_kwargs={},  # Options for optimizer.
    augment_kwargs=None,  # Options for augmentation pipeline, None = disable.
    seed=0,  # Global random seed.
    batch_size=512,  # Total batch size for one training iteration.
    batch_gpu=None,  # Limit batch size per GPU, None = no limit.
    total_kimg=200000,  # Training duration, measured in thousands of training images.
    ema_halflife_kimg=500,  # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio=0.05,  # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg=10000,  # Learning rate ramp-up duration.
    loss_scaling=1,  # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick=50,  # Interval of progress prints.
    snapshot_ticks=50,  # How often to save network snapshots, None = disable.
    state_dump_ticks=500,  # How often to dump training state, None = disable.
    resume_pkl=None,  # Start from the given network snapshot, None = random initialization.
    resume_state_dump=None,  # Start from the given training state, None = reset training state.
    resume_kimg=0,  # Start from the given training progress.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    device=torch.device("cuda"),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0("Loading dataset...")

    dataset_obj = dnnlib.util.construct_class_by_name(
        **dataset_kwargs
    )  # subclass of training.dataset.Dataset

    dist.print0(f"\tlabel_dim: {dataset_obj.label_dim}")

    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=seed,
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            **data_loader_kwargs,
        )
    )

    # Construct network.
    dist.print0("Constructing network...")

    dist.print0(f"\tcond: {network_kwargs.get('cond', False)}")

    interface_kwargs = dict(
        img_resolution=dataset_obj.resolution,
        img_channels=dataset_obj.num_channels,
        label_dim=dataset_obj.label_dim if network_kwargs.get("cond", False) else 0,
    )
    net = dnnlib.util.construct_class_by_name(
        **network_kwargs, **interface_kwargs
    )  # subclass of torch.nn.Module

    dist.print0(f"\tlabel_dim: {net.label_dim}")

    net.train().requires_grad_(True).to(device)
    # if dist.get_rank() == 0:
    #     with torch.no_grad():
    #         images = torch.zeros(
    #             [batch_gpu, net.img_channels, net.img_resolution, net.img_resolution],
    #             device=device,
    #         )
    #         sigma = torch.ones([batch_gpu], device=device)
    #         labels = torch.zeros([batch_gpu, net.label_dim], device=device)
    #         misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0("Setting up optimizer...")
    loss_fn = dnnlib.util.construct_class_by_name(
        **loss_kwargs
    )  # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(
        params=net.parameters(), **optimizer_kwargs
    )  # subclass of torch.optim.Optimizer
    augment_pipe = (
        dnnlib.util.construct_class_by_name(**augment_kwargs)
        if augment_kwargs is not None
        else None
    )  # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()  # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()  # other ranks follow
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=net, require_all=False
        )
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=ema, require_all=False
        )
        del data  # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device("cpu"))
        misc.copy_params_and_buffers(
            src_module=data["net"], dst_module=net, require_all=True
        )
        optimizer.load_state_dict(data["optimizer_state"])
        del data  # conserve memory

    # Train.
    dist.print0(f"Training for {total_kimg} kimg...")
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                # FIX: handle multi hot
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)
                loss = loss_fn(
                    net=ddp, images=images, labels=labels, augment_pipe=augment_pipe
                )
                training_stats.report("Loss/loss", loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Update weights.
        for g in optimizer.param_groups:
            g["lr"] = optimizer_kwargs["lr"] * min(
                cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1
            )
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = cur_nimg >= total_kimg * 1000
        if (
            (not done)
            and (cur_tick != 0)
            and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000)
        ):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"
        ]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"
        ]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"
        ]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"
        ]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(" ".join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0("Aborting...")

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(
                ema=ema,
                loss_fn=loss_fn,
                augment_pipe=augment_pipe,
                dataset_kwargs=dict(dataset_kwargs),
            )
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value  # conserve memory
            if dist.get_rank() == 0:
                with open(
                    os.path.join(
                        run_dir, f"network-snapshot-{cur_nimg // 1000:06d}.pkl"
                    ),
                    "wb",
                ) as f:
                    pickle.dump(data, f)
            del data  # conserve memory

        # Save full dump of the training state.
        if (
            (state_dump_ticks is not None)
            and (done or cur_tick % state_dump_ticks == 0)
            and cur_tick != 0
            and dist.get_rank() == 0
        ):
            torch.save(
                dict(net=net, optimizer_state=optimizer.state_dict()),
                os.path.join(run_dir, f"training-state-{cur_nimg // 1000:06d}.pt"),
            )

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
            stats_jsonl.write(
                json.dumps(
                    dict(
                        training_stats.default_collector.as_dict(),
                        timestamp=time.time(),
                    )
                )
                + "\n"
            )
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0("Exiting...")


def training_loop_weak(
    run_dir=".",  # Output directory.
    dataset_kwargs={},  # Options for training set.
    data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
    network_kwargs={},  # Options for model and preconditioning.
    loss_kwargs={},  # Options for loss function.
    optimizer_kwargs={},  # Options for optimizer.
    augment_kwargs=None,  # Options for augmentation pipeline, None = disable.
    seed=0,  # Global random seed.
    batch_size=512,  # Total batch size for one training iteration.
    batch_gpu=None,  # Limit batch size per GPU, None = no limit.
    total_kimg=200000,  # Training duration, measured in thousands of training images.
    ema_halflife_kimg=500,  # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio=0.05,  # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg=10000,  # Learning rate ramp-up duration.
    loss_scaling=1,  # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick=50,  # Interval of progress prints.
    snapshot_ticks=50,  # How often to save network snapshots, None = disable.
    state_dump_ticks=500,  # How often to dump training state, None = disable.
    resume_pkl=None,  # Start from the given network snapshot, None = random initialization.
    resume_state_dump=None,  # Start from the given training state, None = reset training state.
    resume_kimg=0,  # Start from the given training progress.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    device=torch.device("cuda"),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0("Loading dataset...")
    dataset_obj = dnnlib.util.construct_class_by_name(
        **dataset_kwargs
    )  # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=seed,
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            **data_loader_kwargs,
        )
    )

    # Construct network.
    dist.print0("Constructing network...")
    interface_kwargs = dict(
        img_resolution=dataset_obj.resolution,
        img_channels=dataset_obj.num_channels,
        label_dim=dataset_obj.label_dim if network_kwargs.get("cond", False) else 0,
    )
    net = dnnlib.util.construct_class_by_name(
        **network_kwargs, **interface_kwargs
    )  # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    # if dist.get_rank() == 0:
    #     with torch.no_grad():
    #         images = torch.zeros(
    #             [batch_gpu, net.img_channels, net.img_resolution, net.img_resolution],
    #             device=device,
    #         )
    #         sigma = torch.ones([batch_gpu], device=device)
    #         labels = torch.zeros([batch_gpu, net.label_dim], device=device)
    #         misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0("Setting up optimizer...")
    loss_fn = dnnlib.util.construct_class_by_name(
        **loss_kwargs
    )  # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(
        params=net.parameters(), **optimizer_kwargs
    )  # subclass of torch.optim.Optimizer
    augment_pipe = (
        dnnlib.util.construct_class_by_name(**augment_kwargs)
        if augment_kwargs is not None
        else None
    )  # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()  # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()  # other ranks follow
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=net, require_all=False
        )
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=ema, require_all=False
        )
        del data  # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device("cpu"))
        misc.copy_params_and_buffers(
            src_module=data["net"], dst_module=net, require_all=True
        )
        optimizer.load_state_dict(data["optimizer_state"])
        del data  # conserve memory
    # --- Importance weighting to match a target class prior ---
    # We want the risk under a target prior pi_t (here uniform => [0.5, 0.5])
    # but we sample from the training prior pi_s (empirical). Importance weights:
    #   w(c) = pi_t(c) / pi_s(c)
    # With pi_t summing to 1, E_{pi_s}[w] = 1, keeping loss scale stable in expectation.
    class_counts = torch.tensor([31083.0, 38388.0], device=device)
    train_prior = (class_counts / class_counts.sum()).clamp_min(1e-12)
    target_prior = torch.full_like(train_prior, 1.0 / train_prior.numel())  # [0.5, 0.5]
    class_weights = target_prior / train_prior  # shape: (num_classes,)
    # Train.
    dist.print0(f"Training for {total_kimg} kimg...")
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)
                loss_tensor = loss_fn(
                    net=ddp, images=images, labels=labels, augment_pipe=augment_pipe
                )
                # Per-sample scalar loss (mean over spatial dims if needed)
                per_sample_loss = (
                    loss_tensor.mean(dim=[1, 2, 3])
                    if loss_tensor.dim() > 1
                    else loss_tensor.view(-1)
                )

                # For one‑hot labels, pick the class weight for each sample via dot product
                # (for multi-label cases this becomes the sum of weights of active classes)
                sample_weights = (labels * class_weights).sum(dim=1)

                # Normalize by total weight to keep loss scale stable across batches
                denom = sample_weights.sum().clamp(min=1.0)
                weighted_loss = (per_sample_loss * sample_weights).sum() / denom

                training_stats.report("Loss/loss_weighted", weighted_loss)
                training_stats.report("Loss/avg_sample_weight", sample_weights.mean())
                weighted_loss.mul(loss_scaling / batch_gpu_total).backward()

        # Update weights.
        for g in optimizer.param_groups:
            g["lr"] = optimizer_kwargs["lr"] * min(
                cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1
            )
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = cur_nimg >= total_kimg * 1000
        if (
            (not done)
            and (cur_tick != 0)
            and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000)
        ):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"
        ]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"
        ]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"
        ]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"
        ]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(" ".join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0("Aborting...")

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(
                ema=ema,
                loss_fn=loss_fn,
                augment_pipe=augment_pipe,
                dataset_kwargs=dict(dataset_kwargs),
            )
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value  # conserve memory
            if dist.get_rank() == 0:
                with open(
                    os.path.join(
                        run_dir, f"network-snapshot-{cur_nimg // 1000:06d}.pkl"
                    ),
                    "wb",
                ) as f:
                    pickle.dump(data, f)
            del data  # conserve memory

        # Save full dump of the training state.
        if (
            (state_dump_ticks is not None)
            and (done or cur_tick % state_dump_ticks == 0)
            and cur_tick != 0
            and dist.get_rank() == 0
        ):
            torch.save(
                dict(net=net, optimizer_state=optimizer.state_dict()),
                os.path.join(run_dir, f"training-state-{cur_nimg // 1000:06d}.pt"),
            )

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
            stats_jsonl.write(
                json.dumps(
                    dict(
                        training_stats.default_collector.as_dict(),
                        timestamp=time.time(),
                    )
                )
                + "\n"
            )
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0("Exiting...")


# ----------------------------------------------------------------------------


def training_loop_minmax(
    run_dir=".",  # Output directory.
    dataset_kwargs={},  # Options for training set.
    data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
    network_kwargs={},  # Options for model and preconditioning.
    loss_kwargs={},  # Options for loss function.
    optimizer_kwargs={},  # Options for optimizer.
    augment_kwargs=None,  # Options for augmentation pipeline, None = disable.
    seed=0,  # Global random seed.
    batch_size=512,  # Total batch size for one training iteration.
    batch_gpu=None,  # Limit batch size per GPU, None = no limit.
    total_kimg=200000,  # Training duration, measured in thousands of training images.
    ema_halflife_kimg=500,  # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio=0.05,  # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg=10000,  # Learning rate ramp-up duration.
    loss_scaling=1,  # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick=50,  # Interval of progress prints.
    snapshot_ticks=50,  # How often to save network snapshots, None = disable.
    state_dump_ticks=500,  # How often to dump training state, None = disable.
    resume_pkl=None,  # Start from the given network snapshot, None = random initialization.
    resume_state_dump=None,  # Start from the given training state, None = reset training state.
    resume_kimg=0,  # Start from the given training progress.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    device=torch.device("cuda"),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0("Loading dataset...")
    dataset_obj = dnnlib.util.construct_class_by_name(
        **dataset_kwargs
    )  # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=seed,
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            **data_loader_kwargs,
        )
    )

    # Construct network.
    dist.print0("Constructing network...")
    interface_kwargs = dict(
        img_resolution=dataset_obj.resolution,
        img_channels=dataset_obj.num_channels,
        label_dim=dataset_obj.label_dim if network_kwargs.get("cond", False) else 0,
    )
    net = dnnlib.util.construct_class_by_name(
        **network_kwargs, **interface_kwargs
    )  # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    # if dist.get_rank() == 0:
    #     with torch.no_grad():
    #         images = torch.zeros(
    #             [batch_gpu, net.img_channels, net.img_resolution, net.img_resolution],
    #             device=device,
    #         )
    #         sigma = torch.ones([batch_gpu], device=device)
    #         labels = torch.zeros([batch_gpu, net.label_dim], device=device)
    #         misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0("Setting up optimizer...")
    loss_fn = dnnlib.util.construct_class_by_name(
        **loss_kwargs
    )  # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(
        params=net.parameters(), **optimizer_kwargs
    )  # subclass of torch.optim.Optimizer
    augment_pipe = (
        dnnlib.util.construct_class_by_name(**augment_kwargs)
        if augment_kwargs is not None
        else None
    )  # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # --- MinMax trainer helpers -------------------------------------------------
    # Bins in log10(sigma) space; use numpy for bookkeeping (no grads involved).
    sigma_bins = np.logspace(np.log10(1e-2), np.log10(1e1), 50)  # [1e-2, 1e+1]
    n_bins = len(sigma_bins)
    ema_decay = 0.98  # EMA coefficient for per-(class,bin) loss tracking.
    eps = 1e-8

    # EMA buffers: one for the background/all-class bucket (index 0), then one per class.
    # Shape: (label_dim + 1, n_bins)
    ema_loss_per_bin = np.zeros((dataset_obj.label_dim + 1, n_bins), dtype=np.float64)
    ema_count_per_bin = np.zeros((dataset_obj.label_dim + 1, n_bins), dtype=np.float64)

    def _bin_index(sigmas_np: np.ndarray) -> np.ndarray:
        # Map each sigma to a bin index in [0, n_bins-1].
        idx = np.digitize(sigmas_np, sigma_bins, right=True)
        idx = np.clip(idx, 0, n_bins - 1)
        return idx

    def _update_ema_losses(
        loss_np: np.ndarray, sigmas_np: np.ndarray, labels_np: np.ndarray
    ):
        # loss_np: (N,) per-sample loss (mean over spatial dims)
        # sigmas_np: (N,) per-sample sigma
        # labels_np: (N, C) in {0,1} or floats in [0,1]
        bin_idx = _bin_index(sigmas_np)
        # All-sample bucket at index 0
        for b, l in zip(bin_idx, loss_np):
            ema_loss_per_bin[0, b] = ema_loss_per_bin[0, b] * ema_decay + (
                1 - ema_decay
            ) * float(l)
            ema_count_per_bin[0, b] = ema_count_per_bin[0, b] * ema_decay + (
                1 - ema_decay
            )
        # Per-class buckets (assume multi-label; count every positive class)
        pos = labels_np > 0.5
        for i in range(dataset_obj.label_dim):
            if not np.any(pos[:, i]):
                continue
            for b, l, p in zip(bin_idx, loss_np, pos[:, i]):
                if p:
                    ema_loss_per_bin[i + 1, b] = ema_loss_per_bin[
                        i + 1, b
                    ] * ema_decay + (1 - ema_decay) * float(l)
                    ema_count_per_bin[i + 1, b] = ema_count_per_bin[
                        i + 1, b
                    ] * ema_decay + (1 - ema_decay)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()  # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()  # other ranks follow
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=net, require_all=False
        )
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=ema, require_all=False
        )
        del data  # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device("cpu"))
        misc.copy_params_and_buffers(
            src_module=data["net"], dst_module=net, require_all=True
        )
        optimizer.load_state_dict(data["optimizer_state"])
        del data  # conserve memory

    # Train.
    dist.print0(f"Training for {total_kimg} kimg...")
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    # --- Debug accumulators for logging the new minmax logic ---
    last_kept_frac = 0.0
    last_bins_in_batch = 0
    last_bins_kept = 0
    last_bin_probe = ""
    probe_bins = np.unique(np.round(np.linspace(0, n_bins - 1, 5)).astype(int)).tolist()

    while True:
        # Accumulate gradients with bin-wise worst-class masking.
        optimizer.zero_grad(set_to_none=True)
        # Debug accumulators for this optimizer step
        dbg_kept_sum = 0.0
        dbg_bins_in_batch_sum = 0
        dbg_bins_kept_sum = 0
        dbg_rounds = 0
        dbg_probe_text = None
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                # --- Forward pass with grad (sigma sampled as usual inside loss_fn)
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)
                loss_tensor, sigma = loss_fn(
                    net=ddp,
                    images=images,
                    labels=labels,
                    augment_pipe=augment_pipe,
                    return_sigmas=True,
                )
                # Per-sample scalar losses (detach copy for EMA, keep graph for mask-weighted sum)
                per_sample_loss = loss_tensor.mean(dim=[1, 2, 3])  # shape (N,)
                per_sample_loss_det = per_sample_loss.detach().cpu().numpy()
                sigmas_np = sigma.view(-1).detach().cpu().numpy()  # shape (N,)
                labels_np = labels.detach().cpu().numpy()  # (N,C)

                # --- Update EMA trackers from this batch (no grad bookkeeping)
                _update_ema_losses(per_sample_loss_det, sigmas_np, labels_np)

                # --- Build a mask: for each bin, only the worst EMA class contributes
                # Compute worst class per bin from EMA arrays (ignore bins with tiny counts)
                means = ema_loss_per_bin / (ema_count_per_bin + eps)  # (1+C, B)
                mask_eff = ema_count_per_bin > 0.1
                scores = np.where(mask_eff, means, -np.inf)
                # For class choice, ignore global bucket (index 0)
                class_scores = scores[1:, :]  # (C, B)
                worst_class_per_bin = np.argmax(
                    class_scores, axis=0
                )  # (B,) in [0..C-1]

                # Bin indices for current samples
                bin_idx = _bin_index(sigmas_np)  # (N,)

                # For each sample k, we keep it iff its label includes the worst class of its bin
                # Build a boolean mask on CPU then move to device
                keep = np.zeros_like(bin_idx, dtype=np.float32)
                for k in range(bin_idx.shape[0]):
                    b = bin_idx[k]
                    wc = int(worst_class_per_bin[b])
                    keep[k] = 1.0 if (labels_np[k, wc] > 0.5) else 0.0
                keep_t = torch.from_numpy(keep).to(per_sample_loss.device)

                # Avoid full drop of a microbatch: if all zeros, fall back to using the global bucket decision
                if keep_t.sum() == 0:
                    # Use the global worst bin across classes to keep at least something
                    # Here we simply keep the top-25% highest losses in the batch as a fallback
                    cutoff = torch.quantile(per_sample_loss.detach(), 0.75).item()
                    keep_t = (per_sample_loss.detach() >= cutoff).to(
                        per_sample_loss.dtype
                    )

                # Normalize by the number of kept samples to keep loss scale stable
                denom = torch.clamp(keep_t.sum(), min=1.0)
                masked_loss = (per_sample_loss * keep_t).sum() / denom

                # Report and backprop
                training_stats.report("Loss/loss_masked", masked_loss)
                training_stats.report("Loss/kept_frac", keep_t.mean().item())
                masked_loss.mul(loss_scaling / batch_gpu_total).backward()

                # --- Debug: accumulate kept fraction & bin coverage ---
                kept_frac = float(keep_t.mean().item())
                bins_in_batch = int(np.unique(bin_idx).size)
                kept_mask_np = keep_t.detach().cpu().numpy() > 0.5
                bins_kept = (
                    int(np.unique(bin_idx[kept_mask_np]).size)
                    if kept_mask_np.any()
                    else 0
                )
                dbg_kept_sum += kept_frac
                dbg_bins_in_batch_sum += bins_in_batch
                dbg_bins_kept_sum += bins_kept
                dbg_rounds += 1

                # Build a short probe of worst class per a few representative bins
                means = ema_loss_per_bin / (ema_count_per_bin + eps)  # (1+C, B)
                class_scores = means[1:, :]  # (C, B)
                probe_parts = []
                for b in probe_bins:
                    wc = (
                        int(np.argmax(class_scores[:, b]))
                        if ema_count_per_bin[0, b] > 0
                        else -1
                    )
                    val = float(class_scores[wc, b]) if wc >= 0 else float("nan")
                    # report in the form b=idx:c=wc:l=val
                    probe_parts.append(f"b{b}:c{wc}:l{val:.3f}")
                dbg_probe_text = "|".join(probe_parts)

        # Update weights.
        for g in optimizer.param_groups:
            g["lr"] = optimizer_kwargs["lr"] * min(
                cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1
            )
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )
        optimizer.step()

        # Update EMA of weights.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Finalize debug stats for this step
        if dbg_rounds > 0:
            last_kept_frac = dbg_kept_sum / dbg_rounds
            last_bins_in_batch = int(round(dbg_bins_in_batch_sum / dbg_rounds))
            last_bins_kept = int(round(dbg_bins_kept_sum / dbg_rounds))
            last_bin_probe = dbg_probe_text or last_bin_probe
            training_stats.report("MinMax/kept_frac", last_kept_frac)
            training_stats.report("MinMax/bins_in_batch", float(last_bins_in_batch))
            training_stats.report("MinMax/bins_kept", float(last_bins_kept))

        # Count images for maintenance cadence
        cur_nimg += batch_size

        # Perform maintenance tasks once per tick.
        done = cur_nimg >= total_kimg * 1000
        if (
            (not done)
            and (cur_tick != 0)
            and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000)
        ):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"
        ]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"
        ]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"
        ]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"
        ]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"
        ]
        fields += [f"kept {last_kept_frac:.2f}"]
        fields += [f"bins {last_bins_kept}/{last_bins_in_batch}"]
        # Keep the probe string short in the log line
        if isinstance(last_bin_probe, str) and len(last_bin_probe) > 0:
            fields += [f"probe {last_bin_probe}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(" ".join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0("Aborting...")

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(
                ema=ema,
                loss_fn=loss_fn,
                augment_pipe=augment_pipe,
                dataset_kwargs=dict(dataset_kwargs),
            )
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value  # conserve memory
            if dist.get_rank() == 0:
                with open(
                    os.path.join(
                        run_dir, f"network-snapshot-{cur_nimg // 1000:06d}.pkl"
                    ),
                    "wb",
                ) as f:
                    pickle.dump(data, f)
            del data  # conserve memory

        # Save full dump of the training state.
        if (
            (state_dump_ticks is not None)
            and (done or cur_tick % state_dump_ticks == 0)
            and cur_tick != 0
            and dist.get_rank() == 0
        ):
            torch.save(
                dict(net=net, optimizer_state=optimizer.state_dict()),
                os.path.join(run_dir, f"training-state-{cur_nimg // 1000:06d}.pt"),
            )

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
            stats_jsonl.write(
                json.dumps(
                    dict(
                        training_stats.default_collector.as_dict(),
                        timestamp=time.time(),
                    )
                )
                + "\n"
            )
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0("Exiting...")


# ----------------------------------------------------------------------------


def evaltraining_loop(
    run_dir=".",  # Output directory.
    dataset_kwargs={},  # Options for training set.
    data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
    network_kwargs={},  # Options for model and preconditioning.
    loss_kwargs={},  # Options for loss function.
    augment_kwargs=None,  # Options for augmentation pipeline, None = disable.
    seed=0,  # Global random seed.
    batch_size=512,  # Total batch size for one training iteration.
    batch_gpu=None,  # Limit batch size per GPU, None = no limit.
    kimg_per_tick=50,  # Interval of progress prints.
    resume_pkl=None,  # Start from the given network snapshot, None = random initialization.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    device=torch.device("cuda"),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0("Loading dataset...")
    dataset_obj = dnnlib.util.construct_class_by_name(
        **dataset_kwargs
    )  # subclass of training.dataset.Dataset
    print(
        f"Dataset {dataset_obj.name} has {len(dataset_obj)} images with resolution {dataset_obj.resolution} and {dataset_obj.num_channels} channels and {dataset_obj.label_dim} labels."
    )
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=seed,
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            **data_loader_kwargs,
        )
    )

    # Construct network.
    dist.print0("Constructing network...")
    interface_kwargs = dict(
        img_resolution=dataset_obj.resolution,
        img_channels=dataset_obj.num_channels,
        label_dim=dataset_obj.label_dim,
    )
    net = dnnlib.util.construct_class_by_name(
        **network_kwargs, **interface_kwargs
    )  # subclass of torch.nn.Module
    net.eval().to(device)
    # if dist.get_rank() == 0:
    #     with torch.no_grad():
    #         images = torch.zeros(
    #             [batch_gpu, net.img_channels, net.img_resolution, net.img_resolution],
    #             device=device,
    #         )
    #         sigma = torch.ones([batch_gpu], device=device)
    #         labels = torch.zeros([batch_gpu, net.label_dim], device=device)
    #         misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    loss_fn = dnnlib.util.construct_class_by_name(
        **loss_kwargs
    )  # training.loss.(VP|VE|EDM)Loss
    augment_pipe = (
        dnnlib.util.construct_class_by_name(**augment_kwargs)
        if augment_kwargs is not None
        else None
    )  # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()  # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()  # other ranks follow
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=net, require_all=False
        )

        del data  # conserve memory

    # Eval.
    dist.print0(
        f"Evaluating for {dataset_kwargs.max_size} imgs on {dataset_obj.label_dim} classes..."
    )
    dist.print0()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 3, dataset_kwargs.max_size)
    stats_jsonl = None
    losses = [[] for i in range(dataset_obj.label_dim + 1)]
    sigmas = [[] for i in range(dataset_obj.label_dim + 1)]
    while True:
        with torch.no_grad():
            # Accumulate gradients.
            for round_idx in range(num_accumulation_rounds):
                with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                    images, labels = next(dataset_iterator)
                    images = images.to(device).to(torch.float32) / 127.5 - 1
                    labels = labels.to(device)
                    loss, sigma = loss_fn(
                        net=ddp,
                        images=images,
                        labels=labels,
                        augment_pipe=augment_pipe,
                        return_sigmas=True,
                    )
                    loss = loss.mean(dim=[1, 2, 3])
                    losses[0].append(loss.cpu().numpy())
                    sigmas[0].append(sigma.view(-1).cpu().numpy())
                    for i in range(dataset_obj.label_dim):
                        mask = labels[:, i] > 0.5
                        losses[i + 1].append(loss[mask].cpu().numpy())
                        sigmas[i + 1].append(sigma.view(-1)[mask].cpu().numpy())

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size

        done = cur_nimg >= dataset_kwargs.max_size * 3
        if (
            (not done)
            and (cur_tick != 0)
            and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000)
        ):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"
        ]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"
        ]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"
        ]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(" ".join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0("Aborting...")

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
            stats_jsonl.write(
                json.dumps(
                    dict(
                        training_stats.default_collector.as_dict(),
                        timestamp=time.time(),
                    )
                )
                + "\n"
            )
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 3, dataset_kwargs.max_size)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    losses[0] = dist.gather_tensor_across_gpus(
        torch.tensor(np.concatenate(losses[0])), device=device
    )
    sigmas[0] = dist.gather_tensor_across_gpus(
        torch.tensor(np.concatenate(sigmas[0])), device=device
    )
    for i in range(1, dataset_obj.label_dim + 1):
        losses[i] = dist.gather_tensor_across_gpus(
            torch.tensor(np.concatenate(losses[i])), device=device
        )
        sigmas[i] = dist.gather_tensor_across_gpus(
            torch.tensor(np.concatenate(sigmas[i])), device=device
        )
    if dist.get_rank() == 0:
        dist.print0(f"Loss: {losses[0].mean()} pm {losses[0].std()}")
        for i in range(dataset_obj.label_dim + 1):
            dist.print0(
                f"Loss for class {i}: {losses[i].mean()} pm {losses[i].std()} ({len(losses[i])} samples)"
            )
        losses_to_save = {
            "loss_avg": {"mean": losses[0].mean().item(), "std": losses[0].std().item()}
        }
        for i in range(1, dataset_obj.label_dim + 1):
            losses_to_save[f"loss_avg_class_{i - 1}"] = {
                "mean": losses[i].mean().item(),
                "std": losses[i].std().item(),
                "count": len(losses[i]),
            }
        with open(os.path.join(run_dir, "losses.json"), "w") as f:
            json.dump(losses_to_save, f, indent=4)
        all_loss_to_save = {
            "all": {"loss": losses[0].cpu().numpy(), "sigma": sigmas[0].cpu().numpy()}
        }
        for i in range(1, dataset_obj.label_dim + 1):
            all_loss_to_save[f"class_{i - 1}"] = {
                "loss": losses[i].cpu().numpy(),
                "sigma": sigmas[i].cpu().numpy(),
            }
        with open(os.path.join(run_dir, "all_losses.pkl"), "wb") as f:
            pickle.dump(all_loss_to_save, f)

    dist.print0()
    dist.print0("Exiting...")


# ----------------------------------------------------------------------------


def training_loop_cata(
    run_dir=".",  # Output directory.
    dataset_kwargs={},  # Options for training set.
    data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
    network_kwargs={},  # Options for model and preconditioning.
    loss_kwargs={},  # Options for loss function.
    optimizer_kwargs={},  # Options for optimizer.
    augment_kwargs=None,  # Options for augmentation pipeline, None = disable.
    seed=0,  # Global random seed.
    batch_size=512,  # Total batch size for one training iteration.
    batch_gpu=None,  # Limit batch size per GPU, None = no limit.
    total_kimg=200000,  # Training duration, measured in thousands of training images.
    ema_halflife_kimg=500,  # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio=0.05,  # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg=10000,  # Learning rate ramp-up duration.
    loss_scaling=1,  # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick=50,  # Interval of progress prints.
    snapshot_ticks=50,  # How often to save network snapshots, None = disable.
    state_dump_ticks=500,  # How often to dump training state, None = disable.
    resume_pkl=None,  # Start from the given network snapshot, None = random initialization.
    resume_state_dump=None,  # Start from the given training state, None = reset training state.
    resume_kimg=0,  # Start from the given training progress.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    type="all",
    device=torch.device("cuda"),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0("Loading dataset...")
    dataset_obj = dnnlib.util.construct_class_by_name(
        **dataset_kwargs
    )  # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=seed,
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            **data_loader_kwargs,
        )
    )

    # Construct network.
    dist.print0("Constructing network...")
    interface_kwargs = dict(
        img_resolution=dataset_obj.resolution,
        img_channels=dataset_obj.num_channels,
        label_dim=dataset_obj.label_dim if network_kwargs.get("cond", False) else 0,
    )
    net = dnnlib.util.construct_class_by_name(
        **network_kwargs, **interface_kwargs
    )  # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    # if dist.get_rank() == 0:
    #     with torch.no_grad():
    #         images = torch.zeros(
    #             [batch_gpu, net.img_channels, net.img_resolution, net.img_resolution],
    #             device=device,
    #         )
    #         sigma = torch.ones([batch_gpu], device=device)
    #         labels = torch.zeros([batch_gpu, net.label_dim], device=device)
    #         misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0("Setting up optimizer...")
    loss_fn = dnnlib.util.construct_class_by_name(
        **loss_kwargs
    )  # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(
        params=net.parameters(), **optimizer_kwargs
    )  # subclass of torch.optim.Optimizer
    augment_pipe = (
        dnnlib.util.construct_class_by_name(**augment_kwargs)
        if augment_kwargs is not None
        else None
    )  # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()  # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()  # other ranks follow
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=net, require_all=False
        )
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=ema, require_all=False
        )
        del data  # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device("cpu"))
        misc.copy_params_and_buffers(
            src_module=data["net"], dst_module=net, require_all=True
        )
        optimizer.load_state_dict(data["optimizer_state"])
        del data  # conserve memory

    # Train.
    dist.print0(f"Training for {total_kimg} kimg...")
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    # --- Class weighting setup (robust to label_dim==0 and to index vs one-hot labels) ---
    num_classes = getattr(dataset_obj, "label_dim", 0)
    class_weights = None  # type: Optional[torch.Tensor]
    if num_classes > 0:
        # Default weights for 2-class case; extendable for >2 classes if needed.
        if type == "all":
            base = torch.tensor([-1, -1], dtype=torch.float32)
        elif type == "male":
            base = torch.tensor([-1, 1], dtype=torch.float32)
        elif type == "female":
            base = torch.tensor([1, -1], dtype=torch.float32)
        else:
            # Fallback: uniform negatives encourages minimizing average loss
            base = torch.full((num_classes,), -1.0, dtype=torch.float32)
        if base.numel() != num_classes:
            # If dataset has a different number of classes, broadcast a sensible default
            base = torch.full(
                (num_classes,), float(base.mean().item()), dtype=torch.float32
            )
        class_weights = base.to(device=device)
        dist.print0(f"Using class weights: {class_weights.cpu().numpy()}")

    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)
                loss_tensor = loss_fn(
                    net=ddp, images=images, labels=labels, augment_pipe=augment_pipe
                )
                # Per-sample scalar loss (mean over spatial dims if needed)
                per_sample_loss = (
                    loss_tensor.mean(dim=[1, 2, 3])
                    if loss_tensor.dim() > 1
                    else loss_tensor.view(-1)
                )  # shape (N,)

                # Compute per-sample weights if we have class information; otherwise fall back to unweighted.
                if class_weights is None or labels.numel() == 0:
                    # No labels available (label_dim==0) — use plain mean loss.
                    weighted_loss = per_sample_loss.mean()
                    training_stats.report("Loss/loss", weighted_loss)
                    weighted_loss.mul(loss_scaling / batch_gpu_total).backward()
                else:
                    # Ensure labels are one-hot of shape (N, C)
                    if labels.dim() == 1 or (
                        labels.dim() == 2 and labels.size(1) == 1 and num_classes > 1
                    ):
                        # labels are class indices
                        labels_oh = torch.nn.functional.one_hot(
                            labels.view(-1).long(), num_classes=num_classes
                        ).to(per_sample_loss.dtype)
                    else:
                        labels_oh = labels.to(per_sample_loss.dtype)
                        if labels_oh.dim() == 1:
                            labels_oh = labels_oh.view(-1, 1)
                        if labels_oh.size(1) != num_classes:
                            # As a last resort, try to coerce to expected shape by trimming or padding with zeros
                            if labels_oh.size(1) > num_classes:
                                labels_oh = labels_oh[:, :num_classes]
                            else:
                                pad = torch.zeros(
                                    labels_oh.size(0),
                                    num_classes - labels_oh.size(1),
                                    device=labels_oh.device,
                                    dtype=labels_oh.dtype,
                                )
                                labels_oh = torch.cat([labels_oh, pad], dim=1)

                    # Compute a weight per sample from label vector and class weights.
                    sample_weights = (labels_oh * class_weights).sum(dim=1)

                    # Fallback in case all weights are zero for a microbatch (e.g., unexpected labels)
                    if sample_weights.abs().sum() == 0:
                        sample_weights = torch.ones_like(sample_weights)

                    # Normalize by total absolute weight to keep loss scale stable across batches
                    denom = sample_weights.abs().sum().clamp(min=1.0)
                    weighted_loss = (per_sample_loss * sample_weights).sum() / denom

                    # Log and backprop
                    training_stats.report("Loss/loss_weighted", weighted_loss)
                    training_stats.report(
                        "Loss/avg_sample_weight", sample_weights.mean()
                    )
                    weighted_loss.mul(loss_scaling / batch_gpu_total).backward()

        # Update weights.
        for g in optimizer.param_groups:
            g["lr"] = optimizer_kwargs["lr"] * min(
                cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1
            )
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = cur_nimg >= total_kimg * 1000
        if (
            (not done)
            and (cur_tick != 0)
            and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000)
        ):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"
        ]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"
        ]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"
        ]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"
        ]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(" ".join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0("Aborting...")

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(
                ema=ema,
                loss_fn=loss_fn,
                augment_pipe=augment_pipe,
                dataset_kwargs=dict(dataset_kwargs),
            )
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value  # conserve memory
            if dist.get_rank() == 0:
                with open(
                    os.path.join(
                        run_dir, f"network-snapshot-{cur_nimg // 1000:06d}.pkl"
                    ),
                    "wb",
                ) as f:
                    pickle.dump(data, f)
            del data  # conserve memory

        # Save full dump of the training state.
        if (
            (state_dump_ticks is not None)
            and (done or cur_tick % state_dump_ticks == 0)
            and cur_tick != 0
            and dist.get_rank() == 0
        ):
            torch.save(
                dict(net=net, optimizer_state=optimizer.state_dict()),
                os.path.join(run_dir, f"training-state-{cur_nimg // 1000:06d}.pt"),
            )

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
            stats_jsonl.write(
                json.dumps(
                    dict(
                        training_stats.default_collector.as_dict(),
                        timestamp=time.time(),
                    )
                )
                + "\n"
            )
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0("Exiting...")
