# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os

import idr_torch
import torch

from . import training_stats

# ----------------------------------------------------------------------------


def init():
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=idr_torch.size,
        rank=idr_torch.rank,
    )
    # if 'MASTER_ADDR' not in os.environ:
    #     os.environ['MASTER_ADDR'] = 'localhost'
    # if 'MASTER_PORT' not in os.environ:
    #     os.environ['MASTER_PORT'] = '29500'
    # if 'RANK' not in os.environ:
    #     os.environ['RANK'] = '0'
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = '0'
    # if 'WORLD_SIZE' not in os.environ:
    #     os.environ['WORLD_SIZE'] = '1'
    #
    # backend = 'gloo' if os.name == 'nt' else 'nccl'
    # print(backend)
    # torch.distributed.init_process_group(backend=backend, init_method='env://')
    torch.cuda.set_device(idr_torch.local_rank)
    sync_device = torch.device("cuda") if get_world_size() > 1 else None
    training_stats.init_multiprocessing(rank=get_rank(), sync_device=sync_device)


# ----------------------------------------------------------------------------


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


# ----------------------------------------------------------------------------


def get_world_size():
    return (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )


# ----------------------------------------------------------------------------


def should_stop():
    return False


# ----------------------------------------------------------------------------


def update_progress(cur, total):
    _ = cur, total


# ----------------------------------------------------------------------------


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


# ----------------------------------------------------------------------------


def gather_tensor_across_gpus(tensor, device=None):
    # First, gather sizes from all ranks
    local_size = torch.tensor([tensor.numel()], device=device)
    all_sizes = [
        torch.zeros_like(local_size) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(all_sizes, local_size)

    # Pad tensor to max size
    max_size = int(torch.stack(all_sizes).max())
    padded_tensor = torch.zeros(max_size, device=device)
    padded_tensor[: tensor.numel()] = tensor

    # Gather padded tensors
    gathered = [
        torch.zeros_like(padded_tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(gathered, padded_tensor)

    # Unpad and concatenate
    result = []
    for g, sz in zip(gathered, all_sizes):
        result.append(g[: sz.item()])
    return torch.cat(result)
