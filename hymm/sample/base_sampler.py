import datetime

import deepspeed
import torch
import torch.distributed as dist

from ..utils.torch_utils import set_reproducibility


def setup_distributed_initialize(args, mode, timeout=None):
    print(f"args: {args}, mode: {mode}")
    if mode == 'ddp':
        # Initialize distributed environment. We set a long timeout for unbalanced generation.
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=3600 * 24 * 365))
        # We use DDP separately in each node to avoid between-node communication. So we need calculate the
        # world size and rank manually. We assume that each node has the same number of GPUs.
        world_size = args.num_nodes * torch.cuda.device_count()
        rank = dist.get_rank() + args.node_index * torch.cuda.device_count()
        device = rank % torch.cuda.device_count()

    elif mode == 'deepspeed':
        if timeout is None:
            timeout = datetime.timedelta(seconds=3600 * 3)
        deepspeed.init_distributed(timeout=timeout)

        world_size = dist.get_world_size()
        rank = dist.get_rank()  # Rank of the current process in the cluster.
        device = rank % torch.cuda.device_count()  # Device of the current process in current node.

    else:
        world_size = 1
        rank = 0
        device = 0

    # Set current device for the current process.
    torch.cuda.set_device(device)
    # Disable gradients
    torch.set_grad_enabled(False)
    # Set reproducibility
    set_reproducibility(args.reproduce, args.global_seed)

    return world_size, rank, device
