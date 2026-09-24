import torch
import torch.nn as nn

from hymm.models.modules.mova import MOVABridge

from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)

from .load import get_no_split_modules

from torch.distributed.fsdp import MixedPrecision


def get_mixed_precision(master_weight_type="fp32"):
    weight_type = torch.float32 if master_weight_type == "fp32" else torch.bfloat16
    mixed_precision = MixedPrecision(
        param_dtype=weight_type,
        # Gradient communication precision.
        reduce_dtype=weight_type,
        # Buffer precision.
        buffer_dtype=weight_type,
        cast_forward_inputs=True,
    )
    return mixed_precision


def get_dit_fsdp_kwargs_v2(
    transformer,
    parallel_dims,
    sharding_strategy,
    cpu_offload=False,
    master_weight_type="fp32",
):
    """Resolve the FSDP2 device mesh for the requested sharding strategy.

    Parameter sharding runs on ``parallel_dims.fsdp_mesh``
    (``[dp_replicate, fsdp_shard]``), which deliberately spans SP ranks: SP
    shards activations, FSDP shards parameters, and the two are orthogonal. The
    strategy only chooses how ``dp_replicate`` was sized:

        full   -> [1, world]        pure FULL_SHARD
        hybrid -> [nodes, per-node] HSDP, shard inside a node
        none   -> [world, 1]        replicate only
    """
    if parallel_dims is None:
        raise ValueError("parallel_dims cannot be None; call initialize_parallel_state first.")

    no_split_modules = get_no_split_modules(transformer)

    if sharding_strategy == "full":
        assert not parallel_dims.dp_replicate_enabled, (
            "fsdp_sharding_strategy='full' expects dp_replicate=1"
        )
    elif sharding_strategy == "hybrid":
        assert parallel_dims.dp_replicate_enabled, (
            "fsdp_sharding_strategy='hybrid' expects dp_replicate>1"
        )
    elif sharding_strategy not in ("none",):
        raise ValueError(f"Unsupported fsdp_sharding_strategy: {sharding_strategy}")

    fsdp_kwargs = {
        "device_mesh": parallel_dims.fsdp_mesh,
        "cpu_offload": bool(cpu_offload),
        "mixed_precision": get_mixed_precision(master_weight_type),
    }
    return fsdp_kwargs, no_split_modules


#TODO no working now, I don't figure it out 
def apply_fsdp2(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    if isinstance(model, MOVABridge):
        transformer_block_lst = model.get_fsdp_block_list()
        nested_block_lst = model.get_fsdp_nested_block_list()
        forward_tail_ids = {id(b) for b in model.get_fsdp_forward_tail_blocks()}
    else:
        raise ValueError(f"Unsupported model type for apply_fsdp2: {type(model)}")

    # Children before parents: fully_shard on a parent leaves out any parameter
    # already claimed by a nested unit, so the nested unit has to exist first.
    # These carry the MoE expert that a low-noise step skips; giving them their
    # own group is what lets that step skip their all-gather too.
    for nested_block in nested_block_lst:
        fully_shard(nested_block, **fsdp_config, reshard_after_forward=True)

    for transformer_block in transformer_block_lst:
        if reshard_after_forward_policy == "always":
            reshard_after_forward = True
        elif reshard_after_forward_policy == "never":
            reshard_after_forward = False
        elif reshard_after_forward_policy == "default":
            if pp_enabled:
                # For PP, do not reshard after forward to avoid per-microbatch
                # all-gathers, which can be expensive and non-overlapped
                reshard_after_forward = False
            else:
                # Skip the reshard on whichever block ends the forward, since FSDP
                # prefetches it straight back for backward. The MoE routing decides
                # which block that is, so every candidate tail is exempted.
                reshard_after_forward = id(transformer_block) not in forward_tail_ids
        else:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, **fsdp_config, reshard_after_forward=not pp_enabled)
