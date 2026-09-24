import os
import json
import pickle as pkl
import torch
from hymm.utils.logging_ import main_print
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from safetensors.torch import save_file, load_file
import torch.distributed as dist

from torch.distributed.tensor import DTensor




def fsdp_save_checkpoint_without_optim(transformer, rank, output_dir, scalar_states=None):
    # cpu_state = transformer.state_dict()
    # cpu_state = {k: v.full_tensor().cpu().clone() if isinstance(v, DTensor) else v.cpu().clone() for k, v in cpu_state.items()}
    # For rank 0, full_param.cpu() offloads the tensor to cpu one by one to avoid peaking GPU memory with unsharded parameters.
    sharded_sd = transformer.state_dict()
    cpu_state = {}
    for param_name, sharded_param in sharded_sd.items():
        if isinstance(sharded_param, DTensor):
            full_param = sharded_param.full_tensor()
        else:
            full_param = sharded_param
        if rank <= 0:
            cpu_state[param_name] = full_param.cpu()
        else:
            del full_param

    global_step = scalar_states.train_steps
    # sync scalar_state
    if dist.is_available() and dist.is_initialized():
        scalar_states_list = [None for _ in range(dist.get_world_size())]
        torch.distributed.all_gather_object(scalar_states_list, scalar_states.to_dict())

    if rank <= 0:
        save_dir = os.path.join(output_dir, 'checkpoints', f"global_step-{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        # save using safetensors
        weight_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)

        scalar_states_path = os.path.join(save_dir, "scalar_states.pkl")
        pkl.dump(scalar_states_list, open(scalar_states_path, "wb"))

        # config_dict = dict(transformer.config)
        # if "dtype" in config_dict:
        #     del config_dict["dtype"]  # TODO
        # if "moe_config" in config_dict and hasattr(config_dict["moe_config"], "to_dict"):
        #     config_dict["moe_config"] = config_dict["moe_config"].to_dict()
        # config_path = os.path.join(save_dir, "config.json")
        # # save dict as json
        # with open(config_path, "w") as f:
        #     json.dump(config_dict, f, indent=4)


def resume_mllm_training(model, checkpoint_dir, convert_state_dict_16168_to_16164=False):
    weight_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors")
    scalar_states_path = os.path.join(checkpoint_dir, "scalar_states.pkl")

    data = load_file(weight_path)
    model_weights = {k: v for k, v in data.items() if k != "__metadata__"}

    if convert_state_dict_16168_to_16164:
        tmp_state_dict = {}
        for k in model_weights.keys():
            if k == 'img_in.proj.weight':
                #tmp_state_dict[k] = torch.cat([model_weights['img_in.proj.weight'][:, :32, :, :],
                #                               model_weights['img_in.proj.weight'][:, -33:, :, :]], dim=1)
                tmp_state_dict[k] = model_weights['img_in.proj.weight'][:, :32, :, :]
            elif k == 'final_layer.linear.weight':
                tmp_state_dict[k] = model_weights['final_layer.linear.weight'][:32]
            elif k == 'final_layer.linear.bias':
                tmp_state_dict[k] = model_weights['final_layer.linear.bias'][:32]
            else:
                tmp_state_dict[k] = model_weights[k]
        state_dict = tmp_state_dict
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(model_weights, strict=False)

    scalar_states_list = pkl.load(open(scalar_states_path, "rb"))
    return model, scalar_states_list


def resume_wan_training(model, checkpoint_dir):

    weight_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors")
    scalar_states_path = os.path.join(checkpoint_dir, "scalar_states.pkl")

    data = load_file(weight_path)
    model_weights = {k: v for k, v in data.items() if k != "__metadata__"}

    missing, unexpected = model.load_state_dict(model_weights, strict=False)
    if missing:
        main_print(f"[resume_wan_training] Missing keys ({len(missing)}): {missing[:10]}...")
    if unexpected:
        main_print(f"[resume_wan_training] Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")

    scalar_states_list = pkl.load(open(scalar_states_path, "rb"))
    return model, scalar_states_list


_OPTIM_SUBDIR = "optimizer_state"


def save_optimizer_state(model, optimizer, output_dir, scalar_states):
    """Collective DCP save of the (sharded, reshardable) FSDP2 optimizer state.

    MUST be called by ALL ranks (dcp.save is collective). Writes to
    `<output_dir>/checkpoints/global_step-<step>/optimizer_state/`. Does NOT
    touch the model weights (saved independently as safetensors).
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_optimizer_state_dict,
        StateDictOptions,
    )

    global_step = scalar_states.train_steps
    save_dir = os.path.join(
        output_dir, "checkpoints", f"global_step-{global_step}", _OPTIM_SUBDIR
    )

    # flatten_optimizer_state_dict=True -> FQN-keyed state (world-size agnostic,
    # required for correct resharding across a different node/GPU count). The
    # returned state references the already-resident per-rank optimizer shards
    # (no extra alloc); DCP streams them to disk with internal chunked D2H copies.
    optim_sd = get_optimizer_state_dict(
        model,
        optimizer,
        options=StateDictOptions(flatten_optimizer_state_dict=True),
    )
    dcp.save({"optim": optim_sd}, checkpoint_id=save_dir)
    main_print(
        f"[save_optimizer_state] saved sharded/reshardable optimizer state -> {save_dir}"
    )


def load_optimizer_state(model, optimizer, checkpoint_dir):
    """Collective DCP load of the optimizer state from `<checkpoint_dir>/optimizer_state`.

    Returns True on success, False when the subdir is absent (e.g. resuming from
    a checkpoint saved before optimizer persistence existed, or with the toggle
    off) — the caller then continues with a COLD optimizer.
    MUST be called by ALL ranks, AFTER FSDP wrap + optimizer creation. Handles a
    different world size than the save (DCP reshards) and a not-yet-stepped
    optimizer (get_optimizer_state_dict materializes the state via a lr-0 dummy
    step that leaves params unchanged, giving DCP a typed template to load into).
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_optimizer_state_dict,
        set_optimizer_state_dict,
        StateDictOptions,
    )

    optim_dir = os.path.join(checkpoint_dir, _OPTIM_SUBDIR)

    # dcp.load below is collective, so the decision to call it must not be made
    # per-rank off the filesystem: a metadata visibility skew between nodes would
    # leave some ranks returning early while the rest block inside dcp.load
    # forever. Rank 0 decides and broadcasts.
    present = [os.path.isdir(optim_dir)]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(present, src=0)
    if not present[0]:
        main_print(
            f"[load_optimizer_state] no '{_OPTIM_SUBDIR}' at {optim_dir}; "
            f"continuing with a COLD optimizer (no AdamW moments restored)."
        )
        return False

    options = StateDictOptions(flatten_optimizer_state_dict=True)
    optim_sd = get_optimizer_state_dict(model, optimizer, options=options)
    dcp.load({"optim": optim_sd}, checkpoint_id=optim_dir)
    set_optimizer_state_dict(
        model, optimizer, optim_state_dict=optim_sd, options=options
    )
    main_print(
        f"[load_optimizer_state] restored (reshardable) optimizer state <- {optim_dir}"
    )
    return True


