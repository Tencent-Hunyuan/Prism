"""Distributed / sequence-parallel state for MOVA.

The topology follows HunyuanVideo-1.5: two independent device meshes over the
same world.

    world_mesh = [dp, sp]                 -> data parallel x Ulysses sequence parallel
    fsdp_mesh  = [dp_replicate, fsdp_shard] -> FSDP2 parameter sharding

SP shards activations, FSDP shards parameters, so the two meshes are
orthogonal and ``fsdp_shard`` deliberately spans SP ranks as well.

Model code only ever touches :data:`nccl_info` and
:func:`get_sequence_parallel_state`, which are kept as thin views over the
active :class:`ParallelDims`, so training and inference share one init path.
"""

import os
import random
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

@dataclass
class ParallelDims:
    sp: int = 1
    dp_replicate: int = 1
    world_size: int = -1
    use_dynamic_ring_attention: bool = False

    world_mesh: Any = field(default=None, init=False, repr=False)
    fsdp_mesh: Any = field(default=None, init=False, repr=False)
    sp_group: Any = field(default=None, init=False, repr=False)
    sp_rank: int = field(default=0, init=False)
    dp_rank: int = field(default=0, init=False)
    dp_size: int = field(default=1, init=False)

    def __post_init__(self):
        if self.world_size == -1:
            if dist.is_initialized():
                self.world_size = dist.get_world_size()
            else:
                self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if self.dp_replicate == -1:
            # One replica per node; FSDP shards within a node.
            local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
            assert self.world_size % local_world == 0, (
                f"world_size({self.world_size}) must be divisible by "
                f"LOCAL_WORLD_SIZE({local_world}) for dp_replicate=-1"
            )
            self.dp_replicate = self.world_size // local_world
        assert self.sp >= 1 and self.dp_replicate >= 1
        assert self.world_size % self.sp == 0, (
            f"world_size({self.world_size}) must be divisible by sp({self.sp})"
        )
        assert self.world_size % self.dp_replicate == 0, (
            f"world_size({self.world_size}) must be divisible by "
            f"dp_replicate({self.dp_replicate})"
        )

    # -- mesh -------------------------------------------------------------

    def build_mesh(self, device_type: str = "cuda"):
        self.world_mesh = init_device_mesh(
            device_type,
            (self.world_size // self.sp, self.sp),
            mesh_dim_names=("dp", "sp"),
        )
        self.fsdp_mesh = init_device_mesh(
            device_type,
            (self.dp_replicate, self.world_size // self.dp_replicate),
            mesh_dim_names=("dp_replicate", "fsdp_shard"),
        )

        self.dp_rank = self.world_mesh["dp"].get_local_rank()
        self.dp_size = self.world_mesh["dp"].size()
        if self.sp_enabled:
            self.sp_rank = self.world_mesh["sp"].get_local_rank()
            self.sp_group = self.world_mesh["sp"].get_group()
        else:
            self.sp_rank = 0
            self.sp_group = None
        return self.world_mesh

    # -- properties -------------------------------------------------------

    @property
    def sp_enabled(self) -> bool:
        return self.sp > 1

    @property
    def dp_enabled(self) -> bool:
        return self.dp_size > 1

    @property
    def dp_replicate_enabled(self) -> bool:
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self) -> bool:
        return (self.world_size // self.dp_replicate) > 1

    @property
    def pp_enabled(self) -> bool:
        return False

    @property
    def sp_mesh(self):
        return self.world_mesh["sp"]

    @property
    def dp_mesh(self):
        return self.world_mesh["dp"]

    @property
    def sp_src_rank(self) -> int:
        """Global rank of local rank 0 inside this rank's SP group."""
        if not self.sp_enabled:
            return dist.get_rank() if dist.is_initialized() else 0
        return dist.get_global_rank(self.sp_group, 0)


# ---------------------------------------------------------------------------
# Legacy view used by the model code (nccl_info.sp_group / .sp_size / ...)
# ---------------------------------------------------------------------------

class COMM_INFO:
    def __init__(self):
        self.sp_group = None
        self.sp_size = 1
        self.global_rank = 0
        self.rank_within_spgroup = 0
        self.parallel_dims = None
        self.device_mesh = None
        self.use_dynamic_ring_attention = False
        self.sp_stream = None
        self.sp_rank_list = None


nccl_info = COMM_INFO()
_SEQUENCE_PARALLEL_STATE = False
_PARALLEL_DIMS: Optional[ParallelDims] = None


def set_sequence_parallel_state(state: bool):
    global _SEQUENCE_PARALLEL_STATE
    _SEQUENCE_PARALLEL_STATE = state


def get_sequence_parallel_state() -> bool:
    return _SEQUENCE_PARALLEL_STATE


def get_parallel_state() -> ParallelDims:
    """Active topology, or a mesh-free single-rank default if none was set up.

    The fallback deliberately does NOT build a device mesh: ``init_device_mesh``
    is collective, and creating one from an incidental accessor call would mean
    some ranks entering a collective their peers never reach.
    """
    global _PARALLEL_DIMS
    if _PARALLEL_DIMS is None:
        _PARALLEL_DIMS = ParallelDims(sp=1, dp_replicate=1, world_size=1)
    return _PARALLEL_DIMS


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init_distributed(timeout_seconds: int = 5400) -> int:
    """Initialise the default process group and bind this rank's device.

    ``device_id`` is passed on purpose: it makes PyTorch build the default NCCL
    communicator eagerly during ``init_process_group`` instead of lazily on the
    first collective. Lazy communicator creation is a classic source of hangs
    when different ranks reach their first collective through different code
    paths (e.g. the MoE expert alternating between steps).

    The timeout here covers *startup*, where a rank can legitimately sit in a
    barrier for a long time while a peer streams a 15B checkpoint off a network
    filesystem. Shrink it with :func:`set_collective_timeout` once the training
    loop starts, where a long wait means a genuine desync rather than slow I/O.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        kwargs = {}
        if torch.cuda.is_available():
            kwargs["device_id"] = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=timeout_seconds),
            **kwargs,
        )
    return local_rank


def set_collective_timeout(timeout_seconds: int, logger=None):
    """Re-arm the watchdog with a tighter per-collective deadline.

    Called once the steady-state training loop begins. A short deadline is what
    turns a desync into an abort with a stack trace instead of a job that sits
    idle until someone notices. Applied to every group we created, since the
    watchdog is per-communicator.

    Failures are swallowed deliberately: this touches a private torch API and
    performs no collective, so a rank that cannot re-arm simply keeps the
    startup timeout rather than diverging from its peers.
    """
    if not dist.is_initialized():
        return False
    try:
        from torch.distributed.distributed_c10d import _set_pg_timeout
    except ImportError:
        if logger is not None:
            logger.warning("[dist] _set_pg_timeout unavailable; keeping the startup timeout")
        return False

    timeout = timedelta(seconds=int(timeout_seconds))
    groups = [None]
    pd = _PARALLEL_DIMS
    if pd is not None:
        if pd.world_mesh is not None:
            groups += [pd.world_mesh["sp"].get_group(), pd.world_mesh["dp"].get_group()]
        if pd.fsdp_mesh is not None:
            groups += [pd.fsdp_mesh["fsdp_shard"].get_group(),
                       pd.fsdp_mesh["dp_replicate"].get_group()]
    for group in groups:
        try:
            _set_pg_timeout(timeout, group)
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                logger.warning(f"[dist] could not re-arm timeout on {group}: {exc}")
            return False
    if logger is not None:
        logger.info(f"[dist] per-collective timeout tightened to {timeout_seconds}s for training")
    return True


def initialize_parallel_state(
    sp: int = 1,
    dp_replicate: int = 1,
    use_dynamic_ring_attention: bool = False,
    warmup: bool = True,
) -> ParallelDims:
    """Build the device meshes and publish them to the model-facing globals."""
    global _PARALLEL_DIMS

    parallel_dims = ParallelDims(
        sp=sp,
        dp_replicate=dp_replicate,
        use_dynamic_ring_attention=use_dynamic_ring_attention,
    )
    if dist.is_initialized():
        parallel_dims.build_mesh("cuda")

    nccl_info.sp_size = parallel_dims.sp
    nccl_info.global_rank = dist.get_rank() if dist.is_initialized() else 0
    nccl_info.rank_within_spgroup = parallel_dims.sp_rank
    nccl_info.sp_group = parallel_dims.sp_group
    nccl_info.parallel_dims = parallel_dims
    nccl_info.device_mesh = parallel_dims.world_mesh
    nccl_info.use_dynamic_ring_attention = use_dynamic_ring_attention
    if use_dynamic_ring_attention and parallel_dims.sp_enabled:
        nccl_info.sp_stream = torch.cuda.Stream()
        nccl_info.sp_rank_list = parallel_dims.world_mesh["sp"].mesh.tolist()

    set_sequence_parallel_state(parallel_dims.sp_enabled)
    _PARALLEL_DIMS = parallel_dims

    if warmup and dist.is_initialized():
        warmup_process_groups(parallel_dims)
    return parallel_dims


def warmup_process_groups(parallel_dims: Optional[ParallelDims] = None):
    """Force NCCL communicator creation for every group we will ever use.

    Every rank walks the same group list in the same order, so all communicators
    exist before the training loop starts. Without this, the first collective on
    a lazily-created group can land while other ranks are inside a different
    collective, which NCCL cannot untangle and the watchdog eventually kills.
    """
    if not dist.is_initialized():
        return
    parallel_dims = parallel_dims or get_parallel_state()
    device = torch.device("cuda", torch.cuda.current_device())
    probe = torch.zeros(1, device=device)

    groups = [None]  # default process group
    if parallel_dims.world_mesh is not None:
        groups.append(parallel_dims.world_mesh["sp"].get_group())
        groups.append(parallel_dims.world_mesh["dp"].get_group())
    if parallel_dims.fsdp_mesh is not None:
        groups.append(parallel_dims.fsdp_mesh["fsdp_shard"].get_group())
        groups.append(parallel_dims.fsdp_mesh["dp_replicate"].get_group())

    for group in groups:
        dist.all_reduce(probe, group=group)
    torch.cuda.synchronize()


def destroy_sequence_parallel_group():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Sequence-parallel sequence-length context
# ---------------------------------------------------------------------------

class SPSequenceContext:
    """Full (pre-shard) sequence length of each SP-sharded stream.

    Ulysses' head-scatter all-to-all has to undo the uneven ``torch.chunk``
    split, which means it needs the *global* sequence length -- information a
    rank cannot recover from its own shard. The towers publish it here once per
    forward instead of paying a ``dist.all_gather_object`` inside every single
    attention call (~800 object collectives per step at 40 layers x 2 towers).

    Values stay valid until the next forward overwrites them, so activation
    checkpointing recompute during backward reads the correct lengths.
    """

    __slots__ = ("visual", "audio")

    def __init__(self):
        self.visual = None
        self.audio = None

    def set(self, visual=None, audio=None):
        if visual is not None:
            self.visual = int(visual)
        if audio is not None:
            self.audio = int(audio)

    def get(self, stream: str):
        return getattr(self, stream, None)

    def clear(self):
        self.visual = None
        self.audio = None


sp_seq_ctx = SPSequenceContext()


# ---------------------------------------------------------------------------
# Cross-rank data / RNG alignment (replaces hy_parallelism)
# ---------------------------------------------------------------------------

def _flatten_tensors(obj, out):
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            _flatten_tensors(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _flatten_tensors(value, out)
    return out


def _rebuild_with_tensors(obj, tensors, cursor):
    if isinstance(obj, torch.Tensor):
        value = tensors[cursor[0]]
        cursor[0] += 1
        return value
    if isinstance(obj, dict):
        return {k: _rebuild_with_tensors(v, tensors, cursor) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        rebuilt = [_rebuild_with_tensors(v, tensors, cursor) for v in obj]
        return tuple(rebuilt) if isinstance(obj, tuple) else rebuilt
    return obj


def sync_data_for_sp(obj, parallel_dims: Optional[ParallelDims] = None, force_object: bool = False):
    """Make ``obj`` identical on every rank of the SP group (source = local rank 0).

    Containers are walked recursively, so a whole dataloader batch can be passed
    straight in. Non-tensor leaves go through ``broadcast_object_list``; tensors
    are broadcast on the device.

    The shapes are agreed on FIRST, in a single object broadcast, and any rank
    whose local tensor disagrees reallocates before the data broadcast. That
    matters because ``dist.broadcast`` demands matching shapes on every
    participant: without the agreement step a rank that silently substituted a
    different sample (the dataset retries on an I/O error, and an error need not
    hit every rank) would corrupt the batch or hang the group -- the exact
    failure this call exists to prevent.
    """
    parallel_dims = parallel_dims or get_parallel_state()
    if not parallel_dims.sp_enabled:
        return obj

    src = parallel_dims.sp_src_rank
    group = parallel_dims.sp_group

    if force_object or not isinstance(obj, (torch.Tensor, dict, list, tuple)):
        holder = [obj]
        dist.broadcast_object_list(holder, src=src, group=group)
        return holder[0]

    tensors = _flatten_tensors(obj, [])
    if not tensors:
        holder = [obj]
        dist.broadcast_object_list(holder, src=src, group=group)
        return holder[0]

    specs = [(tuple(t.shape), t.dtype) for t in tensors]
    holder = [specs]
    dist.broadcast_object_list(holder, src=src, group=group)
    specs = holder[0]

    device = torch.device("cuda", torch.cuda.current_device())
    synced = []
    for tensor, (shape, dtype) in zip(tensors, specs):
        target_device = tensor.device
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            staged = torch.empty(shape, dtype=dtype, device=device)
        else:
            staged = tensor if tensor.is_cuda else tensor.to(device)
        dist.broadcast(staged, src=src, group=group)
        synced.append(staged if target_device.type == "cuda" else staged.to(target_device))

    return _rebuild_with_tensors(obj, synced, [0])


def sync_random_states(step: int, base_seed: int = 0, dp_rank: int = 0):
    """Re-seed the RNGs so an SP group samples identical noise.

    Derived purely from ``(base_seed, dp_rank, step)``: every SP rank computes
    the same value locally, so there is no collective and no device sync, while
    distinct DP groups still see distinct noise.
    """
    seed = (int(base_seed) * 1_000_003 + int(dp_rank) * 10_007 + int(step)) % (2 ** 31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
