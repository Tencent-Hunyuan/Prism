
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn import functional as F




def broadcast(input_: torch.Tensor, group: dist.ProcessGroup):
    src = dist.get_global_rank(group, 0)
    dist.broadcast(input_, src=src, group=group)


# ---------------------------------------------------------------------------
# Shard bookkeeping
#
# Sequences are split across SP ranks with ``torch.chunk`` semantics: every rank
# gets ``ceil(S / sp)`` tokens except the last, which gets the remainder. The
# collectives below need those sizes, but a rank cannot infer the global length
# S from its own shard. Rather than negotiating it at runtime with a
# ``dist.all_gather_object`` on every call (which costs a pickle round-trip plus
# a device sync ~800 times per training step), the towers publish S through
# ``sp_seq_ctx`` and pass it down as ``full_seq_len``.
# ---------------------------------------------------------------------------

def sp_chunk_sizes(total: int, world_size: int) -> List[int]:
    """Shard sizes produced by ``torch.chunk(x, world_size)`` along one dim."""
    chunk = (total + world_size - 1) // world_size
    return [min(chunk, max(total - i * chunk, 0)) for i in range(world_size)]


def assert_sp_splittable(total: int, world_size: int, name: str = "sequence"):
    """``torch.chunk`` silently returns fewer than ``world_size`` chunks when the
    sequence is very short, which would make high SP ranks index out of range."""
    if world_size <= 1:
        return
    sizes = sp_chunk_sizes(total, world_size)
    assert sizes[-1] > 0, (
        f"{name} length {total} is too short for sp_size={world_size}: "
        f"the trailing SP rank(s) would receive zero tokens. Reduce sp_size or "
        f"increase the resolution / duration bucket."
    )


def sp_split(x: torch.Tensor, world_size: int, rank: int, dim: int) -> torch.Tensor:
    """``torch.chunk``-compatible shard selection that never indexes past the end."""
    if world_size <= 1:
        return x
    sizes = sp_chunk_sizes(x.shape[dim], world_size)
    return torch.split(x, sizes, dim=dim)[rank]


# ---------------------------------------------------------------------------
# Ulysses head padding (for num_heads not divisible by sp_size, e.g. audio
# DiT = 12 heads with sp_size = 8).
#
# `all_to_all_4D` requires `num_heads % sp_size == 0` because it scatters the
# head dim evenly across SP ranks. These helpers let a module pad its head dim
# with ZERO heads up to the next multiple of sp_size purely at runtime, with a
# strict isolation contract enforced by the call sites:
#
#   1. Padding is appended (never weight-level) -> pretrained checkpoints load
#      unchanged.
#   2. After the head-scatter all_to_all, each rank holds the CONTIGUOUS global
#      head block [r*hpr, (r+1)*hpr); real heads are the prefix [0, num_heads),
#      so the real heads on a rank are always its FIRST `local_real` local heads.
#      Call sites slice those off BEFORE any attention compute, so padded (zero)
#      heads NEVER enter the BSA Triton / flash kernels and cannot produce NaNs
#      or perturb the real heads.
#   3. Ranks that hold only padding heads (local_real == 0) skip the kernel
#      entirely and emit a zero-width head slice; they still join the all_to_all
#      collectives so distributed/FSDP execution stays in lock-step.
#   4. After the inverse all_to_all the head dim is truncated back to num_heads,
#      so forward and backward for the real heads are bit-identical to the
#      unpadded computation (padding contributes exactly zero gradient).
# ---------------------------------------------------------------------------

def sp_pad_num_heads(num_heads: int, sp_size: int) -> int:
    """Smallest multiple of ``sp_size`` that is >= ``num_heads``."""
    remainder = num_heads % sp_size
    if remainder == 0:
        return num_heads
    return num_heads + (sp_size - remainder)


def sp_pad_heads(x: torch.Tensor, padded_heads: int, dim: int = 2) -> torch.Tensor:
    """Append zero-valued heads along ``dim`` so the head count becomes
    ``padded_heads``. Autograd-safe: gradients routed to the appended heads are
    discarded by the caller's later truncation, leaving the real heads' grads
    untouched. Returns ``x`` unchanged when no padding is needed."""
    cur = x.shape[dim]
    if cur >= padded_heads:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = padded_heads - cur
    pad = x.new_zeros(pad_shape)
    return torch.cat([x, pad], dim=dim)


def sp_num_local_real_heads(num_real_heads: int, padded_heads: int,
                            sp_size: int, sp_rank: int) -> int:
    """Number of this rank's local heads that are REAL after ``all_to_all_4D``
    evenly distributes ``padded_heads``. Real heads are the global prefix
    ``[0, num_real_heads)`` and rank ``r`` owns the contiguous global block
    ``[r*hpr, (r+1)*hpr)`` (hpr = padded_heads // sp_size), so the real heads are
    always this rank's FIRST ``k`` local heads."""
    hpr = padded_heads // sp_size
    return max(0, min(num_real_heads - sp_rank * hpr, hpr))


# ---------------------------------------------------------------------------
# Ulysses all-to-all
# ---------------------------------------------------------------------------

_UNKNOWN_SEQLEN_WARNED = False


def _negotiate_seq_lens(local_len: int, group) -> List[int]:
    global _UNKNOWN_SEQLEN_WARNED
    if not _UNKNOWN_SEQLEN_WARNED:
        print(
            "[SP] all_to_all_4D called without full_seq_len; falling back to a "
            "per-call all_gather_object. This is correct but slow -- publish the "
            "sequence length through hymm.utils.parallel_states.sp_seq_ctx."
        )
        _UNKNOWN_SEQLEN_WARNED = True
    seq_lens = [None] * dist.get_world_size(group)
    dist.all_gather_object(seq_lens, local_len, group)
    return seq_lens


def _all_to_all_4D(
    input: torch.Tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group=None,
    full_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """all-to-all for QKV.

    Args:
        input: a 4D tensor sharded along the scatter dim.
        scatter_idx / gather_idx: 2/1 scatters heads and gathers sequence,
            1/2 does the inverse.
        full_seq_len: global (unsharded) sequence length. Required for the 2/1
            direction, where it is the only way to know how much padding the
            trailing shard carries.
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # (bs, seqlen/P, hc, hs) -> (bs, seqlen, hc/P, hs)
        if full_seq_len is not None:
            sizes = sp_chunk_sizes(int(full_seq_len), seq_world_size)
            gap = sizes[0] - sizes[-1]
        else:
            seq_lens = _negotiate_seq_lens(input.shape[1], group)
            gap = seq_lens[0] - seq_lens[-1] if seq_lens[-1] != seq_lens[0] else 0
            assert gap >= 0, "only the trailing SP rank may hold a shorter shard"

        if gap > 0 and dist.get_group_rank(group, dist.get_rank()) == seq_world_size - 1:
            input = F.pad(input, (0, 0, 0, 0, 0, gap))

        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        assert hc % seq_world_size == 0, (
            f'Invalid Head size: {hc}, which should be divisible by spsize {seq_world_size}'
        )
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        if seq_world_size > 1:
            output = torch.empty_like(input_t)
            # Stream-ordered: NCCL enqueues on the current stream and every
            # consumer below is on that same stream, so no host sync is needed.
            dist.all_to_all_single(output, input_t, group=group)
        else:
            output = input_t

        output = output.reshape(seqlen, bs, shard_hc, hs)
        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
        if gap > 0:
            output = output[:, :-gap]

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # (bs, seqlen, hc/P, hs) -> (bs, seqlen/P, hc, hs)
        # `seqlen` here is already the global length, so the split is derived
        # locally with no negotiation.
        bs, seqlen, shard_hc, hs = input.shape

        hc = shard_hc * seq_world_size
        if seqlen % seq_world_size != 0:
            new_seqlen = (seqlen // seq_world_size + 1) * seq_world_size
            gap = new_seqlen - seqlen
            input = F.pad(input, (0, 0, 0, 0, 0, gap))
            bs, seqlen, shard_hc, hs = input.shape
        else:
            gap = 0

        shard_seqlen = seqlen // seq_world_size

        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        if seq_world_size > 1:
            output = torch.empty_like(input_t)
            dist.all_to_all_single(output, input_t, group=group)
        else:
            output = input_t

        output = output.reshape(hc, shard_seqlen, bs, hs)
        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        if gap > 0 and dist.get_group_rank(group, dist.get_rank()) == seq_world_size - 1:
            output = output[:, :-gap]

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
        full_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.full_seq_len = full_seq_len

        return _all_to_all_4D(input, scatter_idx, gather_idx, group=group,
                              full_seq_len=full_seq_len)

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor):
        return (
            None,
            SeqAllToAll4D.apply(
                ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx,
                ctx.full_seq_len,
            ),
            None,
            None,
            None,
        )


def all_to_all_4D(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    full_seq_len: Optional[int] = None,
):
    return SeqAllToAll4D.apply(group, input_, scatter_dim, gather_dim, full_seq_len)


def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [
        t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all(
            input_, ctx.world_size, process_group, scatter_dim, gather_dim
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
    input_: torch.Tensor, group: dist.ProcessGroup, scatter_dim: int = 2, gather_dim: int = 1
):
    return _AllToAll.apply(input_, group, scatter_dim, gather_dim)


class _AllGather(torch.autograd.Function):
    """All-gather along ``dim`` with autograd support.

    ``full_seq_len`` lets both the forward gather and the backward split derive
    the per-rank shard sizes locally, so neither direction needs the shape
    negotiation collective.
    """

    @staticmethod
    def forward(ctx, input_, dim, group, full_seq_len=None):
        world_size = dist.get_world_size(group)
        rank = dist.get_group_rank(group, dist.get_rank())

        if full_seq_len is not None:
            sizes = sp_chunk_sizes(int(full_seq_len), world_size)
        else:
            shapes = [None] * world_size
            dist.all_gather_object(shapes, input_.shape[dim], group)
            sizes = [int(s) for s in shapes]

        ctx.dim = dim
        ctx.group = group
        ctx.sizes = sizes
        ctx.rank = rank

        input_ = input_.contiguous()
        if all(s == sizes[0] for s in sizes):
            # Uniform shards: a single ncclAllGather instead of the
            # broadcast-per-rank fallback PyTorch falls back to for ragged inputs.
            if dim == 0:
                out_shape = list(input_.shape)
                out_shape[0] = sizes[0] * world_size
                flat = torch.empty(out_shape, dtype=input_.dtype, device=input_.device)
                dist.all_gather_into_tensor(flat, input_, group=group)
                return flat
            tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
            dist.all_gather(tensor_list, input_, group=group)
            return torch.cat(tensor_list, dim=dim)

        tensor_list = []
        for size in sizes:
            shape = list(input_.shape)
            shape[dim] = size
            tensor_list.append(
                torch.empty(shape, dtype=input_.dtype, device=input_.device)
            )
        dist.all_gather(tensor_list, input_, group=group)
        return torch.cat(tensor_list, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.split(grad_output, ctx.sizes, dim=ctx.dim)[ctx.rank]
        return grad_input.contiguous(), None, None, None


def all_gather(input_: torch.Tensor, dim: int = 1, group=None,
               full_seq_len: Optional[int] = None):
    """Gather ``input_`` shards along ``dim`` across the SP group.

    Args:
        input_: this rank's shard.
        dim: concatenation dimension.
        group: SP process group.
        full_seq_len: global length along ``dim``. When supplied the shard sizes
            are computed locally, avoiding a shape-negotiation collective in both
            the forward and the backward pass.
    """
    return _AllGather.apply(input_, dim, group, full_seq_len)
