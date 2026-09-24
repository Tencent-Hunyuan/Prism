"""Distributed bucket batch sampler.

Every batch is drawn from a single bucket, so all samples in it share a tensor
shape.

The important property is that **every data-parallel rank yields exactly the
same number of batches**. Batches are planned globally from a shared seed and
then dealt out round-robin, with the tail truncated to a multiple of the DP
size. The previous Arrow-based sampler filled per-rank buckets independently,
so the batch count depended on how a rank's shard happened to distribute over
buckets; whichever rank ran dry first stopped issuing collectives and every
other rank blocked until the NCCL watchdog killed the job.
"""

import random
from typing import Iterator, List, Optional, Sequence

from torch.utils.data import Sampler


class DistributedBucketBatchSampler(Sampler):
    def __init__(
        self,
        bucket_keys: Sequence,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} out of range for num_replicas {num_replicas}")

        self.bucket_keys = list(bucket_keys)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.epoch = 0
        self.start_batch = 0
        self._cached_epoch = None
        self._cached_batches: Optional[List[List[int]]] = None

        groups = {}
        for index, key in enumerate(self.bucket_keys):
            groups.setdefault(key, []).append(index)
        # Sorted so the plan does not depend on dict insertion order.
        self._groups = [(key, groups[key]) for key in sorted(groups, key=repr)]

    # -- planning ---------------------------------------------------------

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_start_batch(self, start_batch: int):
        """Skip the first ``start_batch`` batches of the current epoch on resume."""
        self.start_batch = max(0, int(start_batch))

    def _plan(self) -> List[List[int]]:
        if self._cached_epoch == self.epoch and self._cached_batches is not None:
            return self._cached_batches

        rng = random.Random(self.seed * 1_000_003 + self.epoch)
        batches: List[List[int]] = []
        for _key, indices in self._groups:
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        # Truncate so the deal is exact: no rank can be handed a spare batch.
        usable = (len(batches) // self.num_replicas) * self.num_replicas
        batches = batches[:usable]

        self._cached_epoch = self.epoch
        self._cached_batches = batches
        return batches

    # -- Sampler API ------------------------------------------------------

    def num_global_batches(self) -> int:
        return len(self._plan())

    def batches_per_epoch(self) -> int:
        """Per-rank batch count for a full epoch, ignoring any resume offset."""
        return len(self._plan()) // self.num_replicas

    def __len__(self) -> int:
        # Has to match what __iter__ actually yields: on the epoch a run resumes
        # into, the first `start_batch` batches are skipped. Reporting the full
        # count there would make the DataLoader's length -- and anything derived
        # from it -- overstate the epoch by exactly the number of replayed steps.
        return max(0, self.batches_per_epoch() - self.start_batch)

    def __iter__(self) -> Iterator[List[int]]:
        mine = self._plan()[self.rank::self.num_replicas]
        if self.start_batch:
            mine = mine[self.start_batch:]
        return iter(mine)


class EpochCyclingBatchIterator:
    """Endless iterator over a bucketed DataLoader with synchronous epoch rollover.

    Because :class:`DistributedBucketBatchSampler` reports the same ``__len__``
    on every rank, all ranks exhaust an epoch on the same step and roll over
    together, so no rank ever sits in a collective its peers have already left.
    """

    def __init__(self, dataloader, batch_sampler, dataset=None,
                 start_epoch: int = 0, start_batch: int = 0, logger=None):
        self.dataloader = dataloader
        self.batch_sampler = batch_sampler
        self.dataset = dataset
        self.logger = logger
        self.epoch = int(start_epoch)
        self.batches_in_epoch = 0
        self._pending_start_batch = int(start_batch)
        self._iterator = None

    @property
    def steps_per_epoch(self) -> int:
        return self.batch_sampler.batches_per_epoch()

    def _start_epoch(self):
        self.batch_sampler.set_epoch(self.epoch)
        self.batch_sampler.set_start_batch(self._pending_start_batch)
        if self.dataset is not None and hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self.epoch)
        self.batches_in_epoch = self._pending_start_batch
        self._pending_start_batch = 0
        if self.logger is not None:
            skipped = self.batch_sampler.start_batch
            resumed = f", skipping {skipped} already-consumed" if skipped else ""
            self.logger.info(
                f"[data] epoch {self.epoch}: {len(self.batch_sampler)} batches/rank"
                f"{resumed} ({self.batch_sampler.num_global_batches()} global)"
            )
        self._iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            self._start_epoch()
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self._start_epoch()
            batch = next(self._iterator)
        self.batches_in_epoch += 1
        return batch
