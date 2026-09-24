"""Multi-resolution / multi-duration bucket definitions.

A bucket plan is fully described by four numbers -- ``base_size``,
``patch_size`` and the temporal ``(min, max, interval)`` -- so the training job
and the offline CSV builder can derive an identical plan without shipping an
index file around.

Spatial buckets keep the token count roughly constant: every crop is
``(hp * patch_size) x (wp * patch_size)`` with ``hp * wp <= (base_size /
patch_size) ** 2``, which is the same construction the Arrow pipeline used.
"""

from typing import List, Sequence, Tuple

import numpy as np

VALID_BUCKET_HW_BASE_SIZES = [128, 256, 480, 512, 640, 720, 960, 1440, 1920]

LATENT_RESOLUTION_TO_BASE_SIZE = {
    "480p": 480,
    "640p": 640,
    "720p": 960,
    "1080p": 1440,
    "2k": 1920,
}


def generate_spatial_buckets(base_size: int = 960, patch_size: int = 16,
                             max_ratio: float = 4.0) -> List[Tuple[int, int]]:
    """All ``(height, width)`` crops for one base size, coarsest aspect first."""
    num_patches = round((base_size / patch_size) ** 2)
    assert max_ratio >= 1.0
    sizes = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            sizes.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return sizes


def generate_temporal_buckets(min_length: int = 49, max_length: int = 289,
                              interval: int = 12) -> List[int]:
    """Frame counts of the form ``4k + 1`` between min and max, inclusive."""
    assert max_length > min_length, (
        f"temporal_max_length({max_length}) must exceed temporal_min_length({min_length})"
    )
    assert (min_length - 1) % 4 == 0, (
        f"temporal_min_length-1 must be a multiple of 4, got {min_length}"
    )
    assert (max_length - 1) % 4 == 0, (
        f"temporal_max_length-1 must be a multiple of 4, got {max_length}"
    )
    lengths = [n + 1 for n in range(min_length - 1, max_length - 1, interval)]
    lengths.append(max_length)
    return sorted(set(lengths))


class BucketPlan:
    """Resolves a sample's ``(frames, height, width)`` onto the bucket grid."""

    def __init__(self,
                 base_size: int = 960,
                 patch_size: int = 16,
                 temporal_min_length: int = 49,
                 temporal_max_length: int = 289,
                 temporal_interval: int = 12,
                 multireso: bool = True,
                 multitemp: bool = True):
        assert base_size in VALID_BUCKET_HW_BASE_SIZES, (
            f"base_size must be one of {VALID_BUCKET_HW_BASE_SIZES}, got {base_size}"
        )
        self.base_size = base_size
        self.patch_size = patch_size
        self.multireso = multireso
        self.multitemp = multitemp

        self.spatial_buckets = generate_spatial_buckets(base_size, patch_size)
        self.aspect_ratios = np.array(
            [round(float(h) / float(w), 5) for h, w in self.spatial_buckets]
        )
        if multitemp:
            self.temporal_buckets = generate_temporal_buckets(
                temporal_min_length, temporal_max_length, temporal_interval
            )
        else:
            self.temporal_buckets = [temporal_max_length]

    # -- resolution -------------------------------------------------------

    def closest_temporal(self, num_frames: int) -> int:
        """Largest temporal bucket that still fits, i.e. never upsamples."""
        closest = self.temporal_buckets[0]
        for bucket in self.temporal_buckets:
            if num_frames - bucket >= 0:
                closest = bucket
        return closest

    def closest_spatial(self, height: int, width: int) -> Tuple[int, int]:
        """Nearest spatial bucket.

        The two legacy code paths matched differently and both are reproduced
        here. With temporal buckets on, the old code took a plain global argmin
        over the aspect ratios. Without them it first restricted the candidates
        to one side of the target ratio -- buckets no taller than the source when
        the source is landscape-or-square, no wider when it is portrait -- so the
        crop only ever removed content. The two disagree on roughly 40% of inputs
        at base 960, so picking one for both paths would silently reassign a
        large share of the corpus.
        """
        aspect_ratio = float(height) / float(width)
        if self.multitemp:
            idx = int(np.abs(self.aspect_ratios - aspect_ratio).argmin())
        else:
            diffs = self.aspect_ratios - aspect_ratio
            if aspect_ratio >= 1:
                candidates = [(i, d) for i, d in enumerate(diffs) if d <= 0]
            else:
                candidates = [(i, d) for i, d in enumerate(diffs) if d >= 0]
            if not candidates:
                candidates = list(enumerate(diffs))
            idx = min(candidates, key=lambda pair: abs(pair[1]))[0]
        h, w = self.spatial_buckets[idx]
        return int(h), int(w)

    def assign(self, num_frames: int, height: int, width: int) -> Tuple[int, int, int]:
        """Map a raw sample onto ``(bucket_frames, bucket_height, bucket_width)``."""
        if self.multitemp:
            # Only the multi-temporal path filtered on length; the old
            # single-length path accepted everything.
            if num_frames < self.temporal_buckets[0]:
                raise ValueError(
                    f"video with {num_frames} frames is shorter than the smallest "
                    f"temporal bucket ({self.temporal_buckets[0]})"
                )
            bucket_frames = self.closest_temporal(num_frames)
        else:
            bucket_frames = min(self.temporal_buckets[-1], num_frames)
            bucket_frames = bucket_frames - (bucket_frames - 1) % 4

        if self.multireso:
            bucket_h, bucket_w = self.closest_spatial(height, width)
        else:
            bucket_h, bucket_w = self.base_size, self.base_size
        return int(bucket_frames), bucket_h, bucket_w

    # -- introspection ----------------------------------------------------

    def describe(self) -> str:
        return (
            f"BucketPlan(base_size={self.base_size}, patch_size={self.patch_size}, "
            f"spatial_buckets={len(self.spatial_buckets)}, "
            f"temporal_buckets={self.temporal_buckets})"
        )


def latent_shape_to_pixels(latent_frames: int, latent_height: int, latent_width: int,
                           vae_spatial_ratio: int = 8,
                           vae_temporal_ratio: int = 4) -> Tuple[int, int, int]:
    """Recover the pixel-space ``(frames, height, width)`` of a cached latent."""
    num_frames = (int(latent_frames) - 1) * vae_temporal_ratio + 1
    return num_frames, int(latent_height) * vae_spatial_ratio, int(latent_width) * vae_spatial_ratio


def summarize_buckets(assignments: Sequence[Tuple[int, int, int]]) -> str:
    """Human-readable count per bucket, sorted by population."""
    counts = {}
    for key in assignments:
        counts[key] = counts.get(key, 0) + 1
    lines = [f"  {'frames x height x width':>28}  {'count':>8}"]
    lines.append(f"  {'-' * 28}  {'-' * 8}")
    for (t, h, w), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {f'{t} x {h} x {w}':>28}  {n:>8}")
    return "\n".join(lines)
