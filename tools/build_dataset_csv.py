#!/usr/bin/env python3
"""Assign multi-resolution / multi-duration buckets to a training CSV.

Reads a raw manifest, resolves every row onto the bucket grid, and writes a new
CSV with ``bucket_frames`` / ``bucket_height`` / ``bucket_width`` columns filled
in. Doing it offline keeps the training job from re-scanning the corpus on every
launch, and lets you inspect the bucket histogram before burning GPU hours.

Bucketing is also done lazily by the dataset when these columns are absent, so
this step is an optimisation and a sanity check rather than a hard requirement.

Input columns
-------------
required : latent_path, video_path, audio_path, caption
shape    : latent_frames + latent_height + latent_width  (preferred)
           or num_frames + height + width
optional : video_id, audio_caption, fps, best_frame_index, ref_frame_index

Examples
--------
    # bucket a manifest that already carries latent shapes
    python tools/build_dataset_csv.py \
        --input  /data/i2va_raw.csv \
        --output /data/i2va_720p.csv \
        --latent-resolution 720p \
        --temporal-min-length 49 --temporal-max-length 289 --temporal-interval 12

    # read the shapes straight off the .npy headers first
    python tools/build_dataset_csv.py \
        --input /data/i2va_raw.csv --output /data/i2va_720p.csv \
        --latent-resolution 720p --scan-latents --workers 32
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hymm.dataset.bucket_utils import (  # noqa: E402
    LATENT_RESOLUTION_TO_BASE_SIZE,
    BucketPlan,
    latent_shape_to_pixels,
    summarize_buckets,
)

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

OUTPUT_COLUMNS = [
    "video_id", "latent_path", "video_path", "audio_path",
    "caption", "audio_caption", "fps", "best_frame_index", "ref_frame_index",
    "latent_frames", "latent_height", "latent_width",
    "num_frames", "height", "width",
    "bucket_frames", "bucket_height", "bucket_width",
]


def read_npy_shape(path: str) -> Optional[tuple]:
    """Read a .npy header without loading the array."""
    try:
        with open(path, "rb") as fh:
            version = np.lib.format.read_magic(fh)
            shape, _fortran, _dtype = np.lib.format._read_array_header(fh, version)
        return tuple(shape)
    except Exception:
        return None


def scan_latent_shapes(rows: List[Dict], workers: int) -> int:
    """Fill latent_frames/height/width from the cached .npy files."""
    def worker(row):
        shape = read_npy_shape(row["latent_path"])
        if shape is None or len(shape) < 4:
            return False
        # [C, T, H, W] or [1, C, T, H, W]
        row["latent_frames"] = str(shape[-3])
        row["latent_height"] = str(shape[-2])
        row["latent_width"] = str(shape[-1])
        return True

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(worker, rows))
    return sum(1 for ok in results if ok)


def resolve_shape(row: Dict, spatial_ratio: int, temporal_ratio: int):
    lf, lh, lw = row.get("latent_frames"), row.get("latent_height"), row.get("latent_width")
    if lf and lh and lw:
        return latent_shape_to_pixels(int(float(lf)), int(float(lh)), int(float(lw)),
                                      spatial_ratio, temporal_ratio)
    nf, h, w = row.get("num_frames"), row.get("height"), row.get("width")
    if nf and h and w:
        return int(float(nf)), int(float(h)), int(float(w))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="raw manifest CSV")
    parser.add_argument("--output", required=True, help="bucketed CSV to write")
    parser.add_argument("--latent-resolution", default=None,
                        choices=sorted(LATENT_RESOLUTION_TO_BASE_SIZE),
                        help="shorthand that sets --base-size")
    parser.add_argument("--base-size", type=int, default=960)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--temporal-min-length", type=int, default=49)
    parser.add_argument("--temporal-max-length", type=int, default=289)
    parser.add_argument("--temporal-interval", type=int, default=12)
    parser.add_argument("--vae-spatial-ratio", type=int, default=8)
    parser.add_argument("--vae-temporal-ratio", type=int, default=4)
    parser.add_argument("--no-multireso", action="store_true",
                        help="single spatial bucket at base_size x base_size")
    parser.add_argument("--no-multitemp", action="store_true",
                        help="single temporal bucket at temporal_max_length")
    parser.add_argument("--scan-latents", action="store_true",
                        help="read latent shapes from the .npy headers")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--min-batch-per-bucket", type=int, default=0,
                        help="drop buckets holding fewer than this many samples")
    args = parser.parse_args()

    base_size = args.base_size
    if args.latent_resolution:
        base_size = LATENT_RESOLUTION_TO_BASE_SIZE[args.latent_resolution]

    plan = BucketPlan(
        base_size=base_size,
        patch_size=args.patch_size,
        temporal_min_length=args.temporal_min_length,
        temporal_max_length=args.temporal_max_length,
        temporal_interval=args.temporal_interval,
        multireso=not args.no_multireso,
        multitemp=not args.no_multitemp,
    )

    print("[config]")
    print(f"  input               : {args.input}")
    print(f"  output              : {args.output}")
    print(f"  {plan.describe()}")
    print()

    with open(args.input, "r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"read {len(rows)} rows")

    if args.scan_latents:
        ok = scan_latent_shapes(rows, args.workers)
        print(f"resolved latent shapes for {ok}/{len(rows)} rows")

    kept, skipped_shape, skipped_bucket = [], 0, 0
    assignments = []
    for row in rows:
        shape = resolve_shape(row, args.vae_spatial_ratio, args.vae_temporal_ratio)
        if shape is None:
            skipped_shape += 1
            continue
        num_frames, height, width = shape
        try:
            bucket = plan.assign(num_frames, height, width)
        except ValueError:
            skipped_bucket += 1
            continue

        row["num_frames"] = str(num_frames)
        row["height"] = str(height)
        row["width"] = str(width)
        row["bucket_frames"], row["bucket_height"], row["bucket_width"] = (
            str(bucket[0]), str(bucket[1]), str(bucket[2])
        )
        row.setdefault("video_id", os.path.splitext(os.path.basename(row["latent_path"]))[0])
        kept.append(row)
        assignments.append(bucket)

    if args.min_batch_per_bucket > 0:
        from collections import Counter

        counts = Counter(assignments)
        keep_keys = {k for k, n in counts.items() if n >= args.min_batch_per_bucket}
        before = len(kept)
        filtered = [
            (row, key) for row, key in zip(kept, assignments) if key in keep_keys
        ]
        kept = [row for row, _ in filtered]
        assignments = [key for _, key in filtered]
        print(f"dropped {before - len(kept)} rows in buckets smaller than "
              f"{args.min_batch_per_bucket}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in kept:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})

    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  rows in           : {len(rows)}")
    print(f"  rows out          : {len(kept)}")
    print(f"  skipped (no shape): {skipped_shape}")
    print(f"  skipped (too short for the smallest temporal bucket): {skipped_bucket}")
    print(f"  output            : {args.output}")
    print()
    print(summarize_buckets(assignments))
    print("=" * 60)


if __name__ == "__main__":
    main()
