"""CSV-backed video+audio dataset for MOVA training.

One row per training sample. Required columns:

    video_id      identifier used in logs
    latent_path   cached WAN VAE latent (.npy), shape [C, T, H, W] or [1, C, T, H, W]
    video_path    source video, read only to grab the reference frame
    audio_path    source audio (m4a / wav / ...)
    caption       structured JSON caption, or plain text

Optional columns:

    audio_caption                              separate audio prompt (defaults to `caption`)
    latent_frames / latent_height / latent_width   cached latent dims, preferred for bucketing
    num_frames / height / width                pixel-space fallback when latent dims are absent
    fps                                        source frame rate (defaults to dataloader_config.video_fps)
    best_frame_index                           frame offset where the audio crop starts
    ref_frame_index                            frame used as the i2v reference (default 1)
    bucket_frames / bucket_height / bucket_width   precomputed bucket, written by tools/build_dataset_csv.py

Everything a sample does randomly -- caption assembly, unconditional dropout,
substituting a replacement row when a file is unreadable -- is driven by an RNG
seeded from ``(global_seed, epoch, row index)``. Two ranks in the same sequence
parallel group therefore materialise byte-identical batches without talking to
each other, which is what keeps the two towers in lock-step.
"""

import csv
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from hymm.dataset.bucket_utils import (
    LATENT_RESOLUTION_TO_BASE_SIZE,
    BucketPlan,
    latent_shape_to_pixels,
)

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

REQUIRED_COLUMNS = ("latent_path", "video_path", "audio_path", "caption")


def _as_int(value, default=None):
    if value is None or value == "":
        return default
    return int(float(value))


def _as_float(value, default=None):
    if value is None or value == "":
        return default
    return float(value)


def read_dataset_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"dataset csv {path} is empty")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(
            f"dataset csv {path} is missing required column(s) {missing}; "
            f"found {list(rows[0].keys())}"
        )
    return rows


class CsvVideoAudioDataset(Dataset):
    def __init__(
        self,
        csv_files,
        args=None,
        video_fps: float = 24.0,
        vae_spatial_ratio: int = 8,
        vae_temporal_ratio: int = 4,
        multireso: bool = True,
        multitemp: bool = True,
        bucket_hw_base_size: int = 960,
        bucket_hw_bucket_stride: int = 16,
        bucket_temporal_min_length: int = 49,
        bucket_temporal_max_length: int = 289,
        bucket_temporal_interval: int = 12,
        latent_resolution: Optional[str] = None,
        caption_sample_ratio=None,
        caption_processor: str = "caption_process_v1",
        ocr_only_long_caption: bool = False,
        num_replace_rate: float = 0.0,
        video_uncond_p: float = 0.0,
        audio_uncond_p: float = 0.0,
        audio_sr: int = 48000,
        crop_latent_to_bucket: bool = False,
        global_seed: int = 42,
        max_retries: int = 32,
        logger=None,
    ):
        if logger is None:
            from loguru import logger
        self.logger = logger
        self.args = args

        if isinstance(csv_files, str):
            csv_files = [csv_files]
        self.csv_files = list(csv_files)

        if latent_resolution is not None:
            res = str(latent_resolution).lower()
            if res not in LATENT_RESOLUTION_TO_BASE_SIZE:
                raise ValueError(
                    f"Unsupported latent_resolution '{latent_resolution}'. "
                    f"Choose from {sorted(LATENT_RESOLUTION_TO_BASE_SIZE)}"
                )
            bucket_hw_base_size = LATENT_RESOLUTION_TO_BASE_SIZE[res]
        self.latent_resolution = latent_resolution

        self.video_fps = float(video_fps)
        self.vae_spatial_ratio = int(vae_spatial_ratio)
        self.vae_temporal_ratio = int(vae_temporal_ratio)
        self.video_uncond_p = float(video_uncond_p)
        self.audio_uncond_p = float(audio_uncond_p)
        self.audio_sr = int(audio_sr)
        self.crop_latent_to_bucket = bool(crop_latent_to_bucket)
        self.global_seed = int(global_seed)
        self.max_retries = int(max_retries)
        self.epoch = 0

        self.bucket_plan = BucketPlan(
            base_size=bucket_hw_base_size,
            patch_size=bucket_hw_bucket_stride,
            temporal_min_length=bucket_temporal_min_length,
            temporal_max_length=bucket_temporal_max_length,
            temporal_interval=bucket_temporal_interval,
            multireso=multireso,
            multitemp=multitemp,
        )

        # -- structured caption ------------------------------------------
        self.caption_aug = None
        if caption_sample_ratio:
            if not isinstance(caption_sample_ratio, str):
                try:
                    from omegaconf import DictConfig, OmegaConf

                    if isinstance(caption_sample_ratio, DictConfig):
                        caption_sample_ratio = json.dumps(
                            OmegaConf.to_container(caption_sample_ratio)
                        )
                except ImportError:
                    pass
            if isinstance(caption_sample_ratio, str):
                caption_sample_ratio = json.loads(caption_sample_ratio)
            from hymm.dataset.caps import load_caption_processor

            self.caption_aug = load_caption_processor(
                name=caption_processor,
                caption_sample_ratio=caption_sample_ratio,
                logger=self.logger,
                kwargs={
                    "ocr_only_long_caption": ocr_only_long_caption,
                    "num_replace_rate": num_replace_rate,
                },
            )

        # -- rows ---------------------------------------------------------
        self.records = []
        for csv_file in self.csv_files:
            rows = read_dataset_csv(csv_file)
            for row in rows:
                record = self._build_record(row, csv_file)
                if record is not None:
                    self.records.append(record)

        if not self.records:
            raise ValueError(f"no usable rows found in {self.csv_files}")

        self.total_length = len(self.records)
        self.bucket_keys = [r["bucket"] for r in self.records]
        self._log_summary()

    # -- construction -----------------------------------------------------

    def _build_record(self, row: Dict[str, str], csv_file: str) -> Optional[Dict]:
        latent_frames = _as_int(row.get("latent_frames"))
        latent_height = _as_int(row.get("latent_height"))
        latent_width = _as_int(row.get("latent_width"))

        if latent_frames and latent_height and latent_width:
            num_frames, height, width = latent_shape_to_pixels(
                latent_frames, latent_height, latent_width,
                self.vae_spatial_ratio, self.vae_temporal_ratio,
            )
        else:
            num_frames = _as_int(row.get("num_frames"))
            height = _as_int(row.get("height"))
            width = _as_int(row.get("width"))
            if not (num_frames and height and width):
                self.logger.warning(
                    f"skipping row without latent_* or num_frames/height/width "
                    f"columns in {csv_file}: {row.get('video_id')}"
                )
                return None

        bucket_frames = _as_int(row.get("bucket_frames"))
        bucket_height = _as_int(row.get("bucket_height"))
        bucket_width = _as_int(row.get("bucket_width"))
        if not (bucket_frames and bucket_height and bucket_width):
            try:
                bucket_frames, bucket_height, bucket_width = self.bucket_plan.assign(
                    num_frames, height, width
                )
            except ValueError as exc:
                self.logger.warning(f"skipping {row.get('video_id')}: {exc}")
                return None

        # The bucket key must pin down the exact tensor shape, otherwise two rows
        # in the same batch could disagree on T and the collate would fail.
        #
        # Default (False) keys on the latent's own length, so the tensor is used
        # whole -- byte-identical to the Arrow loader, which never cropped. True
        # keys on the temporal bucket instead and trims the latent down to it,
        # which loses up to `interval - 1` frames but coarsens the buckets enough
        # to fill batches when micro_batch_size > 1.
        latent_frames_from_bucket = (bucket_frames - 1) // self.vae_temporal_ratio + 1
        if self.crop_latent_to_bucket:
            latent_frames_used = latent_frames_from_bucket
        elif latent_frames:
            latent_frames_used = latent_frames
        else:
            raise ValueError(
                f"crop_latent_to_bucket=False needs the exact latent length to key "
                f"the bucket, but row '{row.get('video_id')}' in {csv_file} has no "
                f"'latent_frames' column. Either run tools/build_dataset_csv.py with "
                f"--scan-latents to fill it in, or set crop_latent_to_bucket=true."
            )

        return {
            "video_id": row.get("video_id") or os.path.basename(row["latent_path"]),
            "latent_path": row["latent_path"],
            "video_path": row["video_path"],
            "audio_path": row["audio_path"],
            "caption": row["caption"],
            "audio_caption": row.get("audio_caption") or "",
            "fps": _as_float(row.get("fps"), self.video_fps),
            "best_frame_index": _as_int(row.get("best_frame_index"), 0) or 0,
            "ref_frame_index": _as_int(row.get("ref_frame_index"), 1),
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "bucket_frames": bucket_frames,
            "latent_frames_used": latent_frames_used,
            "bucket": (latent_frames_used, bucket_height, bucket_width),
        }

    def _log_summary(self):
        from collections import Counter

        counts = Counter(self.bucket_keys)
        self.logger.info(
            f"CsvVideoAudioDataset: {self.total_length} samples from "
            f"{len(self.csv_files)} csv file(s), {len(counts)} non-empty buckets"
        )
        self.logger.info(f"  {self.bucket_plan.describe()}")

    # -- Dataset API ------------------------------------------------------

    def __len__(self):
        return self.total_length

    def set_epoch(self, epoch: int):
        """Rotate the per-sample RNG so repeated epochs vary the caption sampling."""
        self.epoch = int(epoch)

    def get_data_info(self, index: int) -> Dict:
        record = self.records[index]
        return {
            "num_frames": record["num_frames"],
            "height": record["height"],
            "width": record["width"],
            "bucket": record["bucket"],
        }

    def get_bucket_key(self, index: int):
        return self.bucket_keys[index]

    def _sample_rng(self, index: int) -> random.Random:
        # Explicit arithmetic rather than hash(): reproducible across processes
        # and Python versions, which is what makes SP ranks agree.
        seed = (self.global_seed * 1_000_003 + self.epoch * 7_919 + index) % (2 ** 31 - 1)
        return random.Random(seed)

    def __getitem__(self, index: int):
        bucket = self.bucket_keys[index]
        rng = self._sample_rng(index)
        # Replacements are drawn from the SAME bucket so a broken row can never
        # change the batch's tensor shape, and from a seeded RNG so every SP rank
        # substitutes the identical row.
        candidates = self._bucket_members(bucket)

        current = index
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self._load(current, rng)
            except Exception as exc:  # noqa: BLE001 - any I/O or decode failure
                last_error = exc
                self.logger.warning(
                    f"[dataset] sample {current} ({self.records[current]['video_id']}) "
                    f"failed on attempt {attempt + 1}: {exc}"
                )
                if not candidates:
                    break
                current = candidates[rng.randrange(len(candidates))]

        raise RuntimeError(
            f"Unable to load a usable sample for index {index} in bucket {bucket} "
            f"after {self.max_retries} attempts. Last error: {last_error}"
        )

    def _bucket_members(self, bucket) -> List[int]:
        if not hasattr(self, "_bucket_index"):
            index_map: Dict[tuple, List[int]] = {}
            for i, key in enumerate(self.bucket_keys):
                index_map.setdefault(key, []).append(i)
            self._bucket_index = index_map
        return self._bucket_index.get(bucket, [])

    # -- loading ----------------------------------------------------------

    def _load(self, index: int, rng: random.Random) -> Dict:
        record = self.records[index]
        latent_frames, bucket_height, bucket_width = record["bucket"]

        latents = torch.from_numpy(np.load(record["latent_path"]))
        # Cached latents are [C, T, H, W], sometimes with leading singleton dims.
        # Only singletons may be dropped: squeeze(0) on a non-unit dim is a no-op,
        # so a loop here would spin forever on a genuinely batched array.
        while latents.dim() > 4 and latents.shape[0] == 1:
            latents = latents.squeeze(0)
        if latents.dim() != 4:
            raise ValueError(
                f"cached latent {record['latent_path']} has shape "
                f"{tuple(latents.shape)}; expected [C, T, H, W]"
            )
        if torch.isnan(latents).any():
            raise ValueError(f"cached latent {record['latent_path']} contains NaN")

        latent_h_px = latents.shape[-2] * self.vae_spatial_ratio
        latent_w_px = latents.shape[-1] * self.vae_spatial_ratio
        if (latent_h_px, latent_w_px) != (bucket_height, bucket_width):
            raise ValueError(
                f"latent {tuple(latents.shape)} maps to {latent_h_px}x{latent_w_px} px "
                f"but the row is bucketed as {bucket_height}x{bucket_width}"
            )
        if latents.shape[1] < latent_frames:
            raise ValueError(
                f"latent {tuple(latents.shape)} has fewer than the {latent_frames} "
                f"frames its temporal bucket requires"
            )
        latents = latents[:, :latent_frames]

        ref_image = self._load_reference_frame(record, latent_h_px, latent_w_px)
        waveform = self._load_audio(record, latents.shape[1])

        video_caption = self._build_caption(record["caption"], rng)
        raw_audio_caption = record["audio_caption"] or record["caption"]
        audio_caption = (
            video_caption if raw_audio_caption == record["caption"]
            else self._build_caption(raw_audio_caption, rng)
        )

        if rng.random() < self.video_uncond_p:
            video_caption = ""
        if rng.random() < self.audio_uncond_p:
            audio_caption = ""

        return {
            "video_latents": latents,
            "waveform": waveform.unsqueeze(0),
            "video_caption": video_caption,
            "audio_caption": audio_caption,
            "language_tag": "",
            "transcript_text": "",
            "videoid": record["video_id"],
            "ref_image": ref_image,
        }

    def _build_caption(self, raw_caption: str, rng: random.Random) -> str:
        if self.caption_aug is None:
            return raw_caption
        # CaptionAug samples through the `random` module; seeding it from the
        # per-sample RNG keeps the assembled prompt identical across SP ranks.
        random.seed(rng.randrange(2 ** 31))
        return self.caption_aug.caption_aug(raw_caption, lang="en")

    def _load_reference_frame(self, record: Dict, target_height: int, target_width: int):
        from decord import VideoReader

        reader = VideoReader(record["video_path"])
        frame_index = min(max(record["ref_frame_index"], 0), len(reader) - 1)
        frames = reader.get_batch([frame_index])
        if isinstance(frames, torch.Tensor):
            pixels = frames.permute(0, 3, 1, 2).contiguous()
        else:
            pixels = torch.from_numpy(frames.asnumpy()).permute(0, 3, 1, 2).contiguous()
        del reader

        ref = pixels[0].float() / 255.0
        orig_h, orig_w = ref.shape[-2], ref.shape[-1]
        scale = max(target_width / orig_w, target_height / orig_h)
        resize_hw = (int(round(orig_h * scale)), int(round(orig_w * scale)))
        preprocess = transforms.Compose([
            transforms.Resize(resize_hw, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((target_height, target_width)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return preprocess(ref)

    def _load_audio(self, record: Dict, latent_frames: int) -> torch.Tensor:
        import torchaudio
        from torchaudio.transforms import Resample

        waveform, sample_rate = torchaudio.load(record["audio_path"])
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != self.audio_sr:
            waveform = Resample(orig_freq=sample_rate, new_freq=self.audio_sr)(waveform)
        audio = waveform.squeeze(0)

        # Clip length comes from the nominal training fps while the crop offset
        # uses the row's own fps, matching how the Arrow loader aligned audio to
        # the cached latents.
        source_fps = record["fps"] or self.video_fps
        duration_s = (latent_frames - 1) * float(self.vae_temporal_ratio) / self.video_fps
        start = round(record["best_frame_index"] / source_fps * self.audio_sr)
        end = round(start + duration_s * self.audio_sr)
        audio = audio[start:end]

        expected = end - start
        if len(audio) < expected:
            shortfall = expected - len(audio)
            # 9600 samples == 0.2s at 48 kHz; anything larger means the audio and
            # the cached latent describe different clips.
            if shortfall >= start + 9600:
                raise ValueError(
                    f"audio {record['audio_path']} is {shortfall} samples short of the "
                    f"{expected} needed for a {latent_frames}-frame latent "
                    f"(source fps {source_fps}, crop offset {start} samples)"
                )
            audio = torch.nn.functional.pad(audio, (0, shortfall), "constant", 0.0)
        return audio


def collate_video_audio(batch: Sequence[Dict]) -> Dict:
    """Stack a bucket-homogeneous batch; captions and ids stay as lists."""
    shapes = {tuple(b["video_latents"].shape) for b in batch}
    if len(shapes) > 1:
        raise RuntimeError(
            f"batch mixes latent shapes {sorted(shapes)}; the bucket sampler is "
            f"supposed to keep a batch inside a single bucket"
        )

    # Clip duration is derived from each row's own fps, so two rows in the same
    # temporal bucket can still differ by a few audio samples. Right-pad with
    # silence to the batch maximum rather than dropping the batch.
    waveforms = [b["waveform"] for b in batch]
    max_samples = max(w.shape[-1] for w in waveforms)
    waveforms = [
        torch.nn.functional.pad(w, (0, max_samples - w.shape[-1])) if w.shape[-1] < max_samples else w
        for w in waveforms
    ]

    return {
        "video_latents": torch.stack([b["video_latents"] for b in batch]),
        "ref_image": torch.stack([b["ref_image"] for b in batch]),
        "waveform": torch.stack(waveforms),
        "video_caption": [b["video_caption"] for b in batch],
        "audio_caption": [b["audio_caption"] for b in batch],
        "videoid": [b["videoid"] for b in batch],
    }
