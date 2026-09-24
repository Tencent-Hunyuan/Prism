import os
import argparse
import numpy as np
import torch
from einops import rearrange
from decord import VideoReader
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor
from diffusers.models.autoencoders import AutoencoderKLWan


SIZE_TAG_MAP = {640: "480p", 960: "720p", 1440: "1080p", 1920: "2k"}


# ==================== Frame Extraction ====================

def extract_frames(video_path, sample_n_frames=289, required_fps=24, vae_time_compression_ratio=4):
    """Extract frames from a video file, resampled at required_fps.

    Frame count is aligned to 4k+1 for Wan VAE temporal compression.
    Returns (frames_np, info_dict).
    """
    video_reader = VideoReader(video_path)
    total_frames = len(video_reader)
    if total_frames < 1:
        raise ValueError(f"Video has no frames: {video_path}")

    fps = video_reader.get_avg_fps()
    if fps <= 0:
        fps = required_fps

    if total_frames == 1:
        video_images = video_reader.get_batch([0]).asnumpy()
        del video_reader
        return video_images, {
            "original_fps": fps, "fps": required_fps,
            "duration": 0, "height": video_images.shape[1],
            "width": video_images.shape[2], "num_frames": 1,
        }

    duration = total_frames / fps
    target_timestamps = np.arange(0, duration, 1.0 / required_fps)
    frame_indices = np.round(target_timestamps * fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)

    if len(frame_indices) < sample_n_frames:
        sample_n_frames = len(frame_indices) - (len(frame_indices) - 1) % vae_time_compression_ratio

    frame_indices = frame_indices[:sample_n_frames].tolist()
    frame_indices = [min(fi, total_frames - 1) for fi in frame_indices]

    video_images = video_reader.get_batch(frame_indices).asnumpy()
    del video_reader

    info = {
        "original_fps": fps, "fps": required_fps,
        "duration": len(video_images) / required_fps,
        "height": video_images.shape[1], "width": video_images.shape[2],
        "num_frames": len(video_images),
    }
    return video_images, info


# ==================== VAE Helpers ====================

def get_closest_ratio(width, height, ratios, buckets):
    aspect_ratio = float(width) / float(height)
    closest_ratio_id = np.abs(ratios - aspect_ratio).argmin()
    return buckets[closest_ratio_id]


def generate_crop_size_list(base_size=960, patch_size=16, max_ratio=4.0):
    if base_size == 480:
        max_ratio = 3.5
    num_patches = round((base_size / patch_size) ** 2)
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


def get_target_size(frames, target_size):
    T, C, H, W = frames.shape
    th, tw = target_size
    r = max(th / H, tw / W)
    return int(H * r), int(W * r)


def _fast_parallel_stack(video_images, device='cuda'):
    n_frames = len(video_images)
    h, w, c = video_images[0].shape
    result = torch.empty((n_frames, h, w, c), dtype=torch.float16, device=device)

    def _copy_frame(idx):
        result[idx] = torch.from_numpy(video_images[idx]).to(device, non_blocking=True, dtype=torch.float16)

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(_copy_frame, range(n_frames))

    return result.permute(0, 3, 1, 2)


def _process_on_gpu(frames, buckets, aspect_ratios):
    height, width = frames.shape[-2:]
    bw, bh = get_closest_ratio(width=width, height=height, ratios=aspect_ratios, buckets=buckets)
    sample_size = bh, bw
    target_size = get_target_size(frames, sample_size)

    frames = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(frames)
    frames = transforms.CenterCrop(sample_size)(frames)

    size_info = {
        "original_h": height, "original_w": width,
        "crop_h": sample_size[0], "crop_w": sample_size[1],
    }
    return frames, size_info


def _process_on_cpu(video_images_np, buckets, aspect_ratios):
    n_frames = video_images_np.shape[0]
    h, w = video_images_np.shape[1], video_images_np.shape[2]

    bw, bh = get_closest_ratio(width=w, height=h, ratios=aspect_ratios, buckets=buckets)
    sample_size = bh, bw
    th, tw = sample_size
    r = max(th / h, tw / w)
    target_size = int(h * r), int(w * r)

    resize_tf = transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(sample_size),
    ])

    result = torch.empty((n_frames, 3, sample_size[0], sample_size[1]), dtype=torch.float32)

    def _resize_frame(idx):
        frame = torch.from_numpy(video_images_np[idx]).permute(2, 0, 1).to(dtype=torch.float32)
        result[idx] = resize_tf(frame)

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(_resize_frame, range(n_frames))

    size_info = {
        "original_h": h, "original_w": w,
        "crop_h": sample_size[0], "crop_w": sample_size[1],
    }
    return result, size_info


def normalize_wan_latents(vae, latents):
    """Apply Wan VAE normalization consistent with MOVA training."""
    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    inv_std = (1.0 / torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype)).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    return (latents - mean) * inv_std


# ==================== VAE Encode ====================

@torch.no_grad()
def encode_video(vae, video_images, sample_size=960, patch_size=16):
    """Encode video frames to normalized Wan VAE latents.

    Args:
        vae: AutoencoderKLWan model
        video_images: numpy array (T, H, W, C) uint8
        sample_size: bucket base size (960=720p, 1440=1080p, 1920=2k)
        patch_size: patch size for bucket generation

    Returns:
        latent_np: numpy array, normalized latent
        latent_shape: list of ints
        size_info: dict with crop/resize info
    """
    buckets = generate_crop_size_list(base_size=sample_size, patch_size=patch_size, max_ratio=4.0)
    aspect_ratios = np.array([float(w) / float(h) for w, h in buckets])

    use_cpu = False
    try:
        sample = _fast_parallel_stack(video_images, device=vae.device)
        sample, size_info = _process_on_gpu(sample, aspect_ratios=aspect_ratios, buckets=buckets)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            print(f"  [WARN] GPU OOM during resize, falling back to CPU")
            use_cpu = True
        else:
            raise

    if use_cpu:
        sample, size_info = _process_on_cpu(video_images, aspect_ratios=aspect_ratios, buckets=buckets)

    pixel_values = sample.to(device=vae.device, dtype=vae.dtype)
    del sample
    pixel_values.mul_(1 / 127.5).sub_(1.0)
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(0)
    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")

    z = vae.encode(pixel_values).latent_dist.mode()
    del pixel_values
    z = normalize_wan_latents(vae, z)

    latent_np = z.detach().cpu().to(torch.float16).numpy()
    latent_shape = list(latent_np.shape)

    return latent_np, latent_shape, size_info


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Extract Wan VAE latent from a single .mp4 video")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the input .mp4 video file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the output .npy file")
    parser.add_argument("--vae_path", type=str,
                        default="/path/Prism/checkpoints/pretrained_models/pretrained_models/MOVA-360p/video_vae",
                        help="Path to Wan VAE model directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_size", type=int, default=960,
                        help="Latent extraction resolution: 720p=960, 1080p=1440, 2k=1920 (default: 960)")
    parser.add_argument("--sample_n_frames", type=int, default=289,
                        help="Max frames to extract (default: 289, ~12s at 24fps)")
    parser.add_argument("--fps", type=int, default=24,
                        help="Target FPS for frame extraction (default: 24)")
    parser.add_argument("--enable_tiling", action="store_true", default=False,
                        help="Enable VAE tiling to reduce VRAM for high-res videos")
    parser.add_argument("--tile_sample_min_size", type=int, default=256,
                        help="Tile size in pixels (default: 256)")
    parser.add_argument("--tile_sample_stride", type=int, default=192,
                        help="Stride between tiles (default: 192)")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    os.makedirs(args.output_dir, exist_ok=True)

    size_tag = SIZE_TAG_MAP.get(args.sample_size, f"{args.sample_size}p")
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_path = os.path.join(args.output_dir, f"{video_name}.npy")

    print(f"[Config] video: {args.video}")
    print(f"[Config] output: {output_path}")
    print(f"[Config] sample_size: {args.sample_size} ({size_tag})")
    print(f"[Config] device: {args.device}")

    # Load VAE
    print(f"[VAE] Loading AutoencoderKLWan from {args.vae_path} ...")
    vae = AutoencoderKLWan.from_pretrained(args.vae_path, torch_dtype=torch.float16).to(args.device)
    vae.eval()

    if args.enable_tiling:
        vae.enable_tiling(
            tile_sample_min_size=args.tile_sample_min_size,
            tile_sample_stride_size=args.tile_sample_stride,
        )
        print(f"[VAE] Tiling enabled: tile_size={args.tile_sample_min_size}, stride={args.tile_sample_stride}")

    # Extract frames
    print(f"[Frames] Extracting frames (max {args.sample_n_frames}, {args.fps}fps) ...")
    video_images, frame_info = extract_frames(
        args.video,
        sample_n_frames=args.sample_n_frames,
        required_fps=args.fps,
    )
    print(f"[Frames] {frame_info['num_frames']} frames extracted "
          f"({frame_info['height']}x{frame_info['width']}, "
          f"duration={frame_info['duration']:.1f}s)")

    # Encode
    print(f"[Encode] Running Wan VAE ...")
    latent_np, latent_shape, size_info = encode_video(
        vae, video_images,
        sample_size=args.sample_size,
    )
    print(f"[Encode] Latent shape: {latent_shape} "
          f"(crop: {size_info['crop_h']}x{size_info['crop_w']})")

    # Save
    np.save(output_path, latent_np)
    print(f"[Done] Saved to {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()


# 720p latent
# python hymm/wan_vae_latent_extraction.py \
#     --video /path/to/input.mp4 \
#     --output_dir /path/to/output_folder \
#     --vae_path /path/to/MOVA-720p/video_vae \
#     --sample_size 960
#
# 1080p
# python hymm/wan_vae_latent_extraction.py \
#     --video /path/to/input.mp4 \
#     --output_dir /path/to/output_folder \
#     --sample_size 1440
#
# 2K（--enable_tiling）
# python hymm/wan_vae_latent_extraction.py \
#     --video /path/to/input.mp4 \
#     --output_dir /path/to/output_folder \
#     --sample_size 1920 \
#     --enable_tiling