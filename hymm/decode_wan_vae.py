import os
import argparse
import numpy as np
import torch
import imageio
from diffusers.models.autoencoders import AutoencoderKLWan


def denormalize_latents(vae, latents):
    """Reverse the normalization applied during encoding (consistent with MOVA denormalize_video_latents)."""
    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    return latents * std + mean


@torch.no_grad()
def decode_and_save(vae, npy_path, output_path, fps=24.0):
    latent = np.load(npy_path)
    latent = torch.from_numpy(latent).to(device=vae.device, dtype=vae.dtype)
    # latent shape: [1, C, T, H, W]
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)

    latent = denormalize_latents(vae, latent)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        video = vae.decode(latent).sample
    # video: [B, C, T, H, W], range [-1, 1]

    video = video[0].permute(1, 2, 3, 0).float().cpu().clamp(-1, 1)  # [T, H, W, C]
    video = ((video + 1.0) * 127.5).to(torch.uint8).numpy()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, quality=9)
    for frame in video:
        writer.append_data(frame)
    writer.close()

    print(f"Saved {video.shape[0]} frames ({video.shape[2]}x{video.shape[1]}) to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_file", type=str, required=True,
                        help="Path to the .npy latent file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: same name as npy with .mp4)")
    parser.add_argument("--vae_path", type=str,
                        default="/path/Prism/checkpoints/pretrained_models/pretrained_models/MOVA-360p/video_vae")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--enable_tiling", action="store_true", default=False,
                        help="Enable VAE tiling mode to reduce VRAM usage for high-res decoding.")
    parser.add_argument("--tile_sample_min_size", type=int, default=None,
                        help="Tile size (height & width) in pixels. (default: diffusers default 256)")
    parser.add_argument("--tile_sample_stride", type=int, default=None,
                        help="Stride between tiles. overlap = tile_size - stride. (default: diffusers default 192)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.npy_file)[0] + ".mp4"

    print(f"[Init] Loading Wan VAE from {args.vae_path} ...")
    vae = AutoencoderKLWan.from_pretrained(
        args.vae_path,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    vae.eval()

    if args.enable_tiling:
        tiling_kwargs = {}
        if args.tile_sample_min_size is not None:
            tiling_kwargs["tile_sample_min_height"] = args.tile_sample_min_size
            tiling_kwargs["tile_sample_min_width"] = args.tile_sample_min_size
        if args.tile_sample_stride is not None:
            tiling_kwargs["tile_sample_stride_height"] = args.tile_sample_stride
            tiling_kwargs["tile_sample_stride_width"] = args.tile_sample_stride
        vae.enable_tiling(**tiling_kwargs)
        tile_sz = args.tile_sample_min_size or 256
        stride = args.tile_sample_stride or 192
        print(f"[Init] VAE tiling mode ENABLED: tile_size={tile_sz}, stride={stride}, "
              f"overlap={tile_sz - stride}px ({(tile_sz - stride) / tile_sz * 100:.0f}%)")

    print(f"[Init] Wan VAE loaded on {args.device}")

    decode_and_save(vae, args.npy_file, args.output, fps=args.fps)


if __name__ == "__main__":
    main()

# Example:
# export node_ip=$(echo ${NODE_IP_LIST} | sed 's/:8//g')
# pdsh -f 256 -w $node_ip "pip install imageio[ffmpeg]"
# pdsh -f 256 -w $node_ip "pip install imageio[pyav]"
# CUDA_VISIBLE_DEVICES=0 python decode_wan_vae.py --npy_file /path/a.npy --output output.mp4
# ---------------------------- 2k latent decode -----------------------------------
# CUDA_VISIBLE_DEVICES=0 python decode_wan_vae.py --npy_file /path/b.npy --output output.mp4 --enable_tiling --tile_sample_min_size 256 --tile_sample_stride 192