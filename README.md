# Prism

<a href='https://francis-rings.github.io/Prism'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2610.05416'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://huggingface.co/FrancisRing/Prism/tree/main'><img src='https://img.shields.io/badge/HuggingFace-Model-orange'></a> <a href='https://youtu.be/XEL2IkSOSQw'><img src='https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube'></a> <a href='https://www.bilibili.com/video/BV18dHZ6iEM8'><img src='https://img.shields.io/badge/Bilibili-Watch-blue?style=flat-square&logo=bilibili'></a> 

Prism:Dynamic Sparse Attention for Native 2K Joint Video-Audio Generation Model Training
<br/>
[Shuyuan Tu](https://github.com/Francis-Rings)<sup>1</sup>, [Qi Tian](https://scholar.google.com/citations?user=Ypu45nIAAAAJ&hl=zh-CN)<sup>2</sup>, [Yinming Huang](https://yinminghuang.github.io/)<sup>1</sup>, [Yue Wu](https://scholar.google.com/citations?user=1xTR6qoAAAAJ&hl=en)<sup>2</sup>, [Xintong Han](https://scholar.google.com/citations?user=FGiWOIAAAAAJ&hl=en)<sup>2</sup>, [Kaihang Pan](https://scholar.google.com/citations?user=lMQADDUAAAAJ&hl=zh-CN)<sup>3</sup>, [Weijie Kong](https://scholar.google.com/citations?user=gsOklKAAAAAJ&hl=zh-CN)<sup>2</sup>, [Jiangfeng Xiong](https://scholar.google.com/citations?user=lHbXg_0AAAAJ&hl=zh-TW)<sup>2</sup>, [Jian-Wei Zhang](https://scholar.google.com/citations?user=nF_klRIAAAAJ&hl=zh-CN)<sup>2</sup>, [Zuxuan Wu](https://scholar.google.com/citations?user=7t12hVkAAAAJ&hl=en)<sup>1</sup>, [Yu-Gang Jiang](https://scholar.google.com/citations?user=f3_FP8AAAAAJ&hl=en)<sup>1</sup>
<br/>
[<sup>1</sup>Fudan University; <sup>2</sup>Tencent Hunyuan; <sup>3</sup>Zhejiang University]


<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/0f2c8471-bd25-4151-82a1-42ae631b325a" width="320" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/39e384ee-b421-447e-aaa5-2db2b4f5c4c2" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/965b05cb-83a9-48b1-8ca6-783a38876db6" width="320" controls loop></video>
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/d58afef2-2f45-40b7-8326-5e64cc6b53fb" width="320" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/39a57fdc-499a-46e5-bb00-1628d6d8780f" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/16443caa-490f-4cf7-8115-e380ab57c106" width="320" controls loop></video>
     </td>
  </tr>
</table>


## Overview

<p align="center">
  <img src="assets/figures/framework.jpg" alt="model architecture" width="1280"/>
  </br>
  <i>The overview of the framework of Prism.</i>
</p>

Native training joint video-audio generation models at higher resolutions empowers them to learn richer visual details and sharper motion dynamics.
However, full attention incurs quadratic cost and, as resolution increases, spreads attention over increasingly redundant tokens, diluting learning signals for informative content and disrupting pretrained priors.
Existing sparse attention methods either target training-free acceleration or overlook the unique structure of joint video-audio data, where cross-modal interactions are inherently concentrated around sound-producing regions. 
To address this, we propose Prism, a dynamic sparse attention framework for natively training joint video-audio generation models at 2K. 
In particular, Prism organizes the token sequence into spatiotemporal macro-zones, enabling the attention structure to adapt to local content. 
For each zone, it estimates local information structure via video feature variance along the channel and feature norms from the audio-to-video cross-attention, jointly capturing how visual content varies directionally and how strongly audio influences each visual region. 
Based on these signals, Prism dynamically assigns a tailored block shape to each zone, applying finer partitioning along axes of rapid visual content variation and strong audio-visual coupling.
This encourages tokens within each block to remain semantically coherent, allowing block-level features to capture both visual content and joint video-audio interaction patterns.
Prism further adopts a hybrid block selection strategy to dynamically determine per-query sparsity. 
Experiments show that Prism achieves 2.5$\times$ training speedup compared to full attention, while surpassing it in generation quality.

## News
* `[2026-x-xx]`:🔥 The project page, training/inference code, technical report and [a preview model checkpoint](https://huggingface.co/FrancisRing/Prism/tree/main) are released. Stay tuned!

## 🛠️ To-Do List
- [x] Prism-preview-alpha (stable)
- [x] Prism-preview-beta (motion)
- [x] Data Pre-Processing Code (Latent Extraction and Latent Decode)
- [x] Training Code
- [x] Full Finetuning Code
- [x] Inference Code
- [ ] Prism-pro

## 🔑 Quickstart

The Preview Model supports native joint video-audio generation at 720p, 1080p, and 2K resolutions, as well as native joint video-audio training at these resolutions.

### 🧱 Environment Setup

**Prerequisites.** Prism requires Python ≥ 3.10, CUDA ≥ 12.4, and a system-level `ffmpeg` installation.
```
# 1 — Install PyTorch (CUDA 12.4)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu124
# 2 — Install core dependencies
pip install -r requirements.txt
# 3 — Install FlashAttention (requires CUDA)
pip install flash-attn --no-build-isolation
# 4 — Install `ffmpeg` (system-level)
sudo apt-get install ffmpeg
# 5 — Verify the installation
python -c "
import torch, diffusers, transformers, flash_attn, triton
from audiotools import AudioSignal
from decord import VideoReader
print(f'torch          {torch.__version__}')
print(f'diffusers      {diffusers.__version__}')
print(f'transformers   {transformers.__version__}')
print(f'flash_attn     {flash_attn.__version__}')
print(f'triton         {triton.__version__}')
print('All imports OK ✅')"
```
**(Multi-node only) Quick cluster-wide install via `pdsh`**

If you are running on a multi-node cluster, the following one-liner installs the key
packages across all nodes in parallel:
```
export node_ip=$(echo ${NODE_IP_LIST} | sed 's/:8//g')
pdsh -f 256 -w $node_ip "pip install ipdb yunchang mmengine"
pdsh -f 256 -w $node_ip "pip uninstall diffusers -y && pip install diffusers==0.33.0"
pdsh -f 256 -w $node_ip "pip install descript-audiotools"
pdsh -f 256 -w $node_ip "pip install imageio[ffmpeg] imageio[pyav]"
pdsh -f 256 -w $node_ip "pip install decord"
pdsh -f 256 -w $node_ip "pip install flash-attn --no-build-isolation"
```

### 🧱 Download Weights

If you encounter connection issues with Hugging Face, you can utilize the mirror endpoint by setting the environment variable: `export HF_ENDPOINT=https://hf-mirror.com`.
Please download weights manually as follows:
```
pip install "huggingface_hub[cli]"
cd Prism
mkdir checkpoints
huggingface-cli download FrancisRing/Prism --local-dir ./checkpoints
```
All the weights should be organized as follows.
The overall file structure of this project should be organized as follows:
```
Prism/
├── configs/
├── hymm/
├── scripts/
├── tools/
├── checkpoints/
│   ├── pretrained_models/
│   │   └── MOVA-360p/          # base MOVA pretrained model
│   ├── preview_alpha/          # Prism-preview-alpha weights
│   │   └── diffusion_pytorch_model.safetensors
│   └── preview_beta/           # Prism-preview-beta weights
│       └── diffusion_pytorch_model.safetensors
├── mova_infer.sh
├── mova_infer_fsdp.sh
├── requirements.txt
└── README.md
```

### 🧱 Training Dataset Preparation

Training reads a plain CSV file (one row per clip). The row details are depicted as follows:

| column | required | meaning |
| --- | --- | --- |
| `latent_path` | yes | cached WAN VAE latent `.npy`, shape `[C, T, H, W]` |
| `video_path` | yes | source video, read only for the reference frame |
| `audio_path` | yes | source audio (`.m4a`, `.wav`, …) |
| `caption` | yes | structured JSON caption, or plain text |
| `num_frames` | yes† | total frame count of the source video (pixel-space) |
| `height` | yes† | video height in pixels |
| `width` | yes† | video width in pixels |
| `fps` | yes† | source frame rate (used for audio-video alignment) |
| `video_id` | no | identifier used in logs |
| `audio_caption` | no | separate audio prompt; defaults to `caption` |
| `latent_frames`, `latent_height`, `latent_width` | no | cached latent dims; when present, used for bucketing instead of `num_frames/height/width` |
| `best_frame_index` | no | frame offset where the audio crop starts (default `0`) |
| `ref_frame_index` | no | frame used as the i2v reference (default `1`) |

> † When `latent_frames/latent_height/latent_width` are present in the CSV, they take priority for bucketing and `num_frames/height/width` become optional. `fps` defaults to `dataloader_config.video_fps` (24) when absent. We recommend always including all four columns (`num_frames`, `height`, `width`, `fps`) for clarity.

`caption` keeps the structured format, so the existing sampling ratios in
`video_caption_sample_ratio` still control which fields get assembled into the
prompt:

```json
{"style_features": "...", "content_summary": "...", "background_audio": "...",
 "shots": [{"time_range": ["0", "9.0"], "static_description": "...", "dynamic_description": "..."}]}
```

**Step 1 — Extract WAN VAE latents** (offline, once per video):

```
python hymm/wan_vae_latent_extraction.py \
    --input_csv /path/i2va_raw.csv \
    --output_dir /path/latents \
    --model_path /path/checkpoints/pretrained_models/MOVA-360p/video_vae \
    --sample_n_frames 289 --fps 24
```

**Step 2 — Assign multi-resolution and multi-duration buckets** (offline):

The temporal bucket range **must match the training config**.
The three resolution configs use:

| config | `latent-resolution` | `temporal-max-length` |
| --- | --- | --- |
| 720p | `720p` | `289` |
| 1080p | `1080p` | `289` |
| 2k | `2k` | `265` |

```bash
# 720p
python tools/build_dataset_csv.py \
    --input  /path/i2va_raw.csv \
    --output /path/i2va_720p.csv \
    --latent-resolution 720p \
    --temporal-min-length 49 --temporal-max-length 289 --temporal-interval 12 \
    --scan-latents --workers 32

# 1080p
python tools/build_dataset_csv.py \
    --input  /path/i2va_raw.csv \
    --output /path/i2va_1080p.csv \
    --latent-resolution 1080p \
    --temporal-min-length 49 --temporal-max-length 289 --temporal-interval 12 \
    --scan-latents --workers 32

# 2k (note: temporal-max-length is 265)
python tools/build_dataset_csv.py \
    --input  /path/i2va_raw.csv \
    --output /path/i2va_2k.csv \
    --latent-resolution 2k \
    --temporal-min-length 49 --temporal-max-length 265 --temporal-interval 12 \
    --scan-latents --workers 32
```

`--scan-latents` fills `latent_frames/height/width` from the `.npy` headers, and
the tool prints the per-bucket histogram so you can see how batches will fill
before launching.
The bucket columns are optional. The dataset derives them on the fly when they are
missing, using the same `bucket_*` settings from the config.

### 🧱 Preview Model Inference

Prism provides two inference scripts: `prism_infer.sh` for single-GPU or DeepSpeed-based multi-GPU inference, and `prism_infer_fsdp.sh` for FSDP-based multi-GPU inference that shards model parameters across GPUs (recommended for 1080p and 2K).

```bash
# Single-GPU / DeepSpeed multi-GPU inference (720p recommended)
bash prism_infer.sh

# FSDP multi-GPU inference (1080p / 2K recommended)
bash prism_infer_fsdp.sh
```

Before running, edit the shell script to set your own paths and generation settings. The core parameters are:

| Parameter | Default                                     | Description |
| --- |---------------------------------------------| --- |
| `--ckpt` | —                                           | Path to the pretrained MOVA base model (e.g. `checkpoints/pretrained_models/MOVA-360p`) |
| `--resume_ckpt` | —                                           | Path to the Prism checkpoint (e.g. `checkpoints/preview_alpha/diffusion_pytorch_model.safetensors`) |
| `--config` | `configs/train/t2va_config/mova_infer.yaml` | Inference config that sets diffusion steps, default resolution, and seed |
| `--height` / `--width` | 480 / 848                                   | Output resolution. Recommended: 720p = `720 × 1280`, 1080p = `1072 × 1920`, 2K = `1440 × 2560` |
| `--num_frames` | `205`                                       | Number of frames to generate; must satisfy `(n − 1) % 4 == 0`, auto-snapped if not |
| `--prompt` / `--audio_prompt` | —                                           | Text prompt for video / audio generation; supports structured tags such as `<music>`, `<sfx>`, `<speech>` |
| `--ref_path` | —                                           | Path to the reference image for image-to-video generation |
| `--output_path` | —                                           | Path to save the generated `.mp4` video |
| `--sp_size` | `4` or `8`                                  | Sequence-parallelism degree (number of GPUs sharing one sample's token sequence) |
| `--offload` | `cpu`                                       | Offload frozen encoders to CPU to save GPU memory; omit for full GPU-resident inference |
| `--seed` | `42`                                        | Random seed for reproducibility |
| `VISUAL_SHIFT` / `AUDIO_SHIFT` | `9.0` / `7.0`                               | Noise schedule shift for video / audio denoising; higher values allocate more steps to high-noise stages. Recommended: 720p `7.0`, 1080p `9.0–13.0`, 2K `13.0–17.0` |
| `CFG_SCALE` | `5.0`                                       | Classifier-free guidance scale; `1.0` = no guidance, higher = stronger prompt adherence |
| `ENABLE_TILING` | `false`                                     | Enable spatial tiling for VAE decode to avoid OOM at high resolutions (e.g. 2K); `TILE_SAMPLE_MIN_SIZE` and `TILE_SAMPLE_STRIDE` control tile size and overlap |
| `ENABLE_BSA` / `BSA_SPARSITY` | `true` / `0.93` or `0.85` or `0.75`         | Enable Prism Block Sparse Attention and set the top-k sparsity ratio during inference |
| `ENABLE_IVPQ_DYNAMIC_BLOCK` | `true`                                      | Enable dynamic block-shape assignment for inference |

> **Tip:** `mova_infer.sh` uses `--deepspeed` launch and is suited for 720p on a single 80 GB GPU (with `--offload cpu`). `mova_infer_fsdp.sh` uses `torchrun` with FSDP `FULL_SHARD` and is recommended for 1080p / 2K on 4+ GPUs, as it shards parameters across all GPUs rather than replicating the full model.

**❤️ Notably, for your convenience, we provide several ready-to-infer cases in the `assets/ti2va_cases` folder. Feel free to try them out and have fun! ❤️**

### 🧱 Native Joint Video-Audio Training at High-Resolution

Prism supports native joint video-audio training at 720p, 1080p, and 2K resolutions. The training pipeline uses FSDP and sequence parallelism via `torchrun` + `pdsh`, controlled by a two-level config system: a top-level YAML (`configs/train/wan_ti2va.yaml`) that holds per-resolution experiment entries (optimizer, block sparse attention flags, etc.) and a per-resolution inject-config YAML (`configs/train/t2va_config/wan_15B_ti2va_*_init_top_k_p.yaml`) that specifies the pretrained model path, loss weights, and all dataloader settings.

Regarding the bucketed CSV files from step2 in 🧱Training Dataset Preparation, they are the sole data manifest consumed by the trainer. 
Each inject-config YAML has a `dataloader_config.video_csv_file` field. This is where you place your CSV path. 
For example, the 720p inject config (`configs/train/t2va_config/wan_15B_ti2va_720p_init_top_k_p.yaml`) contains:
```yaml
dataloader_config:
  video_csv_file:
  - /path/to/your/i2va_720p.csv      # ← put your bucketed CSV path here
```

Replace the path with the actual location of the CSV produced by `tools/build_dataset_csv.py`. The 1080p and 2K inject configs follow the same pattern — update `video_csv_file` in `configs/train/t2va_config/wan_15B_ti2va_1080p_init_top_k_p.yaml` and `configs/train/t2va_config/wan_15B_ti2va_2k_init_top_k_p.yaml` respectively. You may also specify multiple CSV files as a list under `video_csv_file` to merge several datasets. At training launch, the script reads the `inject_config` path from the top-level YAML, loads the inject-config, and passes `video_csv_file` to `CsvVideoAudioDataset`. The dataset reads every row, derives or looks up the `(bucket_frames, bucket_height, bucket_width)` assignment, and groups all samples by bucket key. A `DistributedBucketBatchSampler` then draws each mini-batch from a single bucket so every sample in the batch shares the exact same tensor shape, avoiding shape mismatches across data-parallel ranks. The multi-resolution spatial buckets are generated by enumerating all `(H, W)` crops whose patch count equals `(base_size / patch_size)²`, and each sample is matched to the bucket with the closest aspect ratio; the multi-duration temporal buckets are a series of frame counts of the form `4k + 1` from `temporal_min_length` to `temporal_max_length` at stride `temporal_interval`, and each sample is assigned to the largest temporal bucket that fits.

**Launch training** at each resolution with:

```bash
# 720p native joint video-audio training
bash scripts/pretrain/pdsh_train.sh "scripts/pretrain/train_wan_ti2va.sh configs/train/wan_ti2va.yaml wan_15B_ti2va_720p_init_top_k_p"

# 1080p native joint video-audio training
bash scripts/pretrain/pdsh_train.sh "scripts/pretrain/train_wan_ti2va.sh configs/train/wan_ti2va.yaml wan_15B_ti2va_1080p_init_top_k_p"

# 2K native joint video-audio training
bash scripts/pretrain/pdsh_train.sh "scripts/pretrain/train_wan_ti2va.sh configs/train/wan_ti2va.yaml wan_15B_ti2va_2k_init_top_k_p"
```

`pdsh_train.sh` is the inner script across all nodes listed in `$NODE_IP_LIST` via `pdsh`. 
On each node, `train_wan_ti2va.sh` parses the named config from the top-level YAML, extracts every flag, then launches `torchrun` with `hymm/train_t2va_wan.py`.

**Core training parameters.** The training configuration is split across two YAML files. Parameters in the **top-level config** (`configs/train/wan_ti2va.yaml`) control the optimizer, parallelism, checkpointing, and Prism sparse-attention behaviour. 
Parameters in the **inject config** (`configs/train/t2va_config/wan_15B_ti2va_*_init_top_k_p.yaml`) control the pretrained model, loss weighting, noise schedule, and dataloader.

**Top-level config** (`configs/train/wan_ti2va.yaml`):

| Parameter | Default                | Description |
| --- |------------------------| --- |
| `sp_size` | 8 (1080p/2K), 4 (720p) | Sequence-parallelism degree — number of GPUs that jointly shard a single sample's token sequence to reduce per-GPU memory |
| `lr` | `1e-5`                 | Peak learning rate |
| `lr_scheduler` | `constant_with_warmup` | LR schedule shape; ramps linearly during warmup then holds constant |
| `lr_warmup_steps` | `200`                  | Number of steps for the linear LR warmup |
| `optimizer` | `adamw`                | Optimizer type |
| `weight_decay` | `0.01`                 | L2 regularization weight |
| `max_grad_norm` | `1.0`                  | Maximum gradient norm for gradient clipping to stabilize training |
| `gradient_accumulation_steps` | `1`                    | Number of forward-backward micro-steps accumulated before one optimizer update; effectively multiplies the global batch size without extra memory |
| `fsdp_strategy` | `full`                 | FSDP sharding strategy — `full` shards parameters, gradients, and optimizer states across all ranks for maximum memory savings; `hybrid` (HSDP) shards within each node and replicates across nodes, reducing cross-node communication at the cost of higher per-node memory; `none` replicates the full model on every rank |
| `checkpointing_steps` | `50`                   | Save a checkpoint every N training steps |
| `train_full_model` | `true`                 | Whether all model parameters are trainable; `false` = attention-only |
| `offload_frozen_encoders` | `true`                 | Move frozen text encoder / audio VAE / video VAE back to CPU after each forward pass to free GPU memory; `false` keeps them on GPU |
| `empty_cache_interval` | `1`                    | Call `torch.cuda.empty_cache()` every N steps to release the CUDA allocator pool; `1` is safest but adds a device sync each step |
| `loss_spike_threshold` | `1.5`                  | Skip the optimizer step if loss exceeds this value, preventing catastrophic weight updates from noisy batches |
| `grad_norm_spike_threshold` | `1e6`                  | Skip the optimizer step if gradient norm exceeds this value |
| `enable_bsa` | `true`                 | Activate Prism Block Sparse Attention for video self-attention |
| `bsa_sparsity` | `0.75/0.80/0.85/0.93`  | Top-k sparsity ratio — each query block attends to only 25% of key blocks |
| `bsa_chunk_3d_shape_q` / `bsa_chunk_3d_shape_k` | `4 4 4`                | The 3-D macro-zone shape `(T, H, W)` that tiles the spatiotemporal volume for query / key blocking |
| `bsa_cdf_threshold` | `0.20`                 | Top-p CDF threshold for hybrid block selection; combined with top-k for adaptive per-query sparsity |
| `enable_ivpq_dynamic_block` | `true`                 | Enable the anisotropic dynamic block-shape assignment (Prism core), where each macro-zone gets a tailored block shape based on local content variance and audio-visual coupling |

**Inject config** (`configs/train/t2va_config/wan_15B_ti2va_*_init_top_k_p.yaml`):

| Parameter | Default | Description |
| --- | --- | --- |
| `pretrained_model_name_or_path` | — | Path to the pretrained MOVA base model weights |
| `video_loss_weight` / `audio_loss_weight` | `0.75` / `0.25` | Balance between the video and audio reconstruction losses |
| `visual_shift` / `audio_shift` | `9.0` / `5.0` | Per-tower noise schedule shift in the flow-matching formulation |
| `video_csv_file` | — | Path(s) to the bucketed CSV manifest produced by `tools/build_dataset_csv.py` |
| `video_size` | `[960,960]` / `[1440,1440]` / `[1920,1920]` | Spatial resolution for 720p / 1080p / 2K |
| `sample_n_frames` | `289` (720p/1080p), `265` (2K) | Maximum number of pixel-space frames per training clip |
| `video_micro_batch_size` | `1` | Per-GPU batch size within a data-parallel rank |
| `video_uncond_p` / `audio_uncond_p` | `0.1` / `0.2` | Classifier-free guidance dropout probability for video / audio captions |
| `latent_resolution` | `720p` / `1080p` / `2k` | Shorthand that auto-selects the spatial bucket base size |
| `video_bucket_temporal_min_length` | `49` | Shortest temporal bucket (frames) |
| `video_bucket_temporal_max_length` | `289` (720p/1080p), `265` (2K) | Longest temporal bucket (frames) |
| `video_bucket_temporal_interval` | `12` | Stride between successive temporal buckets |


### 🧱 VRAM Requirement

For native high-resolution training, 720p requires at least 32 × NVIDIA 80 GB GPUs (e.g. A100 or H100), and 1080p / 2K each require at least 64 × NVIDIA 80 GB GPUs. For inference, 720p can run on a single NVIDIA 80 GB GPU, while 1080p and 2K each require 4 or more NVIDIA 80 GB GPUs.

## ⭐ Contact

If you have any suggestions or find our work helpful, feel free to contact me.

Email: francisshuyuan@gmail.com

If you find our work useful, <b>please consider giving a star ⭐ to this github repository and citing it ❤️</b>:
```bib
@article{tu2026prism,
  title={Prism:Dynamic Sparse Attention for Native 2K Joint Video-Audio Generation Model Training},
  author={Tu, Shuyuan and Tian, Qi and Huang, Yinming and Wu, Yue and Han, Xintong and Pan, Kaihang and Kong, Weijie and Xiong, Jiangfeng and Zhang, Jian-Wei and Wu, Zuxuan and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2610.05416},
  year={2026}
}
```
