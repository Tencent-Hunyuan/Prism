import copy
import gc
import html
import re
from typing import Any, List, Optional, Tuple, Union

import ftfy
import torch
import torch.distributed as dist
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.image_processor import PipelineImageInput
from tqdm import tqdm
from transformers import T5TokenizerFast, UMT5EncoderModel

from hymm.models.modules.mova import MOVABridge
from hymm.models.modules.dac_vae import DAC
from hymm.diffusion.schedulers.flow_match_pair import FlowMatchPairScheduler


def _basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def _whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _prompt_clean(text):
    return _whitespace_clean(_basic_clean(text))


def _retrieve_latents(encoder_output, generator=None, sample_mode="sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class MOVAPipeline:
    """
    Pipeline that drives MOVA inference identically to the original
    MOVA/mova/diffusion/pipelines/pipeline_mova.py.

    Differences from the original:
    - Does NOT inherit from DiffusionPipeline (no auto-offload magic).
    - Uses MOVABridge as the transformer, which already holds video_dit(s),
      audio_dit, and bridge.
    - Compatible with the codebase's DeepSpeed / FSDP launcher.
    """

    def __init__(
        self,
        transformer: MOVABridge,
        video_vae: AutoencoderKLWan,
        audio_vae: DAC,
        text_encoder: UMT5EncoderModel,
        tokenizer: T5TokenizerFast,
        scheduler: FlowMatchPairScheduler,
        boundary_ratio: float = 0.9,
        device=None,
    ):
        self.transformer = transformer
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.boundary_ratio = boundary_ratio

        self.vae_scale_factor_spatial = self.video_vae.config.scale_factor_spatial
        self.vae_scale_factor_temporal = self.video_vae.config.scale_factor_temporal
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        self.audio_vae_scale_factor = int(self.audio_vae.hop_length)
        self.audio_sample_rate = self.audio_vae.sample_rate

        self._device = device
        self._cpu_offload = False

    @property
    def device(self):
        if self._device is not None:
            return self._device
        for p in self.transformer.parameters():
            return p.device
        return torch.device("cpu")

    def to(self, device):
        self._device = device
        self.transformer.to(device)
        self.video_vae.to(device)
        self.audio_vae.to(device)
        self.text_encoder.to(device)
        return self

    def enable_cpu_offload(self, gpu_device=None):
        """Enable sequential CPU offload to reduce peak GPU memory.

        Only the active component lives on GPU at any time:
          1. text_encoder on GPU -> encode prompts -> move to CPU
          2. video_vae on GPU -> encode condition latents -> move to CPU
          3. transformer on GPU -> denoising loop -> move to CPU
          4. video_vae + audio_vae on GPU -> decode -> move to CPU
        """
        self._cpu_offload = True
        if gpu_device is not None:
            self._device = gpu_device

    def normalize_video_latents(self, latents):
        mean = torch.tensor(
            self.video_vae.config.latents_mean, device=latents.device, dtype=latents.dtype,
        ).view(1, self.video_vae.config.z_dim, 1, 1, 1)
        inv_std = (1.0 / torch.tensor(
            self.video_vae.config.latents_std, device=latents.device, dtype=latents.dtype,
        )).view(1, self.video_vae.config.z_dim, 1, 1, 1)
        return (latents - mean) * inv_std

    def denormalize_video_latents(self, latents):
        mean = torch.tensor(
            self.video_vae.config.latents_mean, device=latents.device, dtype=latents.dtype,
        ).view(1, self.video_vae.config.z_dim, 1, 1, 1)
        std = torch.tensor(
            self.video_vae.config.latents_std, device=latents.device, dtype=latents.dtype,
        ).view(1, self.video_vae.config.z_dim, 1, 1, 1)
        return latents * std + mean

    def check_inputs(self, height, width, num_frames):
        target_div = self.vae_scale_factor_spatial * 2
        if height % target_div != 0 or width % target_div != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {target_div}, "
                f"got {height} and {width}."
            )
        if num_frames % self.vae_scale_factor_temporal != 1:
            raise ValueError(
                f"`num_frames - 1` must be divisible by {self.vae_scale_factor_temporal}, "
                f"got {num_frames - 1}."
            )

    def prepare_latents(
        self,
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        num_frames,
        dtype,
        device,
        generator=None,
        latents=None,
        last_image=None,
    ):
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)

        if last_image is None:
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2,
            )
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
                dim=2,
            )
        video_condition = video_condition.to(device=device, dtype=self.video_vae.dtype)

        if isinstance(generator, list):
            latent_condition = [
                _retrieve_latents(self.video_vae.encode(video_condition), sample_mode="argmax")
                for _ in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = _retrieve_latents(
                self.video_vae.encode(video_condition), sample_mode="argmax",
            )
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = self.normalize_video_latents(latent_condition)

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)

    def prepare_audio_latents(
        self,
        audio,
        batch_size,
        num_channels,
        num_samples,
        dtype,
        device,
        generator=None,
        latents=None,
    ):
        latent_t = (num_samples - 1) // self.audio_vae_scale_factor + 1
        shape = (batch_size, num_channels, latent_t)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        return latents

    def _get_t5_prompt_embeds(
        self,
        prompt,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [_prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), mask.to(device),
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
            dim=0,
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        return prompt_embeds

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        image,
        audio_prompt=None,
        negative_prompt="",
        seed=42,
        height=360,
        width=640,
        num_frames=193,
        video_fps=24.0,
        num_inference_steps=50,
        visual_shift=5.0,
        audio_shift=5.0,
        cfg_scale=5.0,
        remove_video_dit=False,
        enable_vae_tiling=False,
        vae_tile_sample_min_size=None,
        vae_tile_sample_stride=None,
    ):
        """
        Run the full MOVA denoising loop.
        This replicates the exact logic from MOVA's __call__.

        VAE tiling (Phase 4 decode only, fully isolated / default OFF):
            enable_vae_tiling: when True, the video VAE decodes the latents tile by
                tile (spatial), drastically cutting peak decode VRAM so that high-res
                (e.g. 2K) videos no longer OOM. When False the decode path is byte-for-
                byte identical to the original (VAE tiling is never touched).
            vae_tile_sample_min_size: tile size (pixels, applied to both H and W).
                None -> diffusers/Wan default (256), same as decode_wan_vae.py.
            vae_tile_sample_stride: stride between tiles (pixels, H and W). overlap =
                tile_size - stride. None -> diffusers/Wan default (192), i.e. 25% overlap,
                same as decode_wan_vae.py.

        Note on FSDP + SP: only the transformer is FSDP-sharded; the video VAE is a
        full replica on every rank and the post-denoise latents are identical across
        all SP ranks. VAE tiling is deterministic, so every rank produces the SAME
        decoded video whether or not tiling is enabled.
        """
        self.check_inputs(height, width, num_frames)

        audio_num_samples = int(self.audio_sample_rate * num_frames / video_fps)
        device = self.device
        offload = self._cpu_offload

        # Deterministic noise generator (independent of global random state)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        # ------------------------------------------------------------------
        # Phase 1: Text encoding
        # ------------------------------------------------------------------
        if offload:
            self.text_encoder.to(device)

        self.scheduler.set_timesteps(num_inference_steps, shift=visual_shift, device=device)
        self.scheduler.set_pair_postprocess_by_name(
            "dual_sigma_shift",
            visual_shift=float(visual_shift),
            audio_shift=float(audio_shift),
        )
        paired_timesteps = self.scheduler.get_pairs()

        prompt_embeds = self._get_t5_prompt_embeds(prompt)
        audio_prompt_embeds = self._get_t5_prompt_embeds(audio_prompt) if audio_prompt is not None else None
        negative_prompt_embeds = self._get_t5_prompt_embeds(negative_prompt)

        if offload:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Phase 2: VAE encode (prepare latents)
        # ------------------------------------------------------------------
        if offload:
            self.video_vae.to(device)

        num_channels_latents = self.video_vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)

        latents, condition = self.prepare_latents(
            image, 1, num_channels_latents, height, width, num_frames,
            torch.float32, device, generator=generator, latents=None, last_image=None,
        )
        audio_latents = self.prepare_audio_latents(
            None, 1, self.audio_vae.latent_dim, audio_num_samples,
            torch.float32, device, generator=generator, latents=None,
        )

        if offload:
            self.video_vae.to("cpu")
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Phase 3: Denoising loop (transformer on GPU)
        # ------------------------------------------------------------------
        if offload:
            self.transformer.to(device)

        cur_use_dit_2 = False
        total_steps = paired_timesteps.shape[0]
        switched = False
        boundary_timestep = self.boundary_ratio * self.scheduler.config.num_train_timesteps

        is_main = not dist.is_initialized() or dist.get_rank() == 0

        for idx_step in tqdm(range(total_steps), disable=not is_main):
            timestep, audio_timestep = paired_timesteps[idx_step]

            if not switched and timestep.item() < boundary_timestep:
                cur_use_dit_2 = True
                if remove_video_dit:
                    self.transformer.video_dit = None
                    gc.collect()
                switched = True

            latent_model_input = torch.cat([latents, condition], dim=1)
            timestep_t = timestep.unsqueeze(0).to(device=device, dtype=torch.float32)
            audio_timestep_t = audio_timestep.unsqueeze(0).to(device=device, dtype=torch.float32)

            noise_pred_posi = self.transformer(
                visual_latents=latent_model_input,
                audio_latents=audio_latents,
                context=prompt_embeds,
                timestep=timestep_t,
                audio_context=audio_prompt_embeds,
                audio_timestep=audio_timestep_t,
                video_fps=video_fps,
                num_train_timesteps=self.scheduler.config.num_train_timesteps,
                use_video_dit_2=cur_use_dit_2,
            )

            if cfg_scale == 1.0:
                visual_noise_pred = noise_pred_posi[0].float()
                audio_noise_pred = noise_pred_posi[1].float()
            else:
                noise_pred_nega = self.transformer(
                    visual_latents=latent_model_input,
                    audio_latents=audio_latents,
                    context=negative_prompt_embeds,
                    timestep=timestep_t,
                    audio_context=None,
                    audio_timestep=audio_timestep_t,
                    video_fps=video_fps,
                    num_train_timesteps=self.scheduler.config.num_train_timesteps,
                    use_video_dit_2=cur_use_dit_2,
                )
                visual_noise_pred_nega = noise_pred_nega[0].float()
                audio_noise_pred_nega = noise_pred_nega[1].float()
                visual_noise_pred_posi = noise_pred_posi[0].float()
                audio_noise_pred_posi = noise_pred_posi[1].float()
                visual_noise_pred = visual_noise_pred_nega + cfg_scale * (visual_noise_pred_posi - visual_noise_pred_nega)
                audio_noise_pred = audio_noise_pred_nega + cfg_scale * (audio_noise_pred_posi - audio_noise_pred_nega)

            next_timestep = paired_timesteps[idx_step + 1, 0] if idx_step + 1 < total_steps else None
            next_audio_timestep = paired_timesteps[idx_step + 1, 1] if idx_step + 1 < total_steps else None
            latents = self.scheduler.step_from_to(
                visual_noise_pred, timestep, next_timestep, latents,
            )
            audio_latents = self.scheduler.step_from_to(
                audio_noise_pred, audio_timestep, next_audio_timestep, audio_latents,
            )

        if offload:
            self.transformer.to("cpu")
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Phase 4: VAE decode
        # ------------------------------------------------------------------
        if offload:
            self.video_vae.to(device)

        # Optional tiled VAE decode (default OFF -> original path untouched). When
        # enabled, decode the latents tile-by-tile spatially to cut peak decode VRAM
        # for high-res (e.g. 2K) videos. Mirrors decode_wan_vae.py: min_size/stride
        # map to the AutoencoderKLWan (height & width) tiling knobs; None keeps the
        # diffusers defaults (256 / 192).
        if enable_vae_tiling:
            tiling_kwargs = {}
            if vae_tile_sample_min_size is not None:
                tiling_kwargs["tile_sample_min_height"] = vae_tile_sample_min_size
                tiling_kwargs["tile_sample_min_width"] = vae_tile_sample_min_size
            if vae_tile_sample_stride is not None:
                tiling_kwargs["tile_sample_stride_height"] = vae_tile_sample_stride
                tiling_kwargs["tile_sample_stride_width"] = vae_tile_sample_stride
            self.video_vae.enable_tiling(**tiling_kwargs)

        video_latents = self.denormalize_video_latents(latents)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            video = self.video_vae.decode(video_latents).sample
        video = self.video_processor.postprocess_video(video, output_type="pil")

        if offload:
            self.video_vae.to("cpu")
            self.audio_vae.to(device)
            torch.cuda.empty_cache()

        with torch.autocast("cuda", dtype=torch.float32):
            audio = self.audio_vae.decode(audio_latents)

        if offload:
            self.audio_vae.to("cpu")
            torch.cuda.empty_cache()

        return video, audio
