import math
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin


class FlowMatchSchedulerBase:
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.train_timesteps = None
        self.train_sigmas = None
        self.set_timesteps(num_train_timesteps)
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None, dynamic_shift_len=None, device=None):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = self.calculate_shift(dynamic_shift_len) if dynamic_shift_len is not None else self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if self.train_timesteps is None:
            self.train_timesteps = self.timesteps
            self.train_sigmas = self.sigmas
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep):
        return noise - sample

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu


class FlowMatchPairScheduler(FlowMatchSchedulerBase, SchedulerMixin, ConfigMixin):
    """
    Paired timestep scheduler for dual-tower (video + audio) diffusion.
    Returns tensor with shape [num_timesteps, 2], each row is (visual_t, audio_t).
    """

    @register_to_config
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self._pair_postprocess_fn = None
        self._pair_postprocess_requires_source = False
        self.pair_timesteps = None
        self.pair_sigmas = None
        self.timesteps = None
        self.sigmas = None
        super().__init__(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            inverse_timesteps=inverse_timesteps,
            extra_one_step=extra_one_step,
            reverse_sigmas=reverse_sigmas,
            exponential_shift=exponential_shift,
            exponential_shift_mu=exponential_shift_mu,
            shift_terminal=shift_terminal,
        )

    def set_pair_postprocess(self, fn):
        if fn is not None and not callable(fn):
            raise TypeError("pair_postprocess must be a callable object or None")
        self._pair_postprocess_fn = fn
        self._pair_postprocess_requires_source = False if fn is None else bool(getattr(fn, "_requires_source", False))
        if self.timesteps is None or self.sigmas is None:
            raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")
        self._refresh_pair_cache()

    def set_pair_postprocess_by_name(self, name, **kwargs):
        if name is None or str(name).lower() in ("none", "off", "false", "no"):
            self.set_pair_postprocess(None)
            return
        if name == "dual_sigma_shift":
            visual_shift = float(kwargs.get("visual_shift", self.shift))
            audio_shift = float(kwargs.get("audio_shift", self.shift))
            visual_denoising_strength = float(kwargs.get("visual_denoising_strength", 1.0))
            audio_denoising_strength = float(kwargs.get("audio_denoising_strength", 1.0))
            visual_mu = kwargs.get("visual_exponential_shift_mu", self.exponential_shift_mu)
            audio_mu = kwargs.get("audio_exponential_shift_mu", self.exponential_shift_mu)

            def _dual_sigma_shift(pairs, *, source):
                num_steps = pairs.shape[0]
                device = pairs.device
                dtype = pairs.dtype

                def _build_column(shift_value, denoising_strength, mu_override):
                    sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
                    if self.extra_one_step:
                        base = torch.linspace(sigma_start, self.sigma_min, num_steps + 1, device=device, dtype=dtype)[:-1]
                    else:
                        base = torch.linspace(sigma_start, self.sigma_min, num_steps, device=device, dtype=dtype)
                    if self.inverse_timesteps:
                        base = torch.flip(base, dims=[0])
                    if self.exponential_shift:
                        mu_value = mu_override
                        if mu_value is None:
                            raise RuntimeError("exponential_shift is enabled but exponential_shift_mu is not provided")
                        exp_mu = math.exp(float(mu_value))
                        base = exp_mu / (exp_mu + (1 / base - 1))
                    else:
                        base = shift_value * base / (1 + (shift_value - 1) * base)
                    if self.shift_terminal is not None:
                        one_minus_z = 1 - base
                        scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
                        base = 1 - (one_minus_z / scale_factor)
                    if self.reverse_sigmas:
                        base = 1 - base
                    if source == "timesteps":
                        return base * self.num_train_timesteps
                    return base

                visual_column = _build_column(visual_shift, visual_denoising_strength, visual_mu)
                audio_column = _build_column(audio_shift, audio_denoising_strength, audio_mu)
                return torch.stack([visual_column, audio_column], dim=1)

            _dual_sigma_shift._requires_source = True
            self.set_pair_postprocess(_dual_sigma_shift)
            return
        raise ValueError(f"Unsupported pair postprocessing name: {name}")

    def _make_pairs_from_vector(self, vec):
        return torch.stack([vec, vec], dim=1)

    def get_pairs(self, source="timesteps"):
        if source == "timesteps":
            pairs = self.pair_timesteps
        elif source == "sigmas":
            pairs = self.pair_sigmas
        else:
            raise ValueError("source only supports 'timesteps' or 'sigmas'")
        if pairs is None:
            raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")
        return pairs

    @property
    def visual_timesteps(self):
        if self.pair_timesteps is None:
            raise RuntimeError("Scheduler not initialized")
        return self.pair_timesteps[:, 0]

    @property
    def audio_timesteps(self):
        if self.pair_timesteps is None:
            raise RuntimeError("Scheduler not initialized")
        return self.pair_timesteps[:, 1]

    def set_timesteps(self, *args, **kwargs):
        super().set_timesteps(*args, **kwargs)
        self._refresh_pair_cache()

    def timestep_to_sigma(self, timestep):
        t_value = float(timestep)
        t_cpu = torch.tensor(t_value)
        idx = torch.argmin((self.train_timesteps - t_cpu).abs())
        return self.train_sigmas[idx]

    def step_from_to(self, model_output, timestep_from, timestep_to, sample):
        sigma_from = self.timestep_to_sigma(timestep_from)
        if timestep_to is None:
            sigma_to = torch.tensor(
                1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0,
                device=sigma_from.device, dtype=sigma_from.dtype,
            )
        else:
            sigma_to = self.timestep_to_sigma(timestep_to)
        prev_sample = sample + model_output * (sigma_to - sigma_from)
        return prev_sample

    def _refresh_pair_cache(self):
        if self.timesteps is None or self.sigmas is None:
            return

        def _apply_postprocess(pairs, source):
            if self._pair_postprocess_fn is None:
                return pairs
            if self._pair_postprocess_requires_source:
                modified = self._pair_postprocess_fn(pairs, source=source)
            else:
                modified = self._pair_postprocess_fn(pairs)
            return modified

        base_pairs_timesteps = self._make_pairs_from_vector(self.timesteps)
        base_pairs_sigmas = self._make_pairs_from_vector(self.sigmas)

        self.pair_timesteps = _apply_postprocess(base_pairs_timesteps, "timesteps")
        self.pair_sigmas = _apply_postprocess(base_pairs_sigmas, "sigmas")
