from __future__ import annotations
import logging

from typing import List, Optional, Tuple, Union
import numpy as np
from numpy import ndarray
import torch
from torch import Generator, FloatTensor
from diffusers.schedulers.scheduling_euler_discrete import (
    EulerDiscreteScheduler as DiffusersEulerDiscreteScheduler,
    EulerDiscreteSchedulerOutput,
)
from diffusers.utils.torch_utils import randn_tensor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class EulerDiscreteScheduler(DiffusersEulerDiscreteScheduler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: ndarray | List[float] | None = None,
        prediction_type: str = "epsilon",
        interpolation_type: str = "linear",
        use_karras_sigmas: bool | None = False,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
    ):
        super().__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            trained_betas,
            prediction_type,
            interpolation_type,
            use_karras_sigmas,
            timestep_spacing,
            steps_offset,
        )

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        # The original implementation `noisy_samples = original_samples + noise * sigma`
        # causes an OOM error when `original_samples` (a single reference frame) is broadcast
        # to the size of `noise` (all video frames).
        # By looping over the frames, we avoid creating a massive temporary tensor.

        # Check for the specific img2vid case: single reference frame and multi-frame noise.
        if original_samples.shape[1] == 1 and noise.shape[1] > 1:
            # Add noise frame-by-frame to prevent large temporary tensor allocation.
            noisy_samples = torch.empty_like(noise)
            ref_latents = original_samples.squeeze(1)  # Shape: [B, C, H, W]
            sigma_frame = sigma.squeeze(1)  # Shape: [B, 1, 1, 1]

            for i in range(noise.shape[1]):  # Loop over the frame dimension
                noisy_samples[:, i] = ref_latents + noise[:, i] * sigma_frame
        else:
            # Fallback to the original behavior for all other cases.
            noisy_samples = original_samples + noise * sigma
        
        return noisy_samples

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        w_ind_noise: float = 0.5,
        noise_type: str = "random",
    ) -> Union[EulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        # print(self.step_index, sigma)

        gamma = (
            min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigma <= s_tmax
            else 0.0
        )
        device = model_output.device

        if noise_type == "random":
            noise = randn_tensor(
                model_output.shape,
                dtype=model_output.dtype,
                device=device,
                generator=generator,
            )

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        if (
            self.config.prediction_type == "original_sample"
            or self.config.prediction_type == "sample"
        ):
            pred_original_sample = model_output
        elif self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (
                sample / (sigma**2 + 1)
            )
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat

        dt = self.sigmas[self.step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return EulerDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )

    def step_bk(
        self,
        model_output: FloatTensor,
        timestep: float | FloatTensor,
        sample: FloatTensor,
        s_churn: float = 0,
        s_tmin: float = 0,
        s_tmax: float = float("inf"),
        s_noise: float = 1,
        generator: Generator | None = None,
        return_dict: bool = True,
        w_ind_noise: float = 0.5,
        noise_type: str = "random",
    ) -> EulerDiscreteSchedulerOutput | Tuple:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`float`): current timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            s_churn (`float`)
            s_tmin  (`float`)
            s_tmax  (`float`)
            s_noise (`float`)
            generator (`torch.Generator`, optional): Random number generator.
            return_dict (`bool`): option for returning tuple rather than EulerDiscreteSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.EulerDiscreteSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.EulerDiscreteSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.

        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]

        gamma = (
            min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigma <= s_tmax
            else 0.0
        )

        device = model_output.device
        if noise_type == "random":
            noise = randn_tensor(
                model_output.shape,
                dtype=model_output.dtype,
                device=device,
                generator=generator,
            )

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        if (
            self.config.prediction_type == "original_sample"
            or self.config.prediction_type == "sample"
        ):
            pred_original_sample = model_output
        elif self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (
                sample / (sigma**2 + 1)
            )
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat

        dt = self.sigmas[step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt

        if not return_dict:
            return (prev_sample,)

        return EulerDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )
