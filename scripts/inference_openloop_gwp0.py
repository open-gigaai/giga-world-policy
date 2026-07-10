import argparse
import os
import sys
import time
import types

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THIRD_PARTY_ROOT = os.path.join(_PROJECT_ROOT, "third_party")
_GIGA_ROOT = os.environ.get("WAM_GIGA_ROOT", _THIRD_PARTY_ROOT)
for path in [
    _PROJECT_ROOT,
    os.path.join(_GIGA_ROOT, "giga-datasets"),
    os.path.join(_GIGA_ROOT, "giga-models"),
    os.path.join(_GIGA_ROOT, "giga-train"),
]:
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
os.environ["GIGA_MODELS_LIGHT_IMPORT"] = "1"

# ---------------------------------------------------------------------------
# Default inference paths. Keep empty for the open-source config; pass paths via
# CLI flags or the run_inference_openloop*.sh environment variables.
# ---------------------------------------------------------------------------
DEFAULT_BASE_MODEL = ""
DEFAULT_CHECKPOINT = ""
DEFAULT_NORM_STATS = ""
DEFAULT_DATA_PATHS: list[str] = []
DEFAULT_DATA_IDX = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11411
DEFAULT_DEVICE = "cuda:0"

import copy
import html
import json
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import regex as re
import torch
import torch.nn.functional as torch_F
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from world_action_model.models import CasualWorldActionTransformer, CasualWorldActionTransformer_MoT
from giga_models.sockets import RobotInferenceClient, RobotInferenceServer

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class WAPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->transformer_2->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer", "transformer_2", "image_encoder", "image_processor"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_processor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModel = None,
        transformer: Union[CasualWorldActionTransformer, CasualWorldActionTransformer_MoT] = None,
        transformer_2: Union[CasualWorldActionTransformer, CasualWorldActionTransformer_MoT] = None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            transformer=transformer,
            scheduler=scheduler,
            image_processor=image_processor,
            transformer_2=transformer_2,
        )
        self.register_to_config(boundary_ratio=boundary_ratio, expand_timesteps=expand_timesteps)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.image_processor = image_processor
        self.action_scheduler = copy.deepcopy(scheduler)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
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

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_image(
        self,
        image: PipelineImageInput,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device
        image = self.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !=" f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
    ):
        if image is not None and image_embeds is not None:
            raise ValueError(
                f"Cannot forward both `image`: {image} and `image_embeds`: {image_embeds}. Please make sure to" " only forward one of the two."
            )
        if image is None and image_embeds is None:
            raise ValueError("Provide either `image` or `prompt_embeds`. Cannot leave both `image` and `image_embeds` undefined.")
        if image is not None and not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"`image` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(image)}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to" " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if self.config.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None.")

        if self.config.boundary_ratio is not None and image_embeds is not None:
            raise ValueError("Cannot forward `image_embeds` when the pipeline's `boundary_ratio` is not configured.")

    def prepare_latents(
        self,
        image: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        action_chunk: Optional[torch.Tensor] = None,
        action_dim: Optional[int] = 14,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            action_shape = (batch_size, action_chunk, action_dim)
            action = randn_tensor(action_shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]

        if self.config.expand_timesteps:
            video_condition = image

        elif last_image is None:
            video_condition = torch.cat([image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2)
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
                dim=2,
            )
        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)

        if isinstance(generator, list):
            latent_condition = [retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        if self.config.expand_timesteps:
            first_frame_mask = torch.ones(1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device)
            first_frame_mask[:, :, 0] = 0
            return latents, latent_condition, first_frame_mask, action

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

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        action_chunk: int,
        state: Optional[torch.Tensor] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        action_dim: int = 32,
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number.")
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        # Encode image embedding
        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # only wan 2.1 i2v transformer accepts image_embeds
        if self.transformer is not None and self.transformer.config.image_dim is not None:
            if image_embeds is None:
                if last_image is None:
                    image_embeds = self.encode_image(image, device)
                else:
                    image_embeds = self.encode_image([image, last_image], device)
            image_embeds = image_embeds.repeat(batch_size, 1, 1)
            image_embeds = image_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self.action_scheduler.set_timesteps(num_inference_steps, device=device)
        action_timesteps = self.action_scheduler.timesteps
        assert torch.all(timesteps == action_timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        if last_image is not None:
            last_image = self.video_processor.preprocess(last_image, height=height, width=width).to(device, dtype=torch.float32)
        state = state.unsqueeze(0).to(device=device, dtype=self.dtype)
        latents_outputs = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            last_image,
            action_chunk,
            action_dim=action_dim,
        )
        if self.config.expand_timesteps:
            # wan 2.2 5b i2v use firt_frame_mask to mask timesteps
            latents, condition, first_frame_mask, action = latents_outputs
        else:
            latents, condition = latents_outputs

        # 6. Denoising loop
        action = action.to(dtype=transformer_dtype, device=device)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        profile_core_forward = bool(getattr(self, "_profile_core_forward_latency", False))
        profile_denoising_loop = bool(getattr(self, "_profile_denoising_loop_latency", False))
        denoising_loop_repeat = getattr(self, "_denoising_loop_repeat_override", None)
        if denoising_loop_repeat is not None:
            denoising_loop_repeat = max(1, int(denoising_loop_repeat))
        use_cuda_timing = (
            (profile_core_forward or profile_denoising_loop)
            and str(device).startswith("cuda")
            and torch.cuda.is_available()
        )
        core_forward_events = []
        core_forward_times_s = []
        denoising_loop_events = []
        denoising_loop_wall_times_s = []
        denoising_loop_count = denoising_loop_repeat or 1
        progress_total = denoising_loop_count * len(timesteps)
        initial_action = action.detach().clone() if denoising_loop_repeat is not None else None

        with self.progress_bar(total=progress_total) as progress_bar:
            for denoising_loop_idx in range(denoising_loop_count):
                if denoising_loop_repeat is not None:
                    action = initial_action.clone()
                    self.action_scheduler.set_timesteps(num_inference_steps, device=device)

                if profile_denoising_loop:
                    if str(device).startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    denoising_loop_start_time = time.perf_counter()
                    if use_cuda_timing:
                        denoising_loop_start_event = torch.cuda.Event(enable_timing=True)
                        denoising_loop_end_event = torch.cuda.Event(enable_timing=True)
                        denoising_loop_start_event.record()

                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t

                    if boundary_timestep is None or t >= boundary_timestep:
                        # wan2.1 or high-noise stage in wan2.2
                        current_model = self.transformer
                        current_guidance_scale = guidance_scale
                    else:
                        # low-noise stage in wan2.2
                        current_model = self.transformer_2
                        current_guidance_scale = guidance_scale_2

                    if self.config.expand_timesteps:
                        latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                        latent_model_input = latent_model_input.to(transformer_dtype)

                        # seq_len: num_latent_frames * (latent_height // patch_size) * (latent_width // patch_size)
                        temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                        # batch_size, seq_len
                        timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                    else:
                        latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                        timestep = t.expand(latents.shape[0])

                    num_state_tokens = state.shape[1]
                    num_action_tokens = action.shape[1]
                    noise_t = timestep[:, -2:-1]
                    extra_timestep = torch.zeros(1, num_state_tokens + num_action_tokens, device=timesteps.device, dtype=timesteps.dtype)
                    extra_timestep[:, num_state_tokens:] = noise_t

                    frame_per_tokens = first_frame_mask.shape[-1] * first_frame_mask.shape[-2] // 4
                    num_latent_tokens = frame_per_tokens * first_frame_mask.shape[2]
                    timestep = torch.zeros(
                        1, num_state_tokens + num_action_tokens + num_latent_tokens, device=latent_model_input.device, dtype=latent_model_input.dtype
                    )
                    num_clean_latent_tokens = frame_per_tokens
                    timestep[:, num_state_tokens + num_clean_latent_tokens :] = noise_t

                    with current_model.cache_context("cond"):
                        if profile_core_forward and use_cuda_timing:
                            core_start_event = torch.cuda.Event(enable_timing=True)
                            core_end_event = torch.cuda.Event(enable_timing=True)
                            core_start_event.record()
                        elif profile_core_forward:
                            core_start_time = time.perf_counter()

                        action_pred = current_model(
                            ref_latents=latent_model_input[:, :, :1],
                            noisy_latents=latent_model_input[:, :, 1:],
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            # encoder_hidden_states_image=image_embeds,
                            # attention_kwargs=attention_kwargs,
                            return_dict=False,
                            action=action,
                            state=state,
                            # extra_timestep=extra_timestep,
                            action_only=True,
                        )

                        if profile_core_forward and use_cuda_timing:
                            core_end_event.record()
                            core_forward_events.append((core_start_event, core_end_event))
                        elif profile_core_forward:
                            core_forward_times_s.append(time.perf_counter() - core_start_time)

                    if self.do_classifier_free_guidance:
                        with current_model.cache_context("uncond"):
                            noise_uncond = current_model(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                encoder_hidden_states_image=image_embeds,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                                action=action,
                                action_only=True,
                            )
                            noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    action = self.action_scheduler.step(action_pred, t, action, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if denoising_loop_repeat is not None:
                        progress_bar.update()
                    elif i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()

                if profile_denoising_loop:
                    if use_cuda_timing:
                        denoising_loop_end_event.record()
                        denoising_loop_events.append((denoising_loop_start_event, denoising_loop_end_event))
                    if str(device).startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    denoising_loop_wall_times_s.append(time.perf_counter() - denoising_loop_start_time)

        if profile_core_forward:
            if use_cuda_timing and core_forward_events:
                torch.cuda.synchronize(device)
                core_forward_times_s = [
                    start_event.elapsed_time(end_event) / 1000.0
                    for start_event, end_event in core_forward_events
                ]
            self._last_core_forward_latencies_s = core_forward_times_s
            self._last_core_forward_latency_s = float(sum(core_forward_times_s))
        else:
            self._last_core_forward_latencies_s = []
            self._last_core_forward_latency_s = None

        if profile_denoising_loop:
            denoising_loop_cuda_times_s = []
            if use_cuda_timing and denoising_loop_events:
                torch.cuda.synchronize(device)
                denoising_loop_cuda_times_s = [
                    start_event.elapsed_time(end_event) / 1000.0
                    for start_event, end_event in denoising_loop_events
                ]
            self._last_denoising_loop_latencies_s = denoising_loop_wall_times_s
            self._last_denoising_loop_cuda_latencies_s = denoising_loop_cuda_times_s
            self._last_denoising_loop_latency_s = float(sum(denoising_loop_wall_times_s))
        else:
            self._last_denoising_loop_latencies_s = []
            self._last_denoising_loop_cuda_latencies_s = []
            self._last_denoising_loop_latency_s = None

        if not return_dict:
            return action


def load_video(video, valid_range=None, sample_frames=None, sample_stride=1, sample_method=2, max_frames=None):
    if sample_frames is not None:
        assert max_frames is None
    if valid_range is None:
        valid_range = (0, len(video))
    video_length = valid_range[1] - valid_range[0]
    if sample_frames is None:
        sample_indexes = np.arange(valid_range[0], valid_range[1], sample_stride, dtype=int)
        if max_frames is not None and len(sample_indexes) > max_frames:
            sample_indexes = sample_indexes[:max_frames]
    elif sample_frames >= video_length:
        sample_indexes = np.arange(valid_range[0], valid_range[1], dtype=int)
    else:
        sample_length = min(video_length, (sample_frames - 1) * sample_stride + 1)
        sample_indexes = np.linspace(valid_range[0], valid_range[0] + sample_length - 1, sample_frames, dtype=int)
    images = [video[index] for index in sample_indexes]
    return images, sample_indexes


from giga_datasets import image_utils
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def read_video(video_path):
    from torchvision.io import VideoReader as TorchVideoReader

    reader = TorchVideoReader(video_path, 'video')
    cv_frame_list = []
    frame_count = 0
    for frame in reader:
        frame_count += 1
        last_pts = frame['pts']
        frame_np = frame['data'].cpu().numpy()
        # 将帧转换为 NumPy 数组（HWC 格式）
        # (3, 480, 640) -> (480, 640, 3)
        frame_np = np.transpose(frame_np, (1, 2, 0))
        cv_frame = frame_np
        cv_frame_list.append(cv_frame)  # hdf5 -> mp4
    cv_frame_list = np.array(cv_frame_list)
    return cv_frame_list


def process_video(video, dst_size):
    sample_info = {
        "sample_frames": len(video),
        "sample_stride": 1,
    }
    images, sample_indexes = load_video(video, **sample_info)
    print("sample_indexes:", sample_indexes)
    height, width = images[0].height, images[0].width
    dst_width, dst_height = image_utils.get_image_size((width, height), dst_size, mode='area', multiple=32)
    print(dst_width, dst_height)
    if float(dst_height) / height < float(dst_width) / width:
        new_height = int(round(float(dst_width) / width * height))
        new_width = dst_width
    else:
        new_height = dst_height
        new_width = int(round(float(dst_height) / height * width))
    assert new_width >= dst_width and new_height >= dst_height
    x1 = (new_width - dst_width) // 2
    y1 = (new_height - dst_height) // 2
    input_images = []
    for i in range(len(images)):
        image = F.resize(images[i], (new_height, new_width), InterpolationMode.BILINEAR)
        image = F.crop(image, y1, x1, dst_height, dst_width)
        input_images.append(image)
    # ref_image
    ref_image = input_images[0]
    return ref_image, input_images, sample_indexes

def process_images(input_images, dst_width, dst_height):
    height = input_images.height
    width = input_images.width
    if float(dst_height) / height < float(dst_width) / width:
        new_height = int(round(float(dst_width) / width * height))
        new_width = dst_width
    else:
        new_height = dst_height
        new_width = int(round(float(dst_height) / height * width))
    input_images = F.resize(input_images, (new_height, new_width), InterpolationMode.BILINEAR)
    # center crop
    x1 = (new_width - dst_width) // 2
    y1 = (new_height - dst_height) // 2
    input_images = F.crop(input_images, y1, x1, dst_height, dst_width)
    return input_images

def get_ref_image_3views(images, dst_size, layout="tshape"):
    dst_width, dst_height = dst_size
    print(f"=========== dst_size: {dst_width} {dst_height} ============")
    img_front, img_left, img_right = images

    if layout == "tshape":
        top_h = dst_height//2
        bottom_h = dst_height - top_h
        left_w = dst_width // 2
        right_w = dst_width - left_w

        cam_high = process_images(img_front, dst_width=dst_width, dst_height=top_h)
        cam_left = process_images(img_left, dst_width=left_w, dst_height=bottom_h)
        cam_right = process_images(img_right, dst_width=right_w, dst_height=bottom_h)
        out = Image.new("RGB", (dst_width, dst_height))
        out.paste(cam_high, (0, 0))
        out.paste(cam_left, (0, top_h))
        out.paste(cam_right, (left_w, top_h))
    elif layout in {"horizontal"}:
        target_h = int(img_front.height)
        target_w = int(img_front.width)
        img_front_r = F.resize(img_front, (target_h, target_w), InterpolationMode.BILINEAR)
        img_left_r = F.resize(img_left, (target_h, target_w), InterpolationMode.BILINEAR)
        img_right_r = F.resize(img_right, (target_h, target_w), InterpolationMode.BILINEAR)

        out = Image.new("RGB", (target_w * 3, target_h))
        out.paste(img_front_r, (0, 0))
        out.paste(img_left_r, (target_w, 0))
        out.paste(img_right_r, (target_w * 2, 0))
        out = F.resize(out, (dst_height, dst_width), InterpolationMode.BILINEAR)
    else:
        raise ValueError(f"Unknown layout: {layout}")
    return out


def _require_nonempty_path(value: str | None, name: str) -> str:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{name} must be set")
    return str(value)


def _resolve_data_paths(data_paths: list[str] | None) -> list[str]:
    if data_paths is None:
        data_paths = DEFAULT_DATA_PATHS
    data_paths = [path for path in data_paths if str(path).strip()]
    if not data_paths:
        raise ValueError("--data-path must be set at least once")
    return data_paths

def get_task_dir_from_checkpoint(checkpoint: str) -> str:
    """Resolve experiment task dir from checkpoint path under .../<task>/models/..."""
    checkpoint_path = os.path.abspath(checkpoint)
    if os.path.isfile(checkpoint_path):
        checkpoint_path = os.path.dirname(checkpoint_path)

    parts = checkpoint_path.split(os.sep)
    if "models" in parts:
        models_idx = parts.index("models")
        return os.sep.join(parts[:models_idx])

    return os.path.abspath(os.path.join(checkpoint_path, os.pardir, os.pardir))


def get_step_subdir_from_checkpoint(checkpoint: str) -> str:
    """Extract step subdir name like step_50000 from checkpoint path."""
    checkpoint_path = os.path.abspath(checkpoint)
    if os.path.isfile(checkpoint_path):
        checkpoint_path = os.path.dirname(checkpoint_path)

    for part in reversed(checkpoint_path.split(os.sep)):
        match = re.match(r"checkpoint_.*_step_(\d+)$", part)
        if match:
            return f"step_{match.group(1)}"
        match = re.match(r"step_(\d+)$", part)
        if match:
            return f"step_{match.group(1)}"

    return os.path.basename(checkpoint_path)


def get_output_dirs_from_checkpoint(checkpoint: str) -> Tuple[str, str]:
    task_dir = get_task_dir_from_checkpoint(checkpoint)
    step_subdir = get_step_subdir_from_checkpoint(checkpoint)
    save_dir = os.path.join(task_dir, "open_loopresults", step_subdir)
    ref_image_save_dir = os.path.join(task_dir, "visualization", step_subdir)
    return save_dir, ref_image_save_dir


def load_fixed_prompt_embedding(t5_path: str, device: str) -> torch.Tensor:
    t5 = torch.load(t5_path, map_location="cpu")
    if isinstance(t5, dict):
        t5 = t5.get("t5_embedding", next(iter(t5.values())))
    t5 = t5.float()

    if t5.ndim == 2:
        t5 = t5[:64]
        if t5.shape[0] < 64:
            t5 = torch_F.pad(t5, (0, 0, 0, 64 - t5.shape[0]), value=0.0)
        t5 = t5.unsqueeze(0)
    elif t5.ndim == 3:
        t5 = t5[:, :64]
        if t5.shape[1] < 64:
            t5 = torch_F.pad(t5, (0, 0, 0, 64 - t5.shape[1]), value=0.0)
    else:
        raise ValueError(f"Unsupported T5 embedding shape from {t5_path}: {tuple(t5.shape)}")

    return t5.to(device)


def resolve_torch_dtype(dtype_name):
    if dtype_name is None:
        return None
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    normalized = str(dtype_name).lower().replace("torch.", "")
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported model dtype: {dtype_name}")
    return aliases[normalized]


def get_policy(
    checkpoint=DEFAULT_CHECKPOINT,
    base_model=DEFAULT_BASE_MODEL,
    norm_stats=DEFAULT_NORM_STATS,
    data_paths=None,
    data_idx=DEFAULT_DATA_IDX,
    device=DEFAULT_DEVICE,
    fixed_t5_path=None,
    model_dtype="bf16",
):
    checkpoint = _require_nonempty_path(checkpoint, "--checkpoint")
    base_model = _require_nonempty_path(base_model, "--base-model")
    norm_stats = _require_nonempty_path(norm_stats, "--norm-stats")
    data_paths = _resolve_data_paths(data_paths)

    _, ref_image_save_dir = get_output_dirs_from_checkpoint(checkpoint)
    os.makedirs(ref_image_save_dir, exist_ok=True)
    print(f"Saving ref images to: {ref_image_save_dir}")

    if device.startswith("cuda"):
        torch.cuda.set_device(device)

    torch_dtype = resolve_torch_dtype(model_dtype)
    print(f"Loading base model from: {base_model}")
    print(f"Loading checkpoint from: {checkpoint}")
    print(f"Loading model dtype: {torch_dtype}")
    vae = AutoencoderKLWan.from_pretrained(base_model, subfolder="vae", torch_dtype=torch_dtype)

    # Pick the transformer class according to the checkpoint's own config so that
    # both the plain (CasualWorldActionTransformer) and the Mixture-of-Transformers
    # (CasualWorldActionTransformer_MoT) checkpoints can be loaded transparently.
    model_class = CasualWorldActionTransformer_MoT
    config_path = os.path.join(checkpoint, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            ckpt_class_name = json.load(f).get("_class_name", "")
        if ckpt_class_name == "CasualWorldActionTransformer":
            model_class = CasualWorldActionTransformer
    print(f"Using transformer class: {model_class.__name__}")
    transformer = model_class.from_pretrained(checkpoint, torch_dtype=torch_dtype)
    transformer.eval()
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    pipe = WAPipeline.from_pretrained(base_model, vae=vae, transformer=transformer, scheduler=scheduler, torch_dtype=torch_dtype)
    pipe.to(device=device, dtype=torch_dtype)
    pipe._model_dtype_name = str(torch_dtype)

    print(f"Loading norm stats from: {norm_stats}")
    with open(norm_stats, "r") as f:
        stats_dict = json.load(f)

    dst_size = (320, 384)
    action_chunk = 48
    guidance_scale = 0.0
    num_inference_steps = 10

    state_mean = torch.tensor(stats_dict['norm_stats']['observation.state']['mean']).to(device=device)
    state_std = torch.tensor(stats_dict['norm_stats']['observation.state']['std']).to(device=device)
    state_min = torch.tensor(stats_dict['norm_stats']['observation.state']['q01'])[..., :14].to(device=device)
    state_max = torch.tensor(stats_dict['norm_stats']['observation.state']['q99'])[..., :14].to(device=device)

    delta_mean = torch.tensor(stats_dict['norm_stats']['action']['mean'][:14]).to(device=device)
    delta_std = torch.tensor(stats_dict['norm_stats']['action']['std'][:14]).to(device=device)
    delta_min = torch.tensor(stats_dict['norm_stats']['action']['q01'][:14])[..., :14].to(device=device)
    delta_max = torch.tensor(stats_dict['norm_stats']['action']['q99'][:14])[..., :14].to(device=device)
    eps = 1e-8  # 小的epsilon值防止除零
    state_range = (state_max - state_min).clamp_min(eps)
    delta_range = (delta_max - delta_min).clamp_min(eps)
    # state_min = torch.from_numpy(stats_dict['state_min']).to(device=device, dtype=torch.float32)
    # state_max = torch.from_numpy(stats_dict['state_max']).to(device=device, dtype=torch.float32)

    # The server does not need the dataset: open-loop replay/debug paths are disabled and
    # the client streams observations over the socket. Keep `dataset` as None so the
    # (disabled) replay/debug branches below remain valid closures.
    dataset = None
    replay_action = False
    replay_index = 0

    replay_action = False
    replay_image = False
    debug = False

    if fixed_t5_path is None and data_paths:
        fixed_t5_path = os.path.join(data_paths[0], "t5_embedding", f"episode_{int(data_idx):06d}.pt")

    fixed_prompt_embedding = None
    if fixed_t5_path:
        if not os.path.isfile(fixed_t5_path):
            print(f"Fixed T5 embedding not found, requests must provide prompt_embedding or prompt: {fixed_t5_path}")
        else:
            fixed_prompt_embedding = load_fixed_prompt_embedding(fixed_t5_path, device)
            print(f"Loaded fixed T5 embedding from: {fixed_t5_path}, shape={tuple(fixed_prompt_embedding.shape)}")

    def inference(self, data):
        if replay_action:
            if not hasattr(self, 'replay_index'):
                self.replay_index = 0
            action = dataset[data_idx]['action'][self.replay_index : self.replay_index + action_chunk]
            self.replay_index += action_chunk
            self.replay_index = self.replay_index % len(dataset[data_idx]['action'])
            return None, action

        if replay_image:
            if not hasattr(self, 'replay_index'):
                self.replay_index = 0
            frame = dataset[data_idx]['video'][self.replay_index].asnumpy()  # decord NDArray -> numpy
            ref_image = Image.fromarray(frame)
            state = torch.tensor(dataset[data_idx]['state'][self.replay_index]).to(device)

            # state = data['observation.state'].to(device)

            state = torch.zeros_like(data['observation.state']).to(device)
            self.replay_index += action_chunk

        if not replay_image:
            if debug:
                if not hasattr(self, 'replay_index'):
                    self.replay_index = 0
                frame = dataset[data_idx]['video'][self.replay_index].asnumpy()  # decord NDArray -> numpy
                ref_image_1 = Image.fromarray(frame)
                state_1 = torch.tensor(dataset[data_idx]['state'][self.replay_index]).to(device)
                self.replay_index += action_chunk

            images = {
                'observation.images.cam_high': data['observation.images.cam_high'],  # 3 H W, tensor float64
                'observation.images.cam_left_wrist': data['observation.images.cam_left_wrist'],  # 3 H W, tensor float64
                'observation.images.cam_right_wrist': data['observation.images.cam_right_wrist'],  # 3 H W, tensor float64
            }

            state = data['observation.state'].to(device)

            pil_images = [
                PIL.Image.fromarray((images['observation.images.cam_high'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)),
                PIL.Image.fromarray((images['observation.images.cam_left_wrist'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)),
                PIL.Image.fromarray((images['observation.images.cam_right_wrist'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)),
            ]
            ref_image = get_ref_image_3views(pil_images, dst_size)

            if debug:
                # 图像 PSNR
                import torch.nn.functional as F

                a = np.array(ref_image).astype(np.float64)
                b = np.array(ref_image_1).astype(np.float64)
                mse_img = np.mean((a - b) ** 2)
                psnr = 20 * np.log10(255.0 / np.sqrt(mse_img)) if mse_img > 0 else float('inf')
                print(f"[DEBUG] image  MSE: {mse_img:.4f}, PSNR: {psnr:.2f} dB")

                # state 差距
                s0 = state[..., :14]
                s1 = state_1[..., :14]
                diff = (s0 - s1).abs()
                print(
                    f"[DEBUG] state  max_diff: {diff.max().item():.6f}, mean_diff: {diff.mean().item():.6f}, MSE: {F.mse_loss(s0.float(), s1.float()).item():.6f}"
                )

        state = state[..., :14]

        chunk_idx = data.get("chunk_idx", 0)
        quiet = bool(data.get("_quiet", False))
        if not data.get("_skip_ref_image_save", False):
            episode_idx = data.get("episode_idx", data_idx)
            ref_episode_dir = os.path.join(ref_image_save_dir, f"episode_{int(episode_idx)}")
            os.makedirs(ref_episode_dir, exist_ok=True)
            ref_image.save(os.path.join(ref_episode_dir, f"chunk_{chunk_idx:04d}.png"))
        # min-max
        eps = 1e-8
        norm_state = ((state - state_min) / state_range) * 2 - 1
        if not quiet:
            print(norm_state.max(), norm_state.min())

        norm_state = norm_state.to(device)
        if norm_state.ndim == 1:
            norm_state = norm_state.unsqueeze(0)
        norm_state = torch_F.pad(norm_state, (0, 32 - norm_state.shape[-1]), value=0.0)
        prompt_embedding = data.get('prompt_embedding', None)
        prompt = data.get('prompt', None)
        if prompt_embedding is None and prompt is None:
            prompt_embedding = fixed_prompt_embedding
        if prompt_embedding is None and prompt is None:
            raise ValueError("Request did not include prompt_embedding or prompt, and no fixed T5 embedding was loaded.")
        if prompt_embedding is not None:
            prompt_embedding = prompt_embedding.to(device)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(device))
        model_start_time = time.perf_counter()
        denoising_loop_repeat = data.get("_denoising_loop_repeat", None)
        request_num_inference_steps = int(data.get("_num_inference_steps", num_inference_steps))
        old_denoising_loop_repeat = getattr(pipe, "_denoising_loop_repeat_override", None)
        if denoising_loop_repeat is not None:
            pipe._denoising_loop_repeat_override = int(denoising_loop_repeat)
        try:
            pred_action = pipe(
                height=dst_size[1],
                width=dst_size[0],
                action_chunk=action_chunk,
                state=norm_state,
                num_frames=5,
                guidance_scale=guidance_scale,
                num_inference_steps=request_num_inference_steps,
                image=ref_image,
                return_dict=False,
                prompt_embeds=prompt_embedding,
                prompt=prompt,
                action_dim=32,
            )
        finally:
            if denoising_loop_repeat is not None:
                if old_denoising_loop_repeat is None:
                    try:
                        delattr(pipe, "_denoising_loop_repeat_override")
                    except AttributeError:
                        pass
                else:
                    pipe._denoising_loop_repeat_override = old_denoising_loop_repeat
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(device))
        model_end_time = time.perf_counter()
        self._last_model_inference_s = model_end_time - model_start_time
        if not quiet:
            print(f"Model inference time: {self._last_model_inference_s:.6f} seconds", flush=True)

        # gt_action = torch.tensor(data['gt_action']).to(pred_action.device)
        # gt_action = (gt_action - delta_min) / delta_range
        # mse = torch.nn.functional.mse_loss(pred_action.squeeze(0), gt_action)
        # mse_per_dim = torch.mean((pred_action.squeeze(0) - gt_action) ** 2, dim=0)
        # self._last_norm_mse = mse.item()
        # self._last_norm_mse_per_dim = mse_per_dim.detach().cpu().numpy()
        # print(f"[norm] MSE: {self._last_norm_mse:.6f}, MSE per dim: {self._last_norm_mse_per_dim}")
        # min-max
        pred_action = pred_action[..., :14]
        pred_action = ((pred_action + 1) / 2) * delta_range + delta_min
        if not quiet:
            print('pred_action ', pred_action.max(), pred_action.min())
        # z-score
        # pred_action = pred_action * delta_std + delta_mean
        pred_action = pred_action.cpu().numpy()
        mask = np.array([True] * 6 + [False] + [True] * 6 + [False])
        pred_action = pred_action[0] + state.repeat(action_chunk, 1).cpu().numpy() * mask
        return pred_action

    pipe.inference = types.MethodType(inference, pipe)
    return pipe

def load_lerobot_v3_episode(root, episode_index, action_dim=14, fps_default=30):
    """Load a single episode from a LeRobot v3.0 packed dataset.

    Returns ``(front_frames, left_frames, right_frames, state, action, t5_embedding, task)``
    where the ``*_frames`` are lists of ``PIL.Image``, ``state``/``action`` are float32
    numpy arrays of shape ``[L, action_dim]`` and ``t5_embedding`` is a float tensor.
    """
    import json as _json

    import av  # PyAV; these videos are AV1-encoded which the installed decord build cannot decode
    import pandas as pd

    def _read_frames_pyav(video_path, start_frame, num_frames):
        container = av.open(video_path)
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            end_frame = start_frame + num_frames
            frames = []
            fi = 0
            for frame in container.decode(stream):
                if fi >= end_frame:
                    break
                if fi >= start_frame:
                    frames.append(frame.to_ndarray(format="rgb24"))
                fi += 1
        finally:
            container.close()
        return frames

    view_keys = [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]

    with open(os.path.join(root, "meta", "info.json")) as f:
        info = _json.load(f)
    fps = info.get("fps", fps_default)

    # 1. Episode metadata (one row per episode, may be split across files).
    ep_meta_dir = os.path.join(root, "meta", "episodes")
    ep_files = sorted(
        os.path.join(dp, fn)
        for dp, _, fns in os.walk(ep_meta_dir)
        for fn in fns
        if fn.endswith(".parquet")
    )
    ep_df = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
    row = ep_df[ep_df["episode_index"] == episode_index].iloc[0]
    length = int(row["length"])

    # 2. State / action from the packed data parquet that holds this episode.
    data_chunk = int(row["data/chunk_index"])
    data_file = int(row["data/file_index"])
    data_path = os.path.join(root, "data", f"chunk-{data_chunk:03d}", f"file-{data_file:03d}.parquet")
    ddf = pd.read_parquet(data_path, columns=["observation.state", "action", "episode_index", "frame_index"])
    ddf = ddf[ddf["episode_index"] == episode_index].sort_values("frame_index")
    state = np.stack(ddf["observation.state"].to_numpy())[:, :action_dim].astype(np.float32)
    action = np.stack(ddf["action"].to_numpy())[:, :action_dim].astype(np.float32)

    # 3. Video frames per view, sliced from the packed mp4 by timestamp range.
    frames_per_view = []
    for key in view_keys:
        v_chunk = int(row[f"videos/{key}/chunk_index"])
        v_file = int(row[f"videos/{key}/file_index"])
        from_ts = float(row[f"videos/{key}/from_timestamp"])
        video_path = os.path.join(root, "videos", key, f"chunk-{v_chunk:03d}", f"file-{v_file:03d}.mp4")
        start = int(round(from_ts * fps))
        batch = _read_frames_pyav(video_path, start, length)  # list of [H, W, 3] uint8
        frames_per_view.append([Image.fromarray(f) for f in batch])

    # 4. Per-episode T5 embedding (the .pt file is the tensor itself).
    t5_path = os.path.join(root, "t5_embedding", f"episode_{episode_index:06d}.pt")
    t5 = torch.load(t5_path, map_location="cpu")
    if isinstance(t5, dict):
        t5 = t5.get("t5_embedding", next(iter(t5.values())))
    t5 = t5.float()

    task = row["tasks"]
    if isinstance(task, (list, np.ndarray)):
        task = task[0]

    return frames_per_view[0], frames_per_view[1], frames_per_view[2], state, action, t5, task


def inference_client(
    checkpoint=DEFAULT_CHECKPOINT,
    data_paths=None,
    data_idx=DEFAULT_DATA_IDX,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    replan_steps=None,
):
    checkpoint = _require_nonempty_path(checkpoint, "--checkpoint")
    data_paths = _resolve_data_paths(data_paths)
    save_dir, _ = get_output_dirs_from_checkpoint(checkpoint)

    action_chunk = 48
    # Open-loop replanning horizon: predict the full chunk but only execute/evaluate the
    # first `replan` actions before sliding the window and re-observing. Defaults to the
    # full chunk (replan once per chunk, == previous behaviour).
    replan = action_chunk if not replan_steps else max(1, min(int(replan_steps), action_chunk))
    print(f"Replan steps: {replan} (action_chunk={action_chunk})")

    root = data_paths[0] if isinstance(data_paths, list) else data_paths
    id = data_idx
    print(f"Loading LeRobot v3.0 episode from: {root} (episode index={id})")
    (
        front_view_images,
        left_view_images,
        right_view_images,
        all_state,
        all_action,
        episode_t5_embedding,
        episode_task,
    ) = load_lerobot_v3_episode(root, id)
    print(
        f"Episode {id}: task='{episode_task}', frames={len(front_view_images)}, "
        f"state={all_state.shape}, action={all_action.shape}"
    )
    rollout_num = max(1, (len(front_view_images) - action_chunk) // replan + 1)
    all_pred_actions = []
    all_gt_actions = []
    all_gt_delta = []
    all_pred_delta = []
    all_mse = []
    all_mse_per_dim = []
    all_norm_mse = []
    all_norm_mse_per_dim = []

    pipe = RobotInferenceClient(host=host, port=port)

    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving open-loop results to: {save_dir}")

    def save_action_as_plot(gt_action, pred_action, save_path):
        T = gt_action.shape[0]
        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 10))
        for i in range(gt_action.shape[1]):
            plt.subplot(4, 4, i + 1)
            plt.plot(range(T), gt_action[:, i], label='gt')
            plt.plot(range(T), pred_action[:, i], label='pred')
            plt.title(f'Action Dimension {i}')
            plt.xlabel('Frame')
            plt.ylabel('Value')
            plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    from einops import rearrange
    from tqdm import tqdm

    for i in tqdm(range(0, len(front_view_images) - action_chunk, replan)):
        print(f"Processing rollout {i//replan + 1}/{rollout_num}")
        start_frame = i
        end_frame = i + action_chunk
        state = all_state[start_frame][:14]
        state = torch.from_numpy(state).float().unsqueeze(0)
        gt_action = all_action[start_frame:end_frame,:14]

        # 图像始终从第0帧取
        img_front = front_view_images[start_frame]
        img_left = left_view_images[start_frame]
        img_right = right_view_images[start_frame]

        img_front = np.array(img_front) / 255.0
        img_left = np.array(img_left) / 255.0
        img_right = np.array(img_right) / 255.0

        camera_high_chw = rearrange(img_front, 'h w c -> c h w')
        camera_left_chw = rearrange(img_left, 'h w c -> c h w')
        camera_right_chw = rearrange(img_right, 'h w c -> c h w')
        prompt_embedding = episode_t5_embedding[:64]
        prompt_embedding = torch_F.pad(prompt_embedding, (0, 0, 0, 64 - prompt_embedding.shape[0]), value=0.0)[None]
        print(prompt_embedding.shape)
        observation = {
            'observation.state': state,
            'gt_action': gt_action,
            'observation.images.cam_high': torch.from_numpy(camera_high_chw),
            'observation.images.cam_left_wrist': torch.from_numpy(camera_left_chw),
            'observation.images.cam_right_wrist': torch.from_numpy(camera_right_chw),
            'prompt_embedding': prompt_embedding,
            'chunk_idx': i,
            'episode_idx': id,
            # 'prompt': data_dict['t5_embedding']['prompt_text'],
        }

        pred_action = pipe.inference(observation)

        # Per-chunk metrics/plots use the full predicted horizon (action_chunk).
        chunk_mse = np.mean((pred_action - gt_action) ** 2)
        chunk_mse_per_dim = np.mean((pred_action - gt_action) ** 2, axis=0)
        all_mse.append(chunk_mse)
        all_mse_per_dim.append(chunk_mse_per_dim)
        if hasattr(pipe, '_last_norm_mse'):
            all_norm_mse.append(pipe._last_norm_mse)
            all_norm_mse_per_dim.append(pipe._last_norm_mse_per_dim)
        print(f"[chunk {i:04d}] action MSE: {chunk_mse:.6f}, per dim: {chunk_mse_per_dim}")

        # Only the first `replan` executed actions feed the aggregated open-loop trajectory.
        exec_pred = pred_action[:replan]
        exec_gt = gt_action[:replan]
        all_pred_actions.append(exec_pred)
        all_gt_actions.append(exec_gt)

        exec_state = state.repeat(exec_gt.shape[0], 1).cpu().numpy()
        all_gt_delta.append(exec_gt - exec_state)
        all_pred_delta.append(exec_pred - exec_state)

        # 每个 chunk 单独保存曲线（完整预测 horizon）
        chunk_save_dir = os.path.join(save_dir, 'all', f'episode_{id}', f'chunk_{i:04d}')
        os.makedirs(chunk_save_dir, exist_ok=True)
        save_action_as_plot(gt_action, pred_action, os.path.join(chunk_save_dir, 'action_plot.png'))
        gt_delta_full = gt_action - state.repeat(action_chunk, 1).cpu().numpy()
        pred_delta_full = pred_action - state.repeat(action_chunk, 1).cpu().numpy()
        save_action_as_plot(gt_delta_full, pred_delta_full, os.path.join(chunk_save_dir, 'delta_plot.png'))

    all_gt_actions = np.stack(all_gt_actions).reshape(-1, all_gt_actions[0].shape[-1])
    all_pred_actions = np.stack(all_pred_actions).reshape(-1, all_pred_actions[0].shape[-1])
    all_gt_delta = np.stack(all_gt_delta).reshape(-1, all_gt_delta[0].shape[-1])
    all_pred_delta = np.stack(all_pred_delta).reshape(-1, all_pred_delta[0].shape[-1])

    save_name = f'{id}'
    save_action_as_plot(all_gt_actions, all_pred_actions, os.path.join(save_dir, f"{save_name}_action_plot.png"))
    save_action_as_plot(all_gt_delta, all_pred_delta, os.path.join(save_dir, f"{save_name}_delta_plot.png"))
    print(os.path.join(save_dir, f"{save_name}_action_plot.png"))

    avg_mse = np.mean(all_mse)
    avg_mse_per_dim = np.mean(np.stack(all_mse_per_dim), axis=0)
    overall_mse = np.mean((all_pred_actions - all_gt_actions) ** 2)
    overall_mse_per_dim = np.mean((all_pred_actions - all_gt_actions) ** 2, axis=0)
    print(f"\n=== Action MSE Summary (denormalized, {len(all_mse)} chunks) ===")
    print(f"Average chunk MSE: {avg_mse:.6f}")
    print(f"Overall MSE (all timesteps): {overall_mse:.6f}")
    print(f"Average MSE per dimension: {avg_mse_per_dim}")
    print(f"Overall MSE per dimension: {overall_mse_per_dim}")

    if all_norm_mse:
        avg_norm_mse = np.mean(all_norm_mse)
        avg_norm_mse_per_dim = np.mean(np.stack(all_norm_mse_per_dim), axis=0)
        print(f"\n=== Action MSE Summary (normalized, {len(all_norm_mse)} chunks) ===")
        print(f"Average chunk MSE: {avg_norm_mse:.6f}")
        print(f"Average MSE per dimension: {avg_norm_mse_per_dim}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Open-loop WAM inference. Pass paths via CLI flags or run_inference_openloop*.sh environment variables."
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Fine-tuned transformer checkpoint path (directory or weight file).",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base Wan diffusers model path. Required for server mode.",
    )
    parser.add_argument(
        "--norm-stats",
        default=DEFAULT_NORM_STATS,
        help="Norm stats JSON used for state/action denormalization.",
    )
    parser.add_argument(
        "--data-path",
        dest="data_paths",
        action="append",
        default=None,
        help="Packed dataset root. Repeat for multiple roots. Required for client mode.",
    )
    parser.add_argument(
        "--data-idx",
        type=int,
        default=DEFAULT_DATA_IDX,
        help="Episode index inside the loaded dataset.",
    )
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=None,
        help=(
            "Open-loop replanning horizon: the model still predicts the full action chunk, "
            "but only the first --replan-steps actions are executed/evaluated before the "
            "window slides forward and the model re-observes. Defaults to the full chunk (48), "
            "i.e. replan once per chunk."
        ),
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="RobotInference server/client host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="RobotInference server/client port.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument(
        "--fixed-t5-path",
        default=None,
        help=(
            "T5 embedding .pt file used by the server when a request only contains images and state. "
            "Defaults to <data-path>/t5_embedding/episode_<data-idx>.pt."
        ),
    )
    parser.add_argument("--model-dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--server", action="store_true", help="Run inference server instead of open-loop client.")
    return parser.parse_args()


def main(args):
    policy = get_policy(
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        norm_stats=args.norm_stats,
        data_paths=args.data_paths,
        data_idx=args.data_idx,
        device=args.device,
        fixed_t5_path=args.fixed_t5_path,
        model_dtype=args.model_dtype,
    )
    server = RobotInferenceServer(policy, host=args.host, port=args.port)
    server.run()


def test_policy(args):
    inference_client(
        checkpoint=args.checkpoint,
        data_paths=args.data_paths,
        data_idx=args.data_idx,
        host=args.host,
        port=args.port,
        replan_steps=args.replan_steps,
    )


if __name__ == "__main__":
    args = parse_args()
    if args.server:
        main(args)
    else:
        test_policy(args)
