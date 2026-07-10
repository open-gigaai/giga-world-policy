import functools
import os

import torch
from diffusers.models import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from giga_models import utils as gm_utils
from world_action_model.models import CasualWorldActionTransformer_MoT
from giga_train import ModuleDict, Trainer

class CasualWATrainerMoT(Trainer):
    """WAM pretraining trainer for the Mixture-of-Transformers policy."""

    def get_models(self, model_config):
        pretrained = gm_utils.get_model_path(model_config.pretrained)
        self.visual_flow_shift = float(model_config.visual_flow_shift)
        self.action_flow_shift = float(model_config.action_flow_shift)
        self.expand_timesteps = model_config.get("expand_timesteps", False)
        self.action_repeats = model_config.get("action_repeats", 1)
        self.state_repeats = model_config.get("state_repeats", 1)
        self.action_dim = int(model_config.get("action_dim", 14))
        self.num_embodiments = int(model_config.get("num_embodiments", 1))

        vae_pretrained = model_config.get("vae_pretrained", os.path.join(pretrained, "vae"))
        vae_dtype = model_config.get("vae_dtype", torch.float32)
        vae = AutoencoderKLWan.from_pretrained(vae_pretrained)
        vae.requires_grad_(False)
        vae.to(self.device, dtype=vae_dtype)
        self.vae = vae
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.device, dtype=vae_dtype
        )
        self.latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.device, dtype=vae_dtype
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        transformer_pretrained = model_config.get("transformer_pretrained", os.path.join(pretrained, "transformer"))
        transformer_cfg = model_config.get("transformer")
        transformer = CasualWorldActionTransformer_MoT(**transformer_cfg)
        transformer = load_pretrained_weights(
            transformer,
            transformer_pretrained,
            skip_action_expert=model_config.get("skip_action_expert", False),
            strict_load=model_config.get("strict_load", False),
        )
        transformer.to(self.device)
        if model_config.get("enable_gradient_checkpointing", True):
            transformer.enable_gradient_checkpointing()

        model = dict(transformer=transformer)
        checkpoint = model_config.get("checkpoint", None)
        strict = model_config.get("strict", True)
        self.load_checkpoint(checkpoint, list(model.values()), strict=strict)
        model = ModuleDict(model)
        model.train()
        return model

    def forward_step(self, batch_dict):
        transformer = functools.partial(self.model, "transformer")
        images = batch_dict["images"]
        bs = images.shape[0]
        prompt_embeds = batch_dict["prompt_embeds"]
        action = batch_dict["action"]
        state = batch_dict["state"]
        embodiment_id = batch_dict["embodiment_id"]

        visual_timestep, visual_sigma = self.get_timestep_and_sigma(bs, images.ndim, self.visual_flow_shift)
        action_timestep, action_sigma = self.get_timestep_and_sigma(bs, action.ndim, self.action_flow_shift)

        if self.state_repeats > 1:
            state = state.repeat(1, self.state_repeats, 1)
        if self.action_repeats > 1:
            action = action.repeat(1, self.action_repeats, 1)

        visual_latents = self.forward_vae(images)
        visual_noise = torch.randn_like(visual_latents)
        visual_target = visual_noise - visual_latents
        noisy_latents = visual_noise * visual_sigma + visual_latents * (1 - visual_sigma)

        action_noise = torch.randn_like(action)
        action_target = action_noise - action
        noisy_action = action_noise * action_sigma + action * (1 - action_sigma)

        prompt_embeds = prompt_embeds.to(self.dtype)
        if "ref_images" not in batch_dict:
            raise ValueError("CasualWATrainerMoT requires ref_images in batch_dict")

        if not self.expand_timesteps:
            ref_images = batch_dict["ref_images"]
            ref_latents = self.forward_vae(ref_images)
            num_frames = images.shape[1]
            batch_size = ref_latents.shape[0]
            latent_height = ref_latents.shape[-2]
            latent_width = ref_latents.shape[-1]
            mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
            first_frame_mask = mask_lat_size[:, :, 0:1]
            first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
            mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
            mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
            mask_lat_size = mask_lat_size.transpose(1, 2).to(ref_latents.device)
            condition = torch.concat([mask_lat_size, ref_latents], dim=1)
            insert_noisy_latents = torch.concat([noisy_latents, condition], dim=1)
        else:
            num_latent_frames = visual_latents.shape[2]
            latent_height = visual_latents.shape[-2]
            latent_width = visual_latents.shape[-1]
            ref_images = batch_dict["ref_images"][:, :1]
            ref_latents = self.forward_vae(ref_images)
            first_frame_mask = torch.ones(
                bs,
                1,
                num_latent_frames,
                latent_height,
                latent_width,
                dtype=visual_latents.dtype,
                device=visual_latents.device,
            )
            first_frame_mask[:, :, 0] = 0
            insert_noisy_latents = (1 - first_frame_mask) * ref_latents + first_frame_mask * noisy_latents

        insert_noisy_latents = insert_noisy_latents.to(self.dtype)
        num_state_tokens = state.shape[1]
        num_action_tokens = action.shape[1]
        noisy_action = noisy_action.to(self.dtype)
        state = state.to(self.dtype)
        ref_latents = insert_noisy_latents[:, :, :1]
        noisy_latents = insert_noisy_latents[:, :, 1:]
        frame_per_tokens = first_frame_mask.shape[-1] * first_frame_mask.shape[-2] // 4
        num_latent_tokens = frame_per_tokens * first_frame_mask.shape[2]
        num_clean_latent_tokens = frame_per_tokens
        timestep = torch.zeros(
            bs,
            num_state_tokens + num_action_tokens + num_latent_tokens,
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        )
        timestep[:, num_state_tokens + num_clean_latent_tokens : num_state_tokens + num_clean_latent_tokens + num_action_tokens] = (
            action_timestep[:, None]
        )
        timestep[:, num_state_tokens + num_action_tokens + num_clean_latent_tokens :] = visual_timestep[:, None]

        visual_pred, action_pred = transformer(
            ref_latents=ref_latents,
            noisy_latents=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            action=noisy_action,
            state=state,
            embodiment_id=embodiment_id,
        )

        visual_loss = ((visual_pred.float() - visual_target.float()) * first_frame_mask).pow(2).mean()

        action_loss = (action_pred.float() - action_target.float()).pow(2).mean()
        return {
            "visual_loss": visual_loss,
            "action_loss": action_loss,
        }

    def forward_vae(self, images):
        images = images.to(self.vae.dtype)
        with torch.no_grad():
            images = rearrange(images, "b t c h w -> b c t h w")
            latents = self.vae.encode(images).latent_dist.mode()
        latents = (latents - self.latents_mean) * self.latents_std
        return latents

    def get_timestep_and_sigma(self, batch_size, ndim, flow_shift):
        sigma = torch.rand(batch_size).to(self.device)
        sigma = flow_shift * sigma / (1 + (flow_shift - 1) * sigma)
        timestep = torch.round(sigma * 1000).long()
        sigma = timestep.float() / 1000
        while len(sigma.shape) < ndim:
            sigma = sigma.unsqueeze(-1)
        return timestep, sigma

def load_pretrained_weights(model, pretrained_path, skip_action_expert=False, strict_load=False):
    from safetensors.torch import load_file

    pretrained_path = gm_utils.get_model_path(pretrained_path)
    if os.path.isdir(pretrained_path):
        weight_files = sorted(
            os.path.join(pretrained_path, f)
            for f in os.listdir(pretrained_path)
            if f.endswith((".safetensors", ".bin"))
        )
    else:
        weight_files = [pretrained_path]

    state_dicts = {}
    for weight_file in weight_files:
        print(f"Loading weights from {weight_file}")
        if weight_file.endswith(".bin"):
            checkpoint = torch.load(weight_file, map_location="cpu")
        else:
            checkpoint = load_file(weight_file)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        state_dicts.update(checkpoint)

    if not strict_load and hasattr(model, "load_from_wan_pretrained_state_dict"):
        loaded, skipped, unexpected, missing = model.load_from_wan_pretrained_state_dict(
            state_dicts,
            skip_action_expert=skip_action_expert,
        )
        if unexpected:
            print(f"[WARNING] Unexpected keys in Wan checkpoint: {unexpected[:20]}{'...' if len(unexpected) > 20 else ''}")
        if skipped:
            print(f"[WARNING] Skipped keys during MoT remap: {skipped[:20]}{'...' if len(skipped) > 20 else ''}")
        if missing:
            print(f"[WARNING] Missing keys after MoT remap: {missing[:20]}{'...' if len(missing) > 20 else ''}")
        print(f"Loaded {len(loaded)} tensors from Wan2.2 into MoT.")
        return model

    missing_keys, unexpected_keys = model.load_state_dict(state_dicts, strict=False)
    if unexpected_keys:
        print(f"[WARNING] Unexpected keys in state_dict: {unexpected_keys}")
    if missing_keys:
        print(f"[WARNING] Missing keys in state_dict: {missing_keys}")
    return model
