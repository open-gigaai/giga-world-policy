import functools

import torch

from world_action_model.trainer.wa_casual_trainer import CasualWATrainer

class CasualWATrainerPretrain(CasualWATrainer):
    def forward_step(self, batch_dict):
        transformer = functools.partial(self.model, "transformer")
        images = batch_dict["images"]
        bs = images.shape[0]
        prompt_embeds = batch_dict["prompt_embeds"]
        timestep, sigma = self.get_timestep_and_sigma(images.shape[0], images.ndim)
        action = batch_dict["action"]
        state = batch_dict["state"]
        emb_ids = batch_dict["embodiment_id"]

        if self.state_repeats > 1:
            state = state.repeat(1, self.state_repeats, 1)
        if self.action_repeats > 1:
            action = action.repeat(1, self.action_repeats, 1)

        visual_latents = self.forward_vae(images)
        visual_noise = torch.randn_like(visual_latents)
        visual_target = visual_noise - visual_latents
        noisy_latents = visual_noise * sigma + visual_latents * (1 - sigma)

        action_sigma = sigma.squeeze(-1).squeeze(-1)
        action_noise = torch.randn_like(action)
        action_target = action_noise - action
        noisy_action = action_noise * action_sigma + action * (1 - action_sigma)

        prompt_embeds = prompt_embeds.to(self.dtype)
        if "ref_images" in batch_dict:
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
                mask_lat_size = mask_lat_size.transpose(1, 2)
                mask_lat_size = mask_lat_size.to(ref_latents.device)
                condition = torch.concat([mask_lat_size, ref_latents], dim=1)
                insert_noisy_latents = torch.concat([noisy_latents, condition], dim=1)
            else:
                num_latent_frames = visual_latents.shape[2]
                latent_height = visual_latents.shape[-2]
                latent_width = visual_latents.shape[-1]
                ref_images = batch_dict["ref_images"][:, :1]
                ref_latents = self.forward_vae(ref_images)
                first_frame_mask = torch.ones(
                    bs, 1, num_latent_frames, latent_height, latent_width, dtype=visual_latents.dtype, device=visual_latents.device
                )
                first_frame_mask[:, :, 0] = 0
                insert_noisy_latents = (1 - first_frame_mask) * ref_latents + first_frame_mask * noisy_latents
                temp_ts = (first_frame_mask[:, :, :, ::2, ::2] * timestep[:, None, None, None, None]).reshape(bs, -1)
                timestep = temp_ts
        else:
            raise ValueError("CasualWATrainerPretrain requires ref_images in batch_dict")

        insert_noisy_latents = insert_noisy_latents.to(self.dtype)
        num_state_tokens = state.shape[1]
        num_action_tokens = action.shape[1]
        noise_t = timestep[:, -2:-1]

        noisy_action = noisy_action.to(self.dtype)
        state = state.to(self.dtype)

        ref_latents = insert_noisy_latents[:, :, :1]
        noisy_latents = insert_noisy_latents[:, :, 1:]
        frame_per_tokens = first_frame_mask.shape[-1] * first_frame_mask.shape[-2] // 4
        num_latent_tokens = frame_per_tokens * first_frame_mask.shape[2]
        timestep = torch.zeros(bs, num_state_tokens + num_action_tokens + num_latent_tokens, device=noisy_latents.device, dtype=noisy_latents.dtype)
        num_clean_latent_tokens = frame_per_tokens
        timestep[:, num_state_tokens + num_clean_latent_tokens :] = noise_t

        visual_pred, action_pred = transformer(
            ref_latents=ref_latents,
            noisy_latents=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            action=noisy_action,
            state=state,
            emb_ids=emb_ids,
        )

        visual_loss = ((visual_pred.float() - visual_target.float()) * first_frame_mask).pow(2).mean()

        dim_mask = None
        if "action_dim_mask" in batch_dict:
            dim_mask = _as_dim_mask(
                batch_dict["action_dim_mask"], batch_size=bs, seq_len=action.shape[1], dim=action.shape[2], device=action_pred.device
            )

        time_mask = None
        if "action_loss_mask" in batch_dict:
            time_mask = _as_time_mask(batch_dict["action_loss_mask"], batch_size=bs, seq_len=action.shape[1], device=action_pred.device)

        action_loss = masked_mse(action_pred.float(), action_target.float(), dim_mask=dim_mask, time_mask=time_mask)

        return {
            "visual_loss": visual_loss,
            "action_loss": action_loss,
        }

def _as_dim_mask(mask: torch.Tensor, batch_size: int, seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    if mask.dim() == 1:
        mask = mask[None, None, :]
    elif mask.dim() == 2:
        mask = mask[:, None, :]
    elif mask.dim() != 3:
        raise ValueError(f"action_dim_mask must have 1/2/3 dims, got {mask.shape=}")
    if mask.shape[0] != batch_size or mask.shape[-1] != dim:
        raise ValueError(f"action_dim_mask has incompatible shape {mask.shape}, expected (*,{dim}) with batch {batch_size}")
    if mask.shape[1] not in (1, seq_len):
        raise ValueError(f"action_dim_mask has incompatible seq_len {mask.shape[1]}, expected 1 or {seq_len}")
    return mask.to(device=device)

def masked_mse(pred: torch.Tensor, target: torch.Tensor, dim_mask: torch.Tensor | None, time_mask: torch.Tensor | None) -> torch.Tensor:
    sq = (pred - target).pow(2)

    if dim_mask is None:
        per_token = sq.mean(dim=-1)
    else:
        dim_mask = dim_mask.to(dtype=sq.dtype, device=sq.device)
        sq = sq * dim_mask
        denom = dim_mask.sum(dim=-1).clamp_min(1.0)
        per_token = sq.sum(dim=-1) / denom

    if time_mask is None:
        return per_token.mean()

    time_mask = time_mask.to(dtype=per_token.dtype, device=per_token.device)
    weighted = per_token * time_mask
    denom = time_mask.sum().clamp_min(1.0)
    return weighted.sum() / denom