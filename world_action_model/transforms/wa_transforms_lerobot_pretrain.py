import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as torch_F
from giga_train import TRANSFORMS
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from world_action_model.transforms.wa_transforms import WATransforms


@TRANSFORMS.register
class WALeRobotTransformsPretrain(WATransforms):
    def __init__(
        self,
        is_train=False,
        dst_size=None,
        num_frames=1,
        fps=16,
        norm_path=None,
        robotype_to_embodiment_id=None,
        robotype_default_embodiment_id=0,
        model_action_dim=None,
        delta_mask_by_embodiment_id=None,
        norm_use_quantiles=False,
        norm_enable_clamp=False,
        image_cfg=None,
        num_views=1,
        view_keys=None,
        view_concat_mode="t",
        state_key="observation.state",
        action_key="action",
        task_key="task",
        emit_action_denorm_scale=False,
        max_prompt_len=64,
        subtask_prob=0,
    ):
        if norm_path is None:
            raise ValueError("norm_path list is none")

        if isinstance(norm_path, (str, os.PathLike)):
            norm_specs = [{"path": str(norm_path), "data_paths": None}]
        else:
            norm_specs = []
            for item in norm_path:
                if isinstance(item, (str, os.PathLike)):
                    norm_specs.append({"path": str(item), "data_paths": None})
                elif isinstance(item, dict):
                    path = item.get("path")
                    if path is None:
                        raise ValueError(f"norm_path entry missing path: {item}")
                    data_paths = item.get("data_paths", None)
                    if data_paths is not None:
                        data_paths = {os.path.normpath(str(p)) for p in data_paths}
                    norm_specs.append({"path": str(path), "data_paths": data_paths})
                else:
                    raise TypeError(f"Unsupported norm_path entry type: {type(item).__name__}")
            if len(norm_specs) == 0:
                raise ValueError("norm_path list is empty")
        norm_paths = [spec["path"] for spec in norm_specs]

        super().__init__(
            is_train=is_train,
            dst_size=dst_size,
            num_frames=num_frames,
            fps=fps,
            norm_path=norm_paths[0],
            image_cfg=image_cfg,
            num_views=num_views,
        )

        self.robotype_default_embodiment_id = int(robotype_default_embodiment_id)
        if robotype_to_embodiment_id is None:
            robotype_to_embodiment_id = {"aloha": 0, "agilex": 0, "agibot": 1, "ur5": 3, "arx5": 4}
        self.robotype_to_embodiment_id = {str(k): int(v) for k, v in dict(robotype_to_embodiment_id).items()}
        self.model_action_dim = None if model_action_dim is None else int(model_action_dim)
        if delta_mask_by_embodiment_id is None:
            raise ValueError("delta_mask_by_embodiment_id must be provided")
        self.delta_mask_by_embodiment_id = {
            int(k): np.asarray(v, dtype=bool) for k, v in dict(delta_mask_by_embodiment_id).items()
        }
        if len(self.delta_mask_by_embodiment_id) == 0:
            raise ValueError("delta_mask_by_embodiment_id must not be empty")

        self.norm_use_quantiles = bool(norm_use_quantiles)
        self.norm_enable_clamp = bool(norm_enable_clamp)

        self.norm_paths = norm_paths
        self.norm_specs = norm_specs
        self.stats_dicts = []
        for json_path in self.norm_paths:
            with open(json_path, "r", encoding="utf-8") as f:
                self.stats_dicts.append(json.load(f))
            print("Loading stats dict from:", json_path)

        if view_keys is None:
            view_keys = [
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            ]
        self.view_keys = list(view_keys)
        self.view_concat_mode = str(view_concat_mode)
        self.state_key = state_key
        self.action_key = action_key
        self.task_key = task_key
        self.emit_action_denorm_scale = bool(emit_action_denorm_scale)
        self.max_prompt_len = int(max_prompt_len)
        self.subtask_prob = float(subtask_prob)
        self._warned_unknown_robotype = False
    
    def _parse_robotype(self, robotype):
        if robotype is None:
            return None
        if isinstance(robotype, bytes):
            robotype = robotype.decode("utf-8", errors="ignore")
        if hasattr(robotype, "item"):
            try:
                robotype = robotype.item()
            except Exception:
                pass
        if isinstance(robotype, str):
            robotype = robotype.strip()
        return robotype

    def _get_embodiment_id(self, data_dict) -> int:
        robotype = self._parse_robotype(data_dict.get("robotype", None))
        if robotype in self.robotype_to_embodiment_id:
            return int(self.robotype_to_embodiment_id[robotype])
        if isinstance(robotype, str):
            robotype_l = robotype.lower()
            if "agibot" in robotype_l and "agibot" in self.robotype_to_embodiment_id:
                return int(self.robotype_to_embodiment_id["agibot"])
            if "aloha" in robotype_l and "aloha" in self.robotype_to_embodiment_id:
                return int(self.robotype_to_embodiment_id["aloha"])
            if "agilex" in robotype_l and "agilex" in self.robotype_to_embodiment_id:
                return int(self.robotype_to_embodiment_id["agilex"])
        if not self._warned_unknown_robotype:
            print(f"Unknown robotype={robotype!r}, fallback to {self.robotype_default_embodiment_id}")
            self._warned_unknown_robotype = True
        return self.robotype_default_embodiment_id

    def _get_stats_dict(self, embodiment_id: int, data_dict=None):
        if data_dict is not None:
            data_path = data_dict.get("data_path")
            if data_path is not None:
                data_path = os.path.normpath(str(data_path))
                for spec, stats_dict in zip(self.norm_specs, self.stats_dicts):
                    data_paths = spec.get("data_paths")
                    if data_paths is not None and data_path in data_paths:
                        return stats_dict
        if not self.stats_dicts:
            return self.stats_dict
        if 0 <= embodiment_id < len(self.stats_dicts):
            return self.stats_dicts[embodiment_id]
        if not self._warned_unknown_robotype:
            print(f"embodiment_id={embodiment_id} out of range for norm_paths (len={len(self.stats_dicts)}), fallback to 0")
            self._warned_unknown_robotype = True
        return self.stats_dicts[0]

    def _get_delta_mask(self, embodiment_id: int):
        return self.delta_mask_by_embodiment_id.get(int(embodiment_id))

    def _to_nchw_uint8(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            if x.shape[0] in (1, 3):
                x = x[None, ...]
            elif x.shape[-1] in (1, 3):
                x = x.permute(2, 0, 1)[None, ...]
            else:
                x = x[None, ...]
        if x.dim() != 4:
            raise ValueError(f"Unexpected image tensor shape: {tuple(x.shape)}")
        if x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dtype != torch.uint8:
            x_f = x.to(dtype=torch.float32)
            x_max = float(x_f.max().item()) if x_f.numel() > 0 else 0.0
            if x_max <= 1.0:
                x_f = x_f * 255.0
            x = x_f.clamp(0.0, 255.0).to(dtype=torch.uint8)
        return x

    def _process_images(self, input_images: torch.Tensor, dst_width: int, dst_height: int) -> torch.Tensor:
        input_images = input_images.to(dtype=torch.float32) / 255.0
        height = int(input_images.shape[2])
        width = int(input_images.shape[3])
        if float(dst_height) / height < float(dst_width) / width:
            new_height = int(round(float(dst_width) / width * height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / height * width))
        input_images = F.resize(input_images, (new_height, new_width), InterpolationMode.BILINEAR)
        x1 = random.randint(0, new_width - dst_width)
        y1 = random.randint(0, new_height - dst_height)
        input_images = F.crop(input_images, y1, x1, dst_height, dst_width)
        input_images = self.normalize(input_images)
        return input_images

    def _to_prompt_tensor(self, prompt_embed) -> torch.Tensor:
        if isinstance(prompt_embed, np.ndarray):
            prompt_embed = torch.from_numpy(prompt_embed)
        return prompt_embed.to(dtype=torch.float32)

    def _build_prompt_embeds(self, data_dict) -> torch.Tensor:
        prompt_embed = data_dict.get("t5_embedding", None)
        assert prompt_embed is not None, "t5_embedding must be provided"
        prompt_embed = self._to_prompt_tensor(prompt_embed)
        subtask_prompt_embed = data_dict.get('subtask_t5_embedding', None)
        if (
            self.is_train
            and subtask_prompt_embed is not None
            and random.random() < self.subtask_prob
        ):
            subtask_prompt_embed = self._to_prompt_tensor(subtask_prompt_embed)
            prompt_embed = torch.cat([prompt_embed, subtask_prompt_embed], dim=0)
        prompt_embed = prompt_embed[:self.max_prompt_len]

        return torch_F.pad(prompt_embed, (0, 0, 0, self.max_prompt_len - prompt_embed.shape[0]), value=0)

    def __call__(self, data_dict):
        if self.dst_size is None:
            raise ValueError("dst_size is required")
        dst_width, dst_height = self.dst_size

        if "robotype" not in data_dict:
            raise KeyError("Missing robotype key")
        embodiment_id = self._get_embodiment_id(data_dict)
        stats_dict = self._get_stats_dict(embodiment_id, data_dict=data_dict)

        if self.num_views == 3:
            if self.view_concat_mode == "t":
                cam_high = self._to_nchw_uint8(torch.as_tensor(data_dict[self.view_keys[0]]))
                cam_left = self._to_nchw_uint8(torch.as_tensor(data_dict[self.view_keys[1]]))
                cam_right = self._to_nchw_uint8(torch.as_tensor(data_dict[self.view_keys[2]]))

                top_h = int(dst_height) // 2
                bottom_h = int(dst_height) - top_h
                left_w = int(dst_width) // 2
                right_w = int(dst_width) - left_w

                cam_high = self._process_images(cam_high, dst_width=int(dst_width), dst_height=top_h)
                cam_left = self._process_images(cam_left, dst_width=left_w, dst_height=bottom_h)
                cam_right = self._process_images(cam_right, dst_width=right_w, dst_height=bottom_h)

                cam_bottom = torch.cat([cam_left, cam_right], dim=-1)
                input_images = torch.cat([cam_high, cam_bottom], dim=-2)
            elif self.view_concat_mode == "horizontal":
                views = []
                for k in self.view_keys[:3]:
                    v = data_dict[k]
                    if isinstance(v, np.ndarray):
                        v = torch.from_numpy(v)
                    if not isinstance(v, torch.Tensor):
                        raise TypeError(f"Unsupported image type for {k}: {type(v)}")
                    v = self._to_nchw_uint8(v)
                    v = self._process_images(v, dst_width=dst_width, dst_height=dst_height)
                    views.append(v)
                input_images = torch.cat(views, dim=-1)
            else:
                raise ValueError(f"Unsupported view_concat_mode={self.view_concat_mode!r}")
        else:
            views = []
            for k in self.view_keys[: self.num_views]:
                v = data_dict[k]
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                v = self._to_nchw_uint8(torch.as_tensor(v))
                v = self._process_images(v, dst_width=dst_width, dst_height=dst_height)
                views.append(v)

            if len(views) == 1:
                input_images = views[0]
            else:
                input_images = torch.cat(views, dim=-1)

        data_dict["input_images"] = input_images

        if self.image_cfg is not None:
            ref_masks, ref_latent_masks = self.mask_generator.get_mask(data_dict["input_images"].shape[0])
            ref_masks = ref_masks[:, None, None, None]
            ref_latent_masks = ref_latent_masks[None, :, None, None]
            ref_images = data_dict["input_images"].clone() * ref_masks
            data_dict["input_ref_images"] = ref_images
            data_dict["input_ref_masks"] = ref_latent_masks

        action = data_dict[self.action_key]
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        state = data_dict[self.state_key]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)

        action = action.to(dtype=torch.float32)
        state = state.to(dtype=torch.float32)

        if action.dim() == 1:
            action = action[None, :]
        if state.dim() == 1:
            state = state[None, :]
        if state.dim() == 2 and state.shape[0] > 1:
            state = state[:1]

        if action.shape[0] != self.num_frames:
            t = int(self.num_frames)
            cur_t = int(action.shape[0])
            if cur_t >= t:
                action = action[:t]
            else:
                pad = torch.zeros((t - cur_t, action.shape[1]), dtype=action.dtype, device=action.device)
                action = torch.cat([action, pad], dim=0)
                
        base = self._get_delta_mask(embodiment_id)
        assert base is not None, f"embodiment_id {embodiment_id} not found in delta_mask_by_embodiment_id"
        effective_dim = int(base.shape[-1])
        state = state[:, :effective_dim]
        action = action[:, :effective_dim]

        assert self.model_action_dim is not None, "model_action_dim must be provided"
        d = int(self.model_action_dim)
        if state.shape[-1] > d:
            state = state[..., :d]
        if state.shape[-1] < d:
            state = torch_F.pad(state, (0, d - int(state.shape[-1])), value=0.0)
        if action.shape[-1] > d:
            action = action[..., :d]
        if action.shape[-1] < d:
            action = torch_F.pad(action, (0, d - int(action.shape[-1])), value=0.0)

        if d > len(base):
            base = np.pad(base, (0, d - len(base)), constant_values=False)
        else:
            base = base[:d]

        mask_t = torch.as_tensor(base, dtype=torch.bool, device=action.device)
        idx = torch.nonzero(mask_t, as_tuple=False).flatten()
        delta = action.clone()
        if idx.numel() > 0:
            delta[:, idx] = action[:, idx] - state[:, idx]

        def _to_1d(x, device):
            return torch.as_tensor(x, dtype=torch.float32, device=device).flatten()

        def _normalize_feature(x, stats, eff_dim: int, device):
            out = torch.zeros_like(x)
            eps = 1e-8
            if self.norm_use_quantiles:
                low = _to_1d(stats["q01"], device)
                high = _to_1d(stats["q99"], device)
                if eff_dim > 0:
                    rng = (high[:eff_dim] - low[:eff_dim]).clamp_min(eps)
                    out[..., :eff_dim] = (x[..., :eff_dim] - low[:eff_dim]) / rng * 2.0 - 1.0
            else:
                mean = _to_1d(stats["mean"], device)
                std = _to_1d(stats["std"], device)
                if eff_dim > 0:
                    out[..., :eff_dim] = (x[..., :eff_dim] - mean[:eff_dim]) / std[:eff_dim].clamp_min(eps)
            if self.norm_enable_clamp:
                out = out.clamp(-1.0, 1.0)
            return out

        norm_stats = stats_dict["norm_stats"]
        norm_state = _normalize_feature(
            state,
            norm_stats["observation.state"],
            eff_dim=effective_dim,
            device=state.device,
        )
        norm_delta = _normalize_feature(
            delta,
            norm_stats["action"],
            eff_dim=effective_dim,
            device=action.device,
        )

        prompt_embeds = self._build_prompt_embeds(data_dict)

        out = {}
        out["fps"] = torch.tensor(self.fps, dtype=torch.float32)
        out["images"] = data_dict["input_images"]
        out["ref_images"] = data_dict.get("input_ref_images", None)
        out["ref_masks"] = data_dict.get("input_ref_masks", None)
        out["prompt_embeds"] = prompt_embeds
        out["action"] = norm_delta
        out["state"] = norm_state
        out["embodiment_id"] = torch.tensor(int(embodiment_id), dtype=torch.long)
        dim = out["action"].shape[-1]
        if self.emit_action_denorm_scale:
            scale = torch.ones(dim, dtype=torch.float32, device=action.device)
            action_stats = norm_stats["action"]
            eps = 1e-8
            if self.norm_use_quantiles:
                low = _to_1d(action_stats["q01"], action.device)
                high = _to_1d(action_stats["q99"], action.device)
                
                scale[:effective_dim] = (high[:effective_dim] - low[:effective_dim]).clamp_min(eps) / 2.0
            else:
                std = _to_1d(action_stats["std"], action.device)
                scale[:effective_dim] = std[:effective_dim].clamp_min(eps)
            out["action_denorm_scale"] = scale

        keys = list(out.keys())
        for k in keys:
            if out[k] is None:
                out.pop(k)

        return out
