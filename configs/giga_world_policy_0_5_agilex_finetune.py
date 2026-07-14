import os


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value


def _env_required(name: str) -> str:
    value = _env(name)
    if value is None:
        raise RuntimeError(f"{name} must be set for this training config")
    return value


def _env_paths(name: str) -> list[str]:
    raw = _env_required(name)
    paths = [path.strip() for path in raw.split(",") if path.strip()]
    if not paths:
        raise RuntimeError(f"{name} must contain at least one path")
    return paths


def _env_int_list(name: str, default: list[int]) -> list[int]:
    raw = _env(name)
    if raw is None:
        return default
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _build_t5_cfg(data_paths: list[str]) -> tuple[str, dict]:
    t5_load_from = _env("GWP_T5_LOAD_FROM", "dir")
    if t5_load_from == "dir":
        return t5_load_from, dict(
            t5_embedding_dir=_env("GWP_T5_EMBEDDING_DIR", os.path.join(data_paths[0], "t5_embedding")),
            t5_embedding_pattern=_env("GWP_T5_EMBEDDING_PATTERN", "episode_{episode_index:06d}.pt"),
            t5_embedding_key=_env("GWP_T5_EMBEDDING_KEY", "t5_embedding"),
        )
    if t5_load_from == "path":
        return t5_load_from, dict(
            t5_embedding_path=_env_required("GWP_T5_EMBEDDING_PATH"),
            t5_embedding_key=_env("GWP_T5_EMBEDDING_KEY", "t5_embedding"),
        )
    raise ValueError("GWP_T5_LOAD_FROM must be 'dir' or 'path'")


num_frames = 48
debug_single = os.environ.get("WA_DEBUG_SINGLE_GPU", "0") == "1"
gpu_ids = _env_int_list("GWP_GPU_IDS", [0, 1, 2, 3, 4, 5, 6, 7] if not debug_single else [1])
batch_size_per_gpu = int(_env("GWP_BATCH_SIZE_PER_GPU", "16" if not debug_single else "1"))
num_workers = int(_env("GWP_NUM_WORKERS", "8" if not debug_single else "0"))

agilex_data_paths = _env_paths("GWP_AGILEX_DATA_PATHS")
t5_load_from, t5_cfg = _build_t5_cfg(agilex_data_paths)

transformer_pretrained = _env_required("GWP05_TRANSFORMER_PRETRAINED")
project_dir = _env("GWP05_PROJECT_DIR", os.path.join("outputs", "gwp0_5_agilex_finetune"))
wan_pretrained = _env("GWP_WAN_PRETRAINED", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
norm_stats_path = _env_required("GWP_NORM_STATS_PATH")

data_or_config = []
nrom_path = []
view_keys = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
image_frame_offsets = [0, num_frames // 4, num_frames // 2, (3 * num_frames) // 4, num_frames]

nrom_path.append({"path": norm_stats_path, "data_paths": agilex_data_paths})

data_or_config.append(
    dict(
        _class_name="WAMLeRobotDataset",
        data_path=agilex_data_paths,
        data_size=None,
        delta_info={"action": num_frames},
        delta_frames={k: image_frame_offsets for k in view_keys},
        video_backend="pyav",
        t5_load_from=t5_load_from,
        t5_cfg=t5_cfg,
        t5_cache_size=4096,
    )
)

config = dict(
    runners=['world_action_model.CasualWATrainerMoT'],
    project_dir=project_dir,
    launch=dict(
        gpu_ids=gpu_ids,
        distributed_type='DEEPSPEED',
        deepspeed_config=dict(
            deepspeed_config_file='accelerate_configs/zero2.json',
        ),
        until_completion=True,
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=data_or_config,
            batch_size_per_gpu=batch_size_per_gpu,
            num_workers=num_workers,
            pin_memory=True,
            transform=dict(
                type='WALeRobotTransformsPretrain',
                dst_size=(320, 384),
                num_frames=num_frames,
                is_train=True,
                norm_path=nrom_path,
                robotype_to_embodiment_id={
                  'agilex': 0,
                  'agilex_mobile': 0,
                  'agilex_cobot_magic': 0,
                },
                robotype_default_embodiment_id=0,
                model_action_dim=32,
                delta_mask_by_embodiment_id={
                  '0': [True, True, True, True, True, True, False, True, True, True, True, True, True, False],
                },
                norm_use_quantiles=True,
                norm_enable_clamp=False,
                num_views=len(view_keys),
                view_keys=view_keys,
                image_cfg=dict(
                    mask_generator=dict(
                        max_ref_frames=1,
                        start=1,
                        factor=4,
                    ),
                ),
                max_prompt_len=64,
                subtask_prob=0,
            ),
        ),
        test=dict(),
    ),
    models=dict(
        pretrained=wan_pretrained,
        transformer_pretrained=transformer_pretrained,
        strict_load=True,
        transformer=dict(
            added_kv_proj_dim=None,
            attention_head_dim=128,
            cross_attn_norm=True,
            eps=1e-6,
            ffn_dim=14336,
            freq_dim=256,
            image_dim=None,
            in_channels=48,
            num_attention_heads=24,
            num_layers=30,
            out_channels=48,
            patch_size=[1, 2, 2],
            pos_embed_seq_len=None,
            qk_norm='rms_norm_across_heads',
            rope_max_seq_len=1024,
            text_dim=4096,
            action_expert_dim=1024,
            action_ffn_dim=4096,
            in_action_channels=16,
            out_action_channels=16,
            num_embodiments=2,
        ),
        visual_flow_shift=2.0,
        action_flow_shift=5.0,
        expand_timesteps=True,
        enable_gradient_checkpointing=False,
        skip_action_expert=False,
    ),
    optimizers=dict(
        type='CAME8Bit',
        lr=6e-5,
        weight_decay=1e-2,
    ),
    schedulers=dict(
        type='ConstantScheduler',
    ),
    train=dict(
        resume=False,
        max_epochs=0,
        max_steps=50000,
        gradient_accumulation_steps=1,
        mixed_precision='bf16',
        checkpoint_interval=10000,
        checkpoint_total_limit=-1,
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_with='tensorboard',
        log_interval=100,
        with_ema=True,
        activation_checkpointing=False,
        activation_class_names=['WanAttention'],
    ),
    test=dict(),
)
