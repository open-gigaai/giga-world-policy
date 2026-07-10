# GigaWorld-Policy-0.5

> **GigaWorld-Policy-0.5: An Enhanced Framework for Action-Centered World Action Model**

**arXiv:** Coming soon

**Project Page:** Coming soon

## 📖 Overview

World Action Models (WAMs) improve robot policy learning by jointly modeling robot actions and future visual observations, allowing future scene evolution to provide dense supervision for physically grounded action generation. However, many WAMs require explicit future-video generation or predictive rollout during inference, leading to high computational cost and limiting real-time closed-loop deployment. GigaWorld-Policy addresses this issue with an action-centered formulation, where future visual dynamics are used during training while action-only decoding is used at inference time. Building upon this framework, we present GigaWorld-Policy-0.5, an enhanced action-centered WAM designed for more efficient robot control. GigaWorld-Policy-0.5 introduces a Mixture-of-Transformers architecture that separates visual dynamics modeling and action generation into specialized experts, reducing active computation during action-only inference. During pretraining, we further combine action-conditioned world modeling with WAM training to strengthen the coupling between visual dynamics and robot actions. In addition, we employ an agent-based AutoResearch pipeline to systematically search training configurations and derive a reliable training recipe with reduced manual intervention. Experiments and ablations show that GigaWorld-Policy-0.5 preserves the training benefits of future visual dynamics while improving inference efficiency for robot control.

## 🛠️ Installation

### 1. Create Conda Environment

```bash
conda create -n gigaworld-policy python=3.11 -y
conda activate gigaworld-policy
```

### 2. Install Dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

```bash
python -m pip install --no-deps -e ./third_party/giga-train
python -m pip install --no-deps -e ./third_party/giga-models
python -m pip install --no-deps -e ./third_party/giga-datasets
```

## 📊 Data Preprocessing

Before training, compute the normalization statistics and pre-compute the T5 text embeddings for your LeRobot v3.0 dataset.

### 1. Compute Normalization Statistics

```bash
python scripts/compute_wam_task_norm.py \
  --data-root <lerobot_v3_dataset_root> \
  --output <norm_stats_json> \
  --model-dim 32 \
  --action-horizon 48
```

### 2. Compute T5 Embeddings

```bash
python scripts/compute_t5_embedding.py \
  --root <lerobot_v3_dataset_root> \
  --wan_path <wan_t5_model_root> \
  --device cuda \
  --t5_folder_name t5_embedding
```

## 📦 Model Download

Pretrained weights: Coming soon.

## 🚀 Training

Set the dataset, normalization, WAN base model, and initialization checkpoint paths before launching post-training.

### GigaWorld-Policy-0.5

```bash
export GWP_AGILEX_DATA_PATHS=<lerobot_v3_dataset_root>
export GWP_NORM_STATS_PATH=<norm_stats_json>
export GWP_WAN_PRETRAINED=<wan_diffusers_model_or_path>
export GWP_T5_LOAD_FROM=dir
export GWP_T5_EMBEDDING_DIR=<lerobot_v3_dataset_root>/t5_embedding
export GWP05_TRANSFORMER_PRETRAINED=<gwp05_init_transformer_checkpoint>
export GWP05_PROJECT_DIR=<output_dir_for_gwp05>

python scripts/train.py --config giga_world_policy_0_5_agilex_finetune
```

### GigaWorld-Policy-0

```bash
export GWP_AGILEX_DATA_PATHS=<lerobot_v3_dataset_root>
export GWP_NORM_STATS_PATH=<norm_stats_json>
export GWP_WAN_PRETRAINED=<wan_diffusers_model_or_path>
export GWP_T5_LOAD_FROM=dir
export GWP_T5_EMBEDDING_DIR=<lerobot_v3_dataset_root>/t5_embedding
export GWP0_TRANSFORMER_PRETRAINED=<gwp0_init_transformer_checkpoint>
export GWP0_PROJECT_DIR=<output_dir_for_gwp0>

python scripts/train.py --config giga_world_policy_0_agilex_finetune
```

## 🚀 Inference

Open-loop inference uses a server/client split. Start the server in one terminal, then run the client in another terminal.

```bash
export CHECKPOINT=<trained_transformer_checkpoint>
export NORM_STATS=<norm_stats_json>
export DATA_PATH=<lerobot_v3_dataset_root>
export BASE_MODEL=<wan_diffusers_model_or_path>
```

### GigaWorld-Policy-0.5

Terminal 1:

```bash
cd scripts
./run_inference_openloop.sh server
```

Terminal 2:

```bash
cd scripts
./run_inference_openloop.sh client
```

### GigaWorld-Policy-0

Terminal 1:

```bash
cd scripts
./run_inference_openloop_gwp0.sh server
```

Terminal 2:

```bash
cd scripts
./run_inference_openloop_gwp0.sh client
```

## 📚 Citation
```bibtex
@article{gigaworld-policy-0.5,
  title={GigaWorld-Policy-0.5: An Enhanced Framework for Action-Centered World Action Model},
  author={GigaAI},
  journal={arXiv},
  year={2026}
}