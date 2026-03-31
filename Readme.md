# GigaWorld-Policy

> **GigaWorld-Policy: An Efficient Action-Centered World-Action Model**

[![arXiv](https://img.shields.io/badge/arXiv-2603.17240-b31b1b.svg)](https://arxiv.org/abs/2603.17240)

## 📖 Overview

World-Action Models (WAM) initialized from pre-trained video generation backbones have demonstrated remarkable potential for robot policy learning. **GigaWorld-Policy** is an action-centered WAM that learns 2D pixel-action dynamics while enabling efficient action decoding, with optional video generation.

### Key Features

- 🚀 **9x faster** inference compared to Motus (leading WAM baseline)
- 📈 **7% higher** task success rate than Motus
- 💪 **95% improvement** over pi-0.5 on RoboTwin 2.0

## 🛠️ Installation

### 1. Create Conda Environment

```bash
conda create -n gigaworld-policy python==3.11
conda activate gigaworld-policy
```

### 2. Install Dependencies

```bash
pip install ./third_party/giga-train
pip install ./third_party/giga-models
pip install ./third_party/giga-datasets
```

## ⚙️ Configuration

Before training, modify the config file `world_action_model/configs/example.py`:

| Parameter | Description |
|-----------|-------------|
| `models.pretrained` | Path to your pretrained model weights |
| `transform.norm_path` | Path to dataset normalization statistics file |
| `lerobot_data_paths` (data_dir) | Path to your dataset |

## 🚀 Training

```bash
python -m scripts.train --config world_action_model.configs.example.config
```

## 🚀 Inference

We provide an inference server and a simple open-loop evaluation client. Open-loop here means we sample observations (images/state) from an offline dataset and run inference, without executing actions in a real environment to collect the next observations.

### 1. Start Server

```bash
python -m scripts.inference_server \
  --model_id "/path/to/huggingface_model_dir_or_id" \
  --transformer_path "/path/to/transformer_checkpoint_dir" \
  --stats_path "/path/to/norm_stats_delta.json" \
  --t5_embedding_pkl "/path/to/t5_embedding.pt"
```

Optionally, add `--return_images` to enable video visualization during inference (videos will be saved under `--vis_dir`):

```bash
python -m scripts.inference_server \
  --model_id "/path/to/huggingface_model_dir_or_id" \
  --transformer_path "/path/to/transformer_checkpoint_dir" \
  --stats_path "/path/to/norm_stats_delta.json" \
  --t5_embedding_pkl "/path/to/t5_embedding.pt" \
  --return_images \
  --vis_dir "./vis"
```

### 2. Run Open-loop Client

```bash
python -m scripts.inference_client \
  --dataset_paths "/path/to/dataset_dir" \
  --save_dir "./vis"
```

## 📅 Roadmap

| Component | Status |
|-----------|--------|
| Inference Code | ✅ |
| Training Code | ✅ |
| Pre-trained Weights | 🔲 |

## 📚 Citation

```bibtex
@article{ye2026gigaworld,
  title={GigaWorld-Policy: An Efficient Action-Centered World-Action Model},
  author={Ye, Angen and Wang, Boyuan and Ni, Chaojun and Huang, Guan and Zhao, Guosheng and Li, Hao and Li, Hengtao and Li, Jie and Lv, Jindi and Liu, Jingyu and Cao, Min and Li, Peng and Deng, Qiuping and Mei, Wenjun and Wang, Xiaofeng and Chen, Xinze and Zhou, Xinyu and Wang, Yang and Chang, Yifan and Li, Yifan and Zhou, Yukun and Ye, Yun and Liu, Zhichao and Zhu, Zheng},
  journal={arXiv preprint arXiv:2603.17240},
  year={2026}
}
```
