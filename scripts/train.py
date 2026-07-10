"""Entry for the WAM giga-world-policy-0-5 project.

The project loads configs from the local ./configs package by default and uses
./third_party/giga-{train,datasets,models} as framework sources.
"""

import os
import sys

# ------------------------------------------------------------------ sys.path --
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THIRD_PARTY_ROOT = os.path.join(_PROJECT_ROOT, 'third_party')
_GIGA_ROOT = os.environ.get('WAM_GIGA_ROOT', _THIRD_PARTY_ROOT)
_GIGA_DATASETS_PATH = os.environ.get('WAM_GIGA_DATASETS_PATH', os.path.join(_GIGA_ROOT, 'giga-datasets'))
_GIGA_MODELS_PATH = os.environ.get('WAM_GIGA_MODELS_PATH', os.path.join(_GIGA_ROOT, 'giga-models'))
_GIGA_TRAIN_PATH = os.environ.get('WAM_GIGA_TRAIN_PATH', os.path.join(_GIGA_ROOT, 'giga-train'))

for path in [
    _PROJECT_ROOT,
    _GIGA_DATASETS_PATH,
    _GIGA_MODELS_PATH,
    _GIGA_TRAIN_PATH,
]:
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
        old_pp = os.environ.get('PYTHONPATH', '')
        os.environ['PYTHONPATH'] = f'{path}:{old_pp}' if old_pp else path

# Keep giga-models imports lightweight during training bootstrap.
os.environ.setdefault('GIGA_MODELS_LIGHT_IMPORT', '1')

def _resolve_config(config: str) -> str:
    if not config:
        return config

    config = os.path.expanduser(config)

    if config.startswith('configs.'):
        if config.endswith('.py'):
            config = config[:-3]
        if not config.endswith('.config'):
            return f'{config}.config'
        return config

    config_name = os.path.basename(config)
    if config.endswith('.py') and os.path.dirname(config) in {'', 'configs'}:
        return os.path.join('configs', config_name)

    if config.startswith('configs/'):
        return config

    if os.path.isabs(config):
        return config

    project_relative = os.path.join(_PROJECT_ROOT, config)
    if os.path.exists(project_relative):
        return project_relative

    if config.endswith('.json') or config.endswith('.yaml') or config.endswith('.yml'):
        return os.path.join(_PROJECT_ROOT, 'configs', config)

    if config.endswith('.config'):
        return f'configs.{config}'

    return f'configs.{config}.config'


# ------------------------------------------------------------------ launcher --
import tyro
from giga_train import launch_from_config


def _register_world_action_model() -> None:
    import world_action_model.datasets  # noqa: F401
    import world_action_model.trainer  # noqa: F401
    import world_action_model.transforms  # noqa: F401


def _maybe_disable_cudnn() -> None:
    if os.environ.get('WAM_DISABLE_CUDNN', '0') != '1':
        return
    import torch

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    print('[train.py] WAM_DISABLE_CUDNN=1 -> cuDNN disabled (bit-parity verify mode)')


def train(config: str = 'giga_world_policy_0_5_agilex_finetune'):
    _maybe_disable_cudnn()
    _register_world_action_model()
    launch_from_config(_resolve_config(config))


if __name__ == '__main__':
    tyro.cli(train)
