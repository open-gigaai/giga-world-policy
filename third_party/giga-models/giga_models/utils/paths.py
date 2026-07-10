import os


def get_model_dir() -> str:
    default_model_dir = os.path.join(os.path.expanduser('~'), '.cache', 'giga_models', 'models')
    return os.environ.get('GIGA_MODELS_CACHE', default_model_dir)


def get_repo_dir() -> str:
    default_repo_dir = os.path.join(os.path.expanduser('~'), '.cache', 'giga_models', '3rdparty')
    return os.environ.get('GIGA_MODELS_REPO_CACHE', default_repo_dir)


def get_huggingface_model_path(model_name: str) -> str:
    model_dir = os.environ['HUGGINGFACE_HUB_CACHE']
    if '/' in model_name and len(model_name.split('/')) == 2:
        local_model_name = 'models--' + model_name.replace('/', '--')
        hf_model_name = model_name
    elif '--' in model_name and model_name.startswith('models'):
        local_model_name = model_name
        hf_model_name = model_name[8:]
        hf_model_name = hf_model_name.replace('--', '/')
    else:
        local_model_name = model_name
        hf_model_name = None
    model_path = os.path.join(model_dir, local_model_name)
    if not os.path.exists(model_path):
        raise ValueError(f'{model_path} does not exist')
    if os.path.exists(os.path.join(model_path, 'refs')):
        return hf_model_name
    else:
        return model_path


def get_model_path(model_name_or_path: str | None) -> str | None:
    """Resolve a model identifier to a local path.

    If the value is None or already a valid path, return as-is. If it is a
    relative identifier, try resolving it under the user's model cache,
    then under the HuggingFace hub cache.
    """
    if model_name_or_path is None or os.path.exists(model_name_or_path):
        return model_name_or_path
    if os.path.isabs(model_name_or_path):
        raise ValueError(f'{model_name_or_path} does not exist')
    model_dir = get_model_dir()
    model_path = os.path.join(model_dir, model_name_or_path)
    if os.path.exists(model_path):
        return model_path
    return get_huggingface_model_path(model_name_or_path)
