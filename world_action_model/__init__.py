_MODEL_EXPORTS = {
    "CasualWorldActionTransformer",
    "CasualWorldActionTransformer_MoT",
    "WanRotaryPosEmbed1D",
    "WanTransformer3DModel",
}
_TRAINER_EXPORTS = {
    "CasualWATrainer",
    "CasualWATrainerMoT",
    "CasualWATrainerPretrain",
}
_TRANSFORM_EXPORTS = {"WALeRobotTransformsPretrain", "WATransforms"}
_DATASET_EXPORTS = {"WAMLeRobotDataset"}

__all__ = sorted(_MODEL_EXPORTS | _TRAINER_EXPORTS | _TRANSFORM_EXPORTS | _DATASET_EXPORTS)


def __getattr__(name):
    if name in _MODEL_EXPORTS:
        from . import models

        return getattr(models, name)
    if name in _TRAINER_EXPORTS:
        from . import trainer

        return getattr(trainer, name)
    if name in _TRANSFORM_EXPORTS:
        from . import transformers

        return getattr(transformers, name)
    if name in _DATASET_EXPORTS:
        from . import datasets

        return getattr(datasets, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
