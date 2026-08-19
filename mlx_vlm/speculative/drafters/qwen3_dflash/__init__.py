from .config import DFlashConfig as ModelConfig
from .dflash import DFlash2DraftModel, DFlashDraftModel, DFlashKVCache


def Model(config):
    if getattr(config, "is_dflash2", False):
        return DFlash2DraftModel(config)
    return DFlashDraftModel(config)


__all__ = [
    "Model",
    "ModelConfig",
    "DFlashDraftModel",
    "DFlash2DraftModel",
    "DFlashKVCache",
]
