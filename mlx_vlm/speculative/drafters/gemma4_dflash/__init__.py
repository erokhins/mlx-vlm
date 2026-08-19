from ..qwen3_dflash.config import DFlashConfig
from ..qwen3_dflash.dflash import DFlash2DraftModel, DFlashDraftModel, DFlashKVCache


class Gemma4DFlashConfig(DFlashConfig):
    @classmethod
    def from_dict(cls, params: dict) -> "Gemma4DFlashConfig":
        flat = dict(params)
        dflash_cfg = dict(flat.get("dflash_config", None) or {})
        dflash_cfg.setdefault("mask_token_id", 4)
        flat["dflash_config"] = dflash_cfg
        return super().from_dict(flat)

    from_hf_dict = from_dict


class Gemma4DFlashDraftModel(DFlashDraftModel):
    pass


class Gemma4DFlash2DraftModel(DFlash2DraftModel):
    pass


def Model(config):
    if getattr(config, "is_dflash2", False):
        return Gemma4DFlash2DraftModel(config)
    return Gemma4DFlashDraftModel(config)


ModelConfig = Gemma4DFlashConfig

__all__ = [
    "Gemma4DFlashConfig",
    "Gemma4DFlashDraftModel",
    "Gemma4DFlash2DraftModel",
    "DFlashKVCache",
    "Model",
    "ModelConfig",
]
