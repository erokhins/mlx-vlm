import inspect
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ....models.base import BaseModelConfig

_DFLASH_NESTED_KEYS = (
    "mask_token_id",
    "target_layer_ids",
    "runtime_block_size",
    "draft_window_size",
    "block_size",
    "conv_kernel_size",
    "conv_group_size",
    "selector_rank",
    "selector_top_k",
)


@dataclass
class DFlashConfig(BaseModelConfig):
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    rope_theta: float = 10000000.0
    rope_scaling: Optional[dict[str, Any]] = None
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    block_size: int = 16
    mask_token_id: int = 248070
    target_layer_ids: List[int] = field(default_factory=lambda: [1, 8, 15, 22, 29])
    num_target_layers: int = 32
    layer_types: List[str] = field(default_factory=list)
    sliding_window: Optional[int] = None
    final_logit_softcapping: Optional[float] = None
    runtime_block_size: int | None = None
    draft_window_size: int | None = None
    conv_kernel_size: int = 0
    conv_group_size: int = 0
    selector_rank: int = 0
    selector_top_k: int = 0
    architectures: List[str] = field(default_factory=list)

    @property
    def is_dflash2(self) -> bool:
        if any("DFlash2" in str(name) for name in (self.architectures or [])):
            return True
        return int(self.conv_kernel_size or 0) > 0 and int(self.selector_rank or 0) > 0

    @classmethod
    def from_dict(cls, params: dict) -> "DFlashConfig":
        flat = dict(params)
        dflash_cfg = flat.pop("dflash_config", None) or {}
        for key in _DFLASH_NESTED_KEYS:
            if key in dflash_cfg:
                flat[key] = dflash_cfg[key]
        if "target_layer_ids" in flat:
            flat["target_layer_ids"] = list(flat["target_layer_ids"])
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
