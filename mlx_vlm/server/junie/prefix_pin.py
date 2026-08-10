"""Pin the stable Junie prompt prefix's KV in the APC cache.

Junie's chat/completions bodies share a long stable prefix: the system
message (plus tool schemas the template renders into it). The prefix ends
before either compressed history or the user message that opens with
``## ISSUE DESCRIPTION``. The chat endpoint uses this boundary to pin that
prefix's KV snapshot in APC during the request's own prefill and persist it
to the APC disk tier, which is what makes the first request of a new session
— including right after a restart — start warm instead of re-prefilling
~15k tokens.

Enabled by ``MLX_VLM_PIN_STABLE_PREFIX`` (the number of snapshots the
disk tier keeps, exported by the launcher from the ``pin_stable_prefix``
config; the same count goes to ``APC_DISK_EXACT_MAX``).
"""

import logging
from typing import Any, List, Optional


logger = logging.getLogger("mlx_vlm.server")


# Junie renders the issue as a user message opening with this header. The
# <issue_description> tag itself is no boundary marker — the system prompt
# mentions it too.
ISSUE_HEADER = "## ISSUE DESCRIPTION"
# History compression inserts dynamic user messages before the issue.
HISTORY_PROCESSOR_PREFIX = "History processor:"


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def stable_prefix_messages(messages: List[Any]) -> Optional[List[Any]]:
    """The stable messages before compressed history and the issue, or None."""
    stable_end = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return None
        text = _message_text(message.get("content")).lstrip()
        if text.startswith(HISTORY_PROCESSOR_PREFIX) and stable_end is None:
            stable_end = index
        if text.startswith(ISSUE_HEADER):
            boundary = stable_end if stable_end is not None else index
            return messages[:boundary] if boundary else None
    return None


# Private-use unicode brackets so no real issue text can share a first
# character with the sentinel — the probe's rendering diverges from the
# real prompt exactly where the issue message's content begins.
PIN_BOUNDARY_SENTINEL = "pin-boundary"


def boundary_probe_messages(messages: List[Any]) -> Optional[List[Any]]:
    """The stable prefix plus a sentinel user turn, or None.

    Rendering the bare prefix does not work: Junie's live traffic has only
    the system message before the issue message, and Qwen's chat template
    refuses a conversation without a user query. The sentinel turn keeps
    the template happy; the caller finds the pin boundary as the longest
    common token prefix between the real prompt and this probe's
    rendering, which by construction ends just inside the message that
    follows the stable prefix.
    """
    prefix = stable_prefix_messages(messages)
    if not prefix:
        return None
    return prefix + [{"role": "user", "content": PIN_BOUNDARY_SENTINEL}]


def stable_prefix_probe_text(
    messages: List[Any],
    formatted_prompt: Any,
    has_media: bool,
    processor: Any,
    config: Any,
    tools: Any,
    template_kwargs: dict,
) -> Optional[str]:
    """The rendered boundary probe for this request, or None.

    Rendered with the same template arguments as the real prompt so the
    two tokenizations agree over the whole stable region. Never raises —
    a pin is an optimization, the request must not fail over it.
    """
    try:
        if has_media or not isinstance(formatted_prompt, str):
            return None
        probe_messages = boundary_probe_messages(messages)
        if not probe_messages:
            return None
        from ...prompt_utils import apply_chat_template

        probe_text = apply_chat_template(
            processor,
            config,
            probe_messages,
            tools=tools,
            **(template_kwargs or {}),
        )
        return probe_text if isinstance(probe_text, str) else None
    except Exception:
        logger.warning("Stable-prefix pin setup failed", exc_info=True)
        return None


# Pins below this many tokens are not worth a pinned session and a disk
# snapshot slot; a healthy agent prefix is thousands of tokens.
MIN_PIN_TOKENS = 128


def pin_boundary_token_len(generator, probe_text: str, raw_inputs: dict) -> int:
    """Longest common token prefix of the prompt and the probe, or 0.

    The probe is tokenized through the generator's own preprocessing path.
    The two token streams agree over the whole stable region (plus
    whatever of the following turn header survives tokenizer boundary
    merging) and diverge where the sentinel departs from the request's
    real content, so every common token is stable across sessions — and a
    token-id comparison can never land inside a token. Caller must hold
    the generator's tokenizer lock.
    """
    try:
        full_ids = raw_inputs.get("input_ids")
        if full_ids is None:
            return 0
        cache = getattr(generator, "_pin_probe_ids_cache", None)
        if cache is None:
            cache = generator._pin_probe_ids_cache = {}
        probe_list = cache.get(probe_text)
        if probe_list is None:
            probe_ids = generator._preprocess_request(probe_text).get("input_ids")
            if probe_ids is None:
                return 0
            probe_list = probe_ids.reshape(-1).tolist()
            cache.clear()  # one stable prefix at a time is the norm
            cache[probe_text] = probe_list
        full_list = full_ids.reshape(-1).tolist()
        common = 0
        for a, b in zip(full_list, probe_list):
            if a != b:
                break
            common += 1
        if MIN_PIN_TOKENS <= common < len(full_list):
            return common
        return 0
    except Exception:
        logger.warning("Pin boundary probe tokenization failed", exc_info=True)
        return 0
