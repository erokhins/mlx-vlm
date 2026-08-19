"""Tests for pinning the stable Junie prompt prefix in APC.

The chat endpoint finds the stable prefix boundary before compressed
history and the "## ISSUE DESCRIPTION" user message, maps it to a verified
token position, and the prefill harvest pins that snapshot in memory and
persists it to the APC disk tier — warm-start straight from live traffic,
with no seed request file.
"""

import mlx.core as mx

from mlx_vlm.apc import APCManager, DiskBlockStore
from mlx_vlm.server.junie import prefix_pin
from mlx_vlm.server.junie.prefix_pin import (
    PIN_BOUNDARY_SENTINEL,
    boundary_probe_messages,
    pin_boundary_token_len,
    stable_prefix_messages,
)
from mlx_vlm.tests.test_apc_exact_mode import _make_tiny_qwen35


# ---------------------------------------------------------------- boundary


def _junie_messages():
    return [
        {
            "role": "system",
            # The real system prompt mentions the <issue_description> tag;
            # the boundary detector must not trip on it.
            "content": "## ENVIRONMENT\nYou are Junie. The issue arrives in "
            "<issue_description> tags.",
        },
        {"role": "user", "content": "## GUIDELINES\n<guidelines>hi</guidelines>"},
        {
            "role": "user",
            "content": "## ISSUE DESCRIPTION\n<issue_description>\nfix\n"
            "</issue_description>",
        },
        {"role": "user", "content": "## PROJECT STRUCTURE\nsrc/..."},
    ]


def _compacted_junie_messages():
    messages = _junie_messages()
    return messages[:2] + [
        {
            "role": "user",
            "content": (
                "History processor: The current session included prior "
                "operations, but the history has been compressed."
            ),
        },
        {
            "role": "user",
            "content": (
                "History processor: During the current session, you worked "
                "on the following <previous_issue>...</previous_issue>."
            ),
        },
    ] + messages[2:]


def test_stable_prefix_stops_at_issue_description():
    messages = _junie_messages()
    assert stable_prefix_messages(messages) == messages[:2]


def test_stable_prefix_excludes_compacted_history():
    messages = _compacted_junie_messages()
    assert stable_prefix_messages(messages) == messages[:2]


def test_boundary_probe_excludes_compacted_history():
    messages = _compacted_junie_messages()
    assert boundary_probe_messages(messages) == messages[:2] + [
        {"role": "user", "content": PIN_BOUNDARY_SENTINEL}
    ]


def test_stable_prefix_requires_an_issue_message():
    messages = [m for m in _junie_messages() if "## ISSUE" not in m["content"]]
    assert stable_prefix_messages(messages) is None

    messages = [
        m
        for m in _compacted_junie_messages()
        if "## ISSUE" not in m["content"]
    ]
    assert stable_prefix_messages(messages) is None

    # A leading issue message leaves no stable prefix to pin.
    assert stable_prefix_messages(_junie_messages()[2:]) is None


def test_boundary_probe_appends_sentinel_user_turn():
    messages = _junie_messages()
    probe = boundary_probe_messages(messages)
    # Live traffic can have a system-only prefix; the sentinel user turn is
    # what keeps chat templates that require a user query from raising.
    assert probe == messages[:2] + [
        {"role": "user", "content": PIN_BOUNDARY_SENTINEL}
    ]
    assert boundary_probe_messages(messages[2:]) is None


# ------------------------------------------------ probe -> token boundary


class _FakeGenerator:
    """Just enough of ResponseGenerator for pin_boundary_token_len."""

    def __init__(self, ids_by_text):
        self._ids_by_text = ids_by_text

    def _preprocess_request(self, prompt, images=None, audio=None, videos=None):
        return {"input_ids": mx.array([self._ids_by_text[prompt]])}


def test_pin_boundary_is_the_common_token_prefix(monkeypatch):
    monkeypatch.setattr(prefix_pin, "MIN_PIN_TOKENS", 2)
    gen = _FakeGenerator({"full": [1, 2, 3, 4, 5], "probe": [1, 2, 3, 9, 9, 9]})
    raw_inputs = gen._preprocess_request("full")
    assert pin_boundary_token_len(gen, "probe", raw_inputs) == 3


def test_pin_boundary_rejects_degenerate_prefixes(monkeypatch):
    monkeypatch.setattr(prefix_pin, "MIN_PIN_TOKENS", 2)
    # Below the minimum useful pin size.
    gen = _FakeGenerator({"full": [1, 2, 3, 4], "probe": [1, 9]})
    raw_inputs = gen._preprocess_request("full")
    assert pin_boundary_token_len(gen, "probe", raw_inputs) == 0

    # The probe must not cover the whole prompt — there would be no suffix
    # left to generate from.
    gen = _FakeGenerator({"full": [1, 2, 3], "probe": [1, 2, 3, 4]})
    raw_inputs = gen._preprocess_request("full")
    assert pin_boundary_token_len(gen, "probe", raw_inputs) == 0


def test_pin_boundary_floor_is_agent_prompt_sized():
    assert prefix_pin.MIN_PIN_TOKENS == 128


# --------------------------------------------------- pinned store + disk


class _FakeDisk:
    def __init__(self):
        self.saved = []

    def save_exact_cache(self, cache_hash, token_ids, extra_hash, prompt_cache):
        self.saved.append((int(cache_hash), tuple(token_ids)))

    def has_exact_or_pending(self, cache_hash):
        return any(h == int(cache_hash) for h, _ in self.saved)


PREFIX = [1, 5, 10, 20, 30, 40, 50, 60, 2, 3, 4, 6, 7, 8, 9, 11]
EXT = [12, 13, 14, 15, 16, 17, 18, 19]


def _prefill(lm, tokens, cache=None):
    cache = cache if cache is not None else lm.make_cache()
    lm(mx.array([tokens]), cache=cache)
    mx.eval([c for c in cache if c is not None])
    return cache


def test_pinned_store_pins_session_and_persists_to_disk():
    lm = _make_tiny_qwen35()
    apc = APCManager(num_blocks=64, block_size=16)
    apc.disk = _FakeDisk()

    cache = _prefill(lm, PREFIX)
    assert apc.store_exact_cache(PREFIX, cache, pinned=True)

    sess = next(iter(apc._sessions.values()))
    assert sess.pinned
    # Under the default APC_DISK_EXACT_SCOPE=pinned the snapshot goes to
    # the disk tier because of the pin, with no warmup flag involved.
    assert len(apc.disk.saved) == 1
    assert apc.has_pinned_exact_prefix(PREFIX)
    # Only the stored boundary counts — other lengths still need a store.
    assert not apc.has_pinned_exact_prefix(PREFIX[:8])


def test_unpinned_store_stays_off_disk_and_unpinned():
    lm = _make_tiny_qwen35()
    apc = APCManager(num_blocks=64, block_size=16)
    apc.disk = _FakeDisk()

    cache = _prefill(lm, PREFIX)
    assert apc.store_exact_cache(PREFIX, cache)

    sess = next(iter(apc._sessions.values()))
    assert not sess.pinned
    assert apc.disk.saved == []
    assert not apc.has_pinned_exact_prefix(PREFIX)


def test_conversation_extending_pinned_prefix_gets_own_session():
    lm = _make_tiny_qwen35()
    apc = APCManager(num_blocks=64, block_size=16)
    apc.disk = _FakeDisk()

    cache = _prefill(lm, PREFIX)
    apc.store_exact_cache(PREFIX, cache, pinned=True)

    # The live conversation continues past the boundary; its normal store
    # must not rotate the pinned anchor.
    lm(mx.array([EXT]), cache=cache)
    mx.eval([c for c in cache if c is not None])
    apc.store_exact_cache(PREFIX + EXT, cache)

    sessions = list(apc._sessions.values())
    assert len(sessions) == 2
    pinned = [s for s in sessions if s.pinned]
    assert len(pinned) == 1
    assert pinned[0].token_ids == tuple(PREFIX)
    assert apc.has_pinned_exact_prefix(PREFIX)


# ------------------------------------------------------------- disk cap


def test_disk_prunes_exact_snapshots_beyond_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("APC_DISK_EXACT_MAX", "5")
    store = DiskBlockStore(tmp_path, namespace="t")

    paths = []
    for index in range(7):
        path = store.dir / f"exact_{index:032x}{store.SUFFIX}"
        path.write_bytes(b"x")
        # Deterministic least-recently-used order.
        import os

        os.utime(path, (1000 + index, 1000 + index))
        store._exact_index[index] = path
        paths.append(path)

    assert store._prune_exact_entries() == 2
    assert sorted(store._exact_index) == [2, 3, 4, 5, 6]
    assert [p.exists() for p in paths] == [False, False, True, True, True, True, True]
