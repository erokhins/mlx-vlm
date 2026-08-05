"""Server lifecycle phase tracking for the control-plane endpoints.

The HTTP server starts serving immediately; model loading, seed warmup and
settings-driven model restarts all happen on background threads. This module
is the single source of truth for which phase that machinery is in, so
``GET /status`` can report it and inference paths can reject requests while
the model is being (re)loaded.
"""

import threading
import time
from typing import Optional

PHASE_STARTING = "starting"
PHASE_LOADING_MODEL = "loading_model"
PHASE_WARMING_UP = "warming_up"
PHASE_READY = "ready"
PHASE_RESTARTING = "restarting"
PHASE_ERROR = "error"

# Phases during which the model is not available to serve inference. The
# thread performing the (re)load itself is exempt (it may call
# get_cached_model), as is warming_up: the seed warmup posts a real
# chat-completions request against the server itself. "starting" is not
# busy so that embedding the app without the lifespan (e.g. tests) keeps
# the lazy-load path working.
_MODEL_BUSY_PHASES = (PHASE_LOADING_MODEL, PHASE_RESTARTING)


class LifecycleState:
    def __init__(self):
        self._lock = threading.Lock()
        self._phase = PHASE_STARTING
        self._detail: Optional[str] = None
        self._since = time.time()
        self._loader_thread_id: Optional[int] = None
        self.process_started_at = time.time()

    def set_phase(self, phase: str, detail: Optional[str] = None) -> None:
        with self._lock:
            self._phase = phase
            self._detail = detail
            self._since = time.time()

    def transition(
        self, expected: str, phase: str, detail: Optional[str] = None
    ) -> bool:
        """Set ``phase`` only when the current phase is ``expected``."""
        with self._lock:
            if self._phase != expected:
                return False
            self._phase = phase
            self._detail = detail
            self._since = time.time()
            return True

    def set_loader_thread(self, thread_id: Optional[int]) -> None:
        with self._lock:
            self._loader_thread_id = thread_id

    def phase(self) -> str:
        with self._lock:
            return self._phase

    def busy_phase_for_caller(self) -> Optional[str]:
        """The busy phase blocking this thread from using the model, if any."""
        with self._lock:
            if self._phase not in _MODEL_BUSY_PHASES:
                return None
            if threading.get_ident() == self._loader_thread_id:
                return None
            return self._phase

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "phase": self._phase,
                "detail": self._detail,
                "since_unix": self._since,
            }


lifecycle = LifecycleState()
