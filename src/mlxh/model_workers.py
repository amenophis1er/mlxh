"""Concurrent model-worker lifecycle primitives for the API supervisor.

The manager deliberately knows nothing about MLX or worker transports. A
``start_worker`` callback must return only after the worker is ready, and a
``stop_worker`` callback owns terminating and reaping that worker.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

log = logging.getLogger(__name__)


@dataclass
class _Worker:
    model: str
    handle: Any
    loaded_at: float
    active_requests: int = 0
    expires_at: float | None = None


class ModelWorkers:
    """Own independent worker processes, with optional idle expiry.

    Different models use different lifecycle locks, so they can start and
    serve concurrently. Leases cover the complete proxied request (including
    response streaming); an idle timer starts only after the final lease ends.
    No implicit worker-count limit is applied.
    """

    def __init__(
        self,
        start_worker: Callable[[str], Any],
        stop_worker: Callable[[Any], None],
        *,
        idle_timeout_s: float = 300,
        reap_interval_s: float = 1,
    ):
        if idle_timeout_s < 0:
            raise ValueError("idle_timeout_s must be non-negative")
        if reap_interval_s <= 0:
            raise ValueError("reap_interval_s must be positive")
        self._start_worker = start_worker
        self._stop_worker = stop_worker
        self.idle_timeout_s = idle_timeout_s
        self._workers: dict[str, _Worker] = {}
        self._guard = threading.RLock()
        self._condition = threading.Condition(self._guard)
        self._model_locks: dict[str, threading.Lock] = {}
        self._closed = False
        self._wake = threading.Event()
        self._reap_interval_s = reap_interval_s
        self._reaper = threading.Thread(
            target=self._reaper_loop, name="mlxh-model-reaper", daemon=True,
        )
        self._reaper.start()

    def _model_lock(self, model: str) -> threading.Lock:
        with self._guard:
            return self._model_locks.setdefault(model, threading.Lock())

    @contextmanager
    def lease(self, model: str) -> Iterator[Any]:
        """Yield a ready worker, starting it once if necessary."""
        if not model:
            raise ValueError("model must not be empty")
        model_lock = self._model_lock(model)
        with model_lock:
            with self._guard:
                if self._closed:
                    raise RuntimeError("model worker manager is shutting down")
                worker = self._workers.get(model)
                if worker is not None and self._handle_exited(worker.handle):
                    # A crashed worker must not poison this model's route until
                    # its normal idle timeout expires. Existing leases still
                    # hold their old record and will unwind independently.
                    del self._workers[model]
                    worker = None
            if worker is None:
                handle = self._start_worker(model)
                worker = _Worker(model, handle, time.monotonic())
                with self._guard:
                    if self._closed:
                        self._stop_worker(handle)
                        raise RuntimeError("model worker manager is shutting down")
                    # A per-model lock makes this the unique live instance.
                    self._workers[model] = worker
            with self._guard:
                if self._closed:
                    raise RuntimeError("model worker manager is shutting down")
                worker.active_requests += 1
                worker.expires_at = None
        try:
            yield worker.handle
        finally:
            with self._condition:
                current = self._workers.get(model)
                if current is worker:
                    worker.active_requests -= 1
                    if worker.active_requests == 0:
                        worker.expires_at = time.monotonic() + self.idle_timeout_s
                    self._condition.notify_all()
                    self._wake.set()

    def reap(self, *, now: float | None = None) -> list[str]:
        """Stop and remove expired idle workers; exposed for deterministic tests."""
        now = time.monotonic() if now is None else now
        with self._guard:
            candidates = [
                (name, worker) for name, worker in self._workers.items()
                if worker.active_requests == 0
                and worker.expires_at is not None
                and worker.expires_at <= now
            ]
        stopped = []
        for name, worker in candidates:
            with self._model_lock(name):
                with self._guard:
                    current = self._workers.get(name)
                    if (current is not worker or worker.active_requests
                            or worker.expires_at is None or worker.expires_at > now):
                        continue
                    del self._workers[name]
                try:
                    self._stop_worker(worker.handle)
                except Exception:
                    log.exception("failed to stop worker for model %s", name)
                stopped.append(name)
        return stopped

    def snapshot(self) -> list[dict[str, Any]]:
        """Return manager-owned lifecycle state, without exposing handles."""
        now = time.monotonic()
        with self._guard:
            return [
                {
                    "model": worker.model,
                    "state": "busy" if worker.active_requests else "idle",
                    "active_requests": worker.active_requests,
                    "loaded_for_s": round(now - worker.loaded_at, 3),
                    "idle_expires_in_s": (
                        max(0.0, round(worker.expires_at - now, 3))
                        if worker.expires_at is not None else None
                    ),
                    **self._handle_diagnostics(worker.handle),
                }
                for worker in sorted(self._workers.values(), key=lambda item: item.model)
            ]

    def handles(self) -> dict[str, Any]:
        """Return current handles for supervisor-owned health/diagnostic reads."""
        with self._guard:
            return {name: worker.handle for name, worker in self._workers.items()}

    @staticmethod
    def _handle_exited(handle: Any) -> bool:
        process = getattr(handle, "process", None)
        poll = getattr(process, "poll", None)
        return callable(poll) and poll() is not None

    @staticmethod
    def _handle_diagnostics(handle: Any) -> dict[str, Any]:
        """Copy safe worker diagnostics when the launcher exposes them."""
        info = handle if isinstance(handle, dict) else getattr(handle, "info", None)
        if not isinstance(info, dict):
            return {}
        runtime = info.get("runtime") or {}
        mlx = info.get("mlx") or {}
        return {
            "model_kind": info.get("model_kind"),
            "pid": runtime.get("pid"),
            "queue_depth": runtime.get("queue_depth"),
            "requests": runtime.get("requests"),
            "active_memory_bytes": mlx.get("active_memory_bytes"),
            "cache_memory_bytes": mlx.get("cache_memory_bytes"),
            "peak_memory_bytes": mlx.get("last_peak_memory_bytes"),
        }

    def _reaper_loop(self):
        while not self._closed:
            self._wake.wait(self._reap_interval_s)
            self._wake.clear()
            if not self._closed:
                self.reap()

    def close(self, *, wait_timeout_s: float | None = 30):
        """Reject new leases, wait for active requests, then stop all workers."""
        deadline = (None if wait_timeout_s is None
                    else time.monotonic() + max(0, wait_timeout_s))
        with self._condition:
            self._closed = True
            self._wake.set()
            while any(worker.active_requests for worker in self._workers.values()):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                self._condition.wait(remaining)
            workers = list(self._workers.values())
            self._workers.clear()
        self._reaper.join(timeout=self._reap_interval_s + 1)
        for worker in workers:
            try:
                self._stop_worker(worker.handle)
            except Exception:
                log.exception("failed to stop worker for model %s", worker.model)
