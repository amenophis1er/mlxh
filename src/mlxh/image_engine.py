"""Serial, instrumented diffusion execution on one MLX worker thread."""

import os
import time
import traceback
from importlib.metadata import version

from .engine import EngineError, EngineLifecycle
from .image_models import image_metadata, MFLUX_VERSION
from .images import ImageError, ImageGenerationRequest, ImageGenerationResult, encode_image


class ImageEngine(EngineLifecycle):
    model_kind = "image"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stats["images_generated"] = 0
        self._stopping = False
        self.runner = None
        self.metadata = image_metadata(self.model_path) or {}

    def submit(self, request):
        if not isinstance(request, ImageGenerationRequest):
            raise EngineError(400, "this model generates images and does not support chat")
        if self._stopping:
            raise EngineError(503, "image engine is stopping")
        if not self.ready.is_set() or self.failed:
            raise EngineError(503, "image engine is not ready")
        return super().submit(request)

    def stop(self):
        self._stopping = True
        with self._requests_lock:
            for job in self._requests.values():
                job.cancelled.set()
        self.jobs.put(None)
        if self._thread:
            self._thread.join(timeout=30)

    def _worker(self):
        try:
            import mlx.core as mx
            from .image_runner import ImageRunner

            self._configure_memory(mx)
            self.runner = ImageRunner(self.model_path)
            self._snapshot_peak(mx)
            self.ready.set()
            print(f"Ready: {self.model_id} (image generation)", flush=True)
        except BaseException as exc:
            self.failed = exc
            traceback.print_exc()
            self.ready.set()
            if self.exit_on_load_failure:
                os._exit(1)
            return
        while True:
            job = self.jobs.get()
            if job is None:
                break
            self._execute(job, mx)
            job = None  # do not retain the last prompt while waiting for work

    def _execute(self, job, mx):
        request = job.request
        started = time.monotonic()
        steps = request.steps or self.settings.get("image_steps", 0) or self.runner.default_steps
        result = None
        image = None
        outcome = "completed"
        error = None
        was_active = not job.cancelled.is_set()

        def check(step):
            if job.cancelled.is_set():
                raise ImageError(499, "generation cancelled", code="cancelled")
            timeout = self.settings["gen_timeout_s"]
            if timeout and time.monotonic() - started >= timeout:
                job.cancelled.set()
                raise ImageError(504, "image generation timed out", code="generation_timeout")
            with self._stats_lock:
                if self._stats["current_request"]:
                    self._stats["current_request"]["current_step"] = step

        try:
            check(0)
            print(f"[image] start: {request.width}x{request.height}, steps={steps}, "
                  f"seed={request.seed}, prompt_chars={len(request.prompt)}", flush=True)
            with self._stats_lock:
                self._stats["busy"] = True
                self._stats["current_request"] = {
                    "id": job.request_id, "source": request.source,
                    "started_monotonic": started, "width": request.width,
                    "height": request.height, "steps": steps, "seed": request.seed,
                    "current_step": 0,
                }
            self.runner.validate_prompt(request.prompt)
            check(0)
            image = self.runner.generate(request, steps, check)
            check(steps)
            encoded = encode_image(image, request)
            check(steps)
            result = ImageGenerationResult(encoded, request, steps, time.monotonic() - started)
        except ImageError as exc:
            outcome = {499: "cancelled", 504: "timed_out", 400: "invalid_request"}.get(exc.status_code, "failed")
            # A traceback retains the callback frames, prompt, and MLX latents.
            # Only the sanitized transport error should cross the worker boundary.
            error = exc.with_traceback(None)
        except Exception:
            traceback.print_exc()
            outcome = "failed"
            error = ImageError(500, "image generation failed; see server log", code="generation_failed")
        finally:
            if image is not None:
                try:
                    image.close()
                except Exception:
                    traceback.print_exc()
            if was_active:
                self._snapshot_peak(mx)
            elapsed = time.monotonic() - started if was_active else 0.0
            with self._stats_lock:
                self._stats["busy"] = False
                self._stats["current_request"] = None
                self._stats["images_generated"] += int(result is not None)
                self._stats["last_request"] = {
                    "id": job.request_id, "source": request.source, "outcome": outcome,
                    "width": request.width, "height": request.height, "steps": steps,
                    "seed": request.seed, "generation_s": round(elapsed, 3),
                }
            self._finish_job(job)
            print(f"[image] end: {outcome} in {elapsed:.2f}s", flush=True)
            # Publish only after diagnostics and MLX cleanup are complete.
            job.out.put(error if error is not None else result)

    def snapshot(self):
        with self._stats_lock:
            stats = dict(self._stats)
            current = dict(stats["current_request"]) if stats["current_request"] else None
        if current:
            current["elapsed_s"] = round(time.monotonic() - current.pop("started_monotonic"), 3)
        memory = {"active_memory_bytes": None, "cache_memory_bytes": None,
                  "last_peak_memory_bytes": stats["last_peak_memory_bytes"]}
        try:
            import mlx.core as mx
            memory.update(active_memory_bytes=int(mx.get_active_memory()),
                          cache_memory_bytes=int(mx.get_cache_memory()))
        except Exception:
            pass
        runner = self.runner
        default_width = getattr(runner, "default_width", None)
        default_height = getattr(runner, "default_height", None)
        default_size = (f"{default_width}x{default_height}"
                        if default_width is not None and default_height is not None else None)
        return {
            "model": self.model_id, "model_kind": self.model_kind, "settings": self.settings,
            "capabilities": {"images": False, "image_generation": True,
                             "image_edits": False, "chat_protocol": None},
            "image": {**self.metadata, "backend_version": MFLUX_VERSION,
                      "formats": ["png", "jpeg", "webp"], "default_size": default_size,
                      "default_steps": getattr(runner, "default_steps", None),
                      "max_image_pixels": self.settings.get("max_image_pixels", 4194304),
                      "prompt_limits": getattr(runner, "prompt_limits", {})},
            "mlx": memory,
            "runtime": {"engine_version": 1, "mlxh_version": version("mlxh"),
                        "pid": os.getpid(), "uptime_s": int(time.monotonic() - self.started),
                        "ready": self.ready.is_set() and self.failed is None,
                        "busy": stats["busy"], "queue_depth": self.jobs.qsize(),
                        "requests": stats["requests"], "images_generated": stats["images_generated"],
                        "current_request": current, "last_request": stats["last_request"]},
        }
