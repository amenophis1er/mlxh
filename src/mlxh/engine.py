"""Single instrumented inference engine shared by every mlxh transport."""

from __future__ import annotations

import base64
import binascii
import importlib.metadata
import json
import math
import os
import queue
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .loader import load_runner

MAX_DECODED_IMAGE = 25 * 1024 * 1024


class EngineError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class LogprobsOptions:
    top: int = 0
    ids: list[int] = field(default_factory=list)
    allowed: list[int] = field(default_factory=list)
    as_ids: bool = False


@dataclass
class GenerationRequest:
    messages: list[dict]
    source: str
    max_tokens: int = 1024
    tools: list[dict] | None = None
    temperature: float | None = None
    top_p: float | None = None
    thinking: bool | None = None
    logprobs: LogprobsOptions | None = None


@dataclass
class GenerationTerminal:
    finish_reason: str
    outcome: str
    prompt_tokens: int
    output_tokens: int
    generation_s: float
    tps: float


@dataclass
class Job:
    request: GenerationRequest
    out: queue.Queue
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:24]}")
    cancelled: threading.Event = field(default_factory=threading.Event)


class ReasoningParser:
    """Split pre-opened or inline think blocks across arbitrary chunks."""

    def __init__(self, preopened=False):
        self.reasoning = preopened
        self.buffer = ""
        self._trim_answer_newline = False
        self.events = []

    @staticmethod
    def _tag_prefix_tail(text, tag):
        for size in range(min(len(text), len(tag) - 1), 0, -1):
            if tag.startswith(text[-size:]):
                return size
        return 0

    def feed(self, text, final=False):
        self.buffer += text
        answer, reasoning = [], []
        events = []

        def emit(kind, chunk):
            if not chunk:
                return
            if events and events[-1][0] == kind:
                events[-1] = (kind, events[-1][1] + chunk)
            else:
                events.append((kind, chunk))

        def record(kind, chunk):
            if kind == "text" and self._trim_answer_newline:
                chunk = chunk.lstrip("\n")
                if chunk:
                    self._trim_answer_newline = False
            (reasoning if kind == "reasoning" else answer).append(chunk)
            emit(kind, chunk)

        while self.buffer:
            tag = "</think>" if self.reasoning else "<think>"
            index = self.buffer.find(tag)
            if index >= 0:
                chunk, self.buffer = self.buffer[:index], self.buffer[index + len(tag):]
                record("reasoning" if self.reasoning else "text", chunk)
                was_reasoning = self.reasoning
                self.reasoning = not self.reasoning
                if was_reasoning:
                    self._trim_answer_newline = True
                continue
            keep = 0 if final else self._tag_prefix_tail(self.buffer, tag)
            chunk = self.buffer if not keep else self.buffer[:-keep]
            self.buffer = "" if not keep else self.buffer[-keep:]
            if chunk:
                record("reasoning" if self.reasoning else "text", chunk)
            break
        answer_text = "".join(answer)
        self.events = [(kind, chunk) for kind, chunk in events if chunk]
        return answer_text, "".join(reasoning)


class InferenceEngine:
    """Own one runner, one MLX thread, one queue, and all live statistics."""

    def __init__(self, model_path: str, model_id: str, settings: dict[str, Any],
                 *, exit_on_load_failure=False):
        self.model_path = str(model_path)
        self.model_id = model_id
        self.settings = settings
        self.exit_on_load_failure = exit_on_load_failure
        self.runner = None
        self.jobs: queue.Queue[Job] = queue.Queue()
        self.ready = threading.Event()
        self.failed: BaseException | None = None
        self.started = time.monotonic()
        self._stats = {
            "requests": 0,
            "tokens_generated": 0,
            "prompt_tokens": 0,
            "last_peak_memory_bytes": None,
            "busy": False,
            "current_request": None,
            "last_request": None,
        }
        self._stats_lock = threading.Lock()
        self._requests: dict[str, Job] = {}
        self._requests_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def supports_images(self) -> bool:
        return bool(self.runner and self.runner.supports_images)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def wait_ready(self, timeout=300):
        if self.ready.wait(timeout):
            if self.failed:
                raise EngineError(503, f"model failed to load: {self.failed}")
            return
        if self.failed:
            raise EngineError(503, f"model failed to load: {self.failed}")
        raise EngineError(503, "model is still loading")

    def submit(self, request: GenerationRequest) -> Job:
        self.wait_ready()
        if self.jobs.qsize() >= self.settings["max_queued"]:
            raise EngineError(503, "Server busy: too many queued generations")
        job = Job(request=request, out=queue.Queue())
        with self._requests_lock:
            self._requests[job.request_id] = job
        with self._stats_lock:
            self._stats["requests"] += 1
        self.jobs.put(job)
        return job

    def cancel(self, request_id: str, *, private_only=True) -> bool:
        with self._requests_lock:
            job = self._requests.get(request_id)
        if job is None or (private_only and job.request.source != "chat"):
            return False
        job.cancelled.set()
        return True

    def _finish_job(self, job: Job):
        with self._requests_lock:
            self._requests.pop(job.request_id, None)

    def _set(self, key, value):
        with self._stats_lock:
            self._stats[key] = value

    def _worker(self):
        try:
            import mlx.core as mx
            from .cli import total_ram_bytes

            if self.settings["memory_limit_gb"] > 0:
                mx.set_memory_limit(int(self.settings["memory_limit_gb"] * 1e9))
                print(f"[mlx] memory limit {self.settings['memory_limit_gb']} GB", flush=True)
            elif self.settings["memory_limit_gb"] == 0:
                ram = total_ram_bytes()
                if ram:
                    limit = int(ram * 0.8)
                    mx.set_memory_limit(limit)
                    print(f"[mlx] memory limit {limit / 1e9:.0f} GB "
                          "(auto 80% of RAM; memory_limit_gb overrides, -1 disables)",
                          flush=True)
            if self.settings["cache_limit_gb"] > 0:
                mx.set_cache_limit(int(self.settings["cache_limit_gb"] * 1e9))
            print(f"Loading {self.model_id} from {self.model_path}...", flush=True)
            started = time.perf_counter()
            self.runner = load_runner(self.model_path)
            if (self.settings["prompt_cache"]
                    and getattr(self.runner, "supports_cache", False)):
                self.runner.enable_cache()
                print("[apc] prompt cache enabled (65k-token block pool)", flush=True)
            try:
                self._set("last_peak_memory_bytes", int(mx.get_peak_memory()))
                mx.reset_peak_memory()
            except Exception as exc:
                print(f"[mlx] stats error: {exc}", flush=True)
            print(f"Ready in {time.perf_counter() - started:.1f}s "
                  f"(images: {'yes' if self.runner.supports_images else 'no'})",
                  flush=True)
            self.ready.set()
        except BaseException as exc:
            self.failed = exc
            traceback.print_exc()
            self.ready.set()
            if self.exit_on_load_failure:
                os._exit(1)
            return

        while True:
            job = self.jobs.get()
            if job.cancelled.is_set():
                terminal = GenerationTerminal("cancelled", "cancelled", 0, 0, 0.0, 0.0)
                job.out.put(terminal)
                job.out.put(None)
                with self._stats_lock:
                    self._stats["last_request"] = self._last_snapshot(job, terminal)
                self._finish_job(job)
                continue

            started = time.perf_counter()
            last = None
            outcome = "completed"
            finish_reason = "stop"
            self._set("busy", True)
            with self._stats_lock:
                self._stats["current_request"] = {
                    "id": job.request_id,
                    "source": job.request.source,
                    "started_monotonic": time.monotonic(),
                }
            stream = None
            try:
                stream = self._run_generation(job.request)
                for response in stream:
                    last = response
                    job.out.put(response)
                    if job.cancelled.is_set():
                        outcome, finish_reason = "cancelled", "cancelled"
                        break
                    timeout = self.settings["gen_timeout_s"]
                    if timeout and time.perf_counter() - started > timeout:
                        print(f"[gen] timeout after {timeout}s, stopping", flush=True)
                        outcome, finish_reason = "timed_out", "timeout"
                        break
                else:
                    finish_reason = (getattr(last, "finish_reason", None) or "stop")
            except BaseException as exc:
                outcome, finish_reason = "failed", "error"
                job.out.put(exc)
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                duration = time.perf_counter() - started
                prompt_tokens = int(getattr(last, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(last, "generation_tokens", 0) or 0)
                tps = float(getattr(last, "generation_tps", 0.0) or 0.0)
                terminal = GenerationTerminal(
                    finish_reason, outcome, prompt_tokens, output_tokens, duration, tps
                )
                job.out.put(terminal)
                job.out.put(None)
                with self._stats_lock:
                    self._stats["busy"] = False
                    self._stats["current_request"] = None
                    self._stats["tokens_generated"] += output_tokens
                    self._stats["prompt_tokens"] += prompt_tokens
                    self._stats["last_request"] = self._last_snapshot(job, terminal)
                    try:
                        self._stats["last_peak_memory_bytes"] = int(mx.get_peak_memory())
                    except Exception as exc:
                        print(f"[mlx] stats error: {exc}", flush=True)
                try:
                    mx.reset_peak_memory()
                except Exception as exc:
                    print(f"[mlx] stats error: {exc}", flush=True)
                self._finish_job(job)

    @staticmethod
    def _last_snapshot(job: Job, terminal: GenerationTerminal):
        return {
            "source": job.request.source,
            "duration_s": round(terminal.generation_s, 3),
            "prompt_tokens": terminal.prompt_tokens,
            "output_tokens": terminal.output_tokens,
            "outcome": terminal.outcome,
        }

    def _extract_messages(self, raw_messages):
        messages, images, tmp_files = [], [], []
        try:
            for message in raw_messages:
                content = message.get("content", "")
                if isinstance(content, list):
                    texts = []
                    for part in content:
                        if part.get("type") == "text":
                            texts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            if not self.supports_images:
                                raise EngineError(400, "this model does not support images")
                            url = part.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                try:
                                    payload = base64.b64decode(
                                        url.split(",", 1)[1], validate=True
                                    )
                                except (IndexError, binascii.Error):
                                    raise EngineError(400, "Malformed data: image URL") from None
                                if len(payload) > MAX_DECODED_IMAGE:
                                    raise EngineError(413, "image is larger than the 25 MiB limit")
                                file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                                file.write(payload)
                                file.close()
                                images.append(file.name)
                                tmp_files.append(file.name)
                            elif Path(url).is_file():
                                images.append(url)
                            else:
                                raise EngineError(
                                    400, "image_url must be a data: URL or a local file path"
                                )
                    content = "\n".join(texts)
                normalized = {"role": message.get("role", "user"),
                              "content": content or ""}
                if message.get("tool_calls"):
                    calls = []
                    for call in message["tool_calls"]:
                        function = dict(call.get("function", {}))
                        if isinstance(function.get("arguments"), str):
                            try:
                                function["arguments"] = json.loads(function["arguments"])
                            except json.JSONDecodeError:
                                raise EngineError(
                                    400, "tool_calls arguments must be JSON"
                                ) from None
                        calls.append({"function": function})
                    normalized["tool_calls"] = calls
                messages.append(normalized)
        except BaseException:
            for path in tmp_files:
                Path(path).unlink(missing_ok=True)
            raise
        return messages, images, tmp_files

    def _run_generation(self, request: GenerationRequest):
        messages, images, tmp_files = self._extract_messages(request.messages)
        max_tokens = request.max_tokens
        if self.settings["max_tokens_cap"]:
            max_tokens = min(max_tokens, self.settings["max_tokens_cap"])
        thinking = request.thinking
        if thinking is None:
            thinking = {"on": True, "off": False}.get(self.settings["thinking"])
        prompt = self.runner.template(
            messages, num_images=len(images), tools=request.tools, thinking=thinking
        )
        reasoning_parser = ReasoningParser(
            isinstance(prompt, str) and prompt.rstrip().endswith("<think>")
        )
        cap = self.settings["max_prompt_tokens"]
        if cap:
            n_prompt = len(prompt) if isinstance(prompt, (list, tuple)) else len(prompt) // 4
            if n_prompt > cap:
                raise EngineError(
                    400,
                    f"prompt is ~{n_prompt} tokens, over max_prompt_tokens={cap}. "
                    "Long contexts grow the KV cache and can exhaust unified memory. "
                    f"Raise deliberately: `mlxh config max_prompt_tokens "
                    f"{min(65536, max(cap * 2, n_prompt + 4096))}` "
                    "(or serve --max-prompt-tokens N).",
                )
        print(f"[gen] start: {len(messages)} msgs, {len(images)} images, "
              f"{len(request.tools or [])} tools, max_tokens={max_tokens}", flush=True)
        started = time.perf_counter()
        last = None
        captured, processors = [], None
        if request.logprobs:
            import mlx.core as mx
            captured, capture = capture_float32_logprobs(mx)
            processors = [capture]
        logged_tokens = 0
        try:
            for response in self.runner.stream(
                prompt, images=images, max_tokens=max_tokens,
                temperature=request.temperature, top_p=request.top_p,
                logits_processors=processors,
            ):
                last = response
                answer_text, reasoning_text = reasoning_parser.feed(
                    response.text, final=response.finish_reason is not None
                )
                response.text = answer_text
                response.reasoning_text = reasoning_text
                response.phase_deltas = list(reasoning_parser.events)
                generation_tokens = int(response.generation_tokens)
                is_new = generation_tokens > logged_tokens
                if request.logprobs and is_new:
                    if not captured:
                        raise RuntimeError("model did not expose logits for a generated token")
                    values = captured.pop(0)
                    logged_tokens = generation_tokens
                    if response.finish_reason != "stop":
                        compute_logprobs(response, values, request.logprobs, mx)
                yield response
        finally:
            for path in tmp_files:
                Path(path).unlink(missing_ok=True)
            done = int(getattr(last, "generation_tokens", 0) or 0)
            print(f"[gen] end: {done} tokens in {time.perf_counter() - started:.1f}s", flush=True)

    def snapshot(self):
        with self._stats_lock:
            stats = dict(self._stats)
            current = stats["current_request"]
            if current:
                current = {
                    "id": current["id"],
                    "source": current["source"],
                    "elapsed_s": round(time.monotonic() - current["started_monotonic"], 3),
                }
        try:
            import mlx.core as mx
            mlx_stats = {
                "active_memory_bytes": int(mx.get_active_memory()),
                "cache_memory_bytes": int(mx.get_cache_memory()),
                "last_peak_memory_bytes": stats["last_peak_memory_bytes"],
            }
            mlx_version = importlib.metadata.version("mlx")
        except Exception:
            mlx_stats = {"active_memory_bytes": None, "cache_memory_bytes": None,
                         "last_peak_memory_bytes": stats["last_peak_memory_bytes"]}
            mlx_version = None
        return {
            "model": self.model_id,
            "settings": self.settings,
            "capabilities": {"images": self.supports_images, "chat_protocol": 1},
            "mlx": mlx_stats,
            "runtime": {
                "engine_version": 1,
                "uptime_s": int(time.monotonic() - self.started),
                "pid": os.getpid(),
                "ready": self.ready.is_set() and self.failed is None,
                "busy": stats["busy"],
                "queue_depth": self.jobs.qsize(),
                "requests": stats["requests"],
                "prompt_tokens": stats["prompt_tokens"],
                "tokens_generated": stats["tokens_generated"],
                "mlx_version": mlx_version,
                "current_request": current,
                "last_request": stats["last_request"],
            },
        }


def capture_float32_logprobs(mx):
    captured = []

    def capture(_tokens, logits):
        logits32 = logits.astype(mx.float32)
        normalized = logits32 - mx.logsumexp(logits32, axis=-1, keepdims=True)
        captured.append(normalized.squeeze(0))
        return logits

    return captured, capture


def compute_logprobs(response, logprobs, opts: LogprobsOptions, mx):
    vocab = int(logprobs.shape[-1])
    for field, ids in (("logprob_token_ids", opts.ids),
                       ("allowed_token_ids", opts.allowed)):
        bad = [token_id for token_id in ids if token_id >= vocab]
        if bad:
            raise EngineError(400, f"{field} contains id {bad[0]} outside vocab size {vocab}")
    top_ids = []
    k = min(opts.top, vocab)
    if k:
        indices = mx.argpartition(-logprobs, k - 1)[:k]
        values = logprobs[indices]
        order = mx.argsort(-values)
        top_ids = [int(item) for item in indices[order].tolist()]
    selected = list(dict.fromkeys([*top_ids, *opts.ids]))
    values = [float(value) for value in logprobs[mx.array(selected)].tolist()] if selected else []
    ranked = sorted(zip(selected, values), key=lambda item: item[1], reverse=True)
    selected = [token_id for token_id, _ in ranked]
    values = [value for _, value in ranked]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("model returned a non-finite requested logprob")
    token_lp = float(logprobs[int(response.token)].item())
    if not math.isfinite(token_lp):
        raise RuntimeError("model returned a non-finite generated-token logprob")
    response.logprobs_out = (selected, values, token_lp)
