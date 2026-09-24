import asyncio
import base64
import io
import json
import queue
import threading
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mlxh import cli, serve_app
from mlxh.engine import Job, GenerationRequest
from mlxh.image_engine import ImageEngine
from mlxh.image_models import IMAGE_CATALOG, IMAGE_METADATA, IMAGE_REPO, IMAGE_REVISION, image_metadata
from mlxh.images import (
    ImageEditRequest, ImageError, parse_request, encode_image,
    ImageGenerationResult, stage_reference_image,
)


SETTINGS = {**serve_app.SETTINGS, "memory_limit_gb": -1, "gen_timeout_s": 0}


def request(**kwargs):
    return parse_request({"model": "schnell", "prompt": "a canoe", "size": "256x256",
                          **kwargs}, "schnell", SETTINGS)


@pytest.mark.parametrize("field,value", [
    ("n", True), ("n", 2), ("n", 1.0), ("prompt", " "), ("prompt", "a" * 32001),
    ("steps", True), ("steps", 0), ("steps", 101),
    ("model", "wrong"), ("seed", True), ("seed", -1), ("seed", 2**32),
    ("size", "256X256"), ("size", "255x256"), ("size", "257x256"),
    ("size", "4096x4096"), ("size", None), ("quality", "hd"), ("quality", "high"),
    ("response_format", "url"), ("moderation", "auto"), ("stream", False),
    ("output_format", []), ("output_compression", 50), ("user", 1), ("unknown", 42),
])
def test_invalid_image_fields(field, value):
    with pytest.raises(ImageError) as exc:
        request(**{field: value})
    assert exc.value.status_code == 400
    assert exc.value.param == field


@pytest.mark.parametrize("fmt", ["png", "jpeg", "webp"])
def test_encode_and_response(fmt):
    req = request(output_format=fmt, seed=42, quality="standard")
    image = Image.new("RGB", (256, 256), "red")
    image.info["prompt"] = "secret"
    encoded = encode_image(image, req)
    response = ImageGenerationResult(encoded, req, 4, 1.23).response(123)
    decoded = Image.open(io.BytesIO(base64.b64decode(response["data"][0]["b64_json"])))
    assert decoded.size == (256, 256)
    assert decoded.format == fmt.upper()
    assert "prompt" not in decoded.info
    assert response["mlxh"]["seed"] == 42
    assert response["quality"] == "auto"


def test_auto_size_and_pixel_limit():
    assert request(size="auto").width == 1024
    small = parse_request({"model": "schnell", "prompt": "hi"}, "schnell",
                          {"max_image_pixels": 65536})
    assert (small.width, small.height) == (256, 256)
    capped = parse_request({"model": "schnell", "prompt": "hi"}, "schnell",
                           {"max_image_pixels": 300000})
    assert (capped.width, capped.height) == (544, 544)
    with pytest.raises(ImageError, match="at least 65536"):
        parse_request({"model": "schnell", "prompt": "hi"}, "schnell",
                      {"max_image_pixels": 65535})


def test_result_byte_limit(monkeypatch):
    monkeypatch.setattr("mlxh.images.OUTPUT_LIMIT", 10)
    with pytest.raises(ImageError):
        encode_image(Image.new("RGB", (256, 256)), request())


def test_per_request_steps_are_returned():
    req = request(steps=7)
    result = ImageGenerationResult(b"png", req, 7, 0.5).response(1)
    assert req.steps == result["mlxh"]["steps"] == 7


def test_reference_image_validation_and_normalization(monkeypatch):
    source = io.BytesIO()
    Image.new("RGBA", (20, 10), (255, 0, 0, 10)).save(source, format="PNG")
    staged = stage_reference_image(source.getvalue())
    try:
        with Image.open(staged) as result:
            assert result.format == "PNG"
            assert result.mode == "RGB"
            assert result.size == (20, 10)
            assert not result.info
    finally:
        from pathlib import Path
        Path(staged).unlink()
    with pytest.raises(ImageError, match="invalid reference image"):
        stage_reference_image(b"not an image")
    monkeypatch.setattr("mlxh.images.MAX_REFERENCE_PIXELS", 100)
    too_large = io.BytesIO()
    Image.new("RGB", (11, 10)).save(too_large, format="PNG")
    with pytest.raises(ImageError, match="16-megapixel"):
        stage_reference_image(too_large.getvalue())


def test_edit_request_is_typed_and_enforces_reference_count():
    parsed = parse_request({"model": "klein", "prompt": "edit this"}, "klein",
                           SETTINGS, source="openai-image-edits", input_images=["/tmp/ref.png"])
    assert isinstance(parsed, ImageEditRequest)
    assert parsed.source == "openai-image-edits"
    assert parsed.input_images == ["/tmp/ref.png"]
    with pytest.raises(ImageError, match="between 1 and 4"):
        parse_request({"model": "klein", "prompt": "edit this"}, "klein",
                      SETTINGS, input_images=[])


class Runner:
    default_steps = 4
    supports_images = False
    calls = 0

    def validate_prompt(self, prompt):
        if prompt == "too long":
            raise ImageError(400, "prompt too long", "prompt", "prompt_too_long")

    def generate(self, req, steps, check):
        self.calls += 1
        for step in range(steps):
            check(step + 1)
        return Image.new("RGB", (req.width, req.height), "blue")


class EditRunner(Runner):
    supports_edits = True

    def edit(self, req, steps, check):
        self.calls += 1
        self.references = list(req.input_images)
        for step in range(steps):
            check(step + 1)
        return Image.new("RGB", (req.width, req.height), "green")


def engine(tmp_path):
    e = ImageEngine(str(tmp_path), "schnell", dict(SETTINGS))
    e.runner = Runner()
    e.ready.set()
    return e


class Memory:
    def __init__(self):
        self.events = []

    def get_peak_memory(self):
        self.events.append("snapshot")
        return 123

    def reset_peak_memory(self):
        self.events.append("reset")


def test_execution_counters_and_queue_skip(tmp_path):
    e, mx = engine(tmp_path), Memory()
    job = e.submit(request())
    assert not e.cancel(job.request_id)  # public DELETE remains chat-only
    assert e.cancel(job.request_id, expected_source="openai-images")
    e._execute(e.jobs.get_nowait(), mx)
    assert e.runner.calls == 0
    assert e._stats["last_request"]["outcome"] == "cancelled"
    assert mx.events == []
    job = e.submit(request())
    e._execute(e.jobs.get_nowait(), mx)
    assert isinstance(job.out.get(), ImageGenerationResult)
    assert e._stats["images_generated"] == 1
    assert e._stats["requests"] == 2
    assert not e._stats["busy"]
    assert mx.events == ["snapshot", "reset"]


@pytest.mark.parametrize("mode,outcome", [("timeout", "timed_out"), ("cancel", "cancelled"),
                                         ("error", "failed"), ("prompt", "invalid_request")])
def test_failed_job_finalizes_and_next_job_works(tmp_path, mode, outcome):
    e, mx = engine(tmp_path), Memory()
    job = e.submit(request(prompt="too long" if mode == "prompt" else "hi"))
    generate = e.runner.generate

    def fail(req, steps, check):
        if mode == "cancel":
            e.cancel(job.request_id, expected_source="openai-images")
        elif mode == "timeout":
            e.settings["gen_timeout_s"] = 1e-12
        elif mode == "error":
            raise RuntimeError("private backend path")
        check(1)

    e.runner.generate = fail
    e._execute(e.jobs.get_nowait(), mx)
    result = job.out.get_nowait()
    assert isinstance(result, ImageError)
    assert result.__traceback__ is None
    assert "private backend path" not in result.detail
    assert e._stats["last_request"]["outcome"] == outcome
    assert not e._stats["busy"] and not e._requests
    e.runner.generate = generate
    e.settings["gen_timeout_s"] = 0
    next_job = e.submit(request())
    e._execute(e.jobs.get_nowait(), mx)
    assert isinstance(next_job.out.get_nowait(), ImageGenerationResult)


def test_edit_execution_instruments_and_cleans_private_inputs(tmp_path):
    e, mx = engine(tmp_path), Memory()
    e.runner = EditRunner()
    e.runner.edit_limits = {"effective_reference_pixels": 1_048_576}
    reference = tmp_path / "private-ref.png"
    reference.write_bytes(b"staged")
    req = ImageEditRequest("edit", 256, 256, "png", None, 42, 2,
                           source="openai-image-edits", input_images=[str(reference)],
                           cleanup_input_images=True)
    job = e.submit(req)
    e._execute(e.jobs.get_nowait(), mx)
    result = job.out.get_nowait()
    assert isinstance(result, ImageGenerationResult)
    assert not reference.exists()
    snapshot = e.snapshot()
    assert snapshot["capabilities"]["image_edits"] is True
    assert snapshot["image"]["edit_limits"]["effective_reference_pixels"] == 1_048_576
    assert snapshot["runtime"]["images_generated"] == 1
    assert snapshot["runtime"]["images_edited"] == 1
    assert snapshot["runtime"]["last_request"]["operation"] == "edit"
    assert snapshot["runtime"]["last_request"]["reference_count"] == 1
    response = result.response(1)
    assert response["mlxh"]["operation"] == "edit"
    assert response["mlxh"]["reference_images"] == 1


def test_cancelled_queued_edit_skips_runner_and_cleans_inputs(tmp_path):
    e, mx = engine(tmp_path), Memory()
    e.runner = EditRunner()
    reference = tmp_path / "private-ref.png"
    reference.write_bytes(b"staged")
    req = ImageEditRequest("edit", 256, 256, "png", None, 42, 2,
                           source="openai-image-edits", input_images=[str(reference)],
                           cleanup_input_images=True)
    job = e.submit(req)
    e.cancel(job.request_id, expected_source="openai-image-edits")
    e._execute(e.jobs.get_nowait(), mx)
    assert e.runner.calls == 0
    assert not reference.exists()
    assert e._stats["last_request"]["outcome"] == "cancelled"


def test_queue_full_and_wrong_kind(tmp_path):
    e = engine(tmp_path)
    e.settings["max_queued"] = 1
    e.submit(request())
    with pytest.raises(Exception) as exc:
        e.submit(request())
    assert exc.value.status_code == 503
    assert e._stats["requests"] == 1
    with pytest.raises(Exception) as exc:
        e.submit(GenerationRequest([], "chat"))
        assert exc.value.status_code == 400


def test_snapshot_reports_loaded_runner_limits(tmp_path):
    e = engine(tmp_path)
    e.runner.default_width = 768
    e.runner.default_height = 1024
    e.runner.default_steps = 6
    e.runner.t5_limit = 512
    e.runner.prompt_limits = {"t5_tokens": 512, "tokens": 512}
    info = e.snapshot()["image"]
    assert info["default_size"] == "768x1024"
    assert info["default_steps"] == 6
    assert info["prompt_limits"]["t5_tokens"] == 512


def test_api_success_and_invalid_body(tmp_path, monkeypatch):
    e = engine(tmp_path)
    submit = e.submit

    def immediate(req):
        job = submit(req)
        e._execute(e.jobs.get_nowait(), Memory())
        return job

    e.submit = immediate
    monkeypatch.setattr(serve_app, "engine", e)
    client = TestClient(serve_app.app)
    assert client.post("/v1/images/generations", content=b"bad").status_code == 400
    result = client.post("/v1/images/generations", json={"model": "schnell", "prompt": "hi", "size": "256x256"})
    assert result.status_code == 200
    assert result.json()["size"] == "256x256"
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400
    assert client.post("/v1/images/generations", content=b"{}", headers={
        "content-length": str(serve_app.IMAGE_BODY_LIMIT + 1)}).status_code == 413


def test_api_edit_accepts_multipart_and_cleans_staged_files(tmp_path, monkeypatch):
    e = engine(tmp_path)
    e.runner = EditRunner()
    submit = e.submit

    def immediate(req):
        job = submit(req)
        e._execute(e.jobs.get_nowait(), Memory())
        return job

    e.submit = immediate
    monkeypatch.setattr(serve_app, "engine", e)
    upload = io.BytesIO()
    Image.new("RGBA", (32, 24), (20, 120, 40, 50)).save(upload, format="PNG")
    response = TestClient(serve_app.app).post(
        "/v1/images/edits",
        data={"model": "schnell", "prompt": "Make it watercolor", "size": "256x256",
              "seed": "42", "steps": "4"},
        files=[("image", ("input.png", upload.getvalue(), "image/png"))],
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mlxh"]["operation"] == "edit"
    assert body["mlxh"]["reference_images"] == 1
    assert e._stats["images_edited"] == 1
    assert len(e.runner.references) == 1
    assert not __import__("pathlib").Path(e.runner.references[0]).exists()


def test_api_edit_rejects_unsupported_model_and_fields(tmp_path, monkeypatch):
    e = engine(tmp_path)
    monkeypatch.setattr(serve_app, "engine", e)
    client = TestClient(serve_app.app)
    unsupported = client.post("/v1/images/edits", data={"model": "klein", "prompt": "edit"})
    assert unsupported.status_code == 400
    assert unsupported.json()["error"]["code"] == "unsupported_model_operation"


def test_api_edit_rejects_invalid_image_and_cleans_prior_staging(tmp_path, monkeypatch):
    e = engine(tmp_path)
    e.runner = EditRunner()
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    monkeypatch.setattr(serve_app, "engine", e)
    valid = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(valid, format="PNG")
    response = TestClient(serve_app.app).post(
        "/v1/images/edits", data={"model": "klein", "prompt": "edit"},
        files=[("image", ("one.png", valid.getvalue(), "image/png")),
               ("image", ("two.png", b"bad", "image/png"))],
    )
    assert response.status_code == 400
    assert e._stats["requests"] == 0
    assert not list(tmp_path.glob("mlxh-image-ref-*.png"))


def test_edit_multipart_limit_has_accurate_error(monkeypatch):
    sent = []
    original = serve_app.EDIT_BODY_LIMIT
    monkeypatch.setattr(serve_app, "EDIT_BODY_LIMIT", 1024)

    async def receive():
        pytest.fail("content length should be rejected without reading body")

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/v1/images/edits",
             "headers": [(b"content-length", b"1025")], "query_string": b"",
             "http_version": "1.1", "scheme": "http", "server": ("test", 80),
             "client": ("127.0.0.1", 1000), "root_path": ""}
    asyncio.run(serve_app.app(scope, receive, send))
    assert sent[0]["status"] == 413
    assert b"50 MiB" in sent[1]["body"]
    assert original > 0


def test_api_wrong_model_kind(monkeypatch):
    monkeypatch.setattr(serve_app, "engine", NS(model_kind="language"))
    result = TestClient(serve_app.app).post("/v1/images/generations", json={})
    assert result.status_code == 400
    assert result.json()["error"]["code"] == "unsupported_model_operation"


@pytest.mark.parametrize("status,code", [(500, "generation_failed"), (504, "generation_timeout")])
def test_api_worker_error_envelopes(tmp_path, monkeypatch, status, code):
    e = engine(tmp_path)

    def submit(req):
        job = Job(req, queue.Queue())
        job.out.put(ImageError(status, "generation did not finish", code=code))
        return job

    e.submit = submit
    monkeypatch.setattr(serve_app, "engine", e)
    response = TestClient(serve_app.app).post("/v1/images/generations", json={"model": "schnell", "prompt": "hi"})
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


def test_api_queue_overflow_envelope(tmp_path, monkeypatch):
    e = engine(tmp_path)
    e.settings["max_queued"] = 1
    e.submit(request())
    monkeypatch.setattr(serve_app, "engine", e)
    response = TestClient(serve_app.app).post("/v1/images/generations", json={"model": "schnell", "prompt": "hi"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "server_busy"


def test_chunked_request_limit_before_submission(tmp_path, monkeypatch):
    e = engine(tmp_path)
    monkeypatch.setattr(serve_app, "engine", e)
    sent = []
    parts = iter([b'{"prompt":"' + b'a' * 150000, b'a' * 150000 + b'"}'])

    async def receive():
        return {"type": "http.request", "body": next(parts), "more_body": True}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/v1/images/generations",
             "headers": [(b"content-type", b"application/json")], "query_string": b"",
             "http_version": "1.1", "scheme": "http", "server": ("test", 80),
             "client": ("127.0.0.1", 1000), "root_path": ""}
    asyncio.run(serve_app.app(scope, receive, send))
    assert sent[0]["status"] == 413
    assert e._stats["requests"] == 0


def test_disconnect_sets_image_cancel_flag(tmp_path, monkeypatch):
    e = engine(tmp_path)
    monkeypatch.setattr(serve_app, "engine", e)

    class Request:
        async def json(self):
            return {"model": "schnell", "prompt": "hi"}

        async def is_disconnected(self):
            return True

    response = asyncio.run(serve_app.image_generations(Request()))
    job = e.jobs.get_nowait()
    assert response.status_code == 499
    assert job.cancelled.is_set()
    e._execute(job, Memory())
    assert e.runner.calls == 0


def test_model_metadata_and_language_guard(tmp_path):
    (tmp_path / ".mlxh.json").write_text(json.dumps(IMAGE_METADATA))
    assert cli.is_model(tmp_path)
    assert image_metadata(tmp_path) == IMAGE_METADATA
    with pytest.raises(SystemExit):
        cli._require_language(tmp_path)


def test_legacy_schnell_metadata_remains_supported(tmp_path):
    (tmp_path / ".mlxh.json").write_text(json.dumps({
        "kind": "image", "backend": "mflux", "family": "flux-schnell",
        "quantization_bits": 4,
    }))
    assert image_metadata(tmp_path)["family"] == "flux1-schnell"


@pytest.mark.parametrize("repo,spec", list(IMAGE_CATALOG.items()))
def test_supported_image_repo_metadata(repo, spec, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / ".mlxh.json").write_text(json.dumps({"kind": "image", **spec}))
    assert image_metadata(model) == {"kind": "image", **spec}
    assert cli.model_kind(model) == "image"


def test_image_status(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_fetch_info", lambda _: {
        "model": "schnell", "model_kind": "image", "runtime": {"images_generated": 3, "ready": True}})
    cli.cmd_status(NS(json=False))
    text = capsys.readouterr().out
    assert "IMAGES" in text and "PROMPT" not in text


def test_pull_image_pins_revision_and_records_kind(tmp_path, monkeypatch):
    import huggingface_hub

    cfg = {**cli.DEFAULTS, "models_dir": str(tmp_path)}
    monkeypatch.delenv("MLXH_MODELS_DIR", raising=False)
    monkeypatch.setattr(cli, "_ensure_image_runtime", lambda **kw: "python")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: NS(
        model_info=lambda *a, **kw: NS(siblings=[], sha="unrelated-latest-revision")))
    downloads = []

    def download(repo, local_dir, revision):
        downloads.append((repo, revision))
        from pathlib import Path
        Path(local_dir).mkdir()

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    cli.do_pull(cfg, IMAGE_REPO, "schnell", kind="image", backend="mflux")
    meta = json.loads((tmp_path / "schnell/.mlxh.json").read_text())
    assert meta["revision"] == IMAGE_REVISION
    assert meta["kind"] == "image"
    assert downloads == [(IMAGE_REPO, IMAGE_REVISION)]


@pytest.mark.parametrize("repo,spec", [
    ("black-forest-labs/FLUX.2-klein-4B", IMAGE_CATALOG["black-forest-labs/FLUX.2-klein-4B"]),
    ("Qwen/Qwen-Image-2512", IMAGE_CATALOG["Qwen/Qwen-Image-2512"]),
])
def test_pull_additional_image_families(repo, spec, tmp_path, monkeypatch):
    import huggingface_hub

    cfg = {**cli.DEFAULTS, "models_dir": str(tmp_path)}
    monkeypatch.delenv("MLXH_MODELS_DIR", raising=False)
    monkeypatch.setattr(cli, "_ensure_image_runtime", lambda **kw: "python")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: NS(
        model_info=lambda *a, **kw: NS(siblings=[], sha="resolved-commit")))
    downloads = []

    def download(pulled_repo, local_dir, revision):
        downloads.append((pulled_repo, revision))
        from pathlib import Path
        Path(local_dir).mkdir()

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    cli.do_pull(cfg, repo, "image-model", kind="image", backend="mflux")
    metadata = json.loads((tmp_path / "image-model/.mlxh.json").read_text())
    assert metadata["revision"] == "resolved-commit"
    assert metadata["family"] == spec["family"]
    assert downloads == [(repo, "resolved-commit")]


def test_pull_backend_must_match_repository(tmp_path):
    cfg = {"models_dir": str(tmp_path)}
    with pytest.raises(SystemExit):
        cli.do_pull(cfg, "org/language-model", "model", backend="mflux")


@pytest.mark.parametrize("answer,installed", [("y", True), ("n", False)])
def test_interactive_runtime_install_offer(monkeypatch, answer, installed):
    from mlxh import image_runtime
    calls = []

    def python(_):
        if calls:
            return "private-python"
        raise RuntimeError("missing")

    monkeypatch.setattr(image_runtime, "runtime_python", python)
    monkeypatch.setattr(image_runtime, "install", lambda _: calls.append("installed"))
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: answer)
    if installed:
        assert cli._ensure_image_runtime(interactive=True) == "private-python"
    else:
        with pytest.raises(SystemExit):
            cli._ensure_image_runtime(interactive=True)
    assert bool(calls) == installed


def test_noninteractive_runtime_does_not_install(monkeypatch, capsys):
    from mlxh import image_runtime
    monkeypatch.setattr(image_runtime, "runtime_python", lambda _: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(image_runtime, "install", lambda _: pytest.fail("installed"))
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit):
        cli._ensure_image_runtime(interactive=True)
    assert "mlxh images install" in capsys.readouterr().err
