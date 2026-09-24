import json
from types import SimpleNamespace as NS

import pytest

import mlxh.cli as cli

REV = "a" * 40


@pytest.fixture
def mdir(tmp_path, monkeypatch):
    d = tmp_path / "models"
    d.mkdir()
    monkeypatch.setenv("MLXH_MODELS_DIR", str(d))
    return d


def make_model(mdir, name, repo=None, rev=REV):
    p = mdir / name
    p.mkdir()
    (p / "config.json").write_text("{}")
    if repo:
        (p / ".mlxh.json").write_text(json.dumps({"repo": repo, "revision": rev}))
    return p


def test_defaults_complete():
    cfg = cli.load_config()
    assert set(cli.DEFAULTS) <= set(cfg)
    assert set(cli.KEY_TYPES) == set(cli.DEFAULTS)


def test_bool_converter():
    assert cli._bool("on") and cli._bool("TRUE") and cli._bool("1")
    assert not cli._bool("off") and not cli._bool("False")
    with pytest.raises(ValueError):
        cli._bool("maybe")


def test_discover_and_resolve(mdir):
    make_model(mdir, "m1")
    (mdir / "not-a-model").mkdir()
    cfg = cli.load_config()
    assert list(cli.discover(cfg)) == ["m1"]
    assert cli.resolve(cfg, "m1").endswith("m1")
    with pytest.raises(SystemExit):
        cli.resolve(cfg, "missing")


def test_invalid_image_metadata_remains_discoverable(mdir, capsys):
    model = make_model(mdir, "bad-image")
    (model / ".mlxh.json").write_text(json.dumps({"kind": "image", "backend": "unknown"}))
    cfg = cli.load_config()
    assert "bad-image" in cli.discover(cfg)
    with pytest.raises(SystemExit):
        cli.serve_argv(cfg, "bad-image", str(model))
    assert "unsupported model metadata" in capsys.readouterr().err


def test_run_rejects_image_repo_before_pull(mdir, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "load_config", lambda: {**cli.DEFAULTS, "models_dir": str(mdir)})
    monkeypatch.setattr(cli, "repo_installed_as", lambda *args: None)
    monkeypatch.setattr(cli, "do_pull", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(cli.os, "execv", lambda *args: pytest.fail("must not launch chat"))
    with pytest.raises(SystemExit):
        cli.cmd_run(NS(target=cli.IMAGE_REPO, force=False, rest=[]))
    assert calls == [{"force": False, "kind": "language"}]


def test_cli_image_saves_without_colliding_and_honors_explicit_overwrite(tmp_path):
    first = cli._save_cli_image(b"one", "Fox bookstore", tmp_path, "png")
    second = cli._save_cli_image(b"two", "Fox bookstore", tmp_path, "png")
    assert first.name == "fox-bookstore.png"
    assert second.name == "fox-bookstore-2.png"
    assert first.read_bytes() == b"one" and second.read_bytes() == b"two"

    with pytest.raises(SystemExit):
        cli._save_cli_image(b"three", "ignored", tmp_path, "png", explicit=first)
    assert cli._save_cli_image(b"three", "ignored", tmp_path, "png",
                               explicit=first, force=True) == first
    assert first.read_bytes() == b"three"


def test_image_repl_slash_command_completion():
    from prompt_toolkit.document import Document

    session = cli._image_session()
    completions = list(session.completer.get_completions(Document("/s"), None))
    assert [item.text for item in completions] == ["/size", "/seed", "/steps"]
    assert list(session.completer.get_completions(Document("/size "), None)) == []


def test_image_repl_completes_paths_for_ref_and_output(tmp_path, monkeypatch):
    from prompt_toolkit.document import Document

    monkeypatch.chdir(tmp_path)
    (tmp_path / "input image.png").touch()
    (tmp_path / "folder").mkdir()
    session = cli._image_session()
    ref = list(session.completer.get_completions(Document("/ref input"), None))
    output = list(session.completer.get_completions(Document("/output f"), None))
    assert [(item.text, item.display_text) for item in ref] == [
        (" image.png", "input image.png"),
    ]
    assert [(item.text, item.display_text) for item in output] == [("older", "folder/")]


def test_image_repl_path_completion_expands_tilde(tmp_path, monkeypatch):
    from prompt_toolkit.document import Document

    pictures = tmp_path / "Pictures"
    pictures.mkdir()
    (pictures / "photo.png").touch()
    monkeypatch.setenv("HOME", str(tmp_path))
    session = cli._image_session()
    completions = list(session.completer.get_completions(
        Document("/ref ~/Pictures/pho"), None,
    ))
    assert [(item.text, item.display_text) for item in completions] == [
        ("to.png", "photo.png"),
    ]


def test_image_edit_request_uses_multipart_endpoint(tmp_path, monkeypatch):
    import base64
    import json
    import urllib.request

    reference = tmp_path / "ref.png"
    reference.write_bytes(b"png reference bytes")
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"data": [{"b64_json": base64.b64encode(b"edited").decode()}],
                               "size": "256x256", "mlxh": {"seed": 42}}).encode()

    def open_url(request, timeout):
        captured.update(url=request.full_url, headers=request.headers,
                        body=request.data, timeout=timeout)
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    result = cli._image_api_request(
        9876, "klein", "edit this", size="256x256", seed=42,
        steps=4, output_format="png", input_images=[str(reference)],
    )
    assert result[0] == b"edited"
    assert captured["url"].endswith("/v1/images/edits")
    assert "multipart/form-data; boundary=" in captured["headers"]["Content-type"]
    assert b'name="image"; filename="reference"' in captured["body"]
    assert b"png reference bytes" in captured["body"]
    assert b"name=\"steps\"" in captured["body"]


def test_one_shot_image_cli_uses_server_and_requested_options(tmp_path, monkeypatch):
    cfg = {**cli.DEFAULTS, "models_dir": str(tmp_path), "port": 9876}
    model_path = tmp_path / "klein"
    model_path.mkdir()
    output = tmp_path / "out.png"
    seen = {}
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "resolve", lambda *_: str(model_path))
    monkeypatch.setattr(cli, "checked_model_kind", lambda _: "image")
    monkeypatch.setattr(cli, "_ensure_local_server", lambda *a, **k: ({"model_kind": "image"}, None))

    def generate(port, model, prompt, **kwargs):
        seen.update(port=port, model=model, prompt=prompt, **kwargs)
        return b"image bytes", "768x1024", {"seed": 42, "steps": 4}

    monkeypatch.setattr(cli, "_image_api_request", generate)
    cli.cmd_image(NS(model="klein", prompt=["fox", "reading"], output=str(output),
                     output_dir=None, force=False, size="768x1024", seed=42,
                     steps=4, output_format="png"))
    assert output.read_bytes() == b"image bytes"
    assert seen == {"port": 9876, "model": "klein", "prompt": "fox reading",
                    "size": "768x1024", "seed": 42, "steps": 4, "output_format": "png"}


def test_model_supports_images_from_local_config(mdir):
    prism_vision = make_model(mdir, "prism-vision")
    prism_vision.joinpath("config.json").write_text(json.dumps({
        "model_type": "prism_hadamard_qwen35",
        "components": {"text": True, "vision": True},
    }))
    prism_text = make_model(mdir, "prism-text")
    prism_text.joinpath("config.json").write_text(json.dumps({
        "model_type": "prism_hadamard_qwen35",
        "components": {"text": True, "vision": False},
        "vision_config": {},
    }))
    stock_vision = make_model(mdir, "stock-vision")
    stock_vision.joinpath("config.json").write_text(json.dumps({
        "model_type": "qwen3_vl",
        "vision_config": {},
    }))

    assert cli.model_supports_images(prism_vision)
    assert not cli.model_supports_images(prism_text)
    assert cli.model_supports_images(stock_vision)


def test_source_of(mdir):
    p = make_model(mdir, "m1", repo="org/model")
    assert cli.source_of(p) == "org/model@aaaaaaa"
    q = make_model(mdir, "m2")
    assert cli.source_of(q) == "-"
    meta = q / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (meta / "f.metadata").write_text("b" * 40 + "\netag\n123\n")
    assert cli.source_of(q) == "hf@bbbbbbb"


def test_mv(mdir):
    make_model(mdir, "old", repo="org/model")
    cli.cmd_mv(NS(name="old", new_name="new"))
    assert cli.is_model(mdir / "new") and not (mdir / "old").exists()
    assert cli.source_of(mdir / "new") == "org/model@aaaaaaa"
    with pytest.raises(SystemExit):
        cli.cmd_mv(NS(name="missing", new_name="x"))
    make_model(mdir, "third")
    with pytest.raises(SystemExit):  # refuses to clobber
        cli.cmd_mv(NS(name="third", new_name="new"))


def test_rm_link_keeps_target(mdir, tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "config.json").write_text("{}")
    (mdir / "alias").symlink_to(target)
    cli.cmd_rm(NS(name="alias"))
    assert target.exists()
    assert not (mdir / "alias").is_symlink()


def test_rm_pulled_deletes(mdir):
    make_model(mdir, "gone")
    cli.cmd_rm(NS(name="gone"))
    assert not (mdir / "gone").exists()


def test_chat_args_tools_toggle():
    assert cli._chat_args({"chat_tools": True}, []) == ["--tools"]
    assert cli._chat_args({"chat_tools": True}, ["--no-tools"]) == ["--no-tools"]
    assert cli._chat_args({"chat_tools": True}, ["--tools"]) == ["--tools"]
    assert cli._chat_args({"chat_tools": False}, []) == []


def test_config_set(capsys):
    cli.cmd_config(NS(key="port", value="9000"))
    assert cli.load_config()["port"] == 9000
    cli.cmd_config(NS(key="chat_tools", value="on"))
    assert cli.load_config()["chat_tools"] is True
    with pytest.raises(SystemExit):
        cli.cmd_config(NS(key="bogus", value="1"))
    with pytest.raises(SystemExit):
        cli.cmd_config(NS(key="port", value="abc"))


def test_total_ram_positive():
    assert cli.total_ram_bytes() > 1e9


def test_list_output(mdir, capsys):
    make_model(mdir, "m1", repo="org/model")
    cli.cmd_list(None)
    out = capsys.readouterr().out
    assert "m1" in out
    assert "org/model@aaaaaaa" in out
    assert "pulled" in out


def test_serve_argv_includes_all_knobs(mdir):
    make_model(mdir, "m1")
    cfg = cli.load_config()
    argv = cli.serve_argv(cfg, "m1", "/p", {"port": 9999})
    assert "--max-prompt-tokens" in argv and "--port" in argv
    assert argv[argv.index("--port") + 1] == "9999"
    assert argv[argv.index("--max-prompt-tokens") + 1] == str(cfg["max_prompt_tokens"])


def test_pi_register_provider(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"providers": {"ollama": {"baseUrl": "http://x"}}}))
    out = cli.pi_register_provider(
        1060, "gemma4-12b", path=path, supports_images=True
    )
    data = json.loads(out.read_text())
    assert data["providers"]["ollama"]["baseUrl"] == "http://x"  # untouched
    prov = data["providers"]["mlxh"]
    assert prov["baseUrl"] == "http://127.0.0.1:1060/v1"
    assert prov["compat"]["supportsDeveloperRole"] is False
    assert {"id": "gemma4-12b", "input": ["text", "image"]} in prov["models"]
    # idempotent + adds second model
    cli.pi_register_provider(2000, "qwen-tiny", path=path)
    prov = json.loads(path.read_text())["providers"]["mlxh"]
    assert prov["baseUrl"].endswith(":2000/v1")
    assert len([m for m in prov["models"] if m["id"] == "gemma4-12b"]) == 1
    assert {"id": "qwen-tiny", "input": ["text"]} in prov["models"]


def test_pi_register_provider_updates_existing_model_metadata(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"providers": {"mlxh": {"models": [{
        "id": "bonsai2", "contextWindow": 32768, "input": ["text"],
    }]}}}))

    cli.pi_register_provider(1060, "bonsai2", path=path, supports_images=True)

    model = json.loads(path.read_text())["providers"]["mlxh"]["models"][0]
    assert model == {
        "id": "bonsai2", "contextWindow": 32768,
        "input": ["text", "image"],
    }


def test_ensure_local_server_starts_in_new_session(monkeypatch, tmp_path):
    cfg = cli.load_config()
    calls = []
    responses = iter((OSError("down"), {
        "model": "m1", "capabilities": {"chat_protocol": 1},
        "runtime": {"ready": True, "engine_version": 1},
    }))

    def fetch(_port):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    class Process:
        pid = 123

        def poll(self):
            return None

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return Process()

    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "_fetch_info", fetch)
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    info, process = cli._ensure_local_server(
        cfg, "m1", "/models/m1", 1060,
        require_same_model=True, require_chat_protocol=True,
    )

    assert info["model"] == "m1"
    assert process.pid == 123
    assert calls[0][1]["start_new_session"] is True


def test_chat_refuses_different_model_without_stopping_server(monkeypatch):
    cfg = cli.load_config()
    monkeypatch.setattr(cli, "_fetch_info", lambda _port: {
        "model": "other", "runtime": {"ready": True, "engine_version": 1},
        "capabilities": {"chat_protocol": 1},
    })
    monkeypatch.setattr(
        cli, "_stop_owned_server",
        lambda _process: pytest.fail("reused server must not be stopped"),
    )
    with pytest.raises(SystemExit):
        cli._ensure_local_server(
            cfg, "wanted", "/models/wanted", 1060,
            require_same_model=True, require_chat_protocol=True,
        )
