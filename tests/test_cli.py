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
