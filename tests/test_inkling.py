import json

import pytest

from mlxh import loader
from mlxh.inkling import InklingParser
from mlxh.toolcalls import parse_tool_calls

# what Inkling 2 Mini wrote, verbatim (greedy, on this format)
ANSWER = ("<|message_model|><|content_thinking|>Count: Hi=1 there=2 friend=3. Good.<|end_message|>"
          "<|message_model|><|content_text|>Hi there friend<|end_message|>")
TOOL = ("<|message_model|><|content_thinking|>Need use get_weather tool.<|end_message|>"
        "<|message_model|>get_weather<|content_invoke_tool_json|>"
        '{"name":"get_weather","args":{"city":"Lomé","days":3}}<|end_message|>')


def run(text, size):
    parser = InklingParser()
    answer, reasoning, events = [], [], []
    chunks = [text[i:i + size] for i in range(0, len(text), size)]
    for i, chunk in enumerate(chunks):
        a, r = parser.feed(chunk, final=i == len(chunks) - 1)
        answer.append(a)
        reasoning.append(r)
        events += parser.events
    return "".join(answer), "".join(reasoning), events


@pytest.mark.parametrize("size", [1, 2, 3, 7, 1000])
def test_answer_and_reasoning_split_across_any_chunking(size):
    answer, reasoning, events = run(ANSWER, size)
    assert answer == "Hi there friend"
    assert reasoning == "Count: Hi=1 there=2 friend=3. Good."
    assert "".join(c for k, c in events if k == "text") == answer


@pytest.mark.parametrize("size", [1, 5, 1000])
def test_tool_call_reaches_the_servers_tool_parser(size):
    answer, reasoning, _ = run(TOOL, size)
    assert reasoning == "Need use get_weather tool."
    content, calls = parse_tool_calls(answer)
    assert content == ""
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Lomé", "days": 3}


def test_string_arguments_that_look_like_numbers_stay_strings():
    text = ('<|message_model|>f<|content_invoke_tool_json|>'
            '{"name":"f","args":{"zip":"01234","n":[1,"a"],"s":"</x>"}}<|end_message|>')
    answer, _, _ = run(text, 4)
    _, calls = parse_tool_calls(answer)
    assert json.loads(calls[0]["function"]["arguments"]) == {"zip": "01234", "n": [1, "a"], "s": "</x>"}


def test_text_with_marker_lookalikes_is_kept():
    text = "<|message_model|><|content_text|>use a <|b in code, or <| x |> too<|end_message|>"
    for size in (1, 3, 1000):
        assert run(text, size)[0] == "use a <|b in code, or <| x |> too"


def test_unreadable_tool_json_is_shown_not_lost():
    text = "<|message_model|>f<|content_invoke_tool_json|>{not json<|end_message|>"
    assert run(text, 1000)[0] == "{not json"


def test_cut_off_mid_call_still_gives_what_came():
    text = '<|message_model|>f<|content_invoke_tool_json|>{"name":"f","args":{}}'
    _, calls = parse_tool_calls(run(text, 3)[0])
    assert calls[0]["function"]["name"] == "f"


def test_inkling_loads_with_mlx_lm_not_mlx_vlm(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "inkling2_loop"}))
    import sys
    import types
    # stand-ins: the unit job runs without MLX
    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(load=lambda path: ("model", "tok")))
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.SimpleNamespace(
        load=lambda path: pytest.fail("mlx-vlm asked")))
    runner = loader.load_runner(tmp_path)
    assert isinstance(runner, loader.InklingRunner)
    assert isinstance(runner.output_parser(), InklingParser)


def test_thinking_off_asks_for_no_effort():
    seen = {}

    class Tok:
        def apply_chat_template(self, messages, **kw):
            seen.update(kw)
            return [1]

    runner = loader.InklingRunner(None, Tok())
    runner.template([{"role": "user", "content": "hi"}], thinking=False)
    assert seen["reasoning_effort"] == "none"
    seen.clear()
    runner.template([{"role": "user", "content": "hi"}], thinking=True)
    assert "reasoning_effort" not in seen and "enable_thinking" not in seen


def test_inkling_models_are_served_from_their_own_runtime(tmp_path, monkeypatch):
    from mlxh import cli
    model = tmp_path / "ink"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "inkling2_loop"}))
    other = tmp_path / "qwen"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    monkeypatch.setattr(cli, "_ensure_inkling_runtime", lambda **_: "/inkling/python")
    cfg = {k: 1 for k in ("port", "host", "max_queued", "max_tokens_cap", "memory_limit_gb",
                          "cache_limit_gb", "gen_timeout_s", "max_prompt_tokens", "prompt_cache",
                          "prompt_cache_gb", "thinking", "max_image_pixels", "image_steps")}
    argv = cli.serve_argv(cfg, "ink", model)
    assert argv[:2] == ["/inkling/python", "-I"]
    assert cli.serve_argv(cfg, "qwen", other)[0] == cli.sys.executable


def test_inkling_runtime_pins_the_reviewed_commit():
    from mlxh.image_runtime import INKLING, code_identity
    assert INKLING.version in INKLING.requirements[0] and INKLING.overrides == INKLING.requirements
    assert code_identity(INKLING)["mlx-lm"] == INKLING.version
    assert "mflux" not in code_identity(INKLING)


def test_is_inkling_model_reads_config_safely(tmp_path):
    from mlxh.inkling import is_inkling_model
    assert not is_inkling_model(tmp_path)
    (tmp_path / "config.json").write_text("not json")
    assert not is_inkling_model(tmp_path)


@pytest.mark.parametrize("size", [1, 4, 1000])
def test_tool_arguments_with_xml_and_markers_survive(size):
    args = {"s": "abc</parameter>def", "m": "abc<|xyz|>def", "t": "x<|content_text|>y", "c": "<tool_call>"}
    text = ('<|message_model|>f<|content_invoke_tool_json|>'
            + json.dumps({"name": "f", "args": args}) + "<|end_message|>")
    content, calls = parse_tool_calls(run(text, size)[0])
    assert content == ""
    assert json.loads(calls[0]["function"]["arguments"]) == args


def test_unknown_marker_shaped_text_is_kept_in_answers():
    text = "<|message_model|><|content_text|>abc<|xyz|>def<|end_message|>"
    for size in (1, 1000):
        assert run(text, size)[0] == "abc<|xyz|>def"


def test_tool_results_keep_which_call_they_answer():
    from mlxh.engine import InferenceEngine
    messages, _, _ = InferenceEngine._extract_messages(
        type("E", (), {"supports_images": False})(),
        [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": "{\"city\": \"Lom\\u00e9\"}"}}]},
         {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"}])
    assert messages[0]["tool_calls"][0] == {"id": "call_1", "function": {
        "name": "get_weather", "arguments": {"city": "Lomé"}}}
    assert messages[1]["tool_call_id"] == "call_1"


def test_endoftext_inside_tool_json_is_kept_and_header_name_trimmed():
    text = ('<|message_model|> f\n<|content_invoke_tool_json|>'
            '{"args":{"s":"a<|endoftext|>b"}}<|end_message|>')
    _, calls = parse_tool_calls(run(text, 3)[0])
    assert calls[0]["function"]["name"] == "f"
    assert json.loads(calls[0]["function"]["arguments"]) == {"s": "a<|endoftext|>b"}
