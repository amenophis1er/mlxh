"""Tool-call parsing (Qwen-style XML) and the built-in local tools."""

import ast
import json
import operator
import re
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S
)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_tool_calls(text):
    """Split model output into (content, openai_tool_calls_or_None)."""
    text = re.sub(r"^\s*(<think>)?.*?</think>\s*", "", text, count=1, flags=re.S)
    calls = []
    for name, body in TOOL_CALL_RE.findall(text):
        args = {}
        for key, value in PARAM_RE.findall(body):
            try:
                args[key] = json.loads(value)
            except json.JSONDecodeError:
                args[key] = value
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    content = TOOL_CALL_RE.sub("", text).strip()
    return content, calls or None


# ------------------------------------------------- built-in local tools

def get_weather(city, unit="celsius"):
    def fetch(url):
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.load(r)

    geo = fetch(
        "https://geocoding-api.open-meteo.com/v1/search?count=1&name="
        + urllib.parse.quote(city)
    )
    if not geo.get("results"):
        return {"error": f"city not found: {city}"}
    place = geo["results"][0]
    temp_unit = "fahrenheit" if unit == "fahrenheit" else "celsius"
    wx = fetch(
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={place['latitude']}&longitude={place['longitude']}"
        f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
        f"&temperature_unit={temp_unit}"
    )
    cur = wx.get("current", {})
    return {
        "location": f"{place['name']}, {place.get('country', '')}".strip(", "),
        "temperature": cur.get("temperature_2m"),
        "unit": temp_unit,
        "humidity_pct": cur.get("relative_humidity_2m"),
        "wind_kmh": cur.get("wind_speed_10m"),
        "weather_code": cur.get("weather_code"),
    }


def get_current_time(timezone="local"):
    if timezone == "local":
        now = datetime.now().astimezone()
    else:
        try:
            now = datetime.now(ZoneInfo(timezone))
        except Exception:
            return {"error": f"unknown timezone: {timezone}"}
    return {"time": now.strftime("%Y-%m-%d %H:%M:%S %Z")}


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("only arithmetic expressions are allowed")


def calculate(expression):
    try:
        return {"result": _safe_eval(ast.parse(expression, mode="eval").body)}
    except Exception as e:
        return {"error": str(e)}


TOOL_REGISTRY = {
    "get_weather": get_weather,
    "get_current_time": get_current_time,
    "calculate": calculate,
}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get live current weather for a city",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Get the current date and time. Pass an IANA timezone like 'Asia/Tokyo', or 'local'.",
        "parameters": {"type": "object", "properties": {
            "timezone": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression, e.g. '17 * 23 / 4'",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}}, "required": ["expression"]}}},
]


def run_tool(name, args):
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"bad arguments: {e}"}
    except Exception as e:
        return {"error": str(e)}
