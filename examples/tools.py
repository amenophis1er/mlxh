"""Example tools for mlxh chat.

To use: copy this file to ~/.mlxh/tools.py and start a chat with --tools
(or set it as the default with `mlxh config chat_tools on`):

    cp examples/tools.py ~/.mlxh/tools.py
    mlxh chat <model> -- --tools

The contract: TOOL_REGISTRY maps tool names to callables taking keyword
arguments and returning a JSON-serializable dict; TOOL_SPECS describes the
same tools in the OpenAI function-tool format so the model knows about them.

Note: get_weather calls the free Open-Meteo API over the network.
"""

import ast
import json
import operator
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo


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
