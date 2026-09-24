"""Small standard-library client for mlxh's private terminal transport."""

from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path


class TransportError(Exception):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def image_data_url(path):
    media = mimetypes.guess_type(path)[0] or "image/png"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{media};base64,{encoded}"


class ChatTransport:
    def __init__(self, base_url, timeout=900):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.active_request_id = None

    def info(self):
        try:
            with urllib.request.urlopen(f"{self.base_url}/mlxh/info", timeout=2) as response:
                return json.loads(response.read())
        except Exception as exc:
            raise TransportError(None, f"could not reach mlxh server: {exc}") from None

    @staticmethod
    def _error(exc):
        try:
            payload = json.loads(exc.read())
            message = payload.get("detail") or payload.get("error", {}).get("message")
        except Exception:
            message = None
        return TransportError(exc.code, message or str(exc))

    def generate(self, body):
        data = json.dumps(body, separators=(",", ":")).encode()
        request = urllib.request.Request(
            f"{self.base_url}/mlxh/generate", data=data,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise self._error(exc) from None
        event, data_lines = None, []
        try:
            for raw in response:
                line = raw.decode("utf-8").rstrip("\r\n")
                if not line:
                    if event and data_lines:
                        try:
                            payload = json.loads("\n".join(data_lines))
                        except json.JSONDecodeError as exc:
                            raise TransportError(None, f"malformed server event: {exc}") from None
                        if event == "start":
                            self.active_request_id = payload.get("request_id")
                        yield event, payload
                    event, data_lines = None, []
                elif line.startswith(":"):
                    continue
                elif line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if event or data_lines:
                raise TransportError(None, "server stream ended mid-event")
        except KeyboardInterrupt:
            self.cancel()
            raise
        finally:
            response.close()
            self.active_request_id = None

    def cancel(self):
        request_id = self.active_request_id
        if not request_id:
            return False
        request = urllib.request.Request(
            f"{self.base_url}/mlxh/requests/{request_id}", method="DELETE"
        )
        try:
            with urllib.request.urlopen(request, timeout=2):
                return True
        except Exception:
            return False
