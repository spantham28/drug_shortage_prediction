from http.server import BaseHTTPRequestHandler
import json
import traceback

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from inference import get_income_feature_schema, predict_income, predict_shortage


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        try:
            if self.path.endswith("/features"):
                self._send_json(200, get_income_feature_schema())
            else:
                self._send_json(200, {"status": "ok", "endpoint": "shortage"})
        except Exception as exc:
            self._send_json(500, {"error": str(exc), "trace": traceback.format_exc()})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(raw) if raw else {}
            result = predict_shortage(payload)
            self._send_json(200, result)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": str(exc), "trace": traceback.format_exc()})
