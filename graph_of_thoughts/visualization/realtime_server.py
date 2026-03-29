from __future__ import annotations

import json
import threading
from collections import defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Any
from urllib.parse import parse_qs, urlparse


class EventStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._seq: Dict[str, int] = defaultdict(int)

    def publish(self, run_id: str, payload: Dict[str, Any]) -> int:
        rid = str(run_id or "default")
        with self._lock:
            self._seq[rid] += 1
            item = {"seq": self._seq[rid], "payload": payload}
            self._events[rid].append(item)
            return item["seq"]

    def get_since(self, run_id: str, last_seq: int = 0) -> List[Dict[str, Any]]:
        rid = str(run_id or "default")
        with self._lock:
            return [e for e in self._events.get(rid, []) if int(e["seq"]) > int(last_seq)]

    def list_runs(self) -> Dict[str, Any]:
        with self._lock:
            by_method: Dict[str, List[str]] = defaultdict(list)
            latest = {"method": "", "sample_id": ""}
            for rid, events in self._events.items():
                if ":" not in rid:
                    continue
                method, sample_id = rid.split(":", 1)
                if sample_id not in by_method[method]:
                    by_method[method].append(sample_id)
                if events:
                    latest["method"] = method
                    latest["sample_id"] = sample_id
            return {"by_method": by_method, "latest": latest}


def _build_handler(store: EventStore):
    class RealtimeHandler(BaseHTTPRequestHandler):
        def _send_cors(self) -> None:
            # 允许前后端分离部署（不同 origin）
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _send_json(self, body: Dict[str, Any], status: int = 200) -> None:
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self._send_cors()
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self._send_cors()
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                msg = (
                    "GoT realtime API server is running.\n\n"
                    "This server provides:\n"
                    "- GET /health\n"
                    "- GET /runs\n"
                    "- GET /events?run_id=<method:id>&last_seq=<n>\n\n"
                    "Frontend is separated. Please use the Vue app in `frontend/`.\n"
                ).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self._send_cors()
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                return
            if parsed.path == "/health":
                self._send_json({"ok": True})
                return
            if parsed.path == "/events":
                q = parse_qs(parsed.query)
                run_id = (q.get("run_id") or ["default"])[0]
                try:
                    last_seq = int((q.get("last_seq") or ["0"])[0])
                except Exception:
                    last_seq = 0
                self._send_json({"events": store.get_since(run_id, last_seq)})
                return
            if parsed.path == "/runs":
                self._send_json(store.list_runs())
                return
            self._send_json({"error": "not found"}, status=404)

        def log_message(self, fmt, *args):  # noqa: N802
            return

    return RealtimeHandler


def start_realtime_server(
    store: EventStore,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    handler = _build_handler(store)
    server = ThreadingHTTPServer((host, int(port)), handler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return server
