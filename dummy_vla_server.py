"""Standalone dummy VLA HTTP server for simulator integration tests."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlsplit

from libero_evaluator import Action
from vla_process import VLA_PAYLOAD_SCHEMA_VERSION, action_to_payload


DEFAULT_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
_dummy_action_values = list(DEFAULT_DUMMY_ACTION)
_current_task: dict[str, Any] | None = None
_num_predictions = 0


def configure_dummy_action(action_values: list[float]) -> None:
    global _dummy_action_values
    if len(action_values) != 7:
        raise ValueError(f"Dummy action must have 7 values, got {len(action_values)}")
    _dummy_action_values = [float(value) for value in action_values]


def reset_model_for_episode(task_payload: dict[str, Any]) -> None:
    global _current_task, _num_predictions
    _current_task = task_payload
    _num_predictions = 0


def predict_dummy_action(observation_payload: dict[str, Any]) -> dict[str, Any]:
    global _num_predictions
    validate_observation_payload(observation_payload)
    _num_predictions += 1
    return action_to_payload(Action(_dummy_action_values))


def model_metadata() -> dict[str, Any]:
    return {
        "name": "dummy-vla-server",
        "schema_version": VLA_PAYLOAD_SCHEMA_VERSION,
        "action_dim": 7,
        "action_format": "[dx, dy, dz, droll, dpitch, dyaw, gripper]",
        "image_keys": ["agentview", "wrist"],
        "state_dim": 8,
        "uses_model": False,
        "num_predictions": _num_predictions,
    }


def validate_observation_payload(payload: dict[str, Any]) -> None:
    required_keys = {"instruction", "step", "robot_state", "images"}
    missing_keys = required_keys - payload.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Observation payload missing keys: {missing}")

    robot_state = payload["robot_state"]
    if not isinstance(robot_state, list):
        raise TypeError("Observation robot_state must be a list")

    images = payload["images"]
    if not isinstance(images, dict):
        raise TypeError("Observation images must be an object")


class DummyVLARequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    # 响应 VLAClient 发来的 GET 请求：
    # /health 用于检查 VLA server 是否可用。
    # /metadata 用于返回 dummy VLA 的输入输出约定。
    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/health":
            self._write_json({"ok": True})
            return
        if path == "/metadata":
            self._write_json(model_metadata())
            return
        self.send_error(404, f"Unknown endpoint: {path}")

    # 响应 VLAClient 发来的 POST 请求：
    # /reset_episode 用于开始新 episode 时重置 dummy VLA 状态。
    # /predict_action 用于接收 observation，并返回 dummy action。
    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in {"/reset_episode", "/predict_action"}:
            self.send_error(404, f"Unknown endpoint: {path}")
            return

        try:
            payload = self._read_json()
            if path == "/reset_episode":
                reset_model_for_episode(payload)
                self._write_json({"ok": True})
                return
            self._write_json(predict_dummy_action(payload))
        except json.JSONDecodeError as exc:
            self.send_error(400, f"Invalid JSON: {exc.msg}")
        except (TypeError, ValueError) as exc:
            self.send_error(400, str(exc))

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Request body must be a JSON object")
        return payload

    def _write_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dummy VLA HTTP server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument(
        "--action",
        nargs=7,
        type=float,
        default=DEFAULT_DUMMY_ACTION,
        metavar=("DX", "DY", "DZ", "DROLL", "DPITCH", "DYAW", "GRIPPER"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_dummy_action(list(args.action))
    server = HTTPServer((args.host, args.port), DummyVLARequestHandler)
    print(f"Dummy VLA server listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
