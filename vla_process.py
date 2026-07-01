"""VLA inference process client interface."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib import error, request
from urllib.parse import urljoin

from libero_evaluator import Action, Observation, TaskInfo


VLA_PAYLOAD_SCHEMA_VERSION = 1


class VLAClient(Protocol):
    def check_service_ready(self) -> bool:
        """Return True when the model service is ready."""

    def fetch_model_metadata(self) -> dict[str, Any]:
        """Return model input/output conventions."""

    def reset_model_for_episode(self, task: TaskInfo) -> None:
        """Reset model-side recurrent state or action chunk cache."""

    def request_action_prediction(self, observation: Observation) -> Action:
        """Return one LIBERO-compatible action."""

    def close(self) -> None:
        """Release client resources."""


@dataclass
class HTTPVLAClient:
    base_url: str
    timeout: float = 30.0
    closed: bool = False

    def check_service_ready(self) -> bool:
        if self.closed:
            return False

        try:
            payload = self._request_json("GET", "/health")
        except (OSError, error.URLError, TimeoutError):
            return False
        return bool(payload.get("ok", False))

    def fetch_model_metadata(self) -> dict[str, Any]:
        self._ensure_open()
        return self._request_json("GET", "/metadata")

    def reset_model_for_episode(self, task: TaskInfo) -> None:
        self._ensure_open()
        self._request_json("POST", "/reset_episode", task_to_payload(task))

    def request_action_prediction(self, observation: Observation) -> Action:
        self._ensure_open()
        payload = self._request_json("POST", "/predict_action", observation_to_payload(observation))
        return payload_to_action(payload)

    def close(self) -> None:
        self.closed = True

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("HTTPVLAClient is closed")

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(
            urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/")),
            data=body,
            headers=headers,
            method=method,
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            response_body = response.read()

        if not response_body:
            return {}
        return json.loads(response_body.decode("utf-8"))


def task_to_payload(task: TaskInfo) -> dict[str, Any]:
    return {
        "schema_version": VLA_PAYLOAD_SCHEMA_VERSION,
        "suite": task.suite,
        "task_id": task.task_id,
        "episode_id": task.episode_id,
        "instruction": task.instruction,
        "seed": task.seed,
    }


def observation_to_payload(observation: Observation) -> dict[str, Any]:
    return {
        "schema_version": VLA_PAYLOAD_SCHEMA_VERSION,
        "instruction": observation.instruction,
        "step": observation.step,
        "robot_state": [float(value) for value in observation.robot_state],
        "images": {
            "agentview": image_to_payload(observation.agentview_image),
            "wrist": image_to_payload(observation.wrist_image),
        },
    }


def action_to_payload(action: Action) -> dict[str, Any]:
    return {
        "schema_version": VLA_PAYLOAD_SCHEMA_VERSION,
        "action": _validated_action_values(action.values),
    }


def payload_to_action(payload: dict[str, Any]) -> Action:
    return Action(_validated_action_values(payload["action"]))


def image_to_payload(image: Any) -> dict[str, Any]:
    if not hasattr(image, "shape") or not hasattr(image, "dtype") or not hasattr(image, "tobytes"):
        raise TypeError("Image must expose shape, dtype, and tobytes() for JSON serialization")

    raw_bytes = image.tobytes()
    return {
        "encoding": "raw_base64",
        "shape": [int(value) for value in image.shape],
        "dtype": str(image.dtype),
        "data": base64.b64encode(raw_bytes).decode("ascii"),
    }


def _validated_action_values(values: list[float]) -> list[float]:
    if len(values) != 7:
        raise ValueError(f"VLA action must have 7 values, got {len(values)}")
    return [float(value) for value in values]


@dataclass
class DummyVLAClient:
    action_values: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    current_task: TaskInfo | None = None
    num_predictions: int = 0
    closed: bool = False

    def check_service_ready(self) -> bool:
        return not self.closed

    def fetch_model_metadata(self) -> dict[str, Any]:
        return {
            "name": "dummy-vla",
            "schema_version": VLA_PAYLOAD_SCHEMA_VERSION,
            "action_dim": 7,
            "action_format": "[dx, dy, dz, droll, dpitch, dyaw, gripper]",
            "image_keys": ["agentview", "wrist"],
            "state_dim": 8,
            "uses_model": False,
        }

    def reset_model_for_episode(self, task: TaskInfo) -> None:
        self.current_task = task
        self.num_predictions = 0

    def request_action_prediction(self, observation: Observation) -> Action:
        if self.closed:
            raise RuntimeError("DummyVLAClient is closed")
        if len(self.action_values) != 7:
            raise ValueError(f"Dummy action must have 7 values, got {len(self.action_values)}")

        self.num_predictions += 1
        return Action([float(value) for value in self.action_values])

    def close(self) -> None:
        self.closed = True
