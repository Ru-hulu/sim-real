"""VLA inference process client interface."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error, request
from urllib.parse import urljoin

from libero_evaluator import Action, Observation, TaskInfo


VLA_PAYLOAD_SCHEMA_VERSION = 1

# 仅做接口定义，无具体实现。
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

# 向模型请求推理，获得动作
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
    images = {
        "agentview": array_to_payload(observation.agentview_image),
        "wrist": array_to_payload(observation.wrist_image),
    }
    images.update(array_dict_to_payload(observation.extra_images))

    return {
        "schema_version": VLA_PAYLOAD_SCHEMA_VERSION,
        "instruction": observation.instruction,
        "step": observation.step,
        "robot_state": [float(value) for value in observation.robot_state],
        "robot_joint_state": vector_dict_to_payload(observation.robot_joint_state),
        "gripper_state": vector_dict_to_payload(observation.gripper_state),
        "robot_proprio_state": vector_dict_to_payload(observation.robot_proprio_state),
        "images": images,
        "depth_maps": array_dict_to_payload(observation.depth_maps),
        "segmentation_maps": array_dict_to_payload(observation.segmentation_maps),
    }


def action_to_payload(action: Action) -> dict[str, Any]:
    return {
        "schema_version": VLA_PAYLOAD_SCHEMA_VERSION,
        "action": _validated_action_values(action.values),
    }


def payload_to_action(payload: dict[str, Any]) -> Action:
    return Action(_validated_action_values(payload["action"]))


def vector_dict_to_payload(items: dict[str, list[float]]) -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in items.items()}


def array_dict_to_payload(items: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {key: array_to_payload(value) for key, value in items.items()}


def array_to_payload(value: Any) -> dict[str, Any]:
    if not hasattr(value, "shape") or not hasattr(value, "dtype") or not hasattr(value, "tobytes"):
        raise TypeError("Array must expose shape, dtype, and tobytes() for JSON serialization")

    raw_bytes = value.tobytes()
    return {
        "encoding": "raw_base64",
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "data": base64.b64encode(raw_bytes).decode("ascii"),
    }


def _validated_action_values(values: list[float]) -> list[float]:
    if len(values) != 7:
        raise ValueError(f"VLA action must have 7 values, got {len(values)}")
    return [float(value) for value in values]
