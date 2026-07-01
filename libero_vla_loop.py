"""Communication loop between a LIBERO evaluator process and a VLA service.
This file intentionally keeps the evaluator and model service separated:
- LIBEROEvaluator owns simulation, task reset, stepping, success checks, videos.
- VLAClient owns communication with a model inference process.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class TaskInfo:
    suite: str
    task_id: int
    episode_id: int
    instruction: str
    seed: int


@dataclass(frozen=True)
class Observation:
    instruction: str
    agentview_image: Any
    wrist_image: Any
    robot_state: list[float]
    step: int


@dataclass(frozen=True)
class Action:
    values: list[float]  # [dx, dy, dz, droll, dpitch, dyaw, gripper]


class VLAClient(Protocol):
    def health(self) -> bool:
        """Return True when the model service is ready."""

    def metadata(self) -> dict[str, Any]:
        """Return model input/output conventions."""

    def reset_episode(self, task: TaskInfo) -> None:
        """Reset model-side recurrent state or action chunk cache."""

    def predict_action(self, observation: Observation) -> Action:
        """Return one LIBERO-compatible action."""

    def close(self) -> None:
        """Release client resources."""


class LIBEROEvaluator(Protocol):
    def load_suite(self, suite: str) -> None:
        """Load a LIBERO benchmark suite."""

    def reset_task(self, task_id: int, episode_id: int, seed: int) -> TaskInfo:
        """Reset simulation and return task metadata."""

    def get_observation(self, step: int) -> Observation:
        """Return the current image/state/instruction observation."""

    def step(self, action: Action) -> None:
        """Execute one action in simulation."""

    def is_success(self) -> bool:
        """Return whether current episode has succeeded."""

    def is_timeout(self) -> bool:
        """Return whether current episode has reached max steps."""

    def save_video(self, task: TaskInfo, success: bool) -> None:
        """Persist rollout video."""


def run_episode(
    evaluator: LIBEROEvaluator,
    policy: VLAClient,
    task_id: int,
    episode_id: int,
    seed: int,
) -> bool:
    task = evaluator.reset_task(task_id=task_id, episode_id=episode_id, seed=seed)
    policy.reset_episode(task)

    step = 0
    while not evaluator.is_success() and not evaluator.is_timeout():
        observation = evaluator.get_observation(step=step)
        action = policy.predict_action(observation)
        evaluator.step(action)
        step += 1

    success = evaluator.is_success()
    evaluator.save_video(task=task, success=success)
    return success


def run_benchmark(
    evaluator: LIBEROEvaluator,
    policy: VLAClient,
    suite: str,
    task_ids: list[int],
    episodes_per_task: int,
    seed: int,
) -> dict[str, Any]:
    if not policy.health():
        raise RuntimeError("VLA service is not ready")

    evaluator.load_suite(suite)
    model_info = policy.metadata()

    total = 0
    successes = 0
    results: list[dict[str, Any]] = []

    for task_id in task_ids:
        for episode_id in range(episodes_per_task):
            success = run_episode(
                evaluator=evaluator,
                policy=policy,
                task_id=task_id,
                episode_id=episode_id,
                seed=seed,
            )
            total += 1
            successes += int(success)
            results.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "success": success,
                }
            )

    return {
        "suite": suite,
        "model": model_info,
        "total": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "results": results,
    }
