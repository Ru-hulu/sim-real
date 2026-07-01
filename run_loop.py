"""Runtime loops that connect a LIBERO evaluator with a VLA process."""

from __future__ import annotations

from typing import Any

from libero_evaluator import LIBEROEvaluator
from vla_process import VLAClient


def run_step(evaluator: LIBEROEvaluator, policy: VLAClient, step: int) -> None:
    observation = evaluator.get_observation(step=step)
    action = policy.request_action_prediction(observation)
    evaluator.step(action)


def run_episode(
    evaluator: LIBEROEvaluator,
    policy: VLAClient,
    task_id: int,
    episode_id: int,
    seed: int,
) -> bool:
    task = evaluator.reset_task(task_id=task_id, episode_id=episode_id, seed=seed)
    policy.reset_model_for_episode(task)

    step = 0
    while not evaluator.is_success() and not evaluator.is_timeout():
        run_step(evaluator=evaluator, policy=policy, step=step)
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
    if not policy.check_service_ready():
        raise RuntimeError("VLA service is not ready")

    evaluator.load_suite(suite)
    model_info = policy.fetch_model_metadata()

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
