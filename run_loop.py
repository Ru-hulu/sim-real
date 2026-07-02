"""Runtime loops that connect a LIBERO evaluator with a VLA process."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from libero_evaluator import LIBEROEvaluator, LIBEROSimulationRuntime, LIBEROSimulationRuntimeConfig
from vla_process import HTTPVLAClient, VLAClient


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

def parse_task_ids(raw_task_ids: list[str]) -> list[int]:
    task_ids: list[int] = []
    for raw_task_id in raw_task_ids:
        for item in raw_task_id.split(","):
            item = item.strip()
            if item:
                task_ids.append(int(item))
    if not task_ids:
        raise ValueError("At least one task id is required")
    return task_ids


def run_http_vla_benchmark(
    suite: str,
    task_ids: list[int],
    episodes_per_task: int,
    seed: int,
    max_steps: int,
    output_dir: pathlib.Path,
    libero_home: pathlib.Path | None,
    image_resolution: int,
    vla_server_url: str,
    vla_timeout: float,
) -> dict[str, Any]:
    evaluator = LIBEROSimulationRuntime(
        LIBEROSimulationRuntimeConfig(
            output_dir=output_dir,
            libero_home=libero_home,
            image_resolution=image_resolution,
            max_steps=max_steps,
        )
    )
    policy = HTTPVLAClient(base_url=vla_server_url, timeout=vla_timeout)
    try:
        return run_benchmark(
            evaluator=evaluator,
            policy=policy,
            suite=suite,
            task_ids=task_ids,
            episodes_per_task=episodes_per_task,
            seed=seed,
        )
    finally:
        policy.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LIBERO benchmark episodes with an HTTP VLA server.")
    parser.add_argument("--vla-server-url", required=True, help="Base URL of the VLA HTTP server.")
    parser.add_argument("--vla-timeout", type=float, default=30.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--suite", default="libero_goal", help="LIBERO benchmark suite name.")
    parser.add_argument("--task-ids", nargs="+", default=["0"], help="Task ids, e.g. '0 1 2' or '0,1,2'.")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--image-resolution", type=int, default=256)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/http_vla_benchmark"))
    parser.add_argument("--libero-home", type=pathlib.Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_http_vla_benchmark(
        suite=args.suite,
        task_ids=parse_task_ids(args.task_ids),
        episodes_per_task=args.episodes_per_task,
        seed=args.seed,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        libero_home=args.libero_home,
        image_resolution=args.image_resolution,
        vla_server_url=args.vla_server_url,
        vla_timeout=args.vla_timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
