"""LIBERO simulation process interfaces and implementation."""

from __future__ import annotations

import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LIBEROEvaluatorConfig:
    output_dir: pathlib.Path
    libero_home: pathlib.Path | None = None
    mujoco_gl: str = "egl"
    pyopengl_platform: str = "egl"
    image_resolution: int = 256
    max_steps: int = 300


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


class RealLIBEROEvaluator:
    """Concrete evaluator process owner for LIBERO.

    This first implementation step only owns real process initialization:
    environment variables, import path setup, benchmark registry loading, and
    result directory creation.
    """

    def __init__(self, config: LIBEROEvaluatorConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._configure_process_environment(config)
        self.benchmark_dict = self._load_benchmark_registry()
        self.available_suites = sorted(self.benchmark_dict.keys())

        self.current_suite_name: str | None = None
        self.current_suite: Any | None = None
        self.current_env: Any | None = None
        self.current_task: TaskInfo | None = None
        self.current_task_spec: Any | None = None
        self.current_initial_states: Any | None = None
        self.current_initial_state: Any | None = None
        self.current_episode_dir: pathlib.Path | None = None
        self.current_step = 0
        self.replay_images: list[Any] = []

    @staticmethod
    def _configure_process_environment(config: LIBEROEvaluatorConfig) -> None:
        os.environ["MUJOCO_GL"] = config.mujoco_gl
        os.environ["PYOPENGL_PLATFORM"] = config.pyopengl_platform
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        if config.libero_home is None:
            return

        libero_home = str(config.libero_home.expanduser().resolve())
        if libero_home not in sys.path:
            sys.path.insert(0, libero_home)

    @staticmethod
    def _load_benchmark_registry() -> dict[str, Any]:
        from libero.libero import benchmark
        return benchmark.get_benchmark_dict()

    def load_suite(self, suite: str) -> None:
        if suite not in self.benchmark_dict:
            available = ", ".join(self.available_suites)
            raise ValueError(f"Unknown LIBERO suite '{suite}'. Available suites: {available}")

        suite_instance = self.benchmark_dict[suite]()
        if not hasattr(suite_instance, "n_tasks"):
            raise TypeError(f"LIBERO suite '{suite}' does not expose n_tasks")

        self.current_suite_name = suite
        self.current_suite = suite_instance
        self._clear_current_task()

        suite_dir = self.output_dir / suite
        suite_dir.mkdir(parents=True, exist_ok=True)

    def reset_task(self, task_id: int, episode_id: int, seed: int) -> TaskInfo:
        suite = self._require_current_suite()
        suite_name = self._require_current_suite_name()
        self._validate_task_id(suite=suite, task_id=task_id)

        task_spec = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        self._validate_episode_id(
            episode_id=episode_id,
            num_initial_states=len(initial_states),
            task_id=task_id,
        )

        task = TaskInfo(
            suite=suite_name,
            task_id=task_id,
            episode_id=episode_id,
            instruction=task_spec.language,
            seed=seed,
        )

        self.current_task = task
        self.current_task_spec = task_spec
        self.current_initial_states = initial_states
        self.current_initial_state = initial_states[episode_id]
        self.current_step = 0
        self.replay_images = []
        self.current_episode_dir = self._episode_output_dir(task)
        self.current_episode_dir.mkdir(parents=True, exist_ok=True)
        return task

    def get_observation(self, step: int) -> Observation:
        raise NotImplementedError("Observation extraction will be implemented next.")

    def step(self, action: Action) -> None:
        raise NotImplementedError("Simulation stepping will be implemented next.")

    def is_success(self) -> bool:
        raise NotImplementedError("Success checking will be implemented next.")

    def is_timeout(self) -> bool:
        raise NotImplementedError("Timeout checking will be implemented next.")

    def save_video(self, task: TaskInfo, success: bool) -> None:
        raise NotImplementedError("Video saving will be implemented next.")

    def _clear_current_task(self) -> None:
        self.current_env = None
        self.current_task = None
        self.current_task_spec = None
        self.current_initial_states = None
        self.current_initial_state = None
        self.current_episode_dir = None
        self.current_step = 0
        self.replay_images = []

    def _require_current_suite(self) -> Any:
        if self.current_suite is None:
            raise RuntimeError("No LIBERO suite loaded. Call load_suite() first.")
        return self.current_suite

    def _require_current_suite_name(self) -> str:
        if self.current_suite_name is None:
            raise RuntimeError("No LIBERO suite loaded. Call load_suite() first.")
        return self.current_suite_name

    @staticmethod
    def _validate_task_id(suite: Any, task_id: int) -> None:
        if task_id < 0 or task_id >= suite.n_tasks:
            raise IndexError(f"task_id {task_id} is out of range [0, {suite.n_tasks})")

    @staticmethod
    def _validate_episode_id(episode_id: int, num_initial_states: int, task_id: int) -> None:
        if episode_id < 0 or episode_id >= num_initial_states:
            raise IndexError(
                f"episode_id {episode_id} is out of range [0, {num_initial_states}) "
                f"for task_id {task_id}"
            )

    def _episode_output_dir(self, task: TaskInfo) -> pathlib.Path:
        return self.output_dir / task.suite / f"task_{task.task_id:03d}" / f"episode_{task.episode_id:03d}"
