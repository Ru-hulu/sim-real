"""LIBERO simulation process interfaces and implementation."""

from __future__ import annotations

import math
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

    Owns LIBERO process initialization, benchmark selection, task reset, and
    observation extraction.
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
        self.current_obs: dict[str, Any] | None = None
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
        suite_name, suite = self._require_loaded_suite()
        if task_id < 0 or task_id >= suite.n_tasks:
            raise IndexError(f"task_id {task_id} is out of range [0, {suite.n_tasks})")

        task_spec = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        if episode_id < 0 or episode_id >= len(initial_states):
            raise IndexError(
                f"episode_id {episode_id} is out of range [0, {len(initial_states)}) "
                f"for task_id {task_id}"
            )

        task = TaskInfo(
            suite=suite_name,
            task_id=task_id,
            episode_id=episode_id,
            instruction=task_spec.language,
            seed=seed,
        )

        self.current_task = task
        self._close_current_env()
        self.current_env = self._create_env(task_spec=task_spec, seed=seed)
        self.current_obs = self._reset_env_to_initial_state(initial_states[episode_id])
        self.current_step = 0
        self.replay_images = []
        self.current_episode_dir = self._episode_output_dir(task)
        self.current_episode_dir.mkdir(parents=True, exist_ok=True)
        return task

    def get_observation(self, step: int) -> Observation:
        task = self._require_current_task()
        obs = self._require_current_obs()
        agentview_image = self._extract_image(obs, "agentview_image")
        wrist_image = self._extract_image(obs, "robot0_eye_in_hand_image")
        robot_state = self._extract_robot_state(obs)

        self.current_step = step
        self.replay_images.append(agentview_image)
        return Observation(
            instruction=task.instruction,
            agentview_image=agentview_image,
            wrist_image=wrist_image,
            robot_state=robot_state,
            step=step,
        )

    def step(self, action: Action) -> None:
        raise NotImplementedError("Simulation stepping will be implemented next.")

    def is_success(self) -> bool:
        raise NotImplementedError("Success checking will be implemented next.")

    def is_timeout(self) -> bool:
        raise NotImplementedError("Timeout checking will be implemented next.")

    def save_video(self, task: TaskInfo, success: bool) -> None:
        raise NotImplementedError("Video saving will be implemented next.")

    def _clear_current_task(self) -> None:
        self._close_current_env()
        self.current_task = None
        self.current_obs = None
        self.current_episode_dir = None
        self.current_step = 0
        self.replay_images = []

    def _require_loaded_suite(self) -> tuple[str, Any]:
        if self.current_suite_name is None or self.current_suite is None:
            raise RuntimeError("No LIBERO suite loaded. Call load_suite() first.")
        return self.current_suite_name, self.current_suite

    def _episode_output_dir(self, task: TaskInfo) -> pathlib.Path:
        return self.output_dir / task.suite / f"task_{task.task_id:03d}" / f"episode_{task.episode_id:03d}"

    def _close_current_env(self) -> None:
        if self.current_env is not None and hasattr(self.current_env, "close"):
            self.current_env.close()
        self.current_env = None

    def _create_env(self, task_spec: Any, seed: int) -> Any:
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        task_bddl_file = (
            pathlib.Path(get_libero_path("bddl_files"))
            / task_spec.problem_folder
            / task_spec.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=self.config.image_resolution,
            camera_widths=self.config.image_resolution,
        )
        env.seed(seed)
        return env

    def _reset_env_to_initial_state(self, initial_state: Any) -> dict[str, Any]:
        if self.current_env is None:
            raise RuntimeError("No LIBERO env exists for the current task.")

        self.current_env.reset()
        return self.current_env.set_init_state(initial_state)

    def _require_current_task(self) -> TaskInfo:
        if self.current_task is None:
            raise RuntimeError("No task selected. Call reset_task() first.")
        return self.current_task

    def _require_current_obs(self) -> dict[str, Any]:
        if self.current_obs is None:
            raise RuntimeError("No observation is available. Call reset_task() first.")
        return self.current_obs

    @staticmethod
    def _extract_image(obs: dict[str, Any], key: str) -> Any:
        if key not in obs:
            raise KeyError(f"LIBERO observation does not contain image key '{key}'")
        image = obs[key][::-1, ::-1]
        if hasattr(image, "copy"):
            return image.copy()
        return image

    @classmethod
    def _extract_robot_state(cls, obs: dict[str, Any]) -> list[float]:
        eef_pos = cls._obs_vector(obs=obs, key="robot0_eef_pos")
        eef_axis_angle = cls._quat_to_axis_angle(cls._obs_vector(obs=obs, key="robot0_eef_quat"))
        gripper_qpos = cls._obs_vector(obs=obs, key="robot0_gripper_qpos")
        return eef_pos + eef_axis_angle + gripper_qpos

    @staticmethod
    def _obs_vector(obs: dict[str, Any], key: str) -> list[float]:
        if key not in obs:
            raise KeyError(f"LIBERO observation does not contain state key '{key}'")
        value = obs[key]
        if hasattr(value, "reshape"):
            value = value.reshape(-1)
        return [float(item) for item in value]

    @staticmethod
    def _quat_to_axis_angle(quat: list[float]) -> list[float]:
        if len(quat) != 4:
            raise ValueError(f"Expected quaternion with 4 values, got {len(quat)}")

        clipped_w = max(min(quat[3], 1.0), -1.0)
        denominator = math.sqrt(1.0 - clipped_w * clipped_w)
        if math.isclose(denominator, 0.0):
            return [0.0, 0.0, 0.0]

        scale = 2.0 * math.acos(clipped_w) / denominator
        return [quat[0] * scale, quat[1] * scale, quat[2] * scale]
