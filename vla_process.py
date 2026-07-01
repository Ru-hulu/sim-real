"""VLA inference process client interface."""

from __future__ import annotations

from typing import Any, Protocol

from libero_evaluator import Action, Observation, TaskInfo


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
