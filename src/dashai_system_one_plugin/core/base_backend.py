"""Base backend interface for System One models."""

from abc import ABC, abstractmethod
from typing import Any

from dashai_system_one_plugin.core.contracts import DecisionResult, DecisionSpec


class BaseSystemOneBackend(ABC):
    """Abstract base class that all System One execution engines must implement."""

    def __init__(self) -> None:
        self.checkpoint_id: str | None = None
        self.revision: str | None = None
        self.device: str = "cpu"
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Returns True if the backend model is currently loaded in memory."""
        return self._is_loaded

    @abstractmethod
    def load(
        self,
        checkpoint_id: str,
        revision: str | None = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Load checkpoint and prepare the engine for inference.

        Args:
            checkpoint_id: Identifier or local path of the model checkpoint.
            revision: Optional git revision / tag for reproducibility.
            device: Requested device string ('auto', 'cpu', 'cuda', 'mps').
            **kwargs: Extra backend-specific options (e.g. timeout, batch_size).
        """

    @abstractmethod
    def predict(self, spec: DecisionSpec) -> DecisionResult:
        """Execute decision inference for the given specification.

        Args:
            spec: DecisionSpec with input state and target questions.

        Returns:
            DecisionResult containing typed answers conforming to contracts.
        """

    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory/GPU and release runtime resources."""

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Return a dictionary describing supported capabilities.

        Example return:
            {
                "supported_question_types": ["choice"],
                "probabilities_available": True,
                "multi_question_support": True,
                "streaming": False,
            }
        """
