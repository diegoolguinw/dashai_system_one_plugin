"""dashai-system-one-plugin: System One fast typed decision models for DashAI."""

__version__ = "0.1.0"

from dashai_system_one_plugin.core.contracts import (
    DecisionAnswer,
    DecisionResult,
    DecisionSpec,
    QuestionSpec,
)

__all__ = [
    "DecisionAnswer",
    "DecisionResult",
    "DecisionSpec",
    "QuestionSpec",
    "__version__",
]
