"""Core modules for contracts, registries, devices and error types."""

from dashai_system_one_plugin.core.base_backend import BaseSystemOneBackend
from dashai_system_one_plugin.core.checkpoint_registry import (
    CheckpointInfo,
    CheckpointRegistry,
)
from dashai_system_one_plugin.core.contracts import (
    DecisionAnswer,
    DecisionResult,
    DecisionSpec,
    QuestionSpec,
)
from dashai_system_one_plugin.core.device import resolve_device
from dashai_system_one_plugin.core.errors import (
    BackendMissingError,
    CheckpointMissingError,
    ContractValidationError,
    DeviceError,
    InferenceError,
    SecurityError,
    SystemOnePluginError,
    UnsupportedQuestionError,
)

__all__ = [
    "BackendMissingError",
    "BaseSystemOneBackend",
    "CheckpointInfo",
    "CheckpointMissingError",
    "CheckpointRegistry",
    "ContractValidationError",
    "DecisionAnswer",
    "DecisionResult",
    "DecisionSpec",
    "DeviceError",
    "InferenceError",
    "QuestionSpec",
    "SecurityError",
    "SystemOnePluginError",
    "UnsupportedQuestionError",
    "resolve_device",
]
