"""Typed exceptions for System One plugin."""


class SystemOnePluginError(Exception):
    """Base exception for all system one plugin errors."""


class BackendMissingError(SystemOnePluginError):
    """Raised when an optional backend dependency is not installed."""

    def __init__(self, backend_name: str, install_extra: str):
        self.backend_name = backend_name
        self.install_extra = install_extra
        super().__init__(
            f"Backend '{backend_name}' is not installed. "
            f"Install it with: pip install 'dashai-system-one-plugin[{install_extra}]'"
        )


class CheckpointMissingError(SystemOnePluginError):
    """Raised when a checkpoint is not found locally and cannot be downloaded."""

    def __init__(self, checkpoint_id: str, details: str = ""):
        self.checkpoint_id = checkpoint_id
        message = f"Model checkpoint '{checkpoint_id}' was not found in cache."
        if details:
            message += f" Details: {details}"
        super().__init__(message)


class UnsupportedQuestionError(SystemOnePluginError):
    """Raised when a backend does not support the requested question type."""

    def __init__(self, backend_name: str, question_type: str):
        self.backend_name = backend_name
        self.question_type = question_type
        super().__init__(
            f"Backend '{backend_name}' does not support question type '{question_type}'."
        )


class ContractValidationError(SystemOnePluginError):
    """Raised when inputs or outputs violate the contract invariants."""


class DeviceError(SystemOnePluginError):
    """Raised when a requested compute device is invalid or unavailable."""


class InferenceError(SystemOnePluginError):
    """Raised when an error occurs during model execution."""


class SecurityError(SystemOnePluginError):
    """Raised when a security boundary is violated (e.g. non-local connection)."""
