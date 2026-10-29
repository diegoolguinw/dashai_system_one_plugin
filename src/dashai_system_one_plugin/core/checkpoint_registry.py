"""Registry of approved System One checkpoints with license and offline verification."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class CheckpointInfo:
    """Metadata describing an approved System One checkpoint."""

    checkpoint_id: str
    display_name: str
    backend_type: str
    license_spdx: str
    is_multilingual: bool
    default_revision: str | None = None
    description: str = ""
    approx_size_mb: int = 0


class CheckpointRegistry:
    """Central registry tracking all approved models and validating licensing."""

    _REGISTRY: ClassVar[dict[str, CheckpointInfo]] = {
        "convaiinnovations/laya": CheckpointInfo(
            checkpoint_id="convaiinnovations/laya",
            display_name="Laya Router (English)",
            backend_type="laya",
            license_spdx="Apache-2.0",
            is_multilingual=False,
            default_revision="main",
            description="Compact System One neural router for fast choice decisions.",
            approx_size_mb=807,
        ),
        "convaiinnovations/laya-multilingual": CheckpointInfo(
            checkpoint_id="convaiinnovations/laya-multilingual",
            display_name="Laya Router (Multilingual)",
            backend_type="laya",
            license_spdx="Apache-2.0",
            is_multilingual=True,
            default_revision="main",
            description="Multilingual System One router supporting Spanish, English, and more.",
            approx_size_mb=647,
        ),
        "fastino/GLiNER2.5-Decide": CheckpointInfo(
            checkpoint_id="fastino/GLiNER2.5-Decide",
            display_name="GLiNER 2.5 Decide (English)",
            backend_type="gliner",
            license_spdx="Apache-2.0",
            is_multilingual=False,
            default_revision="main",
            description="Bidirectional encoder model for zero-shot decision classification.",
            approx_size_mb=1800,
        ),
        "fastino/GLiNER2.5-multi-Decide": CheckpointInfo(
            checkpoint_id="fastino/GLiNER2.5-multi-Decide",
            display_name="GLiNER 2.5 Decide (Multilingual)",
            backend_type="gliner",
            license_spdx="Apache-2.0",
            is_multilingual=True,
            default_revision="main",
            description="Multilingual GLiNER variant for zero-shot label extraction and decisions.",
            approx_size_mb=1100,
        ),
    }

    @classmethod
    def get(cls, checkpoint_id: str) -> CheckpointInfo | None:
        """Retrieve metadata for a checkpoint if registered."""
        return cls._REGISTRY.get(checkpoint_id)

    @classmethod
    def list_checkpoints(cls) -> list[CheckpointInfo]:
        """Return a list of all registered checkpoints."""
        return list(cls._REGISTRY.values())

    @classmethod
    def is_approved(cls, checkpoint_id: str) -> bool:
        """Check whether a checkpoint is approved with verified permissive license."""
        return checkpoint_id in cls._REGISTRY

    @classmethod
    def get_license(cls, checkpoint_id: str) -> str:
        """Get the SPDX license identifier for the given checkpoint."""
        info = cls.get(checkpoint_id)
        if not info:
            raise KeyError(f"Checkpoint '{checkpoint_id}' is not in the registry.")
        return info.license_spdx
