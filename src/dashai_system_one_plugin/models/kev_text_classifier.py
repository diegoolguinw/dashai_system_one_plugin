"""DashAI Text Classification model powered by local Kev 0.8B runtime."""

from typing import Any, ClassVar

from dashai_system_one_plugin.backends.kev_backend import KevBackend
from dashai_system_one_plugin.core.base_backend import BaseSystemOneBackend
from dashai_system_one_plugin.models.base_dashai_system_one import (
    BaseSystemOneSchema,
    DashAISystemOneClassifier,
    MultilingualString,
)


class KevTextClassifier(DashAISystemOneClassifier):
    """Text classification using local Kev 0.8B runtime over 127.0.0.1 loopback."""

    SCHEMA = BaseSystemOneSchema
    MODEL_NAME: ClassVar[str] = "jaredpalmer/kev-0.8b"
    BACKEND_TYPE: ClassVar[str] = "kev"

    DISPLAY_NAME: ClassVar[Any] = MultilingualString(
        en="Kev 0.8B Local Classifier",
        es="Clasificador Local Kev 0.8B",
    )
    DESCRIPTION: ClassVar[Any] = MultilingualString(
        en="Qwen-based 0.8B decision model executed strictly on local loopback (127.0.0.1).",
        es="Modelo de decisión de 0.8B basado en Qwen ejecutado estrictamente en loopback local (127.0.0.1).",
    )

    def _create_backend(self) -> BaseSystemOneBackend:
        return KevBackend()
