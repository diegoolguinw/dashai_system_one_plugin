"""DashAI Text Classification models powered by Laya Router."""

from typing import Any, ClassVar

from dashai_system_one_plugin.backends.laya_backend import LayaBackend
from dashai_system_one_plugin.core.base_backend import BaseSystemOneBackend
from dashai_system_one_plugin.models.base_dashai_system_one import (
    BaseDashAISystemOneClassifier,
    BaseSystemOneSchema,
    MultilingualString,
)


class LayaTextClassifier(BaseDashAISystemOneClassifier):
    """Fast zero-shot text classification using Laya Router (English)."""

    SCHEMA = BaseSystemOneSchema
    MODEL_NAME: ClassVar[str] = "convaiinnovations/laya"
    BACKEND_TYPE: ClassVar[str] = "laya"

    DISPLAY_NAME: ClassVar[Any] = MultilingualString(
        en="Laya Router Text Classifier",
        es="Clasificador de Texto Laya Router",
    )
    DESCRIPTION: ClassVar[Any] = MultilingualString(
        en="Fast neural choice router for zero-shot text classification and intent routing (English).",
        es="Router neuronal rápido para clasificación de texto zero-shot y enrutamiento de intención (Inglés).",
    )

    def _create_backend(self) -> BaseSystemOneBackend:
        return LayaBackend()


class LayaMultilingualTextClassifier(BaseDashAISystemOneClassifier):
    """Fast zero-shot text classification using Laya Multilingual Router."""

    SCHEMA = BaseSystemOneSchema
    MODEL_NAME: ClassVar[str] = "convaiinnovations/laya-multilingual"
    BACKEND_TYPE: ClassVar[str] = "laya"

    DISPLAY_NAME: ClassVar[Any] = MultilingualString(
        en="Laya Multilingual Router Text Classifier",
        es="Clasificador de Texto Laya Router Multilingüe",
    )
    DESCRIPTION: ClassVar[Any] = MultilingualString(
        en="Multilingual neural choice router for zero-shot text classification (Spanish, English, etc.).",
        es="Router neuronal multilingüe para clasificación de texto zero-shot (Español, Inglés, etc.).",
    )

    def _create_backend(self) -> BaseSystemOneBackend:
        return LayaBackend()
