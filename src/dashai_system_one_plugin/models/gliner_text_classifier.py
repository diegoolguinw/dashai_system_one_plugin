"""DashAI Text Classification models powered by GLiNER 2.5 Decide."""

from typing import Any, ClassVar

from dashai_system_one_plugin.backends.gliner_backend import GLiNERBackend
from dashai_system_one_plugin.core.base_backend import BaseSystemOneBackend
from dashai_system_one_plugin.models.base_dashai_system_one import (
    BaseSystemOneSchema,
    DashAISystemOneClassifier,
    MultilingualString,
)


class GLiNERDecideTextClassifier(DashAISystemOneClassifier):
    """Zero-shot text classification using GLiNER 2.5 Decide (English)."""

    SCHEMA = BaseSystemOneSchema
    MODEL_NAME: ClassVar[str] = "fastino/GLiNER2.5-Decide"
    BACKEND_TYPE: ClassVar[str] = "gliner"

    DISPLAY_NAME: ClassVar[Any] = MultilingualString(
        en="GLiNER 2.5 Decide Classifier",
        es="Clasificador GLiNER 2.5 Decide",
    )
    DESCRIPTION: ClassVar[Any] = MultilingualString(
        en="Bidirectional representation model for zero-shot text classification and entity decision (English).",
        es="Modelo de representación bidireccional para clasificación zero-shot y decisión sobre entidades (Inglés).",
    )

    def _create_backend(self) -> BaseSystemOneBackend:
        return GLiNERBackend()


class GLiNERMultiDecideTextClassifier(DashAISystemOneClassifier):
    """Zero-shot text classification using GLiNER 2.5 Decide (Multilingual)."""

    SCHEMA = BaseSystemOneSchema
    MODEL_NAME: ClassVar[str] = "fastino/GLiNER2.5-multi-Decide"
    BACKEND_TYPE: ClassVar[str] = "gliner"

    DISPLAY_NAME: ClassVar[Any] = MultilingualString(
        en="GLiNER 2.5 Multilingual Decide Classifier",
        es="Clasificador GLiNER 2.5 Decide Multilingüe",
    )
    DESCRIPTION: ClassVar[Any] = MultilingualString(
        en="Multilingual zero-shot text classifier powered by GLiNER 2.5.",
        es="Clasificador de texto zero-shot multilingüe potenciado por GLiNER 2.5.",
    )

    def _create_backend(self) -> BaseSystemOneBackend:
        return GLiNERBackend()
