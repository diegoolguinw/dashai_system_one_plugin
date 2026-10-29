"""Base DashAI model adapter for System One zero-shot text classifiers."""

import json
from typing import Any, ClassVar

import numpy as np

# DashAI compatibility imports with clean standalone fallbacks
try:
    from DashAI.back.core.schema_fields import (
        enum_field,
        int_field,
        schema_field,
        string_field,
    )
    from DashAI.back.core.schema_fields.base_schema import BaseSchema
    from DashAI.back.core.utils import MultilingualString
    from DashAI.back.models.base_model import BaseModel
except ImportError:
    # Development and offline testing stubs
    class MultilingualString(str):
        def __new__(cls, en: str, es: str | None = None, **kwargs):
            obj = str.__new__(cls, en)
            obj.en = en
            obj.es = es or en
            return obj

    class BaseSchema:
        pass

    def schema_field(field_type, default=None, alias=None, description=None):
        return default

    def string_field():
        return str

    def int_field(ge=None, le=None):
        return int

    def enum_field(enum=None):
        return list(enum or [])

    class BaseModel:
        COMPATIBLE_COMPONENTS: ClassVar[list[str]] = []
        SCHEMA: ClassVar[Any] = None


from dashai_system_one_plugin.core.base_backend import BaseSystemOneBackend
from dashai_system_one_plugin.core.contracts import DecisionSpec, QuestionSpec
from dashai_system_one_plugin.core.device import resolve_device
from dashai_system_one_plugin.core.errors import InferenceError


class BaseSystemOneSchema(BaseSchema):
    """Configuration schema for System One text classification models."""

    instruction: schema_field(
        string_field(),
        "Classify the input text into one of the available categories.",
        alias=MultilingualString(en="Instruction Prompt", es="Instrucción del Prompt"),
        description=MultilingualString(
            en="Natural language guidance for zero-shot decision classification.",
            es="Guía en lenguaje natural para la clasificación de decisiones zero-shot.",
        ),
    )  # type: ignore
    batch_size: schema_field(
        int_field(ge=1),
        32,
        alias=MultilingualString(en="Batch Size", es="Tamaño de Lote"),
        description=MultilingualString(
            en="Number of text samples processed together in each batch.",
            es="Número de muestras de texto procesadas juntas en cada lote.",
        ),
    )  # type: ignore
    device: schema_field(
        enum_field(enum=["auto", "cpu", "cuda", "mps"]),
        "auto",
        alias=MultilingualString(en="Compute Device", es="Dispositivo de Cómputo"),
        description=MultilingualString(
            en="Target hardware: auto-detect, CPU, NVIDIA CUDA, or Apple Silicon MPS.",
            es="Hardware objetivo: autodetección, CPU, NVIDIA CUDA o Apple Silicon MPS.",
        ),
    )  # type: ignore


class DashAISystemOneClassifier(BaseModel):
    """Abstract adapter integrating System One backends with DashAI's TextClassificationTask.

    Note: The class name intentionally avoids 'Base' so DashAI's component
    registry inspection (which filters __mro__ for 'Base' in name) unambiguously
    resolves BaseModel as the single root base type.
    """

    SCHEMA = BaseSystemOneSchema
    COMPATIBLE_COMPONENTS: ClassVar[list[str]] = ["TextClassificationTask"]
    MODEL_NAME: ClassVar[str]
    DISPLAY_NAME: ClassVar[Any]
    DESCRIPTION: ClassVar[Any]
    DEFAULT_INSTRUCTION: ClassVar[str] = (
        "Classify the input text into one of the available categories."
    )

    def __init__(
        self,
        instruction: str | None = None,
        batch_size: int = 32,
        device: str = "auto",
        **kwargs: Any,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.model_name = getattr(self, "MODEL_NAME", "system-one-base")
        self.instruction = instruction or self.DEFAULT_INSTRUCTION
        self.batch_size = int(batch_size)
        self.device_name = str(device)
        self.device = resolve_device(device)
        self.extra_kwargs = kwargs

        self.class_names: list[str] = []
        self.label_to_idx: dict[str, int] = {}
        self.idx_to_label: dict[int, str] = {}
        self.backend: BaseSystemOneBackend | None = None

    def _extract_class_names(self, y_train: Any) -> list[str]:
        """Extract ordered unique categorical labels from training target."""
        if hasattr(y_train, "column_names"):
            if len(y_train.column_names) != 1:
                raise ValueError(
                    f"{type(self).__name__} requires exactly one target column."
                )
            column = y_train.column_names[0]
            output_type = (getattr(y_train, "types", {}) or {}).get(column)
            categories = getattr(output_type, "categories", None)
            if categories is not None:
                names = list(categories)
            else:
                names = list(dict.fromkeys(y_train[column]))
        elif hasattr(y_train, "columns"):
            # Pandas DataFrame
            col = y_train.columns[0]
            names = list(dict.fromkeys(y_train[col].tolist()))
        elif isinstance(y_train, (list, tuple)):
            names = list(dict.fromkeys(y_train))
        else:
            raise ValueError(f"Unsupported dataset format for y_train: {type(y_train)}")

        # Filter out empty/None
        names = [str(n) for n in names if n is not None and str(n).strip()]
        if len(names) < 2:
            raise ValueError(
                f"{type(self).__name__} requires at least 2 distinct class labels for classification."
            )
        return names

    def _create_backend(self) -> BaseSystemOneBackend:
        """Instantiate concrete backend; subclasses must override."""
        raise NotImplementedError

    def _ensure_backend_loaded(self) -> None:
        """Ensure the backend is loaded and ready."""
        if self.backend is None:
            self.backend = self._create_backend()
        if not self.backend.is_loaded:
            self.backend.load(
                checkpoint_id=self.model_name,
                device=self.device,
                **self.extra_kwargs,
            )

    def train(
        self,
        x_train: Any,
        y_train: Any,
        x_validation: Any = None,
        y_validation: Any = None,
    ) -> "DashAISystemOneClassifier":
        """Configure zero-shot labels and initialize the backend without altering weights."""
        self.class_names = self._extract_class_names(y_train)
        self.label_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_label = {idx: name for name, idx in self.label_to_idx.items()}

        self._ensure_backend_loaded()
        return self

    def prepare_output(self, dataset: Any, is_fit: bool = False) -> Any:
        """Encode categorical labels into integer index column for DashAI pipeline."""
        if not self.label_to_idx:
            raise RuntimeError(f"{type(self).__name__} class labels are not initialized.")

        try:
            import pyarrow as pa
            from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset

            if hasattr(dataset, "column_names"):
                column = dataset.column_names[0]
                encoded = [self.label_to_idx[str(val)] for val in dataset[column]]
                return DashAIDataset(pa.table({column: encoded}))
        except (ImportError, KeyError, TypeError, ValueError, AttributeError):
            pass

        return dataset

    def _extract_text_samples(self, x: Any) -> list[str]:
        """Extract list of input strings from x dataset."""
        if hasattr(x, "column_names"):
            col = x.column_names[0]
            return [str(v) for v in x[col]]
        elif hasattr(x, "columns"):
            col = x.columns[0]
            return [str(v) for v in x[col].tolist()]
        elif isinstance(x, (list, tuple)):
            return [str(v) for v in x]
        else:
            raise ValueError(f"Unsupported input feature type: {type(x)}")

    def predict(self, x: Any) -> np.ndarray:
        """Predict class probability distribution of shape (N, K).

        Returns:
            np.ndarray of shape (N, K) containing float probabilities in [0.0, 1.0].
        """
        if not self.class_names:
            raise RuntimeError(
                f"{type(self).__name__} must be initialized with train() or load() before predict()."
            )

        samples = self._extract_text_samples(x)
        num_samples = len(samples)
        num_classes = len(self.class_names)

        if num_samples == 0:
            return np.empty((0, num_classes), dtype=np.float32)

        self._ensure_backend_loaded()
        assert self.backend is not None

        prob_matrix = np.zeros((num_samples, num_classes), dtype=np.float32)

        for i in range(0, num_samples, self.batch_size):
            batch_samples = samples[i : i + self.batch_size]

            for offset, text in enumerate(batch_samples):
                row_idx = i + offset
                spec = DecisionSpec(
                    state=text,
                    questions={
                        "target_class": QuestionSpec(
                            type="choice",
                            instructions=self.instruction,
                            criteria=self.class_names,
                        )
                    },
                )

                decision_res = self.backend.predict(spec)
                ans = decision_res.answers.get("target_class")
                if ans is None:
                    raise InferenceError("Backend failed to produce answer for 'target_class'.")

                if ans.probabilities:
                    for cls_name, p in ans.probabilities.items():
                        idx = self.label_to_idx.get(cls_name)
                        if idx is not None:
                            prob_matrix[row_idx, idx] = float(p)
                else:
                    # One-hot hard decision fallback conforming to ADR-002
                    idx = self.label_to_idx.get(str(ans.value), 0)
                    prob_matrix[row_idx, idx] = 1.0

        return prob_matrix

    def save(self, filename: str) -> None:
        """Save metadata, labels and configuration without dumping large weights."""
        metadata = {
            "format_version": 1,
            "model_name": self.model_name,
            "instruction": self.instruction,
            "batch_size": self.batch_size,
            "device_name": self.device_name,
            "class_names": self.class_names,
            "backend_type": getattr(self, "BACKEND_TYPE", "system_one"),
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filename: str) -> "DashAISystemOneClassifier":
        """Restore instance from saved JSON metadata file."""
        with open(filename, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        required = {
            "format_version",
            "model_name",
            "instruction",
            "batch_size",
            "device_name",
            "class_names",
        }
        if not isinstance(metadata, dict) or required - metadata.keys():
            raise ValueError(f"Invalid {cls.__name__} checkpoint file.")

        if metadata["format_version"] != 1:
            raise ValueError(f"Unsupported checkpoint version: {metadata['format_version']}.")

        if metadata["model_name"] != cls.MODEL_NAME:
            raise ValueError(
                f"Checkpoint model '{metadata['model_name']}' does not match "
                f"expected class model '{cls.MODEL_NAME}'."
            )

        instance = cls(
            instruction=metadata["instruction"],
            batch_size=metadata["batch_size"],
            device=metadata["device_name"],
        )
        instance.class_names = list(metadata["class_names"])
        instance.label_to_idx = {name: idx for idx, name in enumerate(instance.class_names)}
        instance.idx_to_label = {idx: name for name, idx in instance.label_to_idx.items()}
        return instance


# Backward compatibility alias
BaseDashAISystemOneClassifier = DashAISystemOneClassifier
