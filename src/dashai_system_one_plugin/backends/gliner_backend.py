"""GLiNER2 backend adapter for zero-shot decision classification."""

from typing import Any

from dashai_system_one_plugin.core.base_backend import BaseSystemOneBackend
from dashai_system_one_plugin.core.contracts import (
    DecisionAnswer,
    DecisionResult,
    DecisionSpec,
)
from dashai_system_one_plugin.core.device import resolve_device
from dashai_system_one_plugin.core.errors import (
    BackendMissingError,
    InferenceError,
    UnsupportedQuestionError,
)


class GLiNERBackend(BaseSystemOneBackend):
    """Adapter wrapping GLiNER 2.5 AutoExtractor for local classification."""

    def __init__(self) -> None:
        super().__init__()
        self._extractor: Any = None
        self._has_calibrated_probabilities: bool = False

    def load(
        self,
        checkpoint_id: str = "fastino/GLiNER2.5-Decide",
        revision: str | None = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Load GLiNER2 AutoExtractor from pretrained weights."""
        try:
            try:
                from gliner2 import AutoExtractor  # type: ignore
                load_fn = AutoExtractor.from_pretrained
            except ImportError:
                from gliner import GLiNER  # type: ignore
                load_fn = GLiNER.from_pretrained
        except ImportError as exc:
            raise BackendMissingError("gliner2", "gliner") from exc

        self.checkpoint_id = checkpoint_id
        self.revision = revision or "main"
        self.device = resolve_device(device)

        try:
            self._extractor = load_fn(self.checkpoint_id, revision=self.revision, **kwargs)
            if hasattr(self._extractor, "to"):
                self._extractor.to(self.device)
            self._is_loaded = True
        except Exception as exc:
            raise InferenceError(
                f"Failed to load GLiNER checkpoint '{checkpoint_id}': {exc}"
            ) from exc

    def unload(self) -> None:
        self._extractor = None
        self._is_loaded = False
        self.checkpoint_id = None

    def capabilities(self) -> dict[str, Any]:
        return {
            "supported_question_types": ["choice"],
            "probabilities_available": self._has_calibrated_probabilities,
            "multi_question_support": True,
            "streaming": False,
        }

    def predict(self, spec: DecisionSpec) -> DecisionResult:
        if not self._is_loaded or self._extractor is None:
            raise RuntimeError("GLiNERBackend must be loaded before calling predict().")

        answers: dict[str, DecisionAnswer] = {}
        state_str = str(spec.state)

        for q_name, q_spec in spec.questions.items():
            if q_spec.type != "choice":
                raise UnsupportedQuestionError("GLiNERBackend", q_spec.type)

            options: list[str]
            if isinstance(q_spec.criteria, list):
                options = q_spec.criteria
            elif isinstance(q_spec.criteria, dict):
                options = list(q_spec.criteria.keys())
            else:
                raise InferenceError(f"Question '{q_name}' criteria cannot be None.")

            try:
                # GLiNER classification interface
                if hasattr(self._extractor, "classify_text"):
                    result = self._extractor.classify_text(state_str, options)
                elif hasattr(self._extractor, "predict"):
                    result = self._extractor.predict(state_str, options)
                else:
                    # Generic entity/classification fallback
                    result = self._extractor(state_str, labels=options)
            except Exception as exc:
                raise InferenceError(f"GLiNER inference error on question '{q_name}': {exc}") from exc

            # Parse result
            chosen_label: str
            probs: dict[str, float] | None = None

            if isinstance(result, str) and result in options:
                chosen_label = result
            elif isinstance(result, list) and len(result) > 0:
                first = result[0]
                if isinstance(first, dict):
                    chosen_label = first.get("label", options[0])
                elif isinstance(first, str):
                    chosen_label = first
                else:
                    chosen_label = options[0]
            elif isinstance(result, dict):
                chosen_label = result.get("label", result.get("choice", options[0]))
                if "scores" in result or "probabilities" in result:
                    raw_probs = result.get("probabilities", result.get("scores", {}))
                    if isinstance(raw_probs, dict) and all(opt in raw_probs for opt in options):
                        total = sum(raw_probs.values())
                        if total > 0:
                            probs = {opt: round(raw_probs[opt] / total, 4) for opt in options}
            else:
                chosen_label = options[0]

            # In accordance with ADR-002: Do NOT invent fake calibrated confidence
            # If backend only returned a hard label, one-hot representation with calibrated=False
            if probs is None:
                probs = {opt: (1.0 if opt == chosen_label else 0.0) for opt in options}
                is_calibrated = False
            else:
                is_calibrated = True

            answers[q_name] = DecisionAnswer(
                kind="choice",
                value=chosen_label,
                probabilities=probs,
                calibrated=is_calibrated,
            )

        return DecisionResult(
            answers=answers,
            model_id=self.checkpoint_id or "fastino/GLiNER2.5-Decide",
            revision=self.revision,
        )
