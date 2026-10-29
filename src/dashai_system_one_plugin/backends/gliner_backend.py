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
    """Adapter wrapping a local GLiNER2 extractor for zero-shot classification.

    Only the local ``AutoExtractor`` is used; the hosted ``GLiNER2API`` client that
    ``gliner2`` also ships is never imported (local inference only).
    """

    def __init__(self) -> None:
        super().__init__()
        self._extractor: Any = None

    def load(
        self,
        checkpoint_id: str = "fastino/GLiNER2.5-Decide",
        revision: str | None = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Load a GLiNER2 extractor from a Hub repo id or local directory.

        With ``HF_HUB_OFFLINE=1`` the checkpoint is read from the local cache only.
        """
        try:
            from gliner2 import AutoExtractor  # type: ignore
        except ImportError as exc:
            raise BackendMissingError("gliner2", "gliner") from exc

        self.checkpoint_id = checkpoint_id
        self.revision = revision or "main"
        self.device = resolve_device(device)

        try:
            extractor = AutoExtractor.from_pretrained(
                checkpoint_id, revision=self.revision, **kwargs
            )
            extractor.to(self.device)
            extractor.eval()
            self._extractor = extractor
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
        # classify_text yields a hard label plus a confidence for the winner only;
        # there is no full distribution, so none is exposed (ADR-002).
        return {
            "supported_question_types": ["choice"],
            "probabilities_available": False,
            "multi_question_support": True,
            "streaming": False,
        }

    def predict(self, spec: DecisionSpec) -> DecisionResult:
        if not self._is_loaded or self._extractor is None:
            raise RuntimeError("GLiNERBackend must be loaded before calling predict().")

        tasks: dict[str, dict[str, Any]] = {}
        options_by_question: dict[str, list[str]] = {}
        for q_name, q_spec in spec.questions.items():
            if q_spec.type != "choice":
                raise UnsupportedQuestionError("GLiNERBackend", q_spec.type)
            if isinstance(q_spec.criteria, (list, dict)):
                options_by_question[q_name] = list(q_spec.criteria)
            else:
                raise InferenceError(f"Question '{q_name}' criteria cannot be None.")
            tasks[q_name] = {
                "labels": q_spec.criteria,
                "prompt": q_spec.instructions,
            }

        state = spec.state if isinstance(spec.state, str) else str(spec.state)
        try:
            result = self._extractor.classify_text(state, tasks)
        except Exception as exc:
            raise InferenceError(f"GLiNER inference error: {exc}") from exc

        answers: dict[str, DecisionAnswer] = {}
        for q_name, options in options_by_question.items():
            label = result.get(q_name) if isinstance(result, dict) else None
            if label not in options:
                raise InferenceError(
                    f"GLiNER returned no valid label for question '{q_name}': {label!r}"
                )
            # ADR-002: hard label only. One-hot keeps argmax-based metrics working
            # and is flagged uncalibrated rather than inventing confidence.
            answers[q_name] = DecisionAnswer(
                kind="choice",
                value=label,
                probabilities={opt: 1.0 if opt == label else 0.0 for opt in options},
                calibrated=False,
            )

        return DecisionResult(
            answers=answers,
            model_id=self.checkpoint_id or "fastino/GLiNER2.5-Decide",
            revision=self.revision,
        )
