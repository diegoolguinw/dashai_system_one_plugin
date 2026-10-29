"""Laya backend adapter for System One choice routing."""

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


class LayaBackend(BaseSystemOneBackend):
    """Adapter wrapping a Laya Agent for local neural choice decisions."""

    def __init__(self) -> None:
        super().__init__()
        self._agent: Any = None

    def load(
        self,
        checkpoint_id: str = "convaiinnovations/laya",
        revision: str | None = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Load a Laya Agent from a Hub repo id or local directory.

        With ``HF_HUB_OFFLINE=1`` the checkpoint is read from the local cache only.
        """
        try:
            import laya  # type: ignore
        except ImportError as exc:
            raise BackendMissingError("laya", "laya") from exc

        self.checkpoint_id = checkpoint_id
        self.revision = revision or "main"
        self.device = resolve_device(device)

        try:
            self._agent = laya.load(
                checkpoint_id,
                device=self.device,
                revision=revision,
                **kwargs,
            )
            self._is_loaded = True
        except Exception as exc:
            raise InferenceError(
                f"Failed to load Laya checkpoint '{checkpoint_id}': {exc}"
            ) from exc

    def unload(self) -> None:
        self._agent = None
        self._is_loaded = False
        self.checkpoint_id = None

    def capabilities(self) -> dict[str, Any]:
        return {
            "supported_question_types": ["choice"],
            "probabilities_available": True,
            "multi_question_support": True,
            "streaming": False,
        }

    def predict(self, spec: DecisionSpec) -> DecisionResult:
        if not self._is_loaded or self._agent is None:
            raise RuntimeError("LayaBackend must be loaded before calling predict().")

        questions: dict[str, dict[str, Any]] = {}
        options_by_question: dict[str, list[str]] = {}
        for q_name, q_spec in spec.questions.items():
            if q_spec.type != "choice":
                raise UnsupportedQuestionError("LayaBackend", q_spec.type)
            if isinstance(q_spec.criteria, list):
                criteria = {opt: opt for opt in q_spec.criteria}
            elif isinstance(q_spec.criteria, dict):
                criteria = dict(q_spec.criteria)
            else:
                raise InferenceError(f"Question '{q_name}' criteria cannot be None.")
            options_by_question[q_name] = list(criteria)
            questions[q_name] = {
                "type": "choice",
                "instructions": q_spec.instructions,
                "criteria": criteria,
            }

        # One forward pass answers every question.
        try:
            raw = self._agent.predict(spec.state, questions)
        except Exception as exc:
            raise InferenceError(f"Laya inference error: {exc}") from exc

        raw_answers = raw.get("answers", {}) if isinstance(raw, dict) else {}
        answers: dict[str, DecisionAnswer] = {}
        for q_name, options in options_by_question.items():
            raw_q = raw_answers.get(q_name)
            if not isinstance(raw_q, dict) or raw_q.get("choice") not in options:
                raise InferenceError(
                    f"Laya returned no valid choice for question '{q_name}': {raw_q!r}"
                )
            raw_probs = raw_q.get("probabilities") or {}
            total = sum(float(raw_probs.get(opt, 0.0)) for opt in options)
            if total <= 0:
                raise InferenceError(
                    f"Laya returned no probabilities for question '{q_name}'."
                )
            answers[q_name] = DecisionAnswer(
                kind="choice",
                value=raw_q["choice"],
                probabilities={
                    opt: float(raw_probs.get(opt, 0.0)) / total for opt in options
                },
                # Laya documents its shipped checkpoints as over-confident until
                # temperatures are fitted, so these are model scores, not calibrated.
                calibrated=False,
            )

        return DecisionResult(
            answers=answers,
            model_id=self.checkpoint_id or "convaiinnovations/laya",
            revision=self.revision,
        )
