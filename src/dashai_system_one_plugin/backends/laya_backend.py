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
    """Adapter wrapping Laya's Router for local neural choice decisions."""

    def __init__(self) -> None:
        super().__init__()
        self._router: Any = None

    def load(
        self,
        checkpoint_id: str = "convaiinnovations/laya",
        revision: str | None = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Initialize Laya Router with the specified checkpoint and device."""
        try:
            from laya import Router  # type: ignore
        except ImportError as exc:
            raise BackendMissingError("laya", "laya") from exc

        self.checkpoint_id = checkpoint_id
        self.revision = revision or "main"
        self.device = resolve_device(device)

        try:
            # Laya Router constructor accepts model/checkpoint parameters
            self._router = Router(
                model=self.checkpoint_id,
                device=self.device,
                **kwargs,
            )
            self._is_loaded = True
        except Exception as exc:
            raise InferenceError(
                f"Failed to load Laya checkpoint '{checkpoint_id}': {exc}"
            ) from exc

    def unload(self) -> None:
        self._router = None
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
        if not self._is_loaded or self._router is None:
            raise RuntimeError("LayaBackend must be loaded before calling predict().")

        answers: dict[str, DecisionAnswer] = {}
        state_str = str(spec.state)

        for q_name, q_spec in spec.questions.items():
            if q_spec.type != "choice":
                raise UnsupportedQuestionError("LayaBackend", q_spec.type)

            options: list[str]
            if isinstance(q_spec.criteria, list):
                options = q_spec.criteria
            elif isinstance(q_spec.criteria, dict):
                options = list(q_spec.criteria.keys())
            else:
                raise InferenceError(f"Question '{q_name}' criteria cannot be None.")

            try:
                # Call Router.predict with state and candidate options
                raw_response = self._router.predict(
                    state=state_str,
                    questions={q_name: options},
                )
            except Exception as exc:
                raise InferenceError(f"Laya inference error on question '{q_name}': {exc}") from exc

            # Parse Laya response format
            chosen_label: str
            probs: dict[str, float] = {}

            if isinstance(raw_response, dict):
                res_q = raw_response.get(q_name, raw_response)
                if isinstance(res_q, dict):
                    chosen_label = res_q.get("choice", res_q.get("label", options[0]))
                    probs = res_q.get("probabilities", res_q.get("scores", {}))
                elif isinstance(res_q, str):
                    chosen_label = res_q
                else:
                    chosen_label = options[0]
            elif isinstance(raw_response, str):
                chosen_label = raw_response
            else:
                chosen_label = options[0]

            # If raw response did not return full probabilities, normalize or construct
            if not probs:
                probs = {opt: (1.0 if opt == chosen_label else 0.0) for opt in options}
            else:
                # Ensure all options exist in dict and probabilities sum to 1.0
                total = sum(probs.get(opt, 0.0) for opt in options)
                if total > 0:
                    probs = {opt: round(probs.get(opt, 0.0) / total, 4) for opt in options}
                    diff = round(1.0 - sum(probs.values()), 4)
                    probs[chosen_label] = round(probs[chosen_label] + diff, 4)
                else:
                    probs = {opt: (1.0 if opt == chosen_label else 0.0) for opt in options}

            answers[q_name] = DecisionAnswer(
                kind="choice",
                value=chosen_label,
                probabilities=probs,
                calibrated=True,
            )

        return DecisionResult(
            answers=answers,
            model_id=self.checkpoint_id or "convaiinnovations/laya",
            revision=self.revision,
        )
