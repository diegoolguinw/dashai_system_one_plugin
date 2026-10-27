"""Fake backend implementation for fast, offline unit testing without GPU or downloads."""

import hashlib
from typing import Any

from dashai_system_one_plugin.core.base_backend import BaseSystemOneBackend
from dashai_system_one_plugin.core.contracts import (
    DecisionAnswer,
    DecisionResult,
    DecisionSpec,
)
from dashai_system_one_plugin.core.errors import UnsupportedQuestionError


class FakeBackend(BaseSystemOneBackend):
    """Deterministic in-memory backend for testing contracts, interfaces, and pipeline logic."""

    def __init__(self, simulate_probabilities: bool = True):
        super().__init__()
        self.simulate_probabilities = simulate_probabilities

    def load(
        self,
        checkpoint_id: str,
        revision: str | None = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        self.checkpoint_id = checkpoint_id
        self.revision = revision or "main"
        self.device = device
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False
        self.checkpoint_id = None

    def capabilities(self) -> dict[str, Any]:
        return {
            "supported_question_types": ["choice", "score", "noul"],
            "probabilities_available": self.simulate_probabilities,
            "multi_question_support": True,
            "streaming": False,
        }

    def predict(self, spec: DecisionSpec) -> DecisionResult:
        if not self._is_loaded:
            raise RuntimeError("FakeBackend must be loaded before calling predict().")

        answers: dict[str, DecisionAnswer] = {}
        state_str = str(spec.state)

        for q_name, q_spec in spec.questions.items():
            if q_spec.type == "choice":
                options: list[str]
                if isinstance(q_spec.criteria, list):
                    options = q_spec.criteria
                elif isinstance(q_spec.criteria, dict):
                    options = list(q_spec.criteria.keys())
                else:
                    options = ["option_a", "option_b"]

                # Generate deterministic hash-based selection
                seed = int(hashlib.md5(f"{state_str}:{q_name}".encode()).hexdigest(), 16)
                chosen_idx = seed % len(options)
                chosen_val = options[chosen_idx]

                probs: dict[str, float] | None = None
                if self.simulate_probabilities:
                    # Construct valid normalized distribution favoring chosen
                    base_weights = [1.0] * len(options)
                    base_weights[chosen_idx] += 3.0
                    total_w = sum(base_weights)
                    probs = {
                        opt: round(w / total_w, 4)
                        for opt, w in zip(options, base_weights)
                    }
                    # Fix rounding so sum equals 1.0 exactly
                    diff = 1.0 - sum(probs.values())
                    probs[chosen_val] = round(probs[chosen_val] + diff, 4)

                answers[q_name] = DecisionAnswer(
                    kind="choice",
                    value=chosen_val,
                    probabilities=probs,
                    calibrated=self.simulate_probabilities,
                )

            elif q_spec.type == "score":
                seed = int(hashlib.md5(f"{state_str}:{q_name}".encode()).hexdigest(), 16)
                score_val = round((seed % 1000) / 1000.0, 3)
                answers[q_name] = DecisionAnswer(
                    kind="score",
                    value=score_val,
                    probabilities=None,
                    calibrated=True,
                )

            elif q_spec.type == "noul":
                seed = int(hashlib.md5(f"{state_str}:{q_name}".encode()).hexdigest(), 16)
                bool_val = bool(seed % 2 == 0)
                answers[q_name] = DecisionAnswer(
                    kind="noul",
                    value=bool_val,
                    probabilities=None,
                    calibrated=True,
                )
            else:
                raise UnsupportedQuestionError("FakeBackend", q_spec.type)

        return DecisionResult(
            answers=answers,
            model_id=self.checkpoint_id or "fake-model",
            revision=self.revision,
        )
