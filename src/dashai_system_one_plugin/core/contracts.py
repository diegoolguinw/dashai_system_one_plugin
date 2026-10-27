"""Core contracts and data structures for System One decisions."""

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from dashai_system_one_plugin.core.errors import ContractValidationError

QuestionType = Literal["choice", "score", "noul"]


class QuestionSpec(BaseModel):
    """Specification of a question posed to a System One model.

    Attributes:
        type: The question type: 'choice' (categorical selection), 'score' (scalar),
              or 'noul' (null/rejection/binary condition).
        instructions: Natural language prompt or instruction guiding the decision.
        criteria: Options or evaluation criteria (list of category strings,
                  or dict mapping option name to description).
    """

    type: QuestionType
    instructions: str = Field(..., min_length=1)
    criteria: dict[str, str] | list[str] | None = None

    @field_validator("instructions")
    @classmethod
    def validate_instructions(cls, value: str) -> str:
        if not value.strip():
            raise ContractValidationError("Instructions cannot be empty or whitespace only.")
        return value

    @model_validator(mode="after")
    def validate_choice_criteria(self) -> "QuestionSpec":
        if self.type == "choice":
            if not self.criteria:
                raise ContractValidationError(
                    "Questions of type 'choice' must have non-empty 'criteria'."
                )
            if isinstance(self.criteria, list):
                if len(self.criteria) < 2:
                    raise ContractValidationError(
                        "Choice criteria list must have at least 2 distinct options."
                    )
                if len(set(self.criteria)) != len(self.criteria):
                    raise ContractValidationError(
                        "Choice criteria list contains duplicate options."
                    )
            elif isinstance(self.criteria, dict):
                if len(self.criteria) < 2:
                    raise ContractValidationError(
                        "Choice criteria dict must have at least 2 distinct options."
                    )
        return self


class DecisionSpec(BaseModel):
    """Input payload describing the state and the questions to resolve.

    Attributes:
        state: Context or text input to evaluate (string or structured dict).
        questions: Dictionary of named QuestionSpec items.
    """

    state: str | dict[str, Any]
    questions: dict[str, QuestionSpec] = Field(..., min_length=1)

    @field_validator("state")
    @classmethod
    def validate_state(cls, value: str | dict[str, Any]) -> str | dict[str, Any]:
        if isinstance(value, str) and not value.strip():
            raise ContractValidationError("State string cannot be empty or whitespace only.")
        return value


class DecisionAnswer(BaseModel):
    """Result answer to a single question.

    Attributes:
        kind: The answer type matching the question type.
        value: Decided value (string category for choice, float for score, bool for noul).
        probabilities: Optional mapping of option to probability [0.0, 1.0].
        calibrated: Flag indicating whether probabilities are calibrated.
    """

    kind: QuestionType
    value: str | float | bool
    probabilities: dict[str, float] | None = None
    calibrated: bool | None = None

    @model_validator(mode="after")
    def validate_invariants(self) -> "DecisionAnswer":
        # Validate value type matches kind
        if self.kind == "choice" and not isinstance(self.value, str):
            raise ContractValidationError(
                f"Choice answer value must be a string, got {type(self.value).__name__}."
            )
        elif self.kind == "score":
            if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
                raise ContractValidationError(
                    f"Score answer value must be numeric, got {type(self.value).__name__}."
                )
            if math.isnan(self.value) or math.isinf(self.value):
                raise ContractValidationError("Score answer cannot be NaN or Inf.")
        elif self.kind == "noul" and not isinstance(self.value, bool):
            raise ContractValidationError(
                f"Noul answer value must be boolean, got {type(self.value).__name__}."
            )

        # Validate probabilities if provided
        if self.probabilities is not None:
            total_prob = 0.0
            for label, prob in self.probabilities.items():
                if math.isnan(prob) or math.isinf(prob):
                    raise ContractValidationError(
                        f"Probability for label '{label}' is NaN or Inf."
                    )
                if not (0.0 <= prob <= 1.0001):
                    raise ContractValidationError(
                        f"Probability {prob} for label '{label}' outside [0, 1] range."
                    )
                total_prob += prob

            if self.kind == "choice":
                # Check sum equals approx 1.0
                if abs(total_prob - 1.0) > 1e-2:
                    raise ContractValidationError(
                        f"Probabilities for choice must sum to ~1.0, got {total_prob:.4f}."
                    )
                if self.value not in self.probabilities:
                    raise ContractValidationError(
                        f"Selected choice value '{self.value}' missing from probabilities dict."
                    )

        return self


class DecisionResult(BaseModel):
    """Aggregate result from a model for all requested questions."""

    answers: dict[str, DecisionAnswer]
    model_id: str
    revision: str | None = None
