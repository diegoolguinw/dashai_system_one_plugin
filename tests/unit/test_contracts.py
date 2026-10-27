"""Unit tests for core contracts and invariants."""

import pytest

from dashai_system_one_plugin.core.contracts import (
    DecisionAnswer,
    DecisionResult,
    DecisionSpec,
    QuestionSpec,
)
from dashai_system_one_plugin.core.errors import ContractValidationError


def test_valid_question_spec():
    q = QuestionSpec(
        type="choice",
        instructions="Pick category",
        criteria=["cat_a", "cat_b"],
    )
    assert q.type == "choice"
    assert q.criteria == ["cat_a", "cat_b"]


def test_empty_instructions_rejected():
    with pytest.raises(ContractValidationError):
        QuestionSpec(
            type="choice",
            instructions="   ",
            criteria=["cat_a", "cat_b"],
        )


def test_choice_without_criteria_rejected():
    with pytest.raises(ContractValidationError):
        QuestionSpec(
            type="choice",
            instructions="Select category",
            criteria=None,
        )


def test_choice_with_fewer_than_two_options_rejected():
    with pytest.raises(ContractValidationError):
        QuestionSpec(
            type="choice",
            instructions="Select category",
            criteria=["only_one"],
        )


def test_choice_with_duplicate_options_rejected():
    with pytest.raises(ContractValidationError):
        QuestionSpec(
            type="choice",
            instructions="Select category",
            criteria=["opt_a", "opt_a"],
        )


def test_valid_decision_spec():
    q = QuestionSpec(
        type="choice",
        instructions="Classify",
        criteria=["a", "b"],
    )
    spec = DecisionSpec(
        state="Sample input text",
        questions={"q1": q},
    )
    assert spec.state == "Sample input text"
    assert "q1" in spec.questions


def test_decision_spec_empty_state_rejected():
    q = QuestionSpec(type="choice", instructions="Classify", criteria=["a", "b"])
    with pytest.raises(ContractValidationError):
        DecisionSpec(state="   ", questions={"q1": q})


def test_valid_choice_answer():
    ans = DecisionAnswer(
        kind="choice",
        value="opt_a",
        probabilities={"opt_a": 0.8, "opt_b": 0.2},
        calibrated=True,
    )
    assert ans.value == "opt_a"
    assert ans.probabilities["opt_a"] == 0.8


def test_choice_answer_invalid_sum_rejected():
    with pytest.raises(ContractValidationError):
        DecisionAnswer(
            kind="choice",
            value="opt_a",
            probabilities={"opt_a": 0.5, "opt_b": 0.1},  # Sums to 0.6
        )


def test_choice_answer_value_missing_from_probs_rejected():
    with pytest.raises(ContractValidationError):
        DecisionAnswer(
            kind="choice",
            value="opt_c",
            probabilities={"opt_a": 0.6, "opt_b": 0.4},
        )


def test_valid_score_answer():
    ans = DecisionAnswer(kind="score", value=0.75, calibrated=True)
    assert ans.value == 0.75


def test_score_answer_nan_rejected():
    with pytest.raises(ContractValidationError):
        DecisionAnswer(kind="score", value=float("nan"))


def test_valid_noul_answer():
    ans = DecisionAnswer(kind="noul", value=True)
    assert ans.value is True


def test_noul_answer_non_bool_rejected():
    with pytest.raises(ContractValidationError):
        DecisionAnswer(kind="noul", value="not_a_bool")  # type: ignore


def test_decision_result_structure():
    ans = DecisionAnswer(kind="noul", value=False)
    res = DecisionResult(
        answers={"rejection": ans},
        model_id="convaiinnovations/laya",
        revision="main",
    )
    assert res.model_id == "convaiinnovations/laya"
    assert res.answers["rejection"].value is False
