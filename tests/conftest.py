"""Pytest fixtures for dashai-system-one-plugin tests."""

import pytest

from dashai_system_one_plugin.core.contracts import DecisionSpec, QuestionSpec


@pytest.fixture
def sample_choice_question():
    return QuestionSpec(
        type="choice",
        instructions="Classify sentiment into positive, negative, or neutral.",
        criteria=["positive", "negative", "neutral"],
    )


@pytest.fixture
def sample_decision_spec(sample_choice_question):
    return DecisionSpec(
        state="The service was exceptional and the team resolved the issue in minutes.",
        questions={"sentiment": sample_choice_question},
    )
