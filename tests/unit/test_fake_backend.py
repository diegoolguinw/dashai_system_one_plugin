"""Unit tests for FakeBackend."""

import pytest

from dashai_system_one_plugin.backends.fake_backend import FakeBackend
from dashai_system_one_plugin.core.contracts import DecisionSpec, QuestionSpec


def test_fake_backend_lifecycle_and_capabilities():
    backend = FakeBackend()
    assert not backend.is_loaded

    backend.load("test-checkpoint", device="cpu")
    assert backend.is_loaded
    assert backend.checkpoint_id == "test-checkpoint"

    caps = backend.capabilities()
    assert "choice" in caps["supported_question_types"]
    assert caps["probabilities_available"] is True

    backend.unload()
    assert not backend.is_loaded


def test_fake_backend_predict_choice_and_probabilities(sample_decision_spec):
    backend = FakeBackend(simulate_probabilities=True)
    backend.load("test-checkpoint")

    result = backend.predict(sample_decision_spec)
    assert result.model_id == "test-checkpoint"
    assert "sentiment" in result.answers

    ans = result.answers["sentiment"]
    assert ans.kind == "choice"
    assert ans.value in ["positive", "negative", "neutral"]
    assert ans.probabilities is not None
    assert sum(ans.probabilities.values()) == pytest.approx(1.0, abs=1e-2)


def test_fake_backend_score_and_noul():
    backend = FakeBackend()
    backend.load("test-model")

    spec = DecisionSpec(
        state="Test statement",
        questions={
            "score_q": QuestionSpec(type="score", instructions="Score relevance"),
            "noul_q": QuestionSpec(type="noul", instructions="Flag condition"),
        },
    )

    result = backend.predict(spec)
    score_ans = result.answers["score_q"]
    noul_ans = result.answers["noul_q"]

    assert score_ans.kind == "score"
    assert 0.0 <= score_ans.value <= 1.0
    assert noul_ans.kind == "noul"
    assert isinstance(noul_ans.value, bool)
