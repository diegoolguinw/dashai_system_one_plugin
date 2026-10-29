"""Unit tests for Laya backend adapter."""

from unittest.mock import MagicMock, patch

import pytest

from dashai_system_one_plugin.backends.laya_backend import LayaBackend
from dashai_system_one_plugin.core.contracts import DecisionSpec, QuestionSpec
from dashai_system_one_plugin.core.errors import BackendMissingError, InferenceError


def test_laya_missing_backend_raises_typed_error():
    backend = LayaBackend()
    with patch.dict("sys.modules", {"laya": None}):
        with pytest.raises(BackendMissingError) as exc_info:
            backend.load("convaiinnovations/laya")
        assert exc_info.value.backend_name == "laya"


def _spec() -> DecisionSpec:
    return DecisionSpec(
        state="I need a refund for my last invoice.",
        questions={
            "intent": QuestionSpec(
                type="choice",
                instructions="Identify user intent",
                criteria=["billing", "support"],
            )
        },
    )


def _load_with_agent(backend: LayaBackend, agent: MagicMock) -> None:
    mock_laya_module = MagicMock()
    mock_laya_module.load.return_value = agent
    with patch.dict("sys.modules", {"laya": mock_laya_module}):
        backend.load("convaiinnovations/laya", device="cpu")


def test_laya_predict_parses_real_payload_shape():
    backend = LayaBackend()
    agent = MagicMock()
    # Shape returned by laya.Agent.predict: answers nested under "answers".
    agent.predict.return_value = {
        "answers": {
            "intent": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.85, "support": 0.15},
            }
        },
        "usage": {},
    }
    _load_with_agent(backend, agent)
    assert backend.is_loaded

    res = backend.predict(_spec())

    state, questions = agent.predict.call_args.args
    assert state == "I need a refund for my last invoice."
    assert questions["intent"]["instructions"] == "Identify user intent"
    assert questions["intent"]["criteria"] == {"billing": "billing", "support": "support"}

    assert res.model_id == "convaiinnovations/laya"
    ans = res.answers["intent"]
    assert ans.value == "billing"
    assert ans.probabilities["billing"] == pytest.approx(0.85, abs=1e-6)
    assert ans.probabilities["support"] == pytest.approx(0.15, abs=1e-6)
    assert ans.calibrated is False


def test_laya_predict_rejects_choice_outside_options():
    backend = LayaBackend()
    agent = MagicMock()
    agent.predict.return_value = {
        "answers": {"intent": {"choice": "other", "probabilities": {"billing": 1.0}}}
    }
    _load_with_agent(backend, agent)

    with pytest.raises(InferenceError):
        backend.predict(_spec())


def test_laya_predict_rejects_missing_answer():
    backend = LayaBackend()
    agent = MagicMock()
    agent.predict.return_value = {"answers": {}}
    _load_with_agent(backend, agent)

    with pytest.raises(InferenceError):
        backend.predict(_spec())
