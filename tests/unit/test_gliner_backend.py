"""Unit tests for GLiNER backend adapter."""

from unittest.mock import MagicMock, patch

import pytest

from dashai_system_one_plugin.backends.gliner_backend import GLiNERBackend
from dashai_system_one_plugin.core.contracts import DecisionSpec, QuestionSpec
from dashai_system_one_plugin.core.errors import BackendMissingError, InferenceError


def test_gliner_missing_backend_raises_typed_error():
    backend = GLiNERBackend()
    with patch.dict("sys.modules", {"gliner2": None, "gliner": None}):
        with pytest.raises(BackendMissingError) as exc_info:
            backend.load("fastino/GLiNER2.5-Decide")
        assert exc_info.value.backend_name == "gliner2"


def _load_with_extractor(backend: GLiNERBackend, extractor: MagicMock) -> None:
    mock_gliner_module = MagicMock()
    mock_gliner_module.AutoExtractor.from_pretrained.return_value = extractor
    with patch.dict("sys.modules", {"gliner2": mock_gliner_module}):
        backend.load("fastino/GLiNER2.5-Decide", device="cpu")


def _spec() -> DecisionSpec:
    return DecisionSpec(
        state="The screen does not turn on.",
        questions={
            "category": QuestionSpec(
                type="choice",
                instructions="Classify issue",
                criteria=["hardware_issue", "software_bug"],
            )
        },
    )


def test_gliner_predict_uncalibrated_one_hot():
    backend = GLiNERBackend()
    extractor = MagicMock()
    # classify_text returns {task_name: label} for a hard-label result.
    extractor.classify_text.return_value = {"category": "hardware_issue"}
    _load_with_extractor(backend, extractor)

    res = backend.predict(_spec())

    state, tasks = extractor.classify_text.call_args.args
    assert state == "The screen does not turn on."
    assert tasks["category"] == {
        "labels": ["hardware_issue", "software_bug"],
        "prompt": "Classify issue",
    }

    ans = res.answers["category"]
    assert ans.value == "hardware_issue"
    # ADR-002: calibrated must be False when probabilities were not produced by model
    assert ans.calibrated is False
    assert ans.probabilities["hardware_issue"] == 1.0
    assert ans.probabilities["software_bug"] == 0.0
    assert backend.capabilities()["probabilities_available"] is False


def test_gliner_predict_rejects_label_outside_options():
    backend = GLiNERBackend()
    extractor = MagicMock()
    extractor.classify_text.return_value = {"category": "unknown"}
    _load_with_extractor(backend, extractor)

    with pytest.raises(InferenceError):
        backend.predict(_spec())
