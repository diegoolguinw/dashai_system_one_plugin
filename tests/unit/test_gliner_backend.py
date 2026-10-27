"""Unit tests for GLiNER backend adapter."""

from unittest.mock import MagicMock, patch

import pytest

from dashai_system_one_plugin.backends.gliner_backend import GLiNERBackend
from dashai_system_one_plugin.core.contracts import DecisionSpec, QuestionSpec
from dashai_system_one_plugin.core.errors import BackendMissingError


def test_gliner_missing_backend_raises_typed_error():
    backend = GLiNERBackend()
    with patch.dict("sys.modules", {"gliner2": None, "gliner": None}):
        with pytest.raises(BackendMissingError) as exc_info:
            backend.load("fastino/GLiNER2.5-Decide")
        assert exc_info.value.backend_name == "gliner2"


def test_gliner_predict_uncalibrated_one_hot_fallback():
    backend = GLiNERBackend()

    mock_extractor = MagicMock()
    # Mock returning raw label string without probabilities
    mock_extractor.classify_text.return_value = "hardware_issue"

    mock_gliner_module = MagicMock()
    mock_gliner_module.AutoExtractor.from_pretrained.return_value = mock_extractor

    with patch.dict("sys.modules", {"gliner2": mock_gliner_module}):
        backend.load("fastino/GLiNER2.5-Decide", device="cpu")

        spec = DecisionSpec(
            state="The screen does not turn on.",
            questions={
                "category": QuestionSpec(
                    type="choice",
                    instructions="Classify issue",
                    criteria=["hardware_issue", "software_bug"],
                )
            },
        )

        res = backend.predict(spec)
        ans = res.answers["category"]
        assert ans.value == "hardware_issue"
        # ADR-002: calibrated must be False when probabilities were not produced by model
        assert ans.calibrated is False
        assert ans.probabilities["hardware_issue"] == 1.0
        assert ans.probabilities["software_bug"] == 0.0
