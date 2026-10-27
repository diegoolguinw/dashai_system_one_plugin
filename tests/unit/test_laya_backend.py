"""Unit tests for Laya backend adapter."""

from unittest.mock import MagicMock, patch

import pytest

from dashai_system_one_plugin.backends.laya_backend import LayaBackend
from dashai_system_one_plugin.core.contracts import DecisionSpec, QuestionSpec
from dashai_system_one_plugin.core.errors import BackendMissingError


def test_laya_missing_backend_raises_typed_error():
    backend = LayaBackend()
    with patch.dict("sys.modules", {"laya": None}):
        with pytest.raises(BackendMissingError) as exc_info:
            backend.load("convaiinnovations/laya")
        assert exc_info.value.backend_name == "laya"


def test_laya_predict_with_mocked_router():
    backend = LayaBackend()

    mock_router_instance = MagicMock()
    mock_router_instance.predict.return_value = {
        "intent": {
            "choice": "billing",
            "probabilities": {"billing": 0.85, "support": 0.15},
        }
    }

    mock_laya_module = MagicMock()
    mock_laya_module.Router.return_value = mock_router_instance

    with patch.dict("sys.modules", {"laya": mock_laya_module}):
        backend.load("convaiinnovations/laya", device="cpu")
        assert backend.is_loaded

        spec = DecisionSpec(
            state="I need a refund for my last invoice.",
            questions={
                "intent": QuestionSpec(
                    type="choice",
                    instructions="Identify user intent",
                    criteria=["billing", "support"],
                )
            },
        )

        res = backend.predict(spec)
        assert res.model_id == "convaiinnovations/laya"
        ans = res.answers["intent"]
        assert ans.value == "billing"
        assert ans.probabilities["billing"] == pytest.approx(0.85, abs=1e-2)
        assert ans.probabilities["support"] == pytest.approx(0.15, abs=1e-2)
