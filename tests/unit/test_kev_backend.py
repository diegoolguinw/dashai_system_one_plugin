"""Unit tests for Kev backend and loopback security boundaries."""

import json
from unittest.mock import MagicMock, patch

import pytest

from dashai_system_one_plugin.backends.kev_backend import KevBackend
from dashai_system_one_plugin.core.contracts import DecisionSpec, QuestionSpec
from dashai_system_one_plugin.core.errors import SecurityError


def test_kev_rejects_external_hosts():
    with pytest.raises(SecurityError) as exc_info:
        KevBackend(endpoint_url="https://api.external-cloud.com/v1")
    assert "Security violation" in str(exc_info.value)

    with pytest.raises(SecurityError):
        KevBackend(endpoint_url="http://192.168.1.50:8000")


def test_kev_allows_localhost_loopback():
    k1 = KevBackend(endpoint_url="http://127.0.0.1:8000")
    assert k1.endpoint_url == "http://127.0.0.1:8000"

    k2 = KevBackend(endpoint_url="http://localhost:9000")
    assert k2.endpoint_url == "http://localhost:9000"


def test_kev_predict_over_mock_loopback():
    backend = KevBackend(endpoint_url="http://127.0.0.1:8000")
    backend._is_loaded = True

    response_payload = {
        "answers": {
            "ticket": {
                "kind": "choice",
                "value": "billing",
                "probabilities": {"billing": 0.9, "support": 0.1},
                "calibrated": True,
            }
        }
    }
    raw_bytes = json.dumps(response_payload).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = raw_bytes
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        spec = DecisionSpec(
            state="Invoice discrepancy",
            questions={
                "ticket": QuestionSpec(
                    type="choice",
                    instructions="Routing",
                    criteria=["billing", "support"],
                )
            },
        )
        result = backend.predict(spec)
        assert result.answers["ticket"].value == "billing"
        assert result.answers["ticket"].probabilities["billing"] == 0.9
