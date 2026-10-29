"""Unit tests for Kev backend and loopback security boundaries."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from dashai_system_one_plugin.backends.kev_backend import KevBackend
from dashai_system_one_plugin.core.contracts import DecisionSpec, QuestionSpec
from dashai_system_one_plugin.core.errors import InferenceError, SecurityError


class _FakeKev(BaseHTTPRequestHandler):
    """Minimal stand-in for kev.serve: GET /v1/models and POST /v1/systemone."""

    requests: ClassVar[list[dict]] = []
    answers: ClassVar[dict] = {}
    status = 200

    def log_message(self, *args):
        pass

    def _send(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"models": [{"name": "kev-latest", "run": "jaredpalmer/kev-0.8b"}]})
        else:
            self._send(404, {"detail": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "body": body})
        if type(self).status != 200:
            self._send(type(self).status, {"detail": "state too long"})
        else:
            self._send(200, {"model": "kev-latest", "answers": type(self).answers})


@pytest.fixture
def fake_kev():
    _FakeKev.requests = []
    _FakeKev.answers = {}
    _FakeKev.status = 200
    server = HTTPServer(("127.0.0.1", 0), _FakeKev)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", _FakeKev
    server.shutdown()
    server.server_close()


def _loaded(url: str) -> KevBackend:
    backend = KevBackend(endpoint_url=url)
    backend.load("jaredpalmer/kev-0.8b")
    return backend


def test_kev_rejects_external_hosts():
    with pytest.raises(SecurityError) as exc_info:
        KevBackend(endpoint_url="https://api.external-cloud.com/v1")
    assert "Security violation" in str(exc_info.value)

    with pytest.raises(SecurityError):
        KevBackend(endpoint_url="http://192.168.1.50:8000")


def test_kev_rejects_external_endpoint_passed_to_load():
    backend = KevBackend()
    with pytest.raises(SecurityError):
        backend.load(endpoint_url="http://example.com:8008")


def test_kev_allows_localhost_loopback():
    k1 = KevBackend(endpoint_url="http://127.0.0.1:8000")
    assert k1.endpoint_url == "http://127.0.0.1:8000"

    k2 = KevBackend(endpoint_url="http://localhost:9000")
    assert k2.endpoint_url == "http://localhost:9000"


def test_kev_load_without_service_or_python_raises_actionable_error(monkeypatch):
    monkeypatch.delenv("KEV_PYTHON", raising=False)
    backend = KevBackend(endpoint_url="http://127.0.0.1:1")
    with pytest.raises(InferenceError, match="kev.serve"):
        backend.load()
    assert not backend.is_loaded


def test_kev_predict_choice_noul_score_over_loopback(fake_kev):
    url, handler = fake_kev
    handler.answers = {
        "team": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.8,
            "probabilities": {"billing": 0.9, "shipping": 0.1},
        },
        "urgent": {"type": "noul", "noul": 0.75},
        "sev": {
            "type": "score",
            "score": 1.2,
            "legend": {"0": "low", "1": "mid", "2": "high"},
            "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4},
        },
    }
    backend = _loaded(url)
    spec = DecisionSpec(
        state="I was charged twice.",
        questions={
            "team": QuestionSpec(type="choice", instructions="Team?", criteria=["billing", "shipping"]),
            "urgent": QuestionSpec(type="noul", instructions="Reply today?"),
            "sev": QuestionSpec(type="score", instructions="Severity?", criteria=["low", "mid", "high"]),
        },
    )

    result = backend.predict(spec)

    sent = handler.requests[0]
    assert sent["path"] == "/v1/systemone"
    assert sent["body"]["state"] == "I was charged twice."
    assert sent["body"]["questions"]["team"]["criteria"] == {"billing": None, "shipping": None}
    assert sent["body"]["questions"]["sev"]["criteria"] == ["low", "mid", "high"]
    assert "criteria" not in sent["body"]["questions"]["urgent"]

    team = result.answers["team"]
    assert team.value == "billing"
    assert team.probabilities["billing"] == pytest.approx(0.9)
    urgent = result.answers["urgent"]
    assert urgent.kind == "noul" and urgent.value is True
    assert urgent.probabilities == {"false": pytest.approx(0.25), "true": pytest.approx(0.75)}
    sev = result.answers["sev"]
    assert sev.kind == "score" and sev.value == pytest.approx(1.2)
    # Vendor calibration is in-distribution only, so none is asserted.
    assert all(a.calibrated is None for a in result.answers.values())
    assert result.model_id == "jaredpalmer/kev-0.8b"


def test_kev_surfaces_http_errors(fake_kev):
    url, handler = fake_kev
    backend = _loaded(url)
    handler.status = 422
    spec = DecisionSpec(
        state="x",
        questions={"q": QuestionSpec(type="noul", instructions="?")},
    )
    with pytest.raises(InferenceError, match="422"):
        backend.predict(spec)


def test_kev_rejects_missing_answer(fake_kev):
    url, handler = fake_kev
    handler.answers = {}
    backend = _loaded(url)
    spec = DecisionSpec(
        state="x",
        questions={"q": QuestionSpec(type="noul", instructions="?")},
    )
    with pytest.raises(InferenceError, match="no answer"):
        backend.predict(spec)


def test_kev_ignores_proxy_environment(fake_kev, monkeypatch):
    # A proxy variable must not capture loopback traffic.
    monkeypatch.setenv("http_proxy", "http://203.0.113.1:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://203.0.113.1:3128")
    url, _ = fake_kev
    assert _loaded(url).is_loaded
