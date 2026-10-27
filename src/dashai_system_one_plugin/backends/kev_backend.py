"""Kev backend adapter communicating with local process over strict loopback."""

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, ClassVar

from dashai_system_one_plugin.core.base_backend import BaseSystemOneBackend
from dashai_system_one_plugin.core.contracts import (
    DecisionAnswer,
    DecisionResult,
    DecisionSpec,
)
from dashai_system_one_plugin.core.errors import (
    InferenceError,
    SecurityError,
)


class KevBackend(BaseSystemOneBackend):
    """Adapter consuming a strictly local Kev runtime on 127.0.0.1 loopback."""

    ALLOWED_HOSTS: ClassVar[set[str]] = {"127.0.0.1", "localhost", "::1"}

    def __init__(self, endpoint_url: str = "http://127.0.0.1:8000"):
        super().__init__()
        self.endpoint_url = self._validate_local_endpoint(endpoint_url)
        self.timeout_seconds = 10.0

    @classmethod
    def _validate_local_endpoint(cls, url: str) -> str:
        """Enforce strict local loopback to prevent external network leakage."""
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise SecurityError(f"Invalid URL scheme: '{parsed.scheme}'. Only local HTTP allowed.")
        hostname = (parsed.hostname or "").lower()
        if hostname not in cls.ALLOWED_HOSTS:
            raise SecurityError(
                f"Security violation: Kev backend only permits localhost/127.0.0.1 loopback. "
                f"Attempted connection to '{hostname}'."
            )
        return url.rstrip("/")

    def load(
        self,
        checkpoint_id: str = "jaredpalmer/kev-0.8b",
        revision: str | None = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Verify connection to local Kev engine."""
        self.checkpoint_id = checkpoint_id
        self.revision = revision or "main"
        self.device = device

        if "endpoint_url" in kwargs:
            self.endpoint_url = self._validate_local_endpoint(kwargs["endpoint_url"])
        if "timeout" in kwargs:
            self.timeout_seconds = float(kwargs["timeout"])

        # Health check on local loopback
        health_url = f"{self.endpoint_url}/health"
        req = urllib.request.Request(health_url, headers={"User-Agent": "dashai-system-one-plugin"})
        try:
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status != 200:
                    raise InferenceError(f"Kev local service returned status {resp.status}")
        except urllib.error.URLError as exc:
            # Mark unready if service is offline, but do not crash load if configured for delayed launch
            self._is_loaded = False
            raise InferenceError(
                f"Could not connect to local Kev service at {self.endpoint_url}. "
                "Ensure local Kev process is running on 127.0.0.1."
            ) from exc

        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False
        self.checkpoint_id = None

    def capabilities(self) -> dict[str, Any]:
        return {
            "supported_question_types": ["choice", "score", "noul"],
            "probabilities_available": True,
            "multi_question_support": True,
            "streaming": False,
        }

    def predict(self, spec: DecisionSpec) -> DecisionResult:
        if not self._is_loaded:
            raise RuntimeError("KevBackend must be loaded before calling predict().")

        predict_url = f"{self.endpoint_url}/decide"
        payload = spec.model_dump()
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            predict_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "dashai-system-one-plugin",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                if resp.status != 200:
                    raise InferenceError(f"Kev service error HTTP {resp.status}")
                body = resp.read().decode("utf-8")
                raw_result = json.loads(body)
        except urllib.error.URLError as exc:
            raise InferenceError(f"Failed to communicate with local Kev service: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise InferenceError(f"Invalid JSON response from Kev service: {exc}") from exc

        # Construct DecisionResult from raw output
        answers: dict[str, DecisionAnswer] = {}
        for q_name, ans_dict in raw_result.get("answers", {}).items():
            answers[q_name] = DecisionAnswer(
                kind=ans_dict["kind"],
                value=ans_dict["value"],
                probabilities=ans_dict.get("probabilities"),
                calibrated=ans_dict.get("calibrated", True),
            )

        return DecisionResult(
            answers=answers,
            model_id=self.checkpoint_id or "jaredpalmer/kev-0.8b",
            revision=self.revision,
        )
