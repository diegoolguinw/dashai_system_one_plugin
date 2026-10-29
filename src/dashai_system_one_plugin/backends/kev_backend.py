"""Kev backend adapter communicating with a local kev.serve process over loopback."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import warnings
from typing import Any, ClassVar

from dashai_system_one_plugin.core.base_backend import BaseSystemOneBackend
from dashai_system_one_plugin.core.contracts import (
    DecisionAnswer,
    DecisionResult,
    DecisionSpec,
    QuestionSpec,
)
from dashai_system_one_plugin.core.errors import (
    InferenceError,
    SecurityError,
)
from dashai_system_one_plugin.utils.kev_process_manager import (
    HEALTH_PATH,
    LOOPBACK_OPENER,
    KevProcessManager,
)


class KevBackend(BaseSystemOneBackend):
    """Adapter consuming a strictly local Kev runtime on 127.0.0.1 loopback.

    Talks to ``kev.serve`` (TypeSafe-compatible ``POST /v1/systemone``). If the
    service is not up and a Kev interpreter is given (``kev_python`` kwarg or the
    ``KEV_PYTHON`` variable), ``load`` starts it and ``unload`` stops it.
    """

    ALLOWED_HOSTS: ClassVar[set[str]] = {"127.0.0.1", "localhost", "::1"}

    def __init__(self, endpoint_url: str = "http://127.0.0.1:8008"):
        super().__init__()
        self.endpoint_url = self._validate_local_endpoint(endpoint_url)
        self.timeout_seconds = 30.0
        self._manager: KevProcessManager | None = None
        self._server_info: dict[str, Any] = {}

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

    def _request(self, path: str, payload: dict[str, Any] | None, timeout: float) -> Any:
        """One loopback HTTP call returning parsed JSON; never proxied, never redirected."""
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.endpoint_url}{path}",
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "dashai-system-one-plugin",
            },
            method="GET" if data is None else "POST",
        )
        try:
            with LOOPBACK_OPENER.open(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise InferenceError(f"Kev service error HTTP {exc.code}: {detail}") from exc
        except json.JSONDecodeError as exc:
            raise InferenceError(f"Invalid JSON response from Kev service: {exc}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise InferenceError(f"Failed to communicate with local Kev service: {exc}") from exc

    def load(
        self,
        checkpoint_id: str = "jaredpalmer/kev-0.8b",
        revision: str | None = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Connect to the local Kev service, starting it first when configured.

        Recognised kwargs: ``endpoint_url``, ``timeout`` (per request, seconds),
        ``kev_python`` (interpreter of the Kev environment), ``kev_repo_dir``
        (working directory for it), ``startup_timeout`` (seconds).
        """
        self.checkpoint_id = checkpoint_id
        self.revision = revision or "main"
        self.device = device

        if "endpoint_url" in kwargs:
            self.endpoint_url = self._validate_local_endpoint(kwargs["endpoint_url"])
        if "timeout" in kwargs:
            self.timeout_seconds = float(kwargs["timeout"])

        try:
            info = self._request(HEALTH_PATH, None, timeout=2.0)
        except InferenceError as exc:
            self._start_managed(kwargs, exc)
            info = self._request(HEALTH_PATH, None, timeout=2.0)

        self._server_info = info if isinstance(info, dict) else {}
        models = self._server_info.get("models") or [{}]
        served = str(models[0].get("run", ""))
        if served and served.split("@")[0] != checkpoint_id:
            warnings.warn(
                f"Kev service is serving '{served}', not '{checkpoint_id}'.",
                stacklevel=2,
            )
        self._is_loaded = True

    def _start_managed(self, kwargs: dict[str, Any], original: InferenceError) -> None:
        """Launch kev.serve ourselves, or re-raise the connection error if unconfigured."""
        kev_python = kwargs.get("kev_python") or os.environ.get("KEV_PYTHON")
        if not kev_python:
            self._is_loaded = False
            raise InferenceError(
                f"Could not connect to local Kev service at {self.endpoint_url}. "
                "Start it with `python -m kev.serve --run jaredpalmer/kev-0.8b --port 8008` "
                "or pass kev_python / set KEV_PYTHON so the plugin can start it."
            ) from original

        parsed = urllib.parse.urlparse(self.endpoint_url)
        self._manager = KevProcessManager(
            python_executable=kev_python,
            checkpoint=self.checkpoint_id or "jaredpalmer/kev-0.8b",
            port=parsed.port or 8008,
            cwd=kwargs.get("kev_repo_dir") or os.environ.get("KEV_REPO_DIR"),
            startup_timeout=float(kwargs.get("startup_timeout", 300.0)),
        )
        self._manager.start()

    def unload(self) -> None:
        if self._manager is not None:
            self._manager.stop()
            self._manager = None
        self._is_loaded = False
        self.checkpoint_id = None

    def capabilities(self) -> dict[str, Any]:
        return {
            "supported_question_types": ["choice", "score", "noul"],
            "probabilities_available": True,
            "multi_question_support": True,
            "streaming": False,
        }

    @staticmethod
    def _question_payload(q_name: str, q_spec: QuestionSpec) -> dict[str, Any]:
        criteria = q_spec.criteria
        payload: dict[str, Any] = {"type": q_spec.type, "instructions": q_spec.instructions}
        if q_spec.type == "choice":
            # Kev wants {option: description}; a bare list means no descriptions.
            payload["criteria"] = (
                {opt: None for opt in criteria} if isinstance(criteria, list) else dict(criteria or {})
            )
        elif q_spec.type == "score":
            if isinstance(criteria, dict):
                criteria = list(criteria.values())
            if not criteria:
                raise InferenceError(f"Score question '{q_name}' needs ordered level criteria.")
            payload["criteria"] = list(criteria)
        elif isinstance(criteria, dict):  # noul: optional {"false": ..., "true": ...}
            payload["criteria"] = criteria
        return payload

    def predict(self, spec: DecisionSpec) -> DecisionResult:
        if not self._is_loaded:
            raise RuntimeError("KevBackend must be loaded before calling predict().")

        payload = {
            "state": spec.state,
            "questions": {
                name: self._question_payload(name, q) for name, q in spec.questions.items()
            },
        }
        raw = self._request("/v1/systemone", payload, timeout=self.timeout_seconds)
        raw_answers = raw.get("answers") if isinstance(raw, dict) else None
        if not isinstance(raw_answers, dict):
            raise InferenceError("Kev service response has no 'answers' object.")

        answers: dict[str, DecisionAnswer] = {}
        for q_name, q_spec in spec.questions.items():
            raw_q = raw_answers.get(q_name)
            if not isinstance(raw_q, dict):
                raise InferenceError(f"Kev returned no answer for question '{q_name}'.")
            answers[q_name] = self._parse_answer(q_name, q_spec, raw_q)

        return DecisionResult(
            answers=answers,
            model_id=self.checkpoint_id or "jaredpalmer/kev-0.8b",
            revision=self.revision,
        )

    @staticmethod
    def _parse_answer(q_name: str, q_spec: QuestionSpec, raw_q: dict[str, Any]) -> DecisionAnswer:
        # Kev applies a vendor temperature fitted in-distribution; it documents that this
        # does not transfer to every workload, so calibration on user data is not asserted.
        try:
            if q_spec.type == "choice":
                probs = {str(k): float(v) for k, v in raw_q["probabilities"].items()}
                return DecisionAnswer(
                    kind="choice", value=raw_q["choice"], probabilities=probs, calibrated=None
                )
            if q_spec.type == "noul":
                p_true = float(raw_q["noul"])
                return DecisionAnswer(
                    kind="noul",
                    value=p_true >= 0.5,
                    probabilities={"false": 1.0 - p_true, "true": p_true},
                    calibrated=None,
                )
            probs = {str(k): float(v) for k, v in raw_q["probabilities"].items()}
            return DecisionAnswer(
                kind="score", value=float(raw_q["score"]), probabilities=probs, calibrated=None
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise InferenceError(
                f"Malformed Kev answer for question '{q_name}': {raw_q!r}"
            ) from exc
