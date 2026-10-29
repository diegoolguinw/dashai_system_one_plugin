"""Supervisor for the local Kev runtime (``python -m kev.serve``) on loopback."""

import atexit
import contextlib
import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

from dashai_system_one_plugin.core.errors import InferenceError

LOOPBACK_HOST = "127.0.0.1"
HEALTH_PATH = "/v1/models"  # kev.serve has no /health; this is its cheapest route


class _NoProxyNoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so a loopback request can never be bounced elsewhere."""

    def redirect_request(self, *args, **kwargs):  # type: ignore[override]
        return None


# An empty ProxyHandler ignores http_proxy/https_proxy from the environment.
LOOPBACK_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), _NoProxyNoRedirect
)


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((LOOPBACK_HOST, port)) != 0


class KevProcessManager:
    """Start, health-check and stop one ``kev.serve`` subprocess.

    The server is always bound to 127.0.0.1 (the host is not configurable). The
    child runs in its own process group so that stopping it also stops anything it
    spawned, and it is stopped again at interpreter exit if still alive.
    """

    def __init__(
        self,
        python_executable: str | os.PathLike[str],
        checkpoint: str = "jaredpalmer/kev-0.8b",
        port: int = 8008,
        cwd: str | os.PathLike[str] | None = None,
        startup_timeout: float = 300.0,
        grace_seconds: float = 10.0,
        extra_args: Sequence[str] = (),
        env: dict[str, str] | None = None,
    ) -> None:
        self.python_executable = str(python_executable)
        self.checkpoint = checkpoint
        self.port = port
        self.cwd = str(cwd) if cwd is not None else None
        self.startup_timeout = startup_timeout
        self.grace_seconds = grace_seconds
        self.extra_args = list(extra_args)
        self.env = env
        self.log_path: Path | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._atexit_registered = False

    @property
    def endpoint_url(self) -> str:
        return f"http://{LOOPBACK_HOST}:{self.port}"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def command(self) -> list[str]:
        return [
            self.python_executable,
            "-m",
            "kev.serve",
            "--run",
            self.checkpoint,
            "--host",
            LOOPBACK_HOST,
            "--port",
            str(self.port),
            *self.extra_args,
        ]

    def is_healthy(self, timeout: float = 2.0) -> bool:
        try:
            with LOOPBACK_OPENER.open(
                f"{self.endpoint_url}{HEALTH_PATH}", timeout=timeout
            ) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def start(self) -> None:
        """Launch the server and block until it answers or startup_timeout expires."""
        if self.is_running:
            return
        if not _port_is_free(self.port):
            raise InferenceError(
                f"Port {self.port} on {LOOPBACK_HOST} is already in use; "
                "refusing to start Kev there."
            )

        fd, log_name = tempfile.mkstemp(prefix="kev-serve-", suffix=".log")
        self.log_path = Path(log_name)
        env = {**os.environ, **(self.env or {})}
        try:
            with os.fdopen(fd, "wb") as log:
                self._process = subprocess.Popen(
                    self.command(),
                    cwd=self.cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except OSError as exc:
            self._process = None
            raise InferenceError(f"Could not launch Kev runtime: {exc}") from exc

        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                code = self._process.returncode
                self._process = None
                raise InferenceError(
                    f"Kev runtime exited during startup (code {code}).\n{self._log_tail()}"
                )
            if self.is_healthy():
                return
            time.sleep(0.5)

        tail = self._log_tail()
        self.stop()
        raise InferenceError(
            f"Kev runtime not ready after {self.startup_timeout:.0f}s.\n{tail}"
        )

    def stop(self) -> None:
        """Terminate the server and its process group. Safe to call repeatedly."""
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        self._signal_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=self.grace_seconds)
        except subprocess.TimeoutExpired:
            self._signal_group(process, signal.SIGKILL)
            process.wait()

    @staticmethod
    def _signal_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
        with contextlib.suppress(ProcessLookupError):
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(process.pid), sig)
            else:  # pragma: no cover - non-POSIX
                process.send_signal(sig)

    def _log_tail(self, lines: int = 20) -> str:
        if self.log_path is None or not self.log_path.exists():
            return ""
        text = self.log_path.read_text(errors="replace").splitlines()
        return "\n".join(text[-lines:])

    def __enter__(self) -> "KevProcessManager":  # noqa: PYI034 - py3.10 has no Self
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
