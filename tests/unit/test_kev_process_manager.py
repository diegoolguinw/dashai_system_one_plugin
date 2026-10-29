"""Lifecycle tests for KevProcessManager using a toy ``kev.serve`` module."""

import os
import socket
import sys
import textwrap
import time

import pytest

from dashai_system_one_plugin.core.errors import InferenceError
from dashai_system_one_plugin.utils.kev_process_manager import KevProcessManager

TOY_SERVE = textwrap.dedent(
    """
    import argparse, json, os, signal, subprocess, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--host"); ap.add_argument("--port", type=int)
    a = ap.parse_args()
    mode = os.environ.get("TOY_MODE", "ok")
    if mode == "crash":
        print("boom: cannot load checkpoint", flush=True); sys.exit(3)
    if mode == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode == "never_ready":
        time.sleep(600)

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    open(os.environ["TOY_PIDFILE"], "w").write(f"{os.getpid()} {child.pid} {a.host}")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x): pass
        def do_GET(self):
            body = json.dumps({"models": [{"name": "kev-latest", "run": a.run}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

    HTTPServer((a.host, a.port), H).serve_forever()
    """
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def toy(tmp_path, monkeypatch):
    (tmp_path / "kev").mkdir()
    (tmp_path / "kev" / "__init__.py").write_text("")
    (tmp_path / "kev" / "serve.py").write_text(TOY_SERVE)
    pidfile = tmp_path / "pids"
    monkeypatch.setenv("TOY_PIDFILE", str(pidfile))
    monkeypatch.delenv("TOY_MODE", raising=False)

    def make(**kwargs):
        return KevProcessManager(
            sys.executable, port=_free_port(), cwd=tmp_path, **kwargs
        )

    return make, pidfile


def test_start_binds_loopback_and_stop_kills_whole_group(toy):
    make, pidfile = toy
    manager = make(startup_timeout=30)
    with manager:
        assert manager.is_running and manager.is_healthy()
        server_pid, grandchild_pid, host = pidfile.read_text().split()
        assert host == "127.0.0.1"
        assert manager.endpoint_url.startswith("http://127.0.0.1:")
    assert not manager.is_running
    assert not _alive(int(server_pid))
    assert not _alive(int(grandchild_pid))  # no orphaned descendants


def test_stop_is_idempotent(toy):
    make, _ = toy
    manager = make(startup_timeout=30)
    manager.start()
    manager.stop()
    manager.stop()
    assert not manager.is_running


def test_stop_escalates_to_sigkill_when_sigterm_ignored(toy, monkeypatch):
    make, pidfile = toy
    monkeypatch.setenv("TOY_MODE", "stubborn")
    manager = make(startup_timeout=30, grace_seconds=0.5)
    manager.start()
    server_pid = int(pidfile.read_text().split()[0])
    started = time.monotonic()
    manager.stop()
    assert time.monotonic() - started < 10
    assert not _alive(server_pid)


def test_start_reports_crash_with_log_tail(toy, monkeypatch):
    make, _ = toy
    monkeypatch.setenv("TOY_MODE", "crash")
    manager = make(startup_timeout=30)
    with pytest.raises(InferenceError, match="boom: cannot load checkpoint"):
        manager.start()
    assert not manager.is_running


def test_start_times_out_and_cleans_up(toy, monkeypatch):
    make, _ = toy
    monkeypatch.setenv("TOY_MODE", "never_ready")
    manager = make(startup_timeout=1.5, grace_seconds=2)
    with pytest.raises(InferenceError, match="not ready"):
        manager.start()
    assert not manager.is_running


def test_start_refuses_a_port_already_in_use(toy):
    make, _ = toy
    manager = make(startup_timeout=5)
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", manager.port))
        busy.listen()
        with pytest.raises(InferenceError, match="already in use"):
            manager.start()


def test_backend_autostarts_and_unload_stops_the_process(toy, tmp_path):
    from dashai_system_one_plugin.backends.kev_backend import KevBackend

    _, pidfile = toy
    port = _free_port()
    backend = KevBackend(endpoint_url=f"http://127.0.0.1:{port}")
    backend.load(kev_python=sys.executable, kev_repo_dir=tmp_path, startup_timeout=30)
    assert backend.is_loaded
    server_pid, grandchild_pid = (int(x) for x in pidfile.read_text().split()[:2])

    backend.unload()
    assert not backend.is_loaded
    assert not _alive(server_pid)
    assert not _alive(grandchild_pid)
