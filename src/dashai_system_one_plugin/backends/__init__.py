"""Backends package providing execution engines for System One models."""

from dashai_system_one_plugin.backends.fake_backend import FakeBackend
from dashai_system_one_plugin.backends.gliner_backend import GLiNERBackend
from dashai_system_one_plugin.backends.laya_backend import LayaBackend

__all__ = [
    "FakeBackend",
    "GLiNERBackend",
    "LayaBackend",
]
