"""Unit tests for checkpoint registry and license verification."""

import pytest

from dashai_system_one_plugin.core.checkpoint_registry import CheckpointRegistry


def test_registry_contains_approved_models():
    models = CheckpointRegistry.list_checkpoints()
    ids = [m.checkpoint_id for m in models]
    assert "convaiinnovations/laya" in ids
    assert "convaiinnovations/laya-multilingual" in ids
    assert "fastino/GLiNER2.5-Decide" in ids
    assert "fastino/GLiNER2.5-multi-Decide" in ids
    assert len(ids) == 4


def test_registry_license_is_permissive():
    for m in CheckpointRegistry.list_checkpoints():
        assert m.license_spdx in ("Apache-2.0", "MIT")


def test_unknown_checkpoint_license_raises():
    with pytest.raises(KeyError):
        CheckpointRegistry.get_license("unknown/unregistered-model")
