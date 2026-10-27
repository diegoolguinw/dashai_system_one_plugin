"""Unit tests for DashAI model adapters (train, predict, save, load)."""

import os
import tempfile

import numpy as np

from dashai_system_one_plugin.backends.fake_backend import FakeBackend
from dashai_system_one_plugin.models.base_dashai_system_one import (
    BaseDashAISystemOneClassifier,
)
from dashai_system_one_plugin.models.gliner_text_classifier import (
    GLiNERDecideTextClassifier,
)
from dashai_system_one_plugin.models.laya_text_classifier import (
    LayaTextClassifier,
)


class MockClassifier(BaseDashAISystemOneClassifier):
    MODEL_NAME = "mock-model"
    BACKEND_TYPE = "fake"

    def _create_backend(self):
        return FakeBackend(simulate_probabilities=True)


def test_classifier_train_and_predict_shape():
    model = MockClassifier(batch_size=2)
    x_train = ["My password expired", "Where is my invoice?", "Broken login"]
    y_train = ["login", "billing", "login"]

    model.train(x_train, y_train)
    assert model.class_names == ["login", "billing"]
    assert len(model.label_to_idx) == 2

    x_test = ["Cannot authenticate", "Need invoice copy", "System login failure"]
    preds = model.predict(x_test)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (3, 2)
    # Probabilities in each row should sum to ~1.0
    row_sums = preds.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-2)

    # Verify np.argmax (as expected by DashAI ClassificationTask.process_predictions)
    best_indices = np.argmax(preds, axis=1)
    assert len(best_indices) == 3


def test_classifier_save_and_load_roundtrip():
    model = MockClassifier()
    model.train(["text a", "text b"], ["class_1", "class_2"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "model_checkpoint.json")
        model.save(save_path)

        assert os.path.exists(save_path)

        loaded_model = MockClassifier.load(save_path)
        assert loaded_model.model_name == model.model_name
        assert loaded_model.class_names == model.class_names
        assert loaded_model.label_to_idx == model.label_to_idx


def test_concrete_model_attributes():
    laya = LayaTextClassifier()
    assert laya.model_name == "convaiinnovations/laya"
    assert "TextClassificationTask" in laya.COMPATIBLE_COMPONENTS

    gliner = GLiNERDecideTextClassifier()
    assert gliner.model_name == "fastino/GLiNER2.5-Decide"
    assert "TextClassificationTask" in gliner.COMPATIBLE_COMPONENTS
