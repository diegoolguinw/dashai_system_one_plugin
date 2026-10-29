"""Unit tests for classification and calibration metrics."""

import numpy as np
import pytest

from dashai_system_one_plugin.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    log_loss,
    macro_f1,
)

Y = np.array([0, 1, 2, 1])
P = np.array(
    [
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.5, 0.3, 0.2],  # wrong
        [0.2, 0.6, 0.2],
    ]
)


def test_accuracy_and_macro_f1_match_hand_computation():
    assert accuracy(Y, P) == pytest.approx(0.75)
    # class0: tp1 fp1 fn0 -> 2/3 ; class1: tp2 -> 1 ; class2: tp0 fn1 -> 0
    assert macro_f1(Y, P) == pytest.approx((2 / 3 + 1.0 + 0.0) / 3)


def test_macro_f1_agrees_with_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(5), size=200)
    y = rng.integers(0, 5, size=200)
    expected = sklearn_metrics.f1_score(y, probs.argmax(axis=1), average="macro")
    assert macro_f1(y, probs) == pytest.approx(expected)


def test_log_loss_and_brier_hand_computed():
    assert log_loss(Y, P) == pytest.approx(-np.mean(np.log([0.7, 0.8, 0.2, 0.6])))
    first = (0.3**2 + 0.2**2 + 0.1**2)
    assert brier_score(Y[:1], P[:1]) == pytest.approx(first)


def test_log_loss_finite_for_one_hot_wrong_prediction():
    one_hot = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert np.isfinite(log_loss(np.array([0, 1]), one_hot))


def test_ece_zero_when_confidence_equals_accuracy():
    # All predictions confidence 1.0 and all correct.
    probs = np.eye(3)[[0, 1, 2, 0]]
    assert expected_calibration_error(np.array([0, 1, 2, 0]), probs) == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    probs = np.tile([0.99, 0.01], (100, 1))
    y = np.array([0] * 50 + [1] * 50)  # 50% accurate at 99% confidence
    assert expected_calibration_error(y, probs) == pytest.approx(0.49)


def test_shape_validation():
    with pytest.raises(ValueError):
        accuracy(np.array([0, 1]), np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError):
        accuracy(np.array([3]), np.array([[0.5, 0.5]]))
