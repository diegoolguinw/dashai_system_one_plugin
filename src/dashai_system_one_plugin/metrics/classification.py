"""Decision-quality and calibration metrics from docs/benchmark-protocol.md."""

import numpy as np


def _check(y_true: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or len(y_true) != len(probs) or len(y_true) == 0:
        raise ValueError("probs must be (N, K) with N == len(y_true) > 0.")
    if y_true.min() < 0 or y_true.max() >= probs.shape[1]:
        raise ValueError("y_true contains a class index outside [0, K).")
    return y_true, probs


def accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true, probs = _check(y_true, probs)
    return float(np.mean(probs.argmax(axis=1) == y_true))


def macro_f1(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Unweighted mean F1 over classes present in the truth or the predictions."""
    y_true, probs = _check(y_true, probs)
    pred = probs.argmax(axis=1)
    scores = []
    for k in np.union1d(y_true, pred):
        tp = np.sum((pred == k) & (y_true == k))
        fp = np.sum((pred == k) & (y_true != k))
        fn = np.sum((pred != k) & (y_true == k))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def log_loss(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-15) -> float:
    """Mean negative log-likelihood of the true class (probabilities clipped at eps)."""
    y_true, probs = _check(y_true, probs)
    p_true = np.clip(probs[np.arange(len(y_true)), y_true], eps, 1.0)
    return float(-np.mean(np.log(p_true)))


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Multi-class Brier score: mean over samples of the summed squared error."""
    y_true, probs = _check(y_true, probs)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> float:
    """Top-label ECE over uniform confidence bins."""
    y_true, probs = _check(y_true, probs)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)
