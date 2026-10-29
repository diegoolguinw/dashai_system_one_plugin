#!/usr/bin/env python3
"""Benchmark one (model, device, language) configuration in a fresh process.

A fresh process per configuration keeps peak memory attributable to that model.
Prints a single JSON document on the last stdout line.
"""

import argparse
import gzip
import json
import pathlib
import platform
import random
import resource
import subprocess
import sys
import time
from importlib import metadata

import numpy as np

from dashai_system_one_plugin.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    log_loss,
    macro_f1,
)
from dashai_system_one_plugin.models.gliner_text_classifier import (
    GLiNERDecideTextClassifier,
    GLiNERMultiDecideTextClassifier,
)
from dashai_system_one_plugin.models.kev_text_classifier import KevTextClassifier
from dashai_system_one_plugin.models.laya_text_classifier import (
    LayaMultilingualTextClassifier,
    LayaTextClassifier,
)

MODELS = {
    "laya": LayaTextClassifier,
    "laya-multilingual": LayaMultilingualTextClassifier,
    "gliner": GLiNERDecideTextClassifier,
    "gliner-multilingual": GLiNERMultiDecideTextClassifier,
    "kev": KevTextClassifier,
}
INSTRUCTION = "Classify the user utterance by the scenario it belongs to."


def load_samples(path: str, n: int, seed: int) -> tuple[list[str], list[str]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    # Same ids are drawn for every language because the splits are parallel.
    rng = random.Random(seed)
    picked = sorted(rng.sample(range(len(rows)), n))
    return [rows[i]["text"] for i in picked], [rows[i]["label_text"] for i in picked]


def rss_mb(pid: int) -> float | None:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False)
    return int(out.stdout.strip()) / 1024 if out.stdout.strip() else None


def percentiles(values: list[float]) -> dict[str, float]:
    p50, p90, p95 = np.percentile(values, [50, 90, 95])
    return {"p50": p50, "p90": p90, "p95": p95, "mean": float(np.mean(values))}


def checkpoint_revision(repo_id: str) -> str | None:
    """Commit hash the local cache resolved ``main`` to for this checkpoint."""
    from huggingface_hub.constants import HF_HUB_CACHE

    ref = pathlib.Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def hardware() -> dict[str, str]:
    def sysctl(key: str) -> str:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False)
        return out.stdout.strip()

    return {
        "machine": platform.machine(),
        "cpu": sysctl("machdep.cpu.brand_string"),
        "memory_gb": f"{int(sysctl('hw.memsize') or 0) / 2**30:.0f}",
        "os": platform.platform(),
    }


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "laya", "gliner2", "numpy", "transformers", "dashai-system-one-plugin"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "n/a"
    return out


def predict_rows(model, texts: list[str]) -> tuple[np.ndarray, list[float]]:
    """One prediction per call, as DashAI's wrapper does, timing each call."""
    rows, times = [], []
    for text in texts:
        t0 = time.perf_counter()
        rows.append(model.predict([text])[0])
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(rows), times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data", required=True, help="path to a <lang>.json.gz split")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--permute-n", type=int, default=100)
    ap.add_argument("--throughput-n", type=int, default=36)
    ap.add_argument("--kev-port", type=int, default=18008, help="dedicated port, never a shared server")
    a = ap.parse_args()

    texts, labels = load_samples(a.data, a.n, a.seed)
    cls = MODELS[a.model]
    extra = {"endpoint_url": f"http://127.0.0.1:{a.kev_port}"} if a.model == "kev" else {}
    model = cls(instruction=INSTRUCTION, batch_size=1, device=a.device, **extra)

    t0 = time.perf_counter()
    model.train(texts, labels)  # zero-shot: registers labels and loads the backend
    load_s = time.perf_counter() - t0
    if a.model == "kev" and model.backend._manager is None:
        # Timings of a server we did not start would include someone else's load.
        raise SystemExit(f"A Kev server was already listening on port {a.kev_port}.")
    classes = model.class_names
    y = np.array([classes.index(label) for label in labels])

    probs, times = predict_rows(model, texts)
    cold_ms, warm = times[0], times[1:]

    # Position bias: same samples with the label list reversed.
    k = min(a.permute_n, len(texts))
    original = list(classes)
    model.class_names = original[::-1]
    model.label_to_idx = {c: i for i, c in enumerate(model.class_names)}
    rev_probs, _ = predict_rows(model, texts[:k])
    # Map reversed columns back to the original class order.
    rev_probs = rev_probs[:, ::-1]
    model.class_names = original
    model.label_to_idx = {c: i for i, c in enumerate(original)}
    flipped = float(np.mean(rev_probs.argmax(axis=1) != probs[:k].argmax(axis=1)))

    # batch_size sweep: the wrapper loops per sample, so this documents that it is flat.
    throughput = {}
    for bs in (1, 8, 32):
        model.batch_size = bs
        t0 = time.perf_counter()
        model.predict(texts[: a.throughput_n])
        throughput[str(bs)] = a.throughput_n / (time.perf_counter() - t0)

    one_hot = bool(np.all(np.isin(probs, (0.0, 1.0))))  # hard labels only (ADR-002)
    result = {
        "model": a.model, "device": a.device, "lang": a.lang, "n": len(texts), "seed": a.seed,
        "checkpoint": model.model_name, "checkpoint_revision": checkpoint_revision(model.model_name),
        "num_classes": len(classes),
        "accuracy": accuracy(y, probs), "macro_f1": macro_f1(y, probs),
        "probabilities_available": not one_hot,
        "load_s": load_s, "cold_first_inference_ms": cold_ms,
        "warm_ms": percentiles(warm), "warm_calls": len(warm),
        "throughput_samples_per_s": throughput,
        "position_flip_rate": flipped, "position_n": k,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "hardware": hardware(), "versions": versions(),
    }
    rng = np.random.default_rng(a.seed)
    boots = [accuracy(y[i], probs[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
    result["accuracy_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    if not one_hot:
        result.update(
            log_loss=log_loss(y, probs), brier=brier_score(y, probs),
            ece=expected_calibration_error(y, probs),
        )
    if a.device == "mps" and a.model != "kev":
        import torch

        result["mps_driver_mb"] = torch.mps.driver_allocated_memory() / 2**20
    if a.model == "kev" and model.backend._manager is not None:
        result["kev_server_rss_mb_end"] = rss_mb(model.backend._manager._process.pid)
        model.backend.unload()
    print(json.dumps(result))


if __name__ == "__main__":
    sys.exit(main())
