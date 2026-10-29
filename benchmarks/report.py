#!/usr/bin/env python3
"""Render benchmarks/results/*.json as a Markdown report."""

import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ORDER = ["laya", "laya-multilingual", "gliner", "gliner-multilingual"]


def fmt(value: float | None, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def load(results_dir: pathlib.Path) -> list[dict]:
    runs = [json.loads(p.read_text()) for p in results_dir.glob("*.json")]
    return sorted(runs, key=lambda r: (ORDER.index(r["model"]), r["lang"], r["device"]))


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def label(run: dict) -> list[str]:
    return [f"`{run['model']}`", run["lang"], run["device"]]


def render(runs: list[dict]) -> str:
    first = runs[0]
    out = ["# Resultados de Benchmark", ""]
    out.append(
        "Generado por `benchmarks/run_benchmark.py` y `benchmarks/report.py` siguiendo "
        "`docs/benchmark-protocol.md`. Los JSON crudos están en `benchmarks/results/`."
    )
    out += ["", "## Condiciones", ""]
    hw, ver = first["hardware"], first["versions"]
    out.append(f"- Hardware: {hw['cpu']} ({hw['machine']}), {hw['memory_gb']} GB de RAM, {hw['os']}.")
    out.append(
        f"- Python {ver['python']}, torch {ver['torch']}, laya {ver['laya']}, "
        f"gliner2 {ver['gliner2']}, transformers {ver['transformers']}, numpy {ver['numpy']}."
    )
    out.append(
        f"- Dataset: MASSIVE (`mteb/amazon_massive_scenario`, CC-BY-4.0), split `test`, "
        f"{first['n']} muestras por idioma, semilla {first['seed']}, {first['num_classes']} "
        "clases (los mismos ids en inglés y español). Las etiquetas se pasan como nombres "
        "de clase sin descripciones, como lo hace DashAI."
    )
    out.append("- Los modelos solo-inglés no se evalúan sobre español.")
    revs = {}
    for r in runs:
        revs[r["checkpoint"]] = r["checkpoint_revision"]
    out.append("- Checkpoints (commit en el Hub): " + ", ".join(f"`{c}`@`{(v or 'n/a')[:12]}`" for c, v in revs.items()) + ".")

    out += ["", "## Eficacia de decisión", ""]
    rows = [
        label(r) + [f"{r['accuracy']:.3f} [{r['accuracy_ci95'][0]:.3f}, {r['accuracy_ci95'][1]:.3f}]", fmt(r["macro_f1"])]
        for r in runs
    ]
    out.append(table(["Modelo", "Idioma", "Dispositivo", "Accuracy [IC95% bootstrap]", "Macro-F1"], rows))

    out += ["", "## Calibración", ""]
    out.append("Solo para modelos con distribución probabilística real (ADR-002). `n/a` = solo etiquetas duras.")
    out.append("")
    rows = [
        label(r) + [fmt(r.get("log_loss")), fmt(r.get("brier")), fmt(r.get("ece"))]
        for r in runs
        if r["device"] == "cpu"
    ]
    out.append(table(["Modelo", "Idioma", "Dispositivo", "Log-Loss", "Brier", "ECE (10 bins)"], rows))

    out += ["", "## Latencia", ""]
    out.append(
        "Una predicción por llamada a `predict()`. Fría = primera inferencia tras la carga; "
        "caliente = llamadas siguientes (percentiles sobre `warm_calls`). Carga = `train()`, que carga el backend."
    )
    out.append("")
    rows = [
        label(r) + [
            fmt(r["load_s"], ".1f"), fmt(r["cold_first_inference_ms"], ".0f"),
            fmt(r["warm_ms"]["p50"], ".0f"), fmt(r["warm_ms"]["p90"], ".0f"),
            fmt(r["warm_ms"]["p95"], ".0f"), str(r["warm_calls"]),
        ]
        for r in runs
    ]
    out.append(table(["Modelo", "Idioma", "Dispositivo", "Carga (s)", "Fría (ms)", "p50 (ms)", "p90 (ms)", "p95 (ms)", "Llamadas"], rows))

    out += ["", "## Throughput y memoria", ""]
    out.append(
        "`batch_size` no cambia el throughput: `DashAISystemOneClassifier.predict` recorre las muestras "
        "una a una dentro de cada lote. La memoria es el RSS pico del proceso del benchmark. En Apple Silicon la memoria de GPU es "
        "unificada y el RSS puede subestimarla; `Driver MPS` es lo que PyTorch reservó en la GPU."
    )
    out.append("")
    rows = [
        label(r) + [
            fmt(r["throughput_samples_per_s"]["1"], ".1f"), fmt(r["throughput_samples_per_s"]["8"], ".1f"),
            fmt(r["throughput_samples_per_s"]["32"], ".1f"), fmt(r["peak_rss_mb"], ".0f"),
            fmt(r.get("mps_driver_mb"), ".0f"),
        ]
        for r in runs
    ]
    out.append(table(["Modelo", "Idioma", "Dispositivo", "muestras/s (bs=1)", "(bs=8)", "(bs=32)", "RSS pico (MB)", "Driver MPS (MB)"], rows))

    out += ["", "## Sesgo de posición", ""]
    out.append(
        "Fracción de predicciones que cambian al invertir el orden de la lista de etiquetas "
        "(sobre las primeras muestras). Un modelo invariante daría 0."
    )
    out.append("")
    rows = [label(r) + [f"{r['position_flip_rate']:.1%}", str(r["position_n"])] for r in runs]
    out.append(table(["Modelo", "Idioma", "Dispositivo", "Cambian", "Muestras"], rows))
    out += ["", "## Límites de estos resultados", ""]
    out += [
        ("- Una sola máquina (ver Condiciones), una sola ejecución y 300 muestras por idioma: los intervalos "
        "de Accuracy son anchos (~±5 puntos); diferencias menores que eso no son concluyentes."),
        ("- Es clasificación *zero-shot* con 18 nombres de clase sueltos (`play`, `qa`, `general`…) sin "
        "descripciones; varias clases son ambiguas entre sí. Las cifras sirven para comparar modelos en "
        "igualdad de condiciones, no como precisión esperada en otro dominio."),
        ("- No se descarta que los datos de entrenamiento de algún modelo se solapen con MASSIVE; "
        "no se ha comprobado contra los datos de entrenamiento de Laya ni de GLiNER."),
        "- La exactitud es idéntica en CPU y MPS (mismo argmax); lo que cambia es la latencia.",
        ("- La latencia incluye el coste del wrapper de DashAI (`predict` con una muestra)."),
        ("- Durante la ejecución había un servidor de DashAI abierto en la misma máquina y, en algún momento, "
        "se usó su interfaz; no se aisló la máquina, así que parte de la variación de latencia puede ser ruido."),
        ("- Sesgo de posición: se invierte el orden de las etiquetas y se cuenta cuántas predicciones cambian; "
        "no se prueban otras permutaciones."),
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, default=HERE / "results")
    ap.add_argument("--out", type=pathlib.Path, default=HERE.parent / "docs" / "benchmark-results.md")
    args = ap.parse_args()
    runs = load(args.results)
    if not runs:
        raise SystemExit(f"No results in {args.results}")
    args.out.write_text(render(runs))
    print(f"wrote {args.out} from {len(runs)} runs")


if __name__ == "__main__":
    main()
