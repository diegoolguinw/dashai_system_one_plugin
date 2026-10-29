# Resultados de Benchmark

Generado por `benchmarks/run_benchmark.py` y `benchmarks/report.py` siguiendo `docs/benchmark-protocol.md`. Los JSON crudos están en `benchmarks/results/`.

## Condiciones

- Hardware: Apple M1 (arm64), 8 GB de RAM, macOS-26.6.2-arm64-arm-64bit-Mach-O.
- Python 3.13.13, torch 2.13.0, laya 0.4.17, gliner2 2.1.0, transformers 5.15.1, numpy 2.4.6.
- Dataset: MASSIVE (`mteb/amazon_massive_scenario`, CC-BY-4.0), split `test`, 300 muestras por idioma, semilla 0, 18 clases (los mismos ids en inglés y español). Las etiquetas se pasan como nombres de clase sin descripciones, como lo hace DashAI.
- Los modelos solo-inglés no se evalúan sobre español.
- Checkpoints (commit en el Hub): `convaiinnovations/laya`@`7b928d828b7b`, `convaiinnovations/laya-multilingual`@`1720e3e3357c`, `fastino/GLiNER2.5-Decide`@`9d1bfb848cd1`, `fastino/GLiNER2.5-multi-Decide`@`e173bca1f0c4`.

## Eficacia de decisión

| Modelo | Idioma | Dispositivo | Accuracy [IC95% bootstrap] | Macro-F1 |
|---|---|---|---|---|
| `laya` | en | cpu | 0.507 [0.453, 0.560] | 0.505 |
| `laya` | en | mps | 0.507 [0.453, 0.560] | 0.505 |
| `laya-multilingual` | en | cpu | 0.530 [0.473, 0.587] | 0.535 |
| `laya-multilingual` | en | mps | 0.530 [0.473, 0.587] | 0.535 |
| `laya-multilingual` | es | cpu | 0.443 [0.393, 0.497] | 0.437 |
| `laya-multilingual` | es | mps | 0.443 [0.393, 0.497] | 0.437 |
| `gliner` | en | cpu | 0.633 [0.580, 0.683] | 0.629 |
| `gliner` | en | mps | 0.633 [0.580, 0.683] | 0.629 |
| `gliner-multilingual` | en | cpu | 0.530 [0.473, 0.583] | 0.539 |
| `gliner-multilingual` | en | mps | 0.530 [0.473, 0.583] | 0.539 |
| `gliner-multilingual` | es | cpu | 0.440 [0.380, 0.493] | 0.446 |
| `gliner-multilingual` | es | mps | 0.440 [0.380, 0.493] | 0.446 |

## Calibración

Solo para modelos con distribución probabilística real (ADR-002). `n/a` = solo etiquetas duras.

| Modelo | Idioma | Dispositivo | Log-Loss | Brier | ECE (10 bins) |
|---|---|---|---|---|---|
| `laya` | en | cpu | 5.606 | 0.809 | 0.384 |
| `laya-multilingual` | en | cpu | 2.998 | 0.717 | 0.274 |
| `laya-multilingual` | es | cpu | 3.556 | 0.850 | 0.348 |
| `gliner` | en | cpu | n/a | n/a | n/a |
| `gliner-multilingual` | en | cpu | n/a | n/a | n/a |
| `gliner-multilingual` | es | cpu | n/a | n/a | n/a |

## Latencia

Una predicción por llamada a `predict()`. Fría = primera inferencia tras la carga; caliente = llamadas siguientes (percentiles sobre `warm_calls`). Carga = `train()`, que carga el backend.

| Modelo | Idioma | Dispositivo | Carga (s) | Fría (ms) | p50 (ms) | p90 (ms) | p95 (ms) | Llamadas |
|---|---|---|---|---|---|---|---|---|
| `laya` | en | cpu | 5.4 | 613 | 215 | 316 | 431 | 299 |
| `laya` | en | mps | 5.7 | 2393 | 134 | 144 | 150 | 299 |
| `laya-multilingual` | en | cpu | 5.9 | 177 | 79 | 82 | 86 | 299 |
| `laya-multilingual` | en | mps | 6.8 | 1067 | 57 | 60 | 63 | 299 |
| `laya-multilingual` | es | cpu | 6.1 | 177 | 84 | 96 | 120 | 299 |
| `laya-multilingual` | es | mps | 6.5 | 367 | 56 | 60 | 62 | 299 |
| `gliner` | en | cpu | 17.2 | 696 | 253 | 349 | 457 | 299 |
| `gliner` | en | mps | 18.2 | 1970 | 128 | 138 | 143 | 299 |
| `gliner-multilingual` | en | cpu | 14.6 | 179 | 86 | 90 | 92 | 299 |
| `gliner-multilingual` | en | mps | 13.9 | 924 | 48 | 61 | 71 | 299 |
| `gliner-multilingual` | es | cpu | 13.9 | 218 | 90 | 97 | 99 | 299 |
| `gliner-multilingual` | es | mps | 14.6 | 601 | 50 | 61 | 68 | 299 |

## Throughput y memoria

`batch_size` no cambia el throughput: `DashAISystemOneClassifier.predict` recorre las muestras una a una dentro de cada lote. La memoria es el RSS pico del proceso del benchmark. En Apple Silicon la memoria de GPU es unificada y el RSS puede subestimarla; `Driver MPS` es lo que PyTorch reservó en la GPU.

| Modelo | Idioma | Dispositivo | muestras/s (bs=1) | (bs=8) | (bs=32) | RSS pico (MB) | Driver MPS (MB) |
|---|---|---|---|---|---|---|---|
| `laya` | en | cpu | 4.7 | 4.4 | 4.4 | 1609 | n/a |
| `laya` | en | mps | 7.3 | 7.3 | 7.4 | 1741 | 2081 |
| `laya-multilingual` | en | cpu | 12.7 | 12.4 | 12.8 | 2217 | n/a |
| `laya-multilingual` | en | mps | 17.9 | 17.9 | 18.1 | 1257 | 1287 |
| `laya-multilingual` | es | cpu | 12.1 | 12.1 | 12.1 | 1684 | n/a |
| `laya-multilingual` | es | mps | 17.5 | 17.6 | 17.6 | 1786 | 1287 |
| `gliner` | en | cpu | 3.6 | 3.7 | 3.6 | 2034 | n/a |
| `gliner` | en | mps | 7.6 | 7.6 | 7.6 | 1658 | 2089 |
| `gliner-multilingual` | en | cpu | 10.4 | 10.1 | 9.7 | 2489 | n/a |
| `gliner-multilingual` | en | mps | 20.7 | 20.9 | 21.0 | 2264 | 1183 |
| `gliner-multilingual` | es | cpu | 10.5 | 11.3 | 11.0 | 2331 | n/a |
| `gliner-multilingual` | es | mps | 20.0 | 20.0 | 19.3 | 1864 | 1159 |

## Sesgo de posición

Fracción de predicciones que cambian al invertir el orden de la lista de etiquetas (sobre las primeras muestras). Un modelo invariante daría 0.

| Modelo | Idioma | Dispositivo | Cambian | Muestras |
|---|---|---|---|---|
| `laya` | en | cpu | 21.0% | 100 |
| `laya` | en | mps | 21.0% | 100 |
| `laya-multilingual` | en | cpu | 17.0% | 100 |
| `laya-multilingual` | en | mps | 17.0% | 100 |
| `laya-multilingual` | es | cpu | 23.0% | 100 |
| `laya-multilingual` | es | mps | 23.0% | 100 |
| `gliner` | en | cpu | 12.0% | 100 |
| `gliner` | en | mps | 12.0% | 100 |
| `gliner-multilingual` | en | cpu | 18.0% | 100 |
| `gliner-multilingual` | en | mps | 18.0% | 100 |
| `gliner-multilingual` | es | cpu | 18.0% | 100 |
| `gliner-multilingual` | es | mps | 18.0% | 100 |

## Límites de estos resultados

- Una sola máquina (ver Condiciones), una sola ejecución y 300 muestras por idioma: los intervalos de Accuracy son anchos (~±5 puntos); diferencias menores que eso no son concluyentes.
- Es clasificación *zero-shot* con 18 nombres de clase sueltos (`play`, `qa`, `general`…) sin descripciones; varias clases son ambiguas entre sí. Las cifras sirven para comparar modelos en igualdad de condiciones, no como precisión esperada en otro dominio.
- No se descarta que los datos de entrenamiento de algún modelo se solapen con MASSIVE; no se ha comprobado contra los datos de entrenamiento de Laya ni de GLiNER.
- La exactitud es idéntica en CPU y MPS (mismo argmax); lo que cambia es la latencia.
- La latencia incluye el coste del wrapper de DashAI (`predict` con una muestra).
- Durante la ejecución había un servidor de DashAI abierto en la misma máquina y, en algún momento, se usó su interfaz; no se aisló la máquina, así que parte de la variación de latencia puede ser ruido.
- Sesgo de posición: se invierte el orden de las etiquetas y se cuenta cuántas predicciones cambian; no se prueban otras permutaciones.
