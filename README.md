# dashai-system-one-plugin

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Plugin de modelos abiertos **System One** (decisiones neuronales rápidas y tipadas) para
[DashAI](https://github.com/DashAISoftware/dashAI). Añade a la tarea **Text Classification** de DashAI
modelos que clasifican texto *zero-shot* a partir de los nombres de las clases, con inferencia 100 % local
en CPU y Apple Silicon (MPS).

| Modelo | Checkpoint | En DashAI |
|---|---|---|
| Laya (EN / multilingüe) | `convaiinnovations/laya`, `convaiinnovations/laya-multilingual` | `LayaTextClassifier`, `LayaMultilingualTextClassifier` |
| GLiNER 2.5 Decide (EN / multilingüe) | `fastino/GLiNER2.5-Decide`, `fastino/GLiNER2.5-multi-Decide` | `GLiNERDecideTextClassifier`, `GLiNERMultiDecideTextClassifier` |

## Características

1. **Inferencia local.** Ninguna llamada a APIs externas durante la inferencia. Internet se usa una sola vez
   para descargar paquetes y pesos; con `HF_HUB_OFFLINE=1` el plugin funciona sin conexión (verificado con la
   red denegada a nivel de proceso).
2. **Contrato tipado unificado.** `choice`, `score` y `noul` (`core/contracts.py`). Laya y GLiNER implementan
   `choice`; `score` y `noul` quedan en el contrato para otros backends.
3. **Honestidad sobre las probabilidades.** Solo se exponen probabilidades que el modelo calcula. GLiNER solo
   da la etiqueta, así que entrega one-hot con `calibrated=False`; las de Laya se marcan `calibrated=False`
   porque están sobreconfiadas. Ver [ADR-002](docs/adr/ADR-002-unified-contract.md).
4. **Integración con DashAI.** Aparecen en el catálogo de modelos de Text Classification con formulario
   bilingüe (ES/EN). Probado de punta a punta con DashAI 0.10.0.

## Resultados (resumen)

Zero-shot sobre MASSIVE (18 escenarios, 300 muestras por idioma, Apple M1 8 GB). Detalle, intervalos de
confianza, calibración, memoria y límites en [docs/benchmark-results.md](docs/benchmark-results.md).

| Modelo | Idioma | Accuracy | Latencia p50 (CPU / MPS) | ECE |
|---|---|---|---|---|
| Laya | en | 0,51 | 215 / 134 ms | 0,38 |
| Laya multilingüe | en / es | 0,53 / 0,44 | 79 / 57 ms | 0,27 / 0,35 |
| GLiNER Decide | en | 0,63 | 253 / 128 ms | n/a (solo etiqueta) |
| GLiNER multi Decide | en / es | 0,53 / 0,44 | 86 / 48 ms | n/a (solo etiqueta) |

Son modelos pequeños para decisiones rápidas, no clasificadores afinados a tu dominio. Entre el 12 % y el 23 %
de las predicciones cambian al invertir el orden de las etiquetas. **Mide sobre una muestra etiquetada tuya
antes de automatizar decisiones.**

## Instalación

Requiere Python ≥ 3.10. Desde el código fuente:

```bash
git clone <url-del-repositorio> && cd dashai-system-one-plugin
pip install -e ".[all]"          # DashAI + Laya + GLiNER
# o solo lo que necesites:
pip install -e ".[laya]"         # Laya
pip install -e ".[gliner]"       # GLiNER
pip install -e .                 # contratos y backend de prueba, sin modelos
```

Instálalo **en el mismo entorno que DashAI**.

## Uso en DashAI

1. Inicia DashAI con el intérprete de ese entorno: `python -m DashAI --no-browser`.
2. Sube un dataset con una columna de texto y una categórica (hay uno de ejemplo en
   [`examples/routing_sample.csv`](examples/routing_sample.csv)).
3. Crea una sesión de **Text Classification** y elige uno de los modelos de la tabla.
4. Ejecuta y compara métricas con el resto de modelos de DashAI.

Guía paso a paso con capturas: [docs/tutorial.md](docs/tutorial.md).

## Uso en Python

```python
from dashai_system_one_plugin.backends import LayaBackend
from dashai_system_one_plugin.core.contracts import DecisionSpec, QuestionSpec

spec = DecisionSpec(
    state="I was charged twice for order 1182. Please refund one of the charges.",
    questions={
        "team": QuestionSpec(
            type="choice",
            instructions="Which team should handle this?",
            criteria={"billing": "Charges and refunds", "shipping": "Deliveries", "returns": "Exchanges"},
        )
    },
)

backend = LayaBackend()
backend.load("convaiinnovations/laya", device="auto")   # descarga los pesos la primera vez
answer = backend.predict(spec).answers["team"]
print(answer.value, answer.probabilities, answer.calibrated)
backend.unload()
```

Más ejemplos: [`examples/real_backends.py`](examples/real_backends.py) (Laya y GLiNER) y
[`examples/local_inference.py`](examples/local_inference.py) (sin dependencias, con `FakeBackend`).

## Documentación

- [Tutorial](docs/tutorial.md) · [Solución de problemas](docs/troubleshooting.md)
- [Modelos locales y modo sin conexión](docs/local-models.md) · [Compatibilidad](docs/compatibility.md)
- [Arquitectura](docs/architecture.md) · [Decisiones (ADR)](docs/adr/)
- [Protocolo de benchmark](docs/benchmark-protocol.md) · [Resultados](docs/benchmark-results.md)

## Desarrollo

```bash
pip install -e ".[dev]"
ruff check .
pytest                      # suite unitaria (no descarga modelos)
pytest -m integration       # con checkpoints reales en la caché de Hugging Face
python benchmarks/prepare_data.py && python benchmarks/run_benchmark.py && python benchmarks/report.py
```

Las reglas de colaboración de los agentes del proyecto están en [AGENTS.md](AGENTS.md); ver también
[CONTRIBUTING.md](CONTRIBUTING.md).

## Limitaciones conocidas

- `batch_size` no acelera: el adaptador procesa las muestras una a una.
- GLiNER no devuelve probabilidades, solo la etiqueta.
- Solo se ha probado en macOS (Apple M1, CPU y MPS); CUDA no se ha verificado.
- Los modelos en español rinden claramente peor que en inglés en el benchmark.

## Licencia

El código se distribuye bajo la licencia [MIT](LICENSE). Los pesos de los modelos tienen su propia licencia
(todos Apache-2.0 al momento de escribir esto).
