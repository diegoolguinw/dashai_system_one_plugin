# Solución de problemas

Esta guía recoge fallos reales encontrados al integrar los modelos. Para los números
de rendimiento ver [benchmark-results.md](benchmark-results.md).

## Hardware recomendado

Medido en un Apple M1 con 8 GB de RAM (ver `benchmark-results.md`):

| Modelo | Disco (pesos en caché) | RAM pico del proceso | Notas |
|---|---|---|---|
| Laya (EN / multilingüe) | 807 MB / 647 MB | ~1,3–2,2 GB | MPS reduce la latencia p50 a ~55–135 ms |
| GLiNER 2.5 Decide (EN / multilingüe) | ~1,8 GB / ~1,1 GB | ~1,7–2,5 GB | MPS ≈ 2× el throughput de CPU |

- Con 8 GB de RAM conviene cargar **un modelo a la vez**. Cada modelo cargado en DashAI mantiene
  sus pesos en memoria mientras exista la instancia.
- No hay requisito de GPU: todos los modelos corren en CPU, más lento.

## DashAI

### El formulario del modelo queda en blanco
Era un fallo de `BaseSystemOneSchema` (campos declarados como `x: Any = schema_field(...)` en lugar
de anotaciones). Está corregido; si reaparece, comprueba el esquema que sirve la API:

```bash
curl -sL http://127.0.0.1:8000/api/v1/component/LayaTextClassifier/ | python -m json.tool | head -40
```

Cada propiedad debe traer `type` y `placeholder`.

### El modelo no aparece en el catálogo
1. Instala el plugin **en el mismo entorno que DashAI**: `uv pip install --python <venv>/bin/python -e .`
2. Reinicia DashAI. En el log debe aparecer `Se cargaron 4 plugins instalados`.
3. No uses `uv run` para arrancar DashAI tras instalar paquetes a mano: `uv run` sincroniza el entorno y
   puede desinstalar `laya` o `gliner2`. Arranca con `<venv>/bin/python -m DashAI`.

### Al reiniciar DashAI quedan procesos
`DashAI` no siempre termina su consumidor Huey (`huey_consumer`) ni los hijos de `multiprocessing`;
quedan huérfanos (PPID 1) y pueden convivir con el servidor nuevo. Antes de reiniciar:

```bash
pkill -TERM -f "m DashAI --no-browser"; pkill -TERM -f huey_consumer
```

### `batch_size` no cambia la velocidad
Es así por diseño actual: `DashAISystemOneClassifier.predict` recorre las muestras una a una dentro de
cada lote. El parámetro se conserva por compatibilidad con la UI.

## Laya

- **`BackendMissingError: laya`** — falta el paquete: `pip install "dashai-system-one-plugin[laya]"`.
- **`RuntimeWarning: this checkpoint ships invalid temperatures …`** — es un aviso de Laya sobre sus
  checkpoints. Las probabilidades son puntuaciones del modelo, no probabilidades calibradas; el plugin las
  marca con `calibrated=False`. En el benchmark Laya está claramente sobreconfiado (ECE 0,27–0,38).
- **Primera ejecución lenta** — descarga los pesos (unos 800 MB) desde Hugging Face. Después funciona sin red.

## GLiNER

- **`BackendMissingError: gliner2`** — `pip install "dashai-system-one-plugin[gliner]"`.
- **`'ExtractorConfig' object has no attribute 'max_width'`** — se usó `GLiNER2.from_pretrained` en lugar de
  `AutoExtractor.from_pretrained`. El backend ya usa `AutoExtractor`.
- **No hay probabilidades** — GLiNER devuelve solo la etiqueta. El plugin entrega una matriz one-hot con
  `calibrated=False` y `probabilities_available=False`; no se calculan Log-Loss, Brier ni ECE.
- **Preguntas que no son de intención** (p. ej. detectar el idioma `es`/`en`) pueden dar siempre la misma
  respuesta: el modelo no está entrenado para eso.
- `gliner2` incluye un cliente de API en la nube (`GLiNER2API`). El plugin **no** lo usa.

## Sin conexión

Tras la primera descarga, define `HF_HUB_OFFLINE=1` para garantizar que nada intenta conectarse:

```bash
HF_HUB_OFFLINE=1 <venv>/bin/python -m DashAI --no-browser
```

Si falta algún checkpoint en la caché, el error será de carga del modelo (`Failed to load … checkpoint`);
ejecuta una vez con red para descargarlo.

## Dispositivos

- `device="auto"` elige CUDA, luego MPS (Apple Silicon) y, si no, CPU.
- Pedir `cuda` o `mps` en una máquina sin ese dispositivo cae a CPU en silencio. Comprueba el dispositivo
  real con `backend.device`.

## Probabilidades y métricas

| Backend | `probabilities_available` | `calibrated` | Métricas válidas |
|---|---|---|---|
| Laya | sí | `False` | Accuracy, F1; Log-Loss/Brier/ECE solo como diagnóstico |
| GLiNER | no (one-hot) | `False` | Accuracy, F1 |

Con salidas one-hot, Log-Loss y Brier **no** miden calibración (ADR-002).
