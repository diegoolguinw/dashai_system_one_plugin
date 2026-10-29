# Guía de Modelos Locales y Caché Offline

Todo el cómputo ocurre en tu máquina. Internet solo se usa **una vez**, para descargar paquetes y
pesos; después puedes trabajar sin conexión. Las cifras de esta página provienen de
[benchmark-results.md](benchmark-results.md) (Apple M1, 8 GB de RAM).

## 1. Descarga y caché

Los pesos se guardan en la caché estándar de Hugging Face (`~/.cache/huggingface/hub/`, o `HF_HOME`).
La primera carga de cada modelo los descarga; las siguientes los reutilizan.

Para **garantizar** que no se intenta ninguna conexión (p. ej. en un entorno aislado), exporta antes de
iniciar DashAI o tu script:

```bash
export HF_HUB_OFFLINE=1
```

El plugin no activa este modo por sí mismo. Con él, si falta un checkpoint en la caché la carga falla con un
error claro en lugar de intentar descargarlo.

Verificado: los benchmarks y los tests de integración se ejecutaron con `HF_HUB_OFFLINE=1` y con la red
denegada a nivel de proceso (`sandbox-exec` con `(deny network*)`, permitiendo solo `localhost`) y dieron los
mismos resultados.

## 2. Checkpoints soportados

| Modelo | Checkpoint | Disco | RAM pico medida | Idiomas |
|---|---|---|---|---|
| Laya | `convaiinnovations/laya` | 807 MB | ~1,6–1,7 GB | inglés |
| Laya multilingüe | `convaiinnovations/laya-multilingual` | 647 MB | ~1,3–2,2 GB | multilingüe |
| GLiNER 2.5 Decide | `fastino/GLiNER2.5-Decide` | ~1,8 GB | ~1,7–2,0 GB | inglés |
| GLiNER 2.5 multi Decide | `fastino/GLiNER2.5-multi-Decide` | ~1,1 GB | ~1,9–2,5 GB | multilingüe |
| Kev 0.8B | `jaredpalmer/kev-0.8b` + base `Qwen/Qwen3.5-0.8B-Base` | ~63 MB + ~1,6 GB | ~0,4 GB (cliente) + servidor aparte | inglés |

Todos con licencia Apache-2.0 (ver `core/checkpoint_registry.py`).

### 2.1 Laya
Router neuronal que responde varias preguntas tipadas en una sola pasada. Devuelve una distribución de
probabilidad por pregunta, **sin calibrar**: en el benchmark está claramente sobreconfiado.
Latencia p50 medida: 56–215 ms por muestra según modelo y dispositivo (MPS ≈ 1,4–1,6× más rápido que CPU).

### 2.2 GLiNER 2.5 Decide
Clasificación zero-shot sobre etiquetas abiertas. Solo entrega la **etiqueta ganadora**; el plugin la
convierte en one-hot y marca `calibrated=False` (ADR-002). Latencia p50 medida: 48–253 ms por muestra (MPS ≈ 1,8–2× más rápido que CPU).

### 2.3 Kev 0.8B
Modelo de decisión con cabezal de puntero para `choice`, `score` y `noul`. A diferencia de los otros, **corre
en un proceso aparte** (`python -m kev.serve`, repo [jaredpalmer/kev](https://github.com/jaredpalmer/kev)) y el
plugin habla con él por HTTP en `127.0.0.1` (puerto 8008 por defecto). Latencia p50 medida: ~168 ms.

Instalación del runtime (una vez; requiere internet):

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev
uv sync --extra serve        # Python >=3.12,<3.14
```

Uso con arranque automático (el plugin levanta y apaga el servidor):

```python
from dashai_system_one_plugin.backends.kev_backend import KevBackend

backend = KevBackend()  # http://127.0.0.1:8008
backend.load(kev_python="/ruta/kev/.venv/bin/python", kev_repo_dir="/ruta/kev")
# ... backend.predict(spec) ...
backend.unload()        # detiene el proceso
```

O define `KEV_PYTHON` / `KEV_REPO_DIR` en el entorno y llama a `backend.load()` sin argumentos.
La primera carga descarga el modelo base (~4–5 min); con caché tarda ~10 s.

Garantías de aislamiento: el backend rechaza cualquier host que no sea `127.0.0.1`, `localhost` o `::1`
(`SecurityError`), ignora las variables de proxy, no sigue redirecciones y no arranca un servidor si el
puerto ya está ocupado.

## 3. Qué esperar de la calidad

Zero-shot, con solo los nombres de clase, sobre MASSIVE (18 escenarios, 300 muestras): la exactitud ronda
el 44–63 % según modelo e idioma, y entre el 12 % y el 23 % de las predicciones cambian al invertir el orden
de la lista de etiquetas. Son modelos pequeños para decisiones rápidas, no clasificadores afinados a tu
dominio: **mide sobre una muestra etiquetada tuya antes de automatizar decisiones**. Detalles y límites en
[benchmark-results.md](benchmark-results.md).
