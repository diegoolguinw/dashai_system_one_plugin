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

Todos con licencia Apache-2.0 (ver `core/checkpoint_registry.py`).

### 2.1 Laya
Router neuronal que responde varias preguntas tipadas en una sola pasada. Devuelve una distribución de
probabilidad por pregunta, **sin calibrar**: en el benchmark está claramente sobreconfiado.
Latencia p50 medida: 56–215 ms por muestra según modelo y dispositivo (MPS ≈ 1,4–1,6× más rápido que CPU).

### 2.2 GLiNER 2.5 Decide
Clasificación zero-shot sobre etiquetas abiertas. Solo entrega la **etiqueta ganadora**; el plugin la
convierte en one-hot y marca `calibrated=False` (ADR-002). Latencia p50 medida: 48–253 ms por muestra (MPS ≈ 1,8–2× más rápido que CPU).

## 3. Qué esperar de la calidad

Zero-shot, con solo los nombres de clase, sobre MASSIVE (18 escenarios, 300 muestras): la exactitud ronda
el 44–63 % según modelo e idioma, y entre el 12 % y el 23 % de las predicciones cambian al invertir el orden
de la lista de etiquetas. Son modelos pequeños para decisiones rápidas, no clasificadores afinados a tu
dominio: **mide sobre una muestra etiquetada tuya antes de automatizar decisiones**. Detalles y límites en
[benchmark-results.md](benchmark-results.md).
