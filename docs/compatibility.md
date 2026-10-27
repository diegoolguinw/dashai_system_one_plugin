# Compatibilidad y Entorno de Ejecución

## 1. Versiones de DashAI Soportadas

- **Versión mínima probada:** `DashAI >= 0.9.7.post2`
- **Interfaces auditadas:**
  - `DashAI.back.models.base_model.BaseModel`
  - `DashAI.back.tasks.text_classification_task.TextClassificationTask`
  - `DashAI.back.tasks.classification_task.ClassificationTask`
  - `DashAI.back.core.schema_fields` (`BaseSchema`, `schema_field`, `string_field`, `int_field`, `enum_field`)

## 2. Matriz de Dispositivos de Cómputo

| Dispositivo | CPU | Apple Silicon (MPS) | NVIDIA CUDA |
|---|---|---|---|
| Laya Router | Soportado (Default) | Soportado vía PyTorch | Soportado (`cuda:0`) |
| GLiNER 2.5 | Soportado (Default) | Soportado vía PyTorch | Soportado (`cuda:0`) |
| Kev Local | Soportado | Soportado vía MLX/PyTorch | Soportado |

## 3. Matriz de Capacidades por Backend

| Backend | Checkpoint HF | Tipo de Salida | Probabilidades Nativas | Calibrado | Licencia Pesos |
|---|---|---|---|---|---|
| **Laya** | `convaiinnovations/laya` | `choice` | Sí | Sí | Apache-2.0 / MIT |
| **Laya Multi** | `convaiinnovations/laya-multilingual` | `choice` | Sí | Sí | Apache-2.0 / MIT |
| **GLiNER Decide** | `fastino/GLiNER2.5-Decide` | `choice`, `score` | Según variante (one-hot fallback) | Evaluado en benchmark | Apache-2.0 |
| **GLiNER Multi** | `fastino/GLiNER2.5-multi-Decide` | `choice`, `score` | Según variante (one-hot fallback) | Evaluado en benchmark | Apache-2.0 |
| **Kev** (v0.2) | `jaredpalmer/kev-0.8b` | `choice`, `score`, `noul` | Sí | Sí | Apache-2.0 |

## 4. Política de Fallback de Dependencias

Para permitir pruebas unitarias y entornos de CI ultrarrápidos (Gate G1):
- Si `torch`, `dashAI`, `laya` o `gliner2` no están instalados, el paquete provee stubs y fallbacks de desarrollo en `core/` para que los contratos y estructuras de datos puedan ser validados sin dependencias binarias pesadas.
- Al intentar ejecutar una inferencia con un backend ausente, se levanta explícitamente `BackendMissingError` indicando el comando de instalación exacto (por ejemplo: `pip install dashai-system-one-plugin[laya]`).
