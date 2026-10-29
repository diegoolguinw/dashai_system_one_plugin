# Compatibilidad y Entorno de Ejecución

## 1. Versiones de DashAI Soportadas

- **Versión mínima declarada:** `DashAI >= 0.9.7.post2`. **Probado de punta a punta con DashAI 0.10.0** (UI: dataset → sesión de Text Classification → entrenamiento → métricas → guardar/recargar).
- **Interfaces auditadas:**
  - `DashAI.back.models.base_model.BaseModel`
  - `DashAI.back.tasks.text_classification_task.TextClassificationTask`
  - `DashAI.back.tasks.classification_task.ClassificationTask`
  - `DashAI.back.core.schema_fields` (`BaseSchema`, `schema_field`, `string_field`, `int_field`, `enum_field`)

## 2. Matriz de Dispositivos de Cómputo

Verificado en Apple M1 (CPU y MPS); CUDA no se ha probado en este proyecto.

| Dispositivo | CPU | Apple Silicon (MPS) | NVIDIA CUDA |
|---|---|---|---|
| Laya | Verificado | Verificado (PyTorch) | Sin probar |
| GLiNER 2.5 | Verificado | Verificado (PyTorch) | Sin probar |

`device="cuda"` o `"mps"` en una máquina sin ese dispositivo cae a CPU en silencio.

## 3. Matriz de Capacidades por Backend

| Backend | Checkpoint HF | Preguntas | Probabilidades nativas | `calibrated` | Licencia pesos |
|---|---|---|---|---|---|
| **Laya** | `convaiinnovations/laya` | `choice` | Sí (softmax del modelo) | `False` (sobreconfiado, ECE 0,27–0,38 en el benchmark) | Apache-2.0 |
| **Laya Multi** | `convaiinnovations/laya-multilingual` | `choice` | Sí | `False` | Apache-2.0 |
| **GLiNER Decide** | `fastino/GLiNER2.5-Decide` | `choice` | No: solo etiqueta (one-hot) | `False` | Apache-2.0 |
| **GLiNER Multi** | `fastino/GLiNER2.5-multi-Decide` | `choice` | No: solo etiqueta (one-hot) | `False` | Apache-2.0 |

El adaptador de DashAI (Text Classification) solo usa `choice`.

## 3.1 Versiones de dependencias probadas

| Paquete | Versión |
|---|---|
| Python | 3.13 |
| `laya` | 0.4.17 |
| `gliner2` | 2.1.0 |
| `torch` | 2.13 |

## 4. Política de Fallback de Dependencias

Para permitir pruebas unitarias y entornos de CI ultrarrápidos (Gate G1):
- `laya` y `gliner2` se importan de forma diferida al cargar el backend, y los símbolos de DashAI tienen un respaldo mínimo, de modo que los contratos y la suite unitaria funcionan sin dependencias binarias pesadas (la CI lo comprueba en un entorno limpio).
- Al intentar ejecutar una inferencia con un backend ausente, se levanta explícitamente `BackendMissingError` indicando el comando de instalación exacto (por ejemplo: `pip install dashai-system-one-plugin[laya]`).
