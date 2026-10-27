# dashai-system-one-plugin

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Plugin oficial de modelos abiertos **System One** (decisiones neuronales rápidas y tipadas) para [DashAI](https://github.com/DashAISoftware/dashAI).

Integra modelos de pesos abiertos para inferencia 100% local en CPU, GPU (CUDA) y Apple Silicon (MPS):
- **Laya**: `convaiinnovations/laya` y `convaiinnovations/laya-multilingual`
- **GLiNER 2.5 Decide**: `fastino/GLiNER2.5-Decide` y `fastino/GLiNER2.5-multi-Decide`
- **Kev 0.8B (v0.2)**: `jaredpalmer/kev-0.8b`

---

## Características Principales

1. **Inferencia 100% Local y Offline:** Sin dependencias de APIs propietarias ni telemetría externa. Tras descargar los pesos, opera completamente sin conexión.
2. **Contrato Tipado Unificado:** Soporte estructurado para decisiones discretas (`choice`), puntuación (`score`) y detección de nulos/rechazo (`noul`).
3. **Calibración Honesta:** Las distribuciones de probabilidad solo se exponen cuando el modelo y backend las calculan nativamente.
4. **Integración con DashAI:** Aparece como modelos seleccionables en `TextClassificationTask` de la UI web de DashAI con validación de esquemas y bilingüe (EN/ES).

---

## Instalación

```bash
# Instalación base (contratos y stubs ligeros)
pip install dashai-system-one-plugin

# Con soporte de Laya
pip install dashai-system-one-plugin[laya]

# Con soporte de GLiNER
pip install dashai-system-one-plugin[gliner]

# Paquete completo para DashAI
pip install dashai-system-one-plugin[all]
```

---

## Uso Rápido en Python

```python
from dashai_system_one_plugin.core.contracts import DecisionSpec, QuestionSpec
from dashai_system_one_plugin.backends.fake_backend import FakeBackend

# Definir una decisión
spec = DecisionSpec(
    state="No puedo acceder a mi cuenta con mi contraseña habitual.",
    questions={
        "ticket_category": QuestionSpec(
            type="choice",
            instructions="Clasifica el ticket de soporte.",
            criteria=["acceso_y_login", "facturacion", "soporte_tecnico", "otro"],
        )
    },
)

# Ejecutar inferencia local
backend = FakeBackend()
backend.load("fake-checkpoint")
result = backend.predict(spec)

answer = result.answers["ticket_category"]
print(f"Decisión: {answer.value}")
print(f"Probabilidades: {answer.probabilities}")
```

---

## Uso en DashAI

Una vez instalado el plugin en el mismo entorno que DashAI:
1. Inicia DashAI (`dashai run`).
2. Sube un dataset tabular con una columna de texto (ej. consulta del usuario) y una columna categórica (ej. intención o categoría).
3. Selecciona la tarea **Text Classification**.
4. En el catálogo de modelos aparecerán:
   - `LayaTextClassifier`
   - `LayaMultilingualTextClassifier`
   - `GLiNERDecideTextClassifier`
   - `GLiNERMultiDecideTextClassifier`
5. Configura el dispositivo (`auto`, `cpu`, `cuda`) y tamaño de lote.
6. Ejecuta el experimento y compara métricas con el resto de modelos de DashAI.

---

## Licencia

Distribuido bajo la licencia [MIT](LICENSE).
