# ADR-002: Contrato Unificado de Decisiones y Manejo de Probabilidades

## Estado
Aceptado

## Fecha
2026-10-27

## Contexto
Diferentes modelos de System One ofrecen interfaces y salidas heterogéneas:
- **Laya:** Ofrece decisiones de tipo `choice` con distribución de probabilidades bien definidas sobre las opciones provistas (`Router.predict(...)`).
- **GLiNER2.5-Decide:** Extrae o clasifica texto asociando entidades o etiquetas discretas. En ciertas variantes y tareas, solo produce la etiqueta seleccionada sin una distribución probabilística calibrada.
- **Kev:** Expone decisiones tipadas (`choice`, `score`, `noul`) mediante una especificación estructurada en JSON.

DashAI requiere en `ClassificationTask.process_predictions` que la matriz devuelta por `model.predict(x)` tenga forma `(N, K)` para poder realizar `np.argmax(predictions, axis=1)`. Si un modelo solo produce etiquetas discretas sin probabilidades, devolver un vector unidimensional de longitud `N` rompería `process_predictions`.

Adicionalmente, el principio innegociable número 4 del proyecto estipula:
> "Contrato unificado: entradas state + questions; salidas tipadas choice, score y noul solo si el backend las soporta. Nunca simular probabilidades o niveles de confianza ausentes."

## Decisión
1. **Contrato Central Aislado (`contracts.py`):**
   - Definir especificaciones tipadas: `QuestionSpec`, `DecisionSpec`, `DecisionAnswer`, `DecisionResult`.
   - Soporte para tipos: `choice` (selección categórica), `score` (número continuo / score normalizado), `noul` (booleano / nulo / rechazo).
   - Invariantes estrictas:
     - Probabilidades deben pertenecer a `[0.0, 1.0]`.
     - Para `choice` con probabilidades completas, la suma sobre las clases debe ser aproximadamente 1.0 (tolerancia `1e-3`).
     - No permitir valores `NaN` o `Inf`.
     - Si el backend no produce probabilidades, `probabilities` debe ser `None` y `probabilities_available` en las capacidades del backend debe ser `False`.
2. **Compatibilidad con DashAI `(N, K)` sin falsificar confianza:**
   - Si el backend soporta probabilidades nativas (ej. Laya), devolver la matriz de softmax / distribución real `(N, K)` donde la columna $j$ corresponde a la probabilidad de la clase $j$.
   - Si el backend no ofrece probabilidades nativas (ej. variantes discretas de GLiNER):
     - El adaptador construirá una matriz one-hot binaria indicadora de selección categórica `(N, K)` donde la clase seleccionada toma el valor 1.0 y las restantes 0.0.
     - Se marca explícitamente en los metadatos del modelo: `probabilities_available = False` y `is_calibrated = False`.
     - Esto permite que `np.argmax(..., axis=1)` opere correctamente para métricas de etiquetas duras (Accuracy, Macro-F1), mientras que en la documentación se advierte que métricas basadas en log-verosimilitud (LogLoss, Brier Score) no deben interpretarse como calibración de confianza.
3. **Manejo de Etiquetas Desconocidas o Errores de Decisión:**
   - Si el modelo retorna una etiqueta que no figuraba en las clases provistas en `train()`, se mapeará a un fallback explícito o se levantará una excepción descriptiva `InferenceError`, nunca una asignación silenciosa arbitraria.

## Consecuencias
- Se preserva la honestidad matemática y científica exigida por el proyecto.
- Se mantiene 100% de interoperabilidad con la API existente de `DashAI.back.tasks.classification_task.ClassificationTask`.
