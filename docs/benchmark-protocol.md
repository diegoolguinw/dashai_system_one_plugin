# Protocolo de Benchmark y Evaluación Comparativa

## 1. Objetivos

Evaluar de forma rigurosa y reproducible el rendimiento, latencia y calibración de los modelos System One locales integrados en DashAI.

## 2. Métricas de Evaluación

1. **Eficacia de Decisión:**
   - **Accuracy**: Exactitud en la elección de la etiqueta correcta.
   - **Macro-F1**: Balance armónico entre precisión y recall por categoría, minimizando el sesgo de clases desbalanceadas.
2. **Calibración de Incertidumbre (solo para modelos con distribución probabilística real):**
   - **Log-Loss / NLL**: Neg-log likelihood de la predicción.
   - **Brier Score**: Error cuadrático de la probabilidad predicha respecto al indicador real.
   - **ECE (Expected Calibration Error)**: Desviación entre precisión observada y confianza predicha en bins uniformes.
   *(Nota: Se excluyen explícitamente estas métricas para variantes de backend que solo ofrecen salidas discretas hard).*
3. **Eficiencia Computacional:**
   - **Latencia Fría (Cold-Start)**: Tiempo transcurrido para la primera inferencia tras carga de memoria.
   - **Latencia Caliente (Warm-Run)**: p50, p90 y p95 calculados sobre un mínimo de 100 llamadas continuas.
   - **Throughput**: Muestras procesadas por segundo (`samples/sec`) en función del tamaño de lote (`batch_size` 1, 8, 32).
   - **Memoria Pico**: Consumo de memoria RAM (RSS) y VRAM (CUDA/MPS) durante la inferencia.

## 3. Condiciones de Reproducibilidad

- Todos los experimentos deben especificar:
  - Commit y hash exacto del checkpoint en Hugging Face Hub.
  - Versión de Python, PyTorch y librerías de backend.
  - Especificaciones de hardware (CPU, arquitectura, modelo de GPU y tamaño de VRAM).
  - Semilla aleatoria fijada para cualquier muestreo o partición.
  - Invarianza al orden: Realizar pruebas permutando el orden de las etiquetas en la lista de opciones para verificar que el modelo no presenta sesgo de posición ("positional bias").
