# Guía de Modelos Locales y Caché Offline

## 1. Descarga Explícita y Gestión de Caché

Todos los modelos se descargan y almacenan en la caché estándar de Hugging Face (`~/.cache/huggingface/hub/`) o en el directorio especificado por `HF_HOME`.

### Reutilización sin conexión
Una vez que los pesos han sido descargados en el sistema, el plugin activa automáticamente el modo `local_files_only=True` o valida la presencia de los artefactos en disco, garantizando:
- Cero llamadas a servidores remotos durante la inferencia.
- Privacidad total de los datos de entrada.
- Capacidad de operar en entornos con air-gap o sin conectividad de red.

## 2. Checkpoints Soportados

### 2.1 Laya (`convaiinnovations/laya`)
- **Descripción:** Router ultrarrápido entrenado específicamente para toma de decisiones directas y enrutamiento conversacional / de intención.
- **Tamaño:** ~100MB - 350MB según cuantización.
- **Memoria RAM requerida:** < 1 GB.
- **Latencia esperada:** < 15ms en CPU moderna / < 3ms en GPU.

### 2.2 GLiNER 2.5 Decide (`fastino/GLiNER2.5-Decide`)
- **Descripción:** Arquitectura basada en representación bidireccional optimizada para extracción y clasificación zero-shot con conjuntos abiertos de etiquetas.
- **Tamaño:** ~400MB - 800MB.
- **Memoria RAM requerida:** 1.5 - 2 GB.
- **Latencia esperada:** 25ms - 60ms en CPU / 5ms - 10ms en GPU.

### 2.3 Kev 0.8B (`jaredpalmer/kev-0.8b`)
- **Descripción:** Modelo de 0.8B parámetros basado en Qwen con cabezal de decisión tipada para `choice`, `score` y `noul`.
- **Tamaño:** ~1.6 GB.
- **Memoria RAM requerida:** 3 - 4 GB.
- **Modo de ejecución:** Proceso local exclusivo en `127.0.0.1`.
