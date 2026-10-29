# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).

## [0.1.0] - Sin publicar

Primera versión del plugin.

### Añadido
- Modelos de Text Classification para DashAI: `LayaTextClassifier`, `LayaMultilingualTextClassifier`,
  `GLiNERDecideTextClassifier`, `GLiNERMultiDecideTextClassifier` y `KevTextClassifier` (experimental).
- Contrato tipado unificado (`DecisionSpec`, `QuestionSpec`, `DecisionAnswer`, `DecisionResult`) con
  preguntas `choice`, `score` y `noul`.
- Backends locales: Laya (`laya.load`), GLiNER 2.5 (`AutoExtractor`) y Kev (`kev.serve` por HTTP en
  `127.0.0.1`), más un `FakeBackend` para pruebas.
- Supervisor del subproceso de Kev (`utils/kev_process_manager.py`): bind a loopback, grupo de procesos
  propio, apagado SIGTERM → SIGKILL.
- Métricas (`metrics/`): Accuracy, Macro-F1, Log-Loss, Brier y ECE.
- Benchmarks reproducibles (`benchmarks/`) y resultados en `docs/benchmark-results.md`.
- Documentación: tutorial con capturas, solución de problemas, guía de modelos locales, ADR 001–003.
- Tests unitarios, de contrato y de integración con checkpoints reales (`pytest -m integration`).

### Notas
- Las probabilidades de Laya se marcan `calibrated=False` (sobreconfiado); GLiNER solo entrega etiqueta
  (one-hot, `calibrated=False`); Kev usa `calibrated=None`.
- `batch_size` se acepta por compatibilidad con DashAI pero no acelera la inferencia.
- Verificado en Apple M1 (CPU y MPS) con DashAI 0.10.0; CUDA sin probar.
