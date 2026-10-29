# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).

## [0.1.0] - Sin publicar

Primera versión del plugin.

### Añadido
- Modelos de Text Classification para DashAI: `LayaTextClassifier`, `LayaMultilingualTextClassifier`,
  `GLiNERDecideTextClassifier` y `GLiNERMultiDecideTextClassifier`.
- Contrato tipado unificado (`DecisionSpec`, `QuestionSpec`, `DecisionAnswer`, `DecisionResult`) con
  preguntas `choice`, `score` y `noul`.
- Backends locales: Laya (`laya.load`) y GLiNER 2.5 (`AutoExtractor`), más un `FakeBackend` para pruebas.
- Métricas (`metrics/`): Accuracy, Macro-F1, Log-Loss, Brier y ECE.
- Benchmarks reproducibles (`benchmarks/`) y resultados en `docs/benchmark-results.md`.
- Documentación: tutorial con capturas, solución de problemas, guía de modelos locales, ADR 001–003.
- Tests unitarios, de contrato y de integración con checkpoints reales (`pytest -m integration`).

### Notas
- Las probabilidades de Laya se marcan `calibrated=False` (sobreconfiado); GLiNER solo entrega etiqueta
  (one-hot, `calibrated=False`).
- `batch_size` se acepta por compatibilidad con DashAI pero no acelera la inferencia.
- Kev (`jaredpalmer/kev-0.8b`) se implementó y verificó, pero se retiró antes de publicar: exige instalar
  aparte su runtime (Python ≥ 3.12, `torch<2.9`, un clon de ~1 GB), es solo inglés y no encaja bien en la UI de
  DashAI. El código sigue en el historial de git (commit `feat(kev)`).
- Verificado en Apple M1 (CPU y MPS) con DashAI 0.10.0; CUDA sin probar.
