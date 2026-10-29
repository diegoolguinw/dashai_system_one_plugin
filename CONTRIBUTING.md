# Contribuir

Gracias por tu interés. Este proyecto sigue las reglas de [AGENTS.md](AGENTS.md); lo esencial:

1. **Solo modelos de pesos abiertos** con licencia permisiva comprobada (Apache-2.0, MIT).
2. **Inferencia estrictamente local.** Ninguna llamada a APIs externas durante la inferencia; Kev solo habla con
   `127.0.0.1`. Ten cuidado con los clientes de nube que traen algunas librerías (p. ej. `GLiNER2API`).
3. **Honestidad matemática.** No sintetices probabilidades que el modelo no produce; si un backend solo da
   etiquetas, `probabilities_available = False` (ADR-002).
4. **Contratos congelados.** Los tipos de `src/dashai_system_one_plugin/core/contracts.py` no se cambian sin un
   nuevo ADR en `docs/adr/`.
5. Trabaja en una rama propia y abre un pull request; nada de push directo a `main`.

## Entorno

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest                    # suite unitaria; no descarga modelos
```

Los tests de integración usan checkpoints reales y necesitan los extras y la caché de Hugging Face:

```bash
pip install -e ".[laya,gliner]"
pytest -m integration
KEV_PYTHON=/ruta/kev/.venv/bin/python KEV_REPO_DIR=/ruta/kev pytest -m integration tests/integration/test_kev_real.py
```

## Pull requests

- Añade o actualiza tests; los backends se prueban con servidores/objetos falsos y, cuando hace falta, con un
  test `integration`.
- No inventes APIs de terceros: comprueba el formato de salida real de la librería (los backends de este
  repo se reescribieron precisamente por haberse escrito contra APIs supuestas).
- Actualiza `CHANGELOG.md` y la documentación afectada.
- Si cambias un benchmark, regenera `docs/benchmark-results.md` con `benchmarks/report.py` e indica el hardware.
