# Directrices para Agentes de Desarrollo (Orca Multi-Agent System)

Este documento rige la colaboración entre agentes supervisados y el coordinador dentro de Orca (https://www.onorca.dev/) para el repositorio `dashai-system-one-plugin`.

## 1. Principios Innegociables
1. **Solo Modelos de Pesos Abiertos:** Pesos accesibles y descargables con licencias permisivas comprobadas (Apache-2.0, MIT).
2. **Inferencia Local Estricta:** Ninguna llamada a APIs externas o de terceros. Todo cómputo ocurre en localhost / loopback `127.0.0.1`.
3. **Honestidad Matemática:** Nunca sintetizar ni inventar probabilidades de confianza ausentes. Si un backend solo produce etiquetas duras, `probabilities_available = False`.
4. **Respeto a los Contratos:** Los tipos definidos en `src/dashai_system_one_plugin/core/contracts.py` son la fuente única de verdad. No alterar los contratos congelados sin aprobación explícita y creación de un nuevo ADR.
5. **Aislamiento en Worktrees:** Cada agente trabaja en su propio branch y worktree asignado. Nunca realizar push directo a `main`.

## 2. Roles y Propiedad de Archivos

| ID | Rol | Propiedad Exclusiva |
|---|---|---|
| **A0** | Coordinator / Tech Lead | `AGENTS.md`, `pyproject.toml`, `docs/architecture.md`, `docs/adr/`, integración |
| **A1** | DashAI Integration | `src/dashai_system_one_plugin/models/`, compatibilidad DashAI |
| **A2** | Laya Backend | `src/dashai_system_one_plugin/backends/laya_backend.py`, tests Laya |
| **A3** | GLiNER Backend | `src/dashai_system_one_plugin/backends/gliner_backend.py`, tests GLiNER |
| **A4** | Kev Backend | `src/dashai_system_one_plugin/backends/kev_backend.py`, tests Kev |
| **A5** | Quality & Verification | `tests/contract/`, suites de validación, benchmarks |
| **A6** | Documentation | `README.md`, `docs/local-models.md`, `examples/` |

## 3. Protocolo de Entrega (Worker Done)
Al finalizar una tarea asignada, el agente debe reportar:
- Lista de archivos modificados o creados.
- Comandos ejecutados para pruebas unitarias y evidencia de resultados en verde.
- Limitaciones identificadas o riesgos de compatibilidad.
- Confirmación de ausencia de dependencias ocultas o llamadas de red no autorizadas.
