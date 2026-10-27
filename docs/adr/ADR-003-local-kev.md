# ADR-003: Aislamiento Local Estricto para Kev (v0.2)

## Estado
Aceptado

## Fecha
2026-10-27

## Contexto
El modelo `jaredpalmer/kev-0.8b` se basa en un adaptador LoRA y un cabezal especializado de decisiones entrenado sobre el modelo base `Qwen/Qwen3.5-0.8B-Base`.
Cargar únicamente el LoRA mediante la librería `transformers` estándar de Hugging Face no reproduce adecuadamente el sistema de decisión de Kev. El runtime oficial de Kev está diseñado para correr como un proceso que expone un endpoint HTTP de inferencia estructurada.

El requisito innegociable número 2 y 7 del proyecto establece:
- Prohibición estricta de llamadas a APIs remotas o servicios cloud propietarios.
- Inferencia 100% local en la máquina del usuario (CPU, CUDA, MPS).

## Decisión
1. **Loopback Estricto:**
   - La comunicación con el backend Kev en v0.2 se realizará exclusivamente a través de la interfaz de loopback `http://127.0.0.1:<port>` o sockets UNIX locales.
   - Cualquier URL que apunte a una dirección que no sea `127.0.0.1` o `localhost` será rechazada de inmediato con una excepción `SecurityError`.
2. **Ciclo de Vida y Proceso:**
   - El plugin gestionará el ciclo de vida del subproceso local de Kev (arranque, verificación de salud con health-check en `/health`, y terminación limpia mediante `SIGTERM`/`SIGKILL` sin dejar procesos huérfanos).
   - Se implementará un mecanismo de timeout configurable para evitar bloqueos indefinidos si el runtime entra en deadlock o experimenta OOM.
3. **Desacoplamiento de v0.1:**
   - Para no retrasar el lanzamiento del MVP (Laya y GLiNER), el soporte de Kev queda aislado en su propio módulo `backends/kev_backend.py` y `models/kev_text_classifier.py` con una dependencia opcional `[kev]`.

## Consecuencias
- Seguridad garantizada: no hay riesgo de fuga de datos ni telemetría externa.
- Estabilidad de v0.1 protegida sin bloquear la innovación en modelos más complejos.
