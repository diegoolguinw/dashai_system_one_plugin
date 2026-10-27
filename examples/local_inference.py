#!/usr/bin/env python3
"""Example script demonstrating local zero-shot decision inference with System One models."""

import pathlib
import sys

# Ensure src/ is on python path when running script directly
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from dashai_system_one_plugin.backends.fake_backend import FakeBackend
from dashai_system_one_plugin.core.contracts import DecisionSpec, QuestionSpec


def main():
    print("=== System One Local Decision Inference Example ===")

    # Create a decision specification
    spec = DecisionSpec(
        state="El sistema muestra un error 500 al procesar el pago de mi factura.",
        questions={
            "ticket_intent": QuestionSpec(
                type="choice",
                instructions="Determina el departamento responsable para resolver este ticket.",
                criteria=[
                    "facturacion_y_cobros",
                    "soporte_infraestructura",
                    "atencion_general",
                ],
            ),
            "urgency_score": QuestionSpec(
                type="score",
                instructions="Evalúa el nivel de urgencia del problema de 0.0 a 1.0.",
            ),
            "requires_human_escalation": QuestionSpec(
                type="noul",
                instructions="Indica True si se requiere intervención de un agente humano.",
            ),
        },
    )

    # Initialize backend (using FakeBackend for zero-dependency demonstration)
    backend = FakeBackend(simulate_probabilities=True)
    backend.load("convaiinnovations/laya-multilingual")

    print(f"\nEvaluating Input: '{spec.state}'\n")
    result = backend.predict(spec)

    for q_name, answer in result.answers.items():
        print(f"Question: [{q_name}] (kind={answer.kind})")
        print(f"  Decision Value: {answer.value}")
        if answer.probabilities:
            print("  Probabilities:")
            for opt, prob in answer.probabilities.items():
                print(f"    - {opt}: {prob:.4f}")
        print()

    print("Success: Local System One inference completed.")


if __name__ == "__main__":
    main()
