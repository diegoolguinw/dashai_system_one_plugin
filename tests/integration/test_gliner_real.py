"""Integration tests running real GLiNER2 checkpoints from the local HF cache."""

import pytest

from dashai_system_one_plugin.backends.gliner_backend import GLiNERBackend
from dashai_system_one_plugin.core.contracts import DecisionSpec, QuestionSpec

pytest.importorskip("gliner2")

pytestmark = pytest.mark.integration

CASES = [
    ("fastino/GLiNER2.5-Decide", "I was charged twice for my last invoice."),
    (
        "fastino/GLiNER2.5-multi-Decide",
        "Me cobraron dos veces la suscripción mensual en mi tarjeta.",
    ),
]


def _spec(state: str) -> DecisionSpec:
    return DecisionSpec(
        state=state,
        questions={
            "intent": QuestionSpec(
                type="choice",
                instructions="Identify the user's intent.",
                criteria={
                    "billing": "Payment, charges or invoices",
                    "access": "Login or password problems",
                    "sales": "Buying a product or plan",
                },
            )
        },
    )


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize(("checkpoint", "state"), CASES)
def test_gliner_real_inference(checkpoint, state, device):
    if device == "mps":
        torch = pytest.importorskip("torch")
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

    backend = GLiNERBackend()
    backend.load(checkpoint, device=device)
    try:
        assert backend.device == device
        answer = backend.predict(_spec(state)).answers["intent"]
    finally:
        backend.unload()

    assert answer.value == "billing"
    # Hard label only: one-hot, flagged uncalibrated (ADR-002).
    assert answer.probabilities == {"billing": 1.0, "access": 0.0, "sales": 0.0}
    assert answer.calibrated is False
