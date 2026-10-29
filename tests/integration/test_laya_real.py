"""Integration tests running real Laya checkpoints from the local HF cache."""

import pytest

from dashai_system_one_plugin.backends.laya_backend import LayaBackend
from dashai_system_one_plugin.core.contracts import DecisionSpec, QuestionSpec

pytest.importorskip("laya")

pytestmark = pytest.mark.integration

CASES = [
    ("convaiinnovations/laya", "I was charged twice for my last invoice."),
    (
        "convaiinnovations/laya-multilingual",
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
def test_laya_real_inference(checkpoint, state, device):
    if device == "mps":
        torch = pytest.importorskip("torch")
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

    backend = LayaBackend()
    backend.load(checkpoint, device=device)
    try:
        assert backend.device == device
        answer = backend.predict(_spec(state)).answers["intent"]
    finally:
        backend.unload()

    assert answer.value == "billing"
    assert set(answer.probabilities) == {"billing", "access", "sales"}
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert answer.calibrated is False
