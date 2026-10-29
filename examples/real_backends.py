#!/usr/bin/env python3
"""Run the same decision on the real local backends (Laya and GLiNER).

Requires `pip install "dashai-system-one-plugin[laya,gliner]"`; the first run downloads the weights.
Set HF_HUB_OFFLINE=1 afterwards to guarantee that nothing touches the network.
"""

from dashai_system_one_plugin.backends import GLiNERBackend, LayaBackend
from dashai_system_one_plugin.core.contracts import DecisionSpec, QuestionSpec

SPEC = DecisionSpec(
    state="I was charged twice for order 1182. Please refund one of the charges.",
    questions={
        "team": QuestionSpec(
            type="choice",
            instructions="Which team should handle this?",
            criteria={
                "billing": "Charges and refunds",
                "shipping": "Deliveries",
                "returns": "Exchanges",
            },
        )
    },
)

BACKENDS = [
    ("Laya", LayaBackend(), "convaiinnovations/laya"),
    ("GLiNER", GLiNERBackend(), "fastino/GLiNER2.5-Decide"),
]


def main() -> None:
    for name, backend, checkpoint in BACKENDS:
        backend.load(checkpoint, device="auto")
        try:
            answer = backend.predict(SPEC).answers["team"]
        finally:
            backend.unload()
        print(f"{name:7s} -> {answer.value}  calibrated={answer.calibrated}")
        print(f"        probabilities={answer.probabilities}")


if __name__ == "__main__":
    main()
