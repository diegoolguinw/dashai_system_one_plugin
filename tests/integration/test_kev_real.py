"""Integration test against a real kev.serve process (needs KEV_PYTHON and KEV_REPO_DIR)."""

import os

import pytest

from dashai_system_one_plugin.backends.kev_backend import KevBackend
from dashai_system_one_plugin.core.contracts import DecisionSpec, QuestionSpec

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("KEV_PYTHON"), reason="KEV_PYTHON not set"),
]


def test_kev_real_choice_noul_score_and_clean_shutdown():
    backend = KevBackend()
    backend.load(startup_timeout=1500)  # kev_python / kev_repo_dir come from the environment
    process = backend._manager._process
    assert process is not None
    try:
        spec = DecisionSpec(
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
                ),
                "urgent": QuestionSpec(type="noul", instructions="Does this need a reply today?"),
                "sev": QuestionSpec(
                    type="score",
                    instructions="How severe is the issue?",
                    criteria=["minor", "moderate", "serious"],
                ),
            },
        )
        result = backend.predict(spec)
    finally:
        backend.unload()

    team = result.answers["team"]
    assert team.value == "billing"
    assert sum(team.probabilities.values()) == pytest.approx(1.0, abs=1e-2)
    assert result.answers["urgent"].kind == "noul"
    assert 0.0 <= result.answers["sev"].value <= 2.0
    assert process.poll() is not None  # supervised process is gone
