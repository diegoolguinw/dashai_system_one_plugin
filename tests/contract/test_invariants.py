"""Contract invariance verification tests across backends."""

import pytest

from dashai_system_one_plugin.backends.fake_backend import FakeBackend
from dashai_system_one_plugin.core.contracts import DecisionSpec, QuestionSpec


def test_contract_invariance_fake_backend():
    backend = FakeBackend(simulate_probabilities=True)
    backend.load("test-checkpoint")

    spec = DecisionSpec(
        state="Permutation test state",
        questions={
            "q_choice": QuestionSpec(
                type="choice",
                instructions="Choose",
                criteria=["alpha", "beta", "gamma"],
            ),
            "q_score": QuestionSpec(type="score", instructions="Rate"),
            "q_noul": QuestionSpec(type="noul", instructions="Flag"),
        },
    )

    res = backend.predict(spec)

    # Invariant 1: all questions answered
    assert set(res.answers.keys()) == {"q_choice", "q_score", "q_noul"}

    # Invariant 2: choice answer is one of the valid options
    choice_ans = res.answers["q_choice"]
    assert choice_ans.value in ["alpha", "beta", "gamma"]

    # Invariant 3: probabilities exist, sum to ~1.0, and contain the chosen value
    assert choice_ans.probabilities is not None
    assert sum(choice_ans.probabilities.values()) == pytest.approx(1.0, abs=1e-2)
    assert choice_ans.value in choice_ans.probabilities

    # Invariant 4: score answer in valid range without NaN/Inf
    score_ans = res.answers["q_score"]
    assert isinstance(score_ans.value, (int, float))
    assert 0.0 <= score_ans.value <= 1.0

    # Invariant 5: noul answer is boolean
    noul_ans = res.answers["q_noul"]
    assert isinstance(noul_ans.value, bool)
