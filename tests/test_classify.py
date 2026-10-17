import pytest

from rag_abstain.classify import (
    DEFAULT_CONTINUATIONS,
    DEFAULT_REFUSAL_MARKERS,
    RuleClassifier,
    resolve_outcome,
)
from rag_abstain.models import SystemResponse

REFUSAL = "The provided documents do not contain this information."

# From rag-assistant-demo tests/test_classify.py (the FYP's eval/common.classify cases).
HEDGE = (
    "The provided documents do not contain this information. However, based on the "
    "context, it appears that tf-idf weighting is calculated as the product of two statistics."
)

RULES = RuleClassifier(refusal_text=REFUSAL)


# --- ported from rag-assistant-demo/tests/test_classify.py ---------------------------------
# The demo's classify() takes the whole pipeline dict; here the gate case goes through
# resolve_outcome and the text cases through RuleClassifier.classify. "refused" is "declined".


def test_demo_case_gate_decline_ignores_answer_text() -> None:
    response = SystemResponse(answer="anything", decline_stage="gate")
    assert resolve_outcome(response, RULES) == ("declined", "gate")


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("", "declined"),
        (REFUSAL, "declined"),
        (HEDGE, "hedged"),
        ("GPT-4 was released on March 14, 2023.", "answered"),
        ("I don't know.", "declined"),
    ],
)
def test_demo_cases(answer: str, expected: str) -> None:
    assert RULES.classify(answer) == expected


# --- faithfulness of the port ----------------------------------------------------------------


def test_default_markers_and_continuations_match_the_demo() -> None:
    assert len(DEFAULT_REFUSAL_MARKERS) == 12
    assert DEFAULT_REFUSAL_MARKERS[0] == "do not contain"
    assert DEFAULT_REFUSAL_MARKERS[-1] == "i do not know"
    assert "but " in DEFAULT_CONTINUATIONS and "but" not in DEFAULT_CONTINUATIONS
    assert len(DEFAULT_CONTINUATIONS) == 11


def test_markers_are_case_insensitive() -> None:
    assert RULES.classify("I DO NOT KNOW.") == "declined"


def test_long_remainder_without_continuation_is_hedged() -> None:
    answer = "The documents do not contain this. " + "x" * 101
    assert RULES.classify(answer) == "hedged"


def test_remainder_of_exactly_100_chars_is_declined() -> None:
    answer = "The documents do not contain this. " + "x" * 100
    assert RULES.classify(answer) == "declined"


def test_continuation_beyond_first_80_chars_is_not_seen() -> None:
    answer = "The documents do not contain this. " + "x" * 80 + " however"
    assert RULES.classify(answer) == "declined"


def test_bare_but_needs_trailing_space() -> None:
    assert RULES.classify("The documents do not contain this. Butter.") == "declined"
    assert RULES.classify("The documents do not contain this. But it is X.") == "hedged"


def test_marker_without_following_period_is_declined() -> None:
    assert RULES.classify("Sorry, I cannot answer that") == "declined"


def test_first_marker_in_list_order_wins_not_first_in_text() -> None:
    # "insufficient" appears first in the text, but "no information" is earlier in the list,
    # so the remainder is taken after "no information..." (empty) -> declined. Pinned as-is.
    answer = "Insufficient detail. However X is Y. There is no information on Z."
    assert RULES.classify(answer) == "declined"


# --- additions: extra markers and refusal_text -----------------------------------------------


def test_extra_marker_is_detected() -> None:
    answer = "That is outside my knowledge base."
    assert RuleClassifier().classify(answer) == "answered"
    assert RuleClassifier(extra_markers=["outside my knowledge"]).classify(answer) == "declined"


def test_refusal_text_is_a_marker() -> None:
    custom = "Sorry, nothing relevant was found."
    assert RuleClassifier().classify(custom) == "answered"
    assert RuleClassifier(refusal_text=custom).classify(custom) == "declined"


def test_refusal_text_followed_by_answer_is_hedged() -> None:
    custom = "Sorry, nothing relevant was found."
    answer = custom + " However, X is probably Y."
    assert RuleClassifier(refusal_text=custom).classify(answer) == "hedged"


def test_config_lists_markers_without_duplicates() -> None:
    config = RuleClassifier(extra_markers=["Do Not Contain"], refusal_text=REFUSAL).config()
    assert config["name"] == "rules"
    assert config["markers"].count("do not contain") == 1
    assert config["markers"][-1] == REFUSAL.lower()


# --- known limitation (DESIGN 6.1) -----------------------------------------------------------


def test_answer_then_trailing_disclaimer_is_known_limitation() -> None:
    # An answer followed by a trailing disclaimer is read as declined, because the marker is
    # found and nothing follows its sentence. A fix must be a deliberate change to this test.
    answer = (
        "Gradient descent updates parameters by moving against the gradient of the loss. "
        "The documents do not contain the specific formula for the learning rate schedule."
    )
    assert RULES.classify(answer) == "declined"


# --- resolve_outcome (DESIGN 6) --------------------------------------------------------------


def test_error_wins_over_everything() -> None:
    response = SystemResponse(answer=REFUSAL, decline_stage="gate", error="timeout")
    assert resolve_outcome(response, RULES) == ("error", None)


def test_empty_error_string_is_still_an_error() -> None:
    assert resolve_outcome(SystemResponse(error=""), RULES) == ("error", None)


def test_classifier_decline_gets_model_stage() -> None:
    assert resolve_outcome(SystemResponse(answer=REFUSAL), RULES) == ("declined", "model")


def test_classifier_decline_keeps_system_stage() -> None:
    response = SystemResponse(answer=REFUSAL, declined=True, decline_stage="model")
    assert resolve_outcome(response, RULES) == ("declined", "model")


def test_hedge_flagged_declined_by_system_is_still_hedged() -> None:
    # The demo pipeline sets refused=True / model_judgement on hedges; the classifier decides.
    response = SystemResponse(answer=HEDGE, declined=True, decline_stage="model")
    assert resolve_outcome(response, RULES) == ("hedged", None)


def test_answered() -> None:
    response = SystemResponse(answer="GPT-4 was released on March 14, 2023.")
    assert resolve_outcome(response, RULES) == ("answered", None)


@pytest.mark.parametrize("answer", ["", "   \n"])
def test_empty_answer_is_declined(answer: str) -> None:
    assert resolve_outcome(SystemResponse(answer=answer), RULES) == ("declined", "model")


def test_empty_answer_is_not_sent_to_classifier() -> None:
    class Exploding:
        def classify(self, answer: str) -> str:
            raise AssertionError("classifier called")

    assert resolve_outcome(SystemResponse(answer=""), Exploding()) == ("declined", "model")
