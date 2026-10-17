"""Outcome classification (DESIGN section 6).

RuleClassifier is a port of `classify()` in rag-assistant-demo's `rag/classify.py`, itself copied
from the FYP's `eval/common.classify`: same markers, same continuation phrases, same logic. The
only additions are configurable extra markers and the system's own refusal text.
"""

from collections.abc import Iterable
from typing import Any, Literal, Protocol

from rag_abstain.models import DeclineStage, Outcome, SystemResponse

Label = Literal["answered", "declined", "hedged"]

DEFAULT_REFUSAL_MARKERS: tuple[str, ...] = (
    "do not contain",
    "does not contain",
    "doesn't contain",
    "not in the context",
    "no information",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "not provided in the context",
    "insufficient",
    "i don't know",
    "i do not know",
)

DEFAULT_CONTINUATIONS: tuple[str, ...] = (
    "however",
    "but ",
    "although",
    "according to",
    "based on",
    "that said",
    "nevertheless",
    "it appears",
    "it is mentioned",
    "it can be inferred",
    "from the context",
)

# A remainder after the refusal sentence longer than this is an answer given anyway.
HEDGE_REST_CHARS = 100
# Continuation phrases are looked for only this far into the remainder.
CONTINUATION_WINDOW = 80


class Classifier(Protocol):
    """Anything that labels answer text as answered, declined or hedged."""

    def classify(self, answer: str) -> Label:
        """Label one answer."""
        ...

    def config(self) -> dict[str, Any]:
        """Describe this classifier for the run manifest."""
        ...


class RuleClassifier:
    """Keyword classifier: refusal markers, then a hedge check on the text that follows."""

    def __init__(
        self,
        extra_markers: Iterable[str] = (),
        refusal_text: str | None = None,
        markers: Iterable[str] = DEFAULT_REFUSAL_MARKERS,
        continuations: Iterable[str] = DEFAULT_CONTINUATIONS,
    ) -> None:
        """Build the marker list: `markers`, then `extra_markers`, then `refusal_text`.

        Matching is case-insensitive. Order matters as in the original: the first marker in this
        list that occurs anywhere in the answer is the one used.
        """
        combined = [*markers, *extra_markers]
        if refusal_text:
            combined.append(refusal_text)
        self.markers: tuple[str, ...] = tuple(
            dict.fromkeys(m.strip().lower() for m in combined if m.strip())
        )
        self.continuations: tuple[str, ...] = tuple(c.lower() for c in continuations)

    def classify(self, answer: str) -> Label:
        """Return 'answered', 'declined' or 'hedged' for one answer text."""
        answer = (answer or "").strip()
        if not answer:
            return "declined"

        low = answer.lower()
        marker = next((m for m in self.markers if m in low), None)
        if marker is None:
            return "answered"

        tail = answer[low.index(marker) :]
        rest = tail.split(".", 1)[1].strip() if "." in tail else ""
        window = rest.lower()[:CONTINUATION_WINDOW]
        if len(rest) > HEDGE_REST_CHARS or any(c in window for c in self.continuations):
            return "hedged"
        return "declined"

    def config(self) -> dict[str, Any]:
        """Describe this classifier for the run manifest."""
        return {
            "name": "rules",
            "markers": list(self.markers),
            "continuations": list(self.continuations),
            "hedge_rest_chars": HEDGE_REST_CHARS,
            "continuation_window": CONTINUATION_WINDOW,
        }


def resolve_outcome(
    response: SystemResponse, classifier: Classifier
) -> tuple[Outcome, DeclineStage | None]:
    """Resolve a response to (outcome, decline stage), in the order of DESIGN section 6.

    1. `error` set -> error. 2. Gate decline -> declined at `gate`. 3. Empty answer -> declined.
    4. Otherwise the classifier decides; a classifier decline keeps the system's stage, or gets
    `model` if the system gave none. The system's own `declined` flag never overrides the
    classifier, so a hedge the system flags as declined is still reported as hedged.
    The empty-answer check runs before the classifier so a judge is never sent an empty string;
    the outcome is the same as the order written in DESIGN.
    """
    if response.error is not None:
        return "error", None
    if response.decline_stage == "gate":
        return "declined", "gate"
    if not response.answer.strip():
        return "declined", response.decline_stage or "model"
    label = classifier.classify(response.answer)
    if label == "declined":
        return "declined", response.decline_stage or "model"
    return label, None
