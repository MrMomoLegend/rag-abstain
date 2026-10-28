"""Outcome classification (DESIGN section 6).

RuleClassifier (`rule-fyp`) is a port of `classify()` in rag-assistant-demo's `rag/classify.py`,
itself copied from the FYP's `eval/common.classify`: same markers, same continuation phrases, same
logic. The only additions are configurable extra markers and the system's own refusal text.

RuleV2Classifier (`rule-v2`) runs `rule-fyp` first and, when that says `answered`, also looks for
a request for clarification or a correction of the question's premise (DESIGN section 6.1).
"""

import re
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


# Generic phrasings of "which one do you mean?", written from how chat models usually ask,
# not fitted to any question set. Matched case-insensitively after curly quotes are straightened.
CLARIFICATION_PATTERNS: tuple[str, ...] = (
    r"\b(could|can|would) you (please )?(clarify|specify|elaborate)\b",
    r"\bplease (clarify|specify)\b",
    r"\b(could|can|would) you (please )?(provide|give|share) (me )?(some |a bit )?"
    r"(more|additional|further) (context|details|information)\b",
    r"\bplease (provide|give|share) (me )?(some )?(more|additional|further) "
    r"(context|details|information)\b",
    r"\b(i|i'd|i would) need (some )?(more|additional|further) (context|details|information)\b",
    r"\b(are you|you are|you're) referring to\b",
    r"\bwhat do you mean by\b",
    r"\bdo you mean\b",
    r"\b(it is|it's) (not clear|unclear) (which|what|who)\b",
    r"\b(do|did) you have in mind\b",
)

# Generic phrasings of "the question assumes something false". A bare "X, not Y" is left out on
# purpose: it is too common in ordinary answers.
PREMISE_CORRECTION_PATTERNS: tuple[str, ...] = (
    r"\b(the|this|that|your) (question|premise|assumption|claim|statement) (is|seems|appears) "
    r"(to be )?(incorrect|wrong|mistaken|false|inaccurate|based on)\b",
    r"\b(is|are|was|were|does|do|did) (actually|in fact)\b[^.?!]{0,60}\bnot\b",
    r"\bcontrary to (the|your|what|popular)\b",
    r"\bmisconception\b",
    r"\bthere (is|was|are|were) no (record|evidence|indication)\b",
    r"\b(that|this|it) is not (correct|accurate|true)\b",
    r"\bthat's not (correct|accurate|true|right)\b",
    r"\bit is not true that\b",
    r"\b(there may be|there might be|there seems to be|there is) (a |some )?"
    r"(confusion|mix-up|misunderstanding|mistake)\b",
)

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


class RuleV2Classifier(RuleClassifier):
    """`rule-fyp` plus clarification requests and premise corrections.

    When `rule-fyp` labels an answer `answered`: a clarification request is `declined`, or
    `hedged` when an answer follows it (the same length and continuation test `rule-fyp` applies
    after a refusal); a premise correction is `declined`, since the explanation that usually
    follows it is part of the correction. A clarification pattern counts only inside a question
    (its sentence ends in "?") or a please-imperative (its sentence starts with "please"), so a
    statement such as "The paper you're referring to is ..." stays `answered`.
    """

    def __init__(
        self,
        extra_markers: Iterable[str] = (),
        refusal_text: str | None = None,
        clarification: Iterable[str] = CLARIFICATION_PATTERNS,
        premise_correction: Iterable[str] = PREMISE_CORRECTION_PATTERNS,
    ) -> None:
        """Build `rule-fyp` with the same markers, then compile the two pattern lists."""
        super().__init__(extra_markers=extra_markers, refusal_text=refusal_text)
        self.clarification = tuple(clarification)
        self.premise_correction = tuple(premise_correction)
        self._clarification = [re.compile(p, re.IGNORECASE) for p in self.clarification]
        self._premise = [re.compile(p, re.IGNORECASE) for p in self.premise_correction]

    def classify(self, answer: str) -> Label:
        """Return 'answered', 'declined' or 'hedged' for one answer text."""
        label = super().classify(answer)
        if label != "answered":
            return label
        text = answer.strip().translate(_QUOTES)
        rest = self._clarification_rest(text)
        if rest is not None:
            window = rest.lower()[:CONTINUATION_WINDOW]
            if len(rest) > HEDGE_REST_CHARS or any(c in window for c in self.continuations):
                return "hedged"
            return "declined"
        if any(p.search(text) for p in self._premise):
            return "declined"
        return "answered"

    def _clarification_rest(self, text: str) -> str | None:
        """Text after the first sentence that asks for clarification, or None if none does.

        A sentence runs from the previous ".", "?", "!" or line break to the next one.
        """
        found = []
        for pattern in self._clarification:
            for match in pattern.finditer(text):
                start = max(text.rfind(c, 0, match.start()) for c in ".?!\n") + 1
                end = re.search(r"[.?!\n]", text[match.end() :])
                stop = match.end() + end.start() if end else len(text)
                is_question = stop < len(text) and text[stop] == "?"
                if is_question or text[start:stop].lstrip().lower().startswith("please"):
                    found.append(stop)
        if not found:
            return None
        return text[min(found) + 1 :].strip()

    def config(self) -> dict[str, Any]:
        """Describe this classifier for the run manifest."""
        return {
            **super().config(),
            "name": "rule-v2",
            "clarification_patterns": list(self.clarification),
            "premise_correction_patterns": list(self.premise_correction),
        }


CLASSIFIERS: dict[str, type[RuleClassifier]] = {
    "rule-fyp": RuleClassifier,
    "rule-v2": RuleV2Classifier,
}


def make_classifier(
    name: str, extra_markers: Iterable[str] = (), refusal_text: str | None = None
) -> RuleClassifier:
    """Build a classifier by name (`rule-fyp` or `rule-v2`) with a system's markers.

    `rule-fyp` is recorded in manifests as "rules", its name before `rule-v2` existed.
    """
    if name not in CLASSIFIERS:
        raise ValueError(f"unknown classifier {name!r}; choose from {', '.join(CLASSIFIERS)}")
    return CLASSIFIERS[name](extra_markers=extra_markers, refusal_text=refusal_text)


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
