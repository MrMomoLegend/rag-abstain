"""A deterministic fake RAG system for tests and the quickstart. No network, no model.

`answer_question` returns the same dict shape as rag-assistant-demo's
`rag.pipeline.answer_question`, so it is read with `response_map: demo`. Known questions have
a canned answer, sources and nearest-first distances. Anything else gets distances of 0.81 and
up, so it is declined at the gate (threshold 0.53, as in the demo). With `use_threshold=False`
the gate is off and every question reaches the "model", which makes a gate-off run possible.
"""

from dataclasses import dataclass
from typing import Any

REFUSAL = "The provided documents do not contain this information."
DISTANCE_THRESHOLD = 0.53
TOP_K = 3


@dataclass(frozen=True)
class Entry:
    """A canned retrieval result and model answer."""

    answer: str
    sources: tuple[str, ...]
    distances: tuple[float, ...]


HEDGE = (
    "The provided documents do not contain this information. However, based on the context, "
    "it appears that tf-idf weighting is calculated as the product of two statistics."
)

KNOWN: dict[str, Entry] = {
    "which researchers is the cyk algorithm named after?": Entry(
        "The CYK algorithm is named after John Cocke, Daniel Younger and Tadao Kasami.",
        ("CYK algorithm.txt", "CYK algorithm.txt", "Parsing.txt"),
        (0.31, 0.36, 0.49),
    ),
    "when was gpt-4 released?": Entry(
        "GPT-4 was released on March 14, 2023.",
        ("GPT-4.txt", "GPT-4.txt", "Large language model.txt"),
        (0.22, 0.30, 0.45),
    ),
    "what does tf-idf stand for?": Entry(
        "TF-IDF stands for term frequency-inverse document frequency.",
        ("Tf-idf.txt", "Information retrieval.txt", "Tf-idf.txt"),
        (0.27, 0.39, 0.42),
    ),
    "how is tf-idf weighting calculated?": Entry(
        HEDGE,
        ("Tf-idf.txt", "Tf-idf.txt", "Information retrieval.txt"),
        (0.38, 0.41, 0.50),
    ),
    "who introduced the transformer architecture?": Entry(
        "The Transformer architecture was introduced by researchers at OpenAI in 2015.",
        (
            "Transformer (deep learning architecture).txt",
            "Attention (machine learning).txt",
            "GPT-4.txt",
        ),
        (0.29, 0.40, 0.51),
    ),
    "how much did it cost to train gpt-4?": Entry(
        REFUSAL,
        ("GPT-4.txt", "Large language model.txt", "GPT-4.txt"),
        (0.44, 0.50, 0.52),
    ),
    "how many parameters does gpt-4 have?": Entry(
        "GPT-4 has about 1.8 trillion parameters.",
        ("GPT-4.txt", "Large language model.txt", "GPT-4.txt"),
        (0.47, 0.52, 0.55),
    ),
    "what is the capital of brazil?": Entry(
        "The capital of Brazil is Brasilia.",
        ("Natural language processing.txt", "Word embedding.txt", "Parsing.txt"),
        (0.78, 0.80, 0.83),
    ),
    "what is the bleu score used for?": Entry(
        "BLEU is used to evaluate the quality of machine translation output against reference "
        "translations.",
        ("BLEU.txt", "Machine translation.txt", "BLEU.txt"),
        (0.58, 0.61, 0.64),
    ),
}

UNKNOWN = Entry(
    REFUSAL,
    ("Natural language processing.txt", "Word embedding.txt", "Parsing.txt"),
    (0.81, 0.84, 0.86),
)


def _key(question: str) -> str:
    return " ".join(question.lower().split())


def answer_question(
    question: str, use_threshold: bool = True, threshold: float = DISTANCE_THRESHOLD
) -> dict[str, Any]:
    """Answer one question in the demo pipeline's return shape. Same input, same output."""
    entry = KNOWN.get(_key(question), UNKNOWN)
    sources = list(entry.sources)
    distances = list(entry.distances)
    base: dict[str, Any] = {
        "question": question,
        "use_rag": True,
        "top_k": TOP_K,
        "sources": sources,
        "retrieved_chunks": [f"(toy chunk {i} from {s})" for i, s in enumerate(sources, 1)],
        "distances": distances,
        "retrieve_s": 0.01,
    }

    if use_threshold and distances[0] > threshold:
        return {
            **base,
            "answer": REFUSAL,
            "refused": True,
            "refusal_reason": "retrieval_distance",
            "threshold": threshold,
            "generate_s": 0.0,
            "total_s": 0.01,
            "prompt_tokens": None,
            "completion_tokens": None,
            "finish_reason": None,
        }

    # Stands in for the demo's looks_like_refusal(), which also flags hedges as refused.
    refused = "do not contain" in entry.answer.lower()
    return {
        **base,
        "answer": entry.answer,
        "refused": refused,
        "refusal_reason": "model_judgement" if refused else None,
        "threshold": threshold if use_threshold else None,
        "generate_s": 0.8,
        "total_s": 0.81,
        "prompt_tokens": 900,
        "completion_tokens": len(entry.answer.split()),
        "finish_reason": "stop",
    }
