"""
Evaluation harness for the RAG resolution assistant. Runs a fixed set of
test queries through the pipeline and scores each response on
deterministic, local checks: whether cited tickets were actually
retrieved (no hallucinated citations), whether the reply is non-empty
and of reasonable length, and how semantically close the reply is to
the retrieved resolutions it's grounded in (cosine similarity via the
same local embedding model already in the pipeline). No LLM-as-judge —
judging a model's output with the same small local model it came from
is a known-unreliable technique, not a real evaluation signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.ai.rag_assistant import generate_response, get_embedding_model

DEFAULT_REPORT_PATH = Path("data/processed/rag_evaluation_report.json")
MIN_REPLY_LENGTH = 10

TEST_QUERIES = [
    "I was charged twice for my subscription this month, please refund me",
    "My order never arrived and it's been two weeks",
    "The app keeps crashing every time I try to open it after the update",
    "I can't log into my account, it says my password is wrong even though I just reset it",
    "This is the third time I've had this problem and nobody has helped me, I'm furious",
    "What are your customer service hours?",
]


def cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vector_norm = vector / np.linalg.norm(vector)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix_norm @ vector_norm


def score_response(result: dict) -> dict:
    reply = result["reply"]
    retrieved = result["retrieved_examples"]

    reply_embedding = get_embedding_model().encode([reply])[0]
    resolution_embeddings = get_embedding_model().encode(
        [r["resolution_text"] for r in retrieved]
    )
    similarities = cosine_similarity(reply_embedding, resolution_embeddings)

    return {
        "reply_length": len(reply),
        "reply_nonempty": len(reply.strip()) >= MIN_REPLY_LENGTH,
        "num_citations": len(result["cited_ticket_ids"]),
        "num_hallucinated_citations": len(result["hallucinated_citations"]),
        "link_redacted": result["link_redacted"],
        "mention_redacted": result["mention_redacted"],
        "completed_action_claimed": result["completed_action_claimed"],
        "max_groundedness_similarity": float(np.max(similarities)),
        "mean_groundedness_similarity": float(np.mean(similarities)),
        "needs_human_escalation": result["needs_human_escalation"],
    }


def summarize_results(results: list[dict]) -> dict:
    return {
        "num_queries": len(results),
        "hallucination_rate": sum(r["num_hallucinated_citations"] > 0 for r in results) / len(results),
        "link_redaction_rate": sum(r["link_redacted"] for r in results) / len(results),
        "mention_redaction_rate": sum(r["mention_redacted"] for r in results) / len(results),
        "completed_action_claim_rate": sum(r["completed_action_claimed"] for r in results) / len(results),
        "escalation_rate": sum(r["needs_human_escalation"] for r in results) / len(results),
        "mean_groundedness_similarity": float(
            np.mean([r["mean_groundedness_similarity"] for r in results])
        ),
        "empty_or_short_replies": sum(not r["reply_nonempty"] for r in results),
    }


def run(
    queries: list[str] = TEST_QUERIES,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict:
    results = []
    for query in queries:
        response = generate_response(query)
        scores = score_response(response)
        results.append({"query": query, "reply": response["reply"], **scores})

    report = {**summarize_results(results), "results": results}

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    report = run()
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
