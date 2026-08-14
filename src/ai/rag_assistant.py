"""
RAG resolution assistant. Retrieves the most similar past tickets from
the local vector index and prompts a local LLM (via Ollama) to draft a
response grounded in those retrieved resolutions, returned as a
validated structured object rather than free text so a serving layer
can consume it safely. Citations the model claims are cross-checked
against what was actually retrieved — an LLM citing a ticket it wasn't
shown is a hallucination, not a real citation, and is reported as such
rather than trusted. Any URL in the generated reply is stripped before
it's returned, whether copied from a retrieved example or fabricated —
neither can be verified safe, so neither is sent to a customer. A reply
claiming an action was already completed (a refund issued, a DM already
sent) for the new customer is flagged and forced to human escalation
rather than silently edited, since rewriting a natural-language claim
safely is not as reliable as stripping a URL. No external API calls —
embeddings and generation both run locally.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import chromadb
import ollama
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path("data/vector_store")
COLLECTION_NAME = "ticket_resolutions"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "llama3.2:3b"
DEFAULT_TOP_K = 3
URL_PATTERN = re.compile(r"https?://\S+")
COMPLETED_ACTION_PATTERN = re.compile(
    r"\bwe(?:'ve| have) (?:already )?(?:responded|refunded|resolved|fixed|sent)\b"
    r"|\balready (?:responded|refunded|resolved|fixed|sent|contacted)\b"
    r"|\brefund has been (?:issued|processed)\b"
    r"|\baccount has been (?:fixed|resolved)\b"
    r"|\bresponded to you via dm\b",
    re.IGNORECASE,
)


class ResolutionDraft(BaseModel):
    reply: str = Field(description="The drafted customer support reply")
    cited_ticket_ids: list[str] = Field(
        default_factory=list,
        description="Ticket numbers from the examples that informed the reply",
    )
    needs_human_escalation: bool = Field(
        description="True if this ticket is too complex, sensitive, or ambiguous for a templated reply"
    )

_embedding_model: SentenceTransformer | None = None
_collection = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(query_text: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    embedding = get_embedding_model().encode([query_text]).tolist()
    results = get_collection().query(query_embeddings=embedding, n_results=top_k)

    retrieved = []
    for ticket_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append(
            {
                "ticket_id": ticket_id,
                "category": metadata["category"],
                "priority": metadata["priority"],
                "customer_message": document,
                "resolution_text": metadata["resolution_text"],
                "distance": distance,
            }
        )
    return retrieved


def build_prompt(query_text: str, retrieved: list[dict]) -> str:
    examples = "\n\n".join(
        f"Example {i + 1} (ticket {r['ticket_id']}, category: {r['category']}):\n"
        f"Customer: {r['customer_message']}\n"
        f"Support reply: {r['resolution_text']}"
        for i, r in enumerate(retrieved)
    )
    return (
        "You are a customer support agent drafting a reply to a new ticket. "
        "Use only the past examples below as grounding for tone and content — "
        "do not invent policies, refunds, or facts not present in the examples. "
        "The examples belong to other customers: do not copy their links, "
        "order numbers, case numbers, or usernames into your reply — write a "
        "new reply in the same style instead. Do not state that any action "
        "has already been completed for this new customer (a refund issued, "
        "a DM already sent, an account already fixed) — nothing has been "
        "done yet, so only offer to help or ask for more information.\n\n"
        f"{examples}\n\n"
        f"New ticket:\n{query_text}\n\n"
        "Respond with a JSON object with these fields: "
        '"reply" (the drafted reply text), '
        '"cited_ticket_ids" (array of ticket numbers from the examples above '
        "that you drew on), and "
        '"needs_human_escalation" (true if this ticket is too complex, '
        "sensitive, or ambiguous for a templated reply, false otherwise)."
    )


def redact_urls(reply: str) -> tuple[str, bool]:
    redacted = URL_PATTERN.sub("[link removed]", reply)
    return redacted, redacted != reply


def claims_completed_action(reply: str) -> bool:
    return COMPLETED_ACTION_PATTERN.search(reply) is not None


def verify_citations(cited_ticket_ids: list[str], retrieved: list[dict]) -> tuple[list[str], list[str]]:
    retrieved_ids = {r["ticket_id"] for r in retrieved}
    verified = [t for t in cited_ticket_ids if t in retrieved_ids]
    hallucinated = [t for t in cited_ticket_ids if t not in retrieved_ids]
    return verified, hallucinated


def generate_response(query_text: str, top_k: int = DEFAULT_TOP_K) -> dict:
    retrieved = retrieve(query_text, top_k=top_k)
    prompt = build_prompt(query_text, retrieved)

    response = ollama.chat(
        model=GENERATION_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        format=ResolutionDraft.model_json_schema(),
    )
    draft = ResolutionDraft.model_validate_json(response["message"]["content"])
    verified_citations, hallucinated_citations = verify_citations(draft.cited_ticket_ids, retrieved)
    redacted_reply, link_redacted = redact_urls(draft.reply)
    completed_action_claimed = claims_completed_action(redacted_reply)

    return {
        "query": query_text,
        "reply": redacted_reply,
        "cited_ticket_ids": verified_citations,
        "hallucinated_citations": hallucinated_citations,
        "link_redacted": link_redacted,
        "completed_action_claimed": completed_action_claimed,
        "needs_human_escalation": draft.needs_human_escalation or completed_action_claimed,
        "retrieved_examples": retrieved,
        "generation_model": GENERATION_MODEL_NAME,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    query = sys.argv[1] if len(sys.argv) > 1 else "My order never arrived and it's been two weeks."
    result = generate_response(query)

    print(f"Query: {result['query']}\n")
    print(f"Draft reply:\n{result['reply']}\n")
    print(f"Needs human escalation: {result['needs_human_escalation']}")
    print(f"Cited tickets: {result['cited_ticket_ids']}")
    if result["hallucinated_citations"]:
        print(f"Hallucinated citations (dropped): {result['hallucinated_citations']}")
    if result["link_redacted"]:
        print("A URL in the generated reply was redacted before returning it.")
    if result["completed_action_claimed"]:
        print("Reply claims an action was already completed — forced to human escalation.")
    print("\nRetrieved examples:")
    for example in result["retrieved_examples"]:
        print(f"  ticket {example['ticket_id']} ({example['category']}, distance {example['distance']:.3f})")


if __name__ == "__main__":
    main()
