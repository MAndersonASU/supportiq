"""
RAG resolution assistant. Retrieves the most similar past tickets from
the local vector index and prompts a local LLM (via Ollama) to draft a
response grounded in those retrieved resolutions, citing which past
ticket(s) informed it. No external API calls — embeddings and
generation both run locally.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path("data/vector_store")
COLLECTION_NAME = "ticket_resolutions"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "llama3.2:3b"
DEFAULT_TOP_K = 3

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
        "new reply in the same style instead. "
        "Reference the ticket number(s) of the example(s) you drew on.\n\n"
        f"{examples}\n\n"
        f"New ticket:\n{query_text}\n\n"
        "Draft reply:"
    )


def generate_response(query_text: str, top_k: int = DEFAULT_TOP_K) -> dict:
    retrieved = retrieve(query_text, top_k=top_k)
    prompt = build_prompt(query_text, retrieved)

    response = ollama.chat(
        model=GENERATION_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "query": query_text,
        "draft_response": response["message"]["content"],
        "retrieved_examples": retrieved,
        "generation_model": GENERATION_MODEL_NAME,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    query = sys.argv[1] if len(sys.argv) > 1 else "My order never arrived and it's been two weeks."
    result = generate_response(query)

    print(f"Query: {result['query']}\n")
    print(f"Draft reply:\n{result['draft_response']}\n")
    print("Retrieved examples:")
    for example in result["retrieved_examples"]:
        print(f"  ticket {example['ticket_id']} ({example['category']}, distance {example['distance']:.3f})")


if __name__ == "__main__":
    main()
