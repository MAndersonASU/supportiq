"""
Builds the vector index the RAG assistant retrieves from. Embeds a
capped, per-category sample of the knowledge base with a local
sentence-transformers model and stores it in a persistent local Chroma
collection. Embedding the full knowledge base (789,547 entries) takes
about an hour on CPU for limited retrieval benefit, since it is
dominated by one majority category (General Inquiry, 77% of tickets) —
capping per category keeps every category represented for retrieval
while keeping the pipeline fast to run and reproduce.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

DEFAULT_INPUT_PATH = Path("data/processed/knowledge_base.parquet")
DEFAULT_REPORT_PATH = Path("data/processed/vector_index_report.json")
CHROMA_DIR = Path("data/vector_store")
COLLECTION_NAME = "ticket_resolutions"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_PER_CATEGORY = 15_000
ADD_BATCH_SIZE = 5_000
RANDOM_STATE = 42


def sample_knowledge_base(
    df: pd.DataFrame,
    max_per_category: int = MAX_PER_CATEGORY,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    parts = [
        group.sample(n=min(len(group), max_per_category), random_state=random_state)
        for _, group in df.groupby("category")
    ]
    return pd.concat(parts, ignore_index=True)


def build_index(
    input_path: Path = DEFAULT_INPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict:
    df = pd.read_parquet(input_path)
    sampled = sample_knowledge_base(df)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(
        sampled["customer_message"].tolist(), batch_size=64, show_progress_bar=True
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids = [str(t) for t in sampled["ticket_id"]]
    documents = sampled["customer_message"].tolist()
    metadatas = [
        {"category": row.category, "priority": row.priority, "resolution_text": row.resolution_text}
        for row in sampled.itertuples()
    ]

    for start in range(0, len(sampled), ADD_BATCH_SIZE):
        end = start + ADD_BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    report = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "knowledge_base_size": len(df),
        "indexed_size": len(sampled),
        "category_counts": {
            k: int(v) for k, v in sampled["category"].value_counts().items()
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    report = build_index()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
