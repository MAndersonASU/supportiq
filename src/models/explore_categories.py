"""
Diagnostic script, not a pipeline stage. Runs TF-IDF + SVD (LSA) + KMeans
over a sample of ticket text, with brand handles and anonymized IDs
stripped, and reports each cluster's top terms. This is the analysis that
grounded the category keyword lists in label_tickets.py: several clusters
here correspond to real issue-type vocabulary (account/password,
order/delivery, service complaints), while roughly half of tickets fall
into one linguistically generic cluster KMeans cannot usefully subdivide
— which is why raw cluster assignment is not used as the category label.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

DEFAULT_INPUT_PATH = Path("data/processed/ticket_features.parquet")
DEFAULT_REPORT_PATH = Path("data/processed/cluster_exploration_report.json")
SAMPLE_SIZE = 15_000
N_CLUSTERS = 12
RANDOM_STATE = 42

ID_PATTERN = re.compile(r"\b\d{4,}\b")
MENTION_PATTERN = re.compile(r"@\w+")
URL_PATTERN = re.compile(r"https?://\S+")
AMP_PATTERN = re.compile(r"&amp;?")


def strip_brand_and_id_tokens(text: str) -> str:
    text = MENTION_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = ID_PATTERN.sub(" ", text)
    text = AMP_PATTERN.sub(" ", text)
    return text


def explore(
    input_path: Path = DEFAULT_INPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict:
    df = pd.read_parquet(input_path, columns=["text", "brand_id"])
    sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    brand_handles = [b.lower() for b in df["brand_id"].dropna().unique()]

    sample["clean_text"] = sample["text"].map(strip_brand_and_id_tokens)

    base_stop = list(TfidfVectorizer(stop_words="english").get_stop_words())
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=base_stop + brand_handles + ["amp", "https"],
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.5,
    )
    tfidf = vectorizer.fit_transform(sample["clean_text"])

    svd = TruncatedSVD(n_components=100, random_state=RANDOM_STATE)
    lsa = Normalizer(copy=False).fit_transform(svd.fit_transform(tfidf))

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=5)
    cluster_labels = kmeans.fit_predict(lsa)

    terms = np.array(vectorizer.get_feature_names_out())
    clusters = []
    for cluster_id in range(N_CLUSTERS):
        mask = cluster_labels == cluster_id
        indices = np.where(mask)[0]
        mean_tfidf = np.asarray(tfidf[indices].mean(axis=0)).ravel()
        top_term_indices = mean_tfidf.argsort()[::-1][:10]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "size": int(mask.sum()),
                "top_terms": terms[top_term_indices].tolist(),
            }
        )

    report = {
        "sample_size": SAMPLE_SIZE,
        "n_clusters": N_CLUSTERS,
        "explained_variance_ratio": float(svd.explained_variance_ratio_.sum()),
        "clusters": clusters,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    report = explore()
    print(json.dumps({k: v for k, v in report.items() if k != "clusters"}, indent=2))
    for cluster in report["clusters"]:
        print(f"cluster {cluster['cluster_id']} (n={cluster['size']}): {cluster['top_terms']}")


if __name__ == "__main__":
    main()
