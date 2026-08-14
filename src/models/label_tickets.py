"""
Rule-based labeling functions for ticket category and priority. The
dataset has no ground-truth labels and no budget for LLM-based labeling,
so this is weak supervision: keyword labeling functions instead of a
supervised model. Category keywords were calibrated by inspecting an
unsupervised TF-IDF + KMeans clustering pass over ticket text (with brand
handles and anonymized IDs stripped) — several clusters aligned with real
issue-type vocabulary (account/password, order/delivery, service
complaints), which grounded the keyword lists below in the data rather
than guesswork. Raw cluster assignment was not used directly as the
label: roughly half of tickets fell into one linguistically generic
cluster that KMeans could not usefully subdivide.

Priority is scored from ticket text only. It deliberately excludes any
post-open outcome field (e.g. first_response_seconds) — using an outcome
to construct a label that a future model might be trained to predict
would leak the outcome into its own target.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

DEFAULT_INPUT_PATH = Path("data/processed/ticket_features.parquet")
DEFAULT_OUTPUT_PATH = Path("data/processed/labeled_tickets.parquet")
DEFAULT_REPORT_PATH = Path("data/processed/labeling_report.json")

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Account Access": [
        "password", "log in", "login", "logged out", "locked out",
        "can't access", "cannot access", "verify my account",
        "verification code", "reset my password", "hacked", "sign in",
        "authenticate", "account access",
    ],
    "Order / Delivery / Refund": [
        "order", "delivery", "delivered", "package", "shipment", "shipped",
        "tracking", "refund", "return my", "courier", "parcel",
        "missing package", "never arrived", "wrong item",
    ],
    "Billing & Payments": [
        "charged", "overcharged", "billing", "bill", "invoice",
        "subscription", "payment", "autopay", "credit card",
        "double charged", "wrong amount", "cancel my subscription",
    ],
    "Technical Support": [
        "not working", "broken", "glitch", "bug", "crash", "crashed",
        "error", "won't load", "wont load", "outage", "down again",
        "reset", "reinstall", "freeze", "frozen", "lagging",
        "connection issue", "wifi",
    ],
    "Product Complaint": [
        "defective", "damaged", "poor quality", "low quality",
        "disappointed", "doesn't work", "does not work", "arrived broken",
        "wrong size", "not as described",
    ],
    "Customer Service Complaint": [
        "customer service", "rude", "unhelpful", "worst service",
        "poor service", "terrible service", "no response", "ignored",
        "on hold", "still waiting", "worst experience",
    ],
}
DEFAULT_CATEGORY = "General Inquiry"

URGENCY_KEYWORDS = [
    "urgent", "immediately", "asap", "emergency", "right now", "cancel my",
    "cancelling", "unacceptable", "furious", "never again", "scam",
    "fraud", "stolen", "lawsuit", "legal action", "final warning",
    "charged twice", "double charged", "unauthorized charge",
    "fraudulent charge", "overcharged",
]


def _compile_keyword_pattern(keywords: list[str]) -> re.Pattern:
    escaped = [re.escape(kw) for kw in keywords]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b")


CATEGORY_PATTERNS = {
    category: _compile_keyword_pattern(keywords)
    for category, keywords in CATEGORY_KEYWORDS.items()
}
URGENCY_PATTERN = _compile_keyword_pattern(URGENCY_KEYWORDS)
ALL_CAPS_PATTERN = re.compile(r"\b[A-Z]{4,}\b")


def _normalize_apostrophes(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'")


def score_categories(text: str) -> dict[str, int]:
    lowered = _normalize_apostrophes(text.lower())
    return {
        category: len(pattern.findall(lowered))
        for category, pattern in CATEGORY_PATTERNS.items()
    }


def assign_category(text: str) -> str:
    scores = score_categories(text)
    best_category = max(scores, key=scores.get)
    return best_category if scores[best_category] > 0 else DEFAULT_CATEGORY


def score_priority(text: str) -> int:
    lowered = _normalize_apostrophes(text.lower())
    score = len(URGENCY_PATTERN.findall(lowered))
    score += text.count("!")
    score += len(ALL_CAPS_PATTERN.findall(text))
    return score


def assign_priority(text: str) -> str:
    score = score_priority(text)
    if score >= 3:
        return "High"
    if score >= 1:
        return "Medium"
    return "Low"


def label_tickets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["category"] = df["text"].map(assign_category)
    df["priority"] = df["text"].map(assign_priority)

    report = {
        "tickets_labeled": len(df),
        "category_distribution": {
            k: int(v) for k, v in df["category"].value_counts().items()
        },
        "priority_distribution": {
            k: int(v) for k, v in df["priority"].value_counts().items()
        },
    }
    return df, report


def run(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict:
    df = pd.read_parquet(input_path)
    labeled_df, report = label_tickets(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_parquet(output_path, index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    return report


def main() -> None:
    report = run()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
