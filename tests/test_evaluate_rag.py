"""
Tests for the RAG evaluation harness's pure logic: cosine similarity
math and result summarization, both independent of the embedding model
or the live generation pipeline so they run fast and deterministically.
"""

import numpy as np

from src.ai.evaluate_rag import cosine_similarity, summarize_results


def test_cosine_similarity_of_identical_vectors_is_one():
    vector = np.array([1.0, 2.0, 3.0])
    matrix = np.array([[1.0, 2.0, 3.0]])

    result = cosine_similarity(vector, matrix)

    assert np.isclose(result[0], 1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    vector = np.array([1.0, 0.0])
    matrix = np.array([[0.0, 1.0]])

    result = cosine_similarity(vector, matrix)

    assert np.isclose(result[0], 0.0)


def test_cosine_similarity_scores_each_row_of_the_matrix():
    vector = np.array([1.0, 0.0])
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    result = cosine_similarity(vector, matrix)

    assert np.allclose(result, [1.0, 0.0, -1.0])


def make_result(**overrides: object) -> dict:
    result = {
        "num_hallucinated_citations": 0,
        "link_redacted": False,
        "completed_action_claimed": False,
        "needs_human_escalation": False,
        "mean_groundedness_similarity": 0.5,
        "reply_nonempty": True,
    }
    result.update(overrides)
    return result


def test_summarize_reports_hallucination_rate():
    results = [
        make_result(num_hallucinated_citations=1),
        make_result(num_hallucinated_citations=0),
    ]

    summary = summarize_results(results)

    assert summary["hallucination_rate"] == 0.5


def test_summarize_reports_escalation_rate():
    results = [
        make_result(needs_human_escalation=True),
        make_result(needs_human_escalation=False),
        make_result(needs_human_escalation=False),
        make_result(needs_human_escalation=False),
    ]

    summary = summarize_results(results)

    assert summary["escalation_rate"] == 0.25


def test_summarize_reports_link_redaction_rate():
    results = [
        make_result(link_redacted=True),
        make_result(link_redacted=False),
        make_result(link_redacted=False),
        make_result(link_redacted=False),
    ]

    summary = summarize_results(results)

    assert summary["link_redaction_rate"] == 0.25


def test_summarize_reports_completed_action_claim_rate():
    results = [
        make_result(completed_action_claimed=True),
        make_result(completed_action_claimed=False),
        make_result(completed_action_claimed=False),
        make_result(completed_action_claimed=False),
    ]

    summary = summarize_results(results)

    assert summary["completed_action_claim_rate"] == 0.25


def test_summarize_counts_empty_or_short_replies():
    results = [
        make_result(reply_nonempty=False),
        make_result(reply_nonempty=True),
    ]

    summary = summarize_results(results)

    assert summary["empty_or_short_replies"] == 1
