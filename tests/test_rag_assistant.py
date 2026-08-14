"""
Tests for prompt construction (retrieved examples and the new ticket
text both appear, the model is told not to invent ungrounded facts or
copy other customers' details, and to respond with the structured JSON
fields) and for citation verification (a cited ticket ID is only trusted
if it was actually retrieved — anything else is a hallucination, not a
real citation).
"""

from src.ai.rag_assistant import (
    build_prompt,
    claims_completed_action,
    redact_mentions,
    redact_urls,
    ticket_signals_severity,
    verify_citations,
)


def make_example(**overrides: object) -> dict:
    example = {
        "ticket_id": "123",
        "category": "Billing & Payments",
        "priority": "Medium",
        "customer_message": "I was charged twice this month.",
        "resolution_text": "Sorry about that, refund issued.",
        "distance": 0.3,
    }
    example.update(overrides)
    return example


def test_prompt_includes_new_ticket_text():
    prompt = build_prompt("My package never arrived.", [make_example()])

    assert "My package never arrived." in prompt


def test_prompt_includes_retrieved_ticket_id_and_resolution():
    prompt = build_prompt("query", [make_example(ticket_id="999")])

    assert "999" in prompt
    assert "Sorry about that, refund issued." in prompt


def test_prompt_instructs_against_inventing_facts():
    prompt = build_prompt("query", [make_example()])

    assert "do not invent" in prompt.lower()


def test_prompt_instructs_against_copying_other_customers_details():
    prompt = build_prompt("query", [make_example()])

    assert "do not copy" in prompt.lower()


def test_prompt_instructs_against_claiming_completed_actions():
    prompt = build_prompt("query", [make_example()])

    assert "already been completed" in prompt.lower()


def test_prompt_includes_all_retrieved_examples():
    examples = [make_example(ticket_id="1"), make_example(ticket_id="2"), make_example(ticket_id="3")]

    prompt = build_prompt("query", examples)

    assert "Example 1" in prompt
    assert "Example 2" in prompt
    assert "Example 3" in prompt


def test_prompt_requests_structured_json_fields():
    prompt = build_prompt("query", [make_example()])

    assert "cited_ticket_ids" in prompt
    assert "needs_human_escalation" in prompt


def test_verify_citations_accepts_ids_that_were_actually_retrieved():
    retrieved = [make_example(ticket_id="1"), make_example(ticket_id="2")]

    verified, hallucinated = verify_citations(["1"], retrieved)

    assert verified == ["1"]
    assert hallucinated == []


def test_verify_citations_flags_ids_that_were_not_retrieved():
    retrieved = [make_example(ticket_id="1")]

    verified, hallucinated = verify_citations(["1", "999"], retrieved)

    assert verified == ["1"]
    assert hallucinated == ["999"]


def test_verify_citations_handles_no_citations():
    retrieved = [make_example(ticket_id="1")]

    verified, hallucinated = verify_citations([], retrieved)

    assert verified == []
    assert hallucinated == []


def test_redact_urls_removes_https_link():
    reply, redacted = redact_urls("DM us at https://t.co/abc123 for help")

    assert "https://t.co/abc123" not in reply
    assert "[link removed]" in reply
    assert redacted is True


def test_redact_urls_leaves_plain_text_unchanged():
    reply, redacted = redact_urls("Sorry to hear that, please DM us the details")

    assert reply == "Sorry to hear that, please DM us the details"
    assert redacted is False


def test_redact_urls_handles_multiple_links():
    reply, redacted = redact_urls("See https://example.com/a and https://example.com/b")

    assert "https://example.com" not in reply
    assert redacted is True


def test_claims_completed_action_detects_already_responded():
    assert claims_completed_action("We've responded to you via DM. Kindly check.") is True


def test_claims_completed_action_detects_refund_issued():
    assert claims_completed_action("Your refund has been issued, please allow 3-5 days.") is True


def test_claims_completed_action_ignores_offers_to_help():
    assert claims_completed_action("Can you please DM us so we can look into this?") is False


def test_redact_mentions_removes_handle():
    reply, redacted = redact_mentions("@428091 Oh no! How can we help today?")

    assert "@428091" not in reply
    assert "[handle removed]" in reply
    assert redacted is True


def test_redact_mentions_leaves_plain_text_unchanged():
    reply, redacted = redact_mentions("Sorry to hear that, please DM us the details")

    assert reply == "Sorry to hear that, please DM us the details"
    assert redacted is False


def test_redact_mentions_handles_multiple_handles():
    reply, redacted = redact_mentions("@700420 hello\n\n@777348 more text")

    assert "@700420" not in reply
    assert "@777348" not in reply
    assert redacted is True


def test_ticket_signals_severity_detects_legal_threat():
    assert ticket_signals_severity("This is urgent, I will sue you, I am furious.") is True


def test_ticket_signals_severity_detects_safety_language():
    assert ticket_signals_severity("Your product injured me, this is a safety issue.") is True


def test_ticket_signals_severity_ignores_neutral_text():
    assert ticket_signals_severity("What are your business hours?") is False
