"""
Tests for prompt construction: retrieved examples and the new ticket
text both appear in the prompt, ticket numbers are referenced so the
model can cite them, and the model is told not to invent ungrounded
facts.
"""

from src.ai.rag_assistant import build_prompt


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


def test_prompt_includes_all_retrieved_examples():
    examples = [make_example(ticket_id="1"), make_example(ticket_id="2"), make_example(ticket_id="3")]

    prompt = build_prompt("query", examples)

    assert "Example 1" in prompt
    assert "Example 2" in prompt
    assert "Example 3" in prompt
