"""
Tests for the weak-supervision labeling functions: each category's
keywords route correctly, unmatched text falls back to the default
category, and priority scoring buckets as expected.
"""

from src.models.label_tickets import assign_category, assign_priority


def test_account_access_keywords_route_correctly():
    assert assign_category("I can't log in and my password reset never came") == "Account Access"


def test_order_delivery_keywords_route_correctly():
    assert assign_category("My package was never delivered, where is my order") == "Order / Delivery / Refund"


def test_billing_keywords_route_correctly():
    assert assign_category("I was charged twice on my billing statement this month") == "Billing & Payments"


def test_technical_support_keywords_route_correctly():
    assert assign_category("The app keeps crashing and won't load after the update") == "Technical Support"


def test_product_complaint_keywords_route_correctly():
    assert assign_category("The item arrived damaged and is clearly defective") == "Product Complaint"


def test_customer_service_complaint_keywords_route_correctly():
    assert assign_category("Your customer service was rude and unhelpful") == "Customer Service Complaint"


def test_unmatched_text_falls_back_to_general_inquiry():
    assert assign_category("What are your store hours on weekends?") == "General Inquiry"


def test_substring_false_positive_is_avoided_by_word_boundaries():
    # "order" should not match inside "disorder" or "reorder"
    assert assign_category("I have a sleep disorder and need advice") == "General Inquiry"


def test_typographic_apostrophe_still_matches_contraction_keywords():
    # Twitter text commonly uses U+2019 (’) instead of a plain ASCII apostrophe
    assert assign_category("still won’t load, this app is broken") == "Technical Support"


def test_low_priority_for_plain_text():
    assert assign_priority("Can you tell me your return policy?") == "Low"


def test_medium_priority_for_single_urgency_signal():
    assert assign_priority("This is urgent, please help.") == "Medium"


def test_high_priority_for_multiple_urgency_signals():
    text = "This is UNACCEPTABLE!!! I am FURIOUS and want this fixed immediately!"
    assert assign_priority(text) == "High"


def test_medium_priority_for_billing_severity_language():
    assert assign_priority("My card was charged twice this month") == "Medium"
