"""
Tests for the JSON log formatter: output is valid, parseable JSON, and
extra structured fields attached to a log record are merged into it.
"""

import json
import logging

from src.serving.logging_config import JsonFormatter


def make_record(message: str = "test message", extra_fields: dict | None = None) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    if extra_fields is not None:
        record.extra_fields = extra_fields
    return record


def test_output_is_valid_json():
    formatter = JsonFormatter()
    record = make_record()

    parsed = json.loads(formatter.format(record))

    assert parsed["message"] == "test message"
    assert parsed["level"] == "INFO"


def test_extra_fields_are_merged_into_output():
    formatter = JsonFormatter()
    record = make_record(extra_fields={"status_code": 200, "duration_ms": 12.5})

    parsed = json.loads(formatter.format(record))

    assert parsed["status_code"] == 200
    assert parsed["duration_ms"] == 12.5


def test_no_extra_fields_still_produces_valid_json():
    formatter = JsonFormatter()
    record = make_record()

    parsed = json.loads(formatter.format(record))

    assert "timestamp" in parsed
    assert "logger" in parsed
