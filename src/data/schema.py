"""
Data contract for a single raw record from the Twitter customer-support
dataset. Defines and coerces the exact fields and types the ingestion
stage requires; records that don't conform are rejected.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, field_validator

TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"


class RawTweet(BaseModel):
    tweet_id: int
    author_id: str
    inbound: bool
    created_at: datetime
    text: str
    response_tweet_id: list[int] | None
    in_response_to_tweet_id: int | None

    @field_validator("inbound", mode="before")
    @classmethod
    def parse_inbound(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return value

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_created_at(cls, value: object) -> object:
        if isinstance(value, str):
            return datetime.strptime(value, TWITTER_DATE_FORMAT)
        return value

    @field_validator("response_tweet_id", mode="before")
    @classmethod
    def parse_response_tweet_id(cls, value: object) -> object:
        if isinstance(value, str):
            return [int(part) for part in value.split(",")] if value else None
        return value

    @field_validator("in_response_to_tweet_id", mode="before")
    @classmethod
    def parse_in_response_to_tweet_id(cls, value: object) -> object:
        if isinstance(value, str):
            return int(value) if value else None
        return value
