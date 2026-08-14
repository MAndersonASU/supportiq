"""
FastAPI service wrapping the production classifiers and the RAG
resolution assistant behind a single triage endpoint. Models are not
eagerly loaded at startup — triage_pipeline and rag_assistant already
lazy-load and cache the classifiers, embedding model, and vector index
on first use, so the service starts instantly and the first real request
pays the one-time load cost instead of every deployment paying it
upfront. The triage function is injected via a FastAPI dependency rather
than called directly, so the API layer can be tested without a live
model stack (Ollama, MLflow registry, vector index) behind it. A static
single-page UI is served at the root path for agents to use the
service without calling the API directly.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.ai.triage_pipeline import triage
from src.serving.logging_config import get_logger

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="SupportIQ", description="Ticket triage and resolution drafting")

request_logger = get_logger("supportiq.request")
triage_logger = get_logger("supportiq.triage")


class TicketRequest(BaseModel):
    text: str = Field(min_length=1, description="The raw ticket text to triage")


class TriageResponse(BaseModel):
    ticket_text: str
    predicted_category: str
    predicted_priority: str
    draft_reply: str
    cited_ticket_ids: list[str]
    link_redacted: bool
    mention_redacted: bool
    completed_action_claimed: bool
    ticket_severity_signaled: bool
    needs_human_escalation: bool


def get_triage_fn():
    return triage


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    request_logger.info(
        "request completed",
        extra={
            "extra_fields": {
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            }
        },
    )
    return response


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/triage", response_model=TriageResponse)
def triage_ticket(payload: TicketRequest, triage_fn=Depends(get_triage_fn)) -> TriageResponse:
    result = triage_fn(payload.text)

    triage_logger.info(
        "ticket triaged",
        extra={
            "extra_fields": {
                "text_length": len(payload.text),
                "predicted_category": result["predicted_category"],
                "predicted_priority": result["predicted_priority"],
                "link_redacted": result["link_redacted"],
                "mention_redacted": result["mention_redacted"],
                "completed_action_claimed": result["completed_action_claimed"],
                "ticket_severity_signaled": result["ticket_severity_signaled"],
                "needs_human_escalation": result["needs_human_escalation"],
            }
        },
    )
    return TriageResponse(**result)
