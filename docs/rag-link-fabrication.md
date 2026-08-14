# RAG reply link fabrication: observation and fix

## Summary

Live-testing the deployed API surfaced a draft reply containing a
fabricated, non-functional URL. Reproduced the behavior deliberately
(3 of 8 runs on the same query), confirmed it is a distinct and more
frequent failure mode than the copied-real-URL bug already documented
in `docs/architecture.md`, and fixed it by stripping any URL-shaped
text from generated replies before they're returned, regardless of
whether the link is real or fabricated.

## Observation

Phase 3 already documented one link-related grounding failure: the
local LLM copying a real tracking URL from a retrieved example into a
draft for a different customer, fixed at the time with an explicit
prompt instruction not to copy other customers' links, order numbers,
case numbers, or usernames.

Testing the running FastAPI service against the query "my card was
charged twice this month" produced this reply:

```
Sorry to hear that your card was charged twice this month. Can you DM
us the details of the transactions so we can look into it further?
*CAH https://t.co/your_transaction_link
```

`your_transaction_link` is not a valid t.co slug (real Twitter short
links are random alphanumeric strings), and no ticket in the retrieved
context contained this URL. This is not the same bug as the Phase 3
finding — the model isn't leaking a real customer's link, it's
generating a plausible-looking but entirely fictitious one, along with
an agent sign-off pattern (`*CAH`) mimicking the style of real
brand-support tweets in its training context.

## Reproduction

A single retry of the same query returned a clean reply with no link,
confirming the behavior is stochastic — this local model runs without
a fixed generation seed, so identical prompts can produce different
output. To get a real measurement rather than a single anecdote, the
same query was run 8 times in one script:

```
run 0: link_redacted=True   "...*CAH [link removed]"
run 1: link_redacted=False  (no link)
run 2: link_redacted=False  (no link)
run 3: link_redacted=False  (no link)
run 4: link_redacted=True   "...*CAH [link removed]"
run 5: link_redacted=True   "...[link removed]"
run 6: link_redacted=False  (no link)
run 7: link_redacted=False  (no link)
```

(`link_redacted` values shown are from the post-fix run, which also
confirms the fix — see Verification below. Pre-fix, the same 3 runs
produced the fabricated links verbatim.)

3 of 8 runs (37.5%) fabricated a link for this query. Not a rare edge
case for this query shape.

## Root cause

The existing prompt instruction addresses copying real data from
retrieved examples, but has no mechanism to prevent the model from
generating new text that merely resembles the structure of a link. A
prompt instruction is a probabilistic nudge on a 3B-parameter local
model, not an enforced constraint — it measurably reduces this class of
behavior but cannot guarantee against it.

## Severity assessment

Less severe than the original Phase 3 bug: no real customer data is
exposed, since the fabricated link isn't tied to any actual retrieved
ticket. But a customer-facing reply containing a broken or misleading
link is a real product-quality problem, independent of whether the
underlying data is sensitive — worth fixing outright, not just noting.

## Fix

Added `redact_urls()` — a pure function in `src/ai/rag_assistant.py` —
applied to every generated reply before it's returned:

```python
URL_PATTERN = re.compile(r"https?://\S+")

def redact_urls(reply: str) -> tuple[str, bool]:
    redacted = URL_PATTERN.sub("[link removed]", reply)
    return redacted, redacted != reply
```

Any URL-shaped text is stripped unconditionally, whether real or
fabricated — neither can be distinguished or verified safe without a
human in the loop, so both are treated the same way rather than
attempting to tell them apart.

The resulting `link_redacted` boolean is propagated through the whole
stack rather than only fixed silently at the source:

- `generate_response()` returns it alongside the redacted reply.
- `triage_pipeline.triage()` passes it through.
- The `/triage` API response (`TriageResponse.link_redacted`) exposes
  it, and the structured request log records it — a boolean, not the
  redacted content itself, consistent with this service's existing
  privacy-conscious logging.
- `evaluate_rag.py` tracks an aggregate `link_redaction_rate` across
  the fixed 6-query evaluation set, the same way citation hallucination
  is already tracked, so the real frequency of this behavior is
  measured going forward instead of relying on anecdote.

Three new unit tests cover `redact_urls` directly (removes a link,
leaves plain text untouched, handles multiple links in one reply), and
existing tests for the evaluation harness and the FastAPI layer were
updated for the new field.

## Verification

- Full suite: 79/79 tests passing, `ruff check` clean.
- Re-ran the same 8-iteration reproduction script after the fix: all
  runs that previously fabricated a link now return it as
  `[link removed]`, with the rest of the reply text unchanged — see the
  Reproduction section above, which shows the post-fix output.
- Verified through the actual running FastAPI service, not just the
  underlying function — `link_redacted` appears correctly in real HTTP
  responses from a fresh server restart running the fixed code.
- Re-ran `evaluate_rag`'s fixed 6-query set: 0% link redaction rate on
  that particular run, which is expected given the behavior is
  query-shape-dependent and stochastic, not deterministic. The metric's
  value is in tracking this rate over time, not in any single run
  reading zero.

## Residual risk

The fix strips all URL-shaped text unconditionally, including the
hypothetical case of a reply that legitimately needed to preserve a
real, safe URL verbatim (e.g., a help-center link present in a
retrieved example). Given this dataset's brand-support replies
essentially only ever contain tracking or case links tied to individual
customers, blanket removal is the safer default here. A system that
needed to preserve specific safe links would require an explicit
allowlist rather than a blanket strip, which is out of scope for what
this project's data actually contains.
