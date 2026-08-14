# RAG reply completed-action fabrication: observation and fix

## Summary

Testing the RAG resolution assistant for the same failure class as the
earlier link-fabrication finding — a draft reply stating something
happened that didn't — turned up a real, reproducible case: the model
copying a real past reply's completed-action claim ("we've already
responded via DM") into a draft for a different customer, for whom no
such action was ever taken. Measured at 4/12 test runs before any fix,
reduced to 1/12 after strengthening the prompt, and now caught reliably
by a code-level check that forces human escalation whenever it fires —
since editing the claim out safely, the way a URL can be stripped, is
not reliable for free-form natural language.

## Observation

The RAG prompt already instructs the model not to invent facts and not
to copy another customer's links, order numbers, case numbers, or
usernames. It said nothing about copying a *claim of completed action*.
Querying "I was told I'd get a refund last week and still haven't
received it" retrieved a real past ticket whose actual resolution text
was:

```
We've responded to you via DM. Kindly check. ^ST
```

That reply was true for the original customer. The model reused it
almost verbatim as the draft for the new customer — 4 out of 4 times in
an initial test run. For this system, no DM was ever sent to the new
customer, so the claim is false in that context, even though it's
copied real text rather than invented from nothing.

Two other test queries ("you said my account would be fixed," "where is
the compensation I was promised") did not reproduce this pattern across
4 runs each. The difference: this query's single nearest retrieved
match was unusually short and templated, which made the model far more
likely to reproduce it close to verbatim rather than paraphrase.

## Root cause

Same underlying issue as the link-fabrication finding: a prompt
instruction is a probabilistic nudge on a small local model, not an
enforced constraint, and the existing instruction didn't cover this
specific case (claims about actions taken) at all — it only addressed
identifiers (links, order numbers, case numbers, usernames).

## Fix, in two layers

**1. Prompt instruction, extended.** Added an explicit line: "Do not
state that any action has already been completed for this new customer
(a refund issued, a DM already sent, an account already fixed) —
nothing has been done yet, so only offer to help or ask for more
information."

Re-ran the same 12-query battery after this change alone: the rate
dropped from 4/12 to 1/12. Real improvement, not a fix — the same
lesson as the link case, where a prompt instruction alone reduced but
did not eliminate the behavior.

**2. Code-level detection, since editing text safely isn't as simple as
stripping a URL.** For links, replacing the URL with `[link removed]`
leaves the rest of the sentence intact and readable. There's no
equivalent safe edit for a sentence like "we've responded to you via
DM" — deleting or rewriting a clause risks producing a broken or
misleading sentence, which could be worse than leaving the original
text for a human to catch.

Instead, `claims_completed_action()` (`src/ai/rag_assistant.py`) checks
the reply against a small set of high-confidence phrase patterns
(`we've responded`, `already refunded`, `refund has been issued`,
`account has been fixed`, and close variants) and, if any match, forces
`needs_human_escalation` to `True` rather than silently editing the
reply. The reply is left as-is but never treated as a safe, unreviewed
draft.

## Verification

- Re-ran the same 12-query battery a third time with the fix live: the
  pattern fired once more (1/12, consistent with the prompt-only
  result), and that one case correctly came back with
  `needs_human_escalation: True` — confirming the code-level layer
  catches what the prompt alone still lets through.
- Full suite: 84/84 tests passing (5 new: one prompt-content test, three
  for `claims_completed_action`, one for the new evaluation metric),
  `ruff check` clean.
- Added `completed_action_claimed` to `generate_response()`'s return
  value, propagated through `triage_pipeline`, the `/triage` API
  response, and a new `completed_action_claim_rate` metric in
  `evaluate_rag.py` — the same treatment already given to citation
  hallucination and link redaction, so this rate is tracked over time
  rather than assumed.

## Residual risk

The phrase-pattern list is deliberately narrow and high-precision,
which means it has real false negatives. Testing turned up at least one
related claim it doesn't catch — "I've checked on the status of your
refund" also asserts an action (checking) that the system never took,
but wasn't added to the pattern list because "checked" appears in many
completely legitimate phrasings too (e.g. offering to check something
going forward), and a broader match would risk flagging clean replies
just as often as catching bad ones. This is the same class of tradeoff
already documented for `redact_urls`, except harder here: a URL has an
unambiguous structural signature; a false claim about completed work
does not, and no keyword list will catch every phrasing. The fix
reduces the risk on two independent layers (a better prompt, plus a
narrow but reliable escalation trigger) rather than claiming either one
solves it completely.
