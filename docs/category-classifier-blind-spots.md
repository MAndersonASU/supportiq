# Category classifier blind spots: measurement, partial fix, and honest limits

## Summary

Tested the trained category classifier against 15 realistically-phrased
tickets, not the exact keyword phrasings the labeling rule was built
from. 10 of 15 were miscategorized — a 67% miss rate, far below the
98.7% test macro F1 the model reports. This isn't a new defect: it's a
large-scale, quantified confirmation of an already-documented
limitation (the classifier learned to reconstruct a keyword rule, not
real category understanding). Two of the ten gaps were safe, narrow
fixes (missing verb-tense variants); the rest are structurally
different problems that a keyword-list edit can't responsibly close,
and are documented as such rather than papered over with more keywords.

## Observation

Fifteen tickets, phrased the way a real customer would write them
rather than matching the labeling rule's exact keyword list, were run
through the production category classifier:

| Ticket | Predicted | Expected | Result |
|---|---|---|---|
| "my package got lost somewhere in transit" | Order/Delivery/Refund | Order/Delivery/Refund | OK |
| "cant sign into my profile, keeps saying wrong password" | Account Access | Account Access | OK |
| "the app froze up on me again during checkout" | General Inquiry | Technical Support | MISS |
| "got billed twice this cycle for the same subscription" | Billing & Payments | Billing & Payments | OK |
| "box showed up smashed and the item inside is broken" | Technical Support | Product Complaint | MISS |
| "your rep hung up on me mid-call, unbelievable" | General Inquiry | Customer Service Complaint | MISS |
| "do you guys ship internationally?" | General Inquiry | General Inquiry | OK |
| "locked myself out, need to reset access" | Technical Support | Account Access | MISS |
| "the item i received is not what i ordered at all" | General Inquiry | Order/Delivery/Refund | MISS |
| "keeps freezing whenever i open the settings page" | General Inquiry | Technical Support | MISS |
| "i think i was double billed, can you check" | General Inquiry | Billing & Payments | MISS |
| "the shirt arrived torn and stained" | General Inquiry | Product Complaint | MISS |
| "nobody from support has responded to my last three messages" | General Inquiry | Customer Service Complaint | MISS |
| "this thing just won't boot up anymore" | General Inquiry | Technical Support | MISS |

(A 15th case, "what payment methods do you accept," was initially
logged as a miss against an expected label of General Inquiry — that
expectation was wrong. `payment` is a genuine Billing & Payments
keyword, so the classifier's answer was correct and the test case was
corrected before drawing conclusions.)

## Root cause

Every one of these misses traces back to the weak-supervision labeling
rule itself (`src/models/label_tickets.py`), not the trained model —
confirmed by running the same tickets through the rule directly and
getting identical results. The classifier is doing exactly what it was
trained to do: reproduce the rule. This is the same root cause already
documented for the category metric (`docs/architecture.md`) and for the
one previously-known instance of it ("crashing" not matching
`\bcrash\b`).

Inspecting the specific misses split them into two different kinds of
gap:

- **Word-form gaps** — the concept is already covered, but not this
  exact inflection: `"freeze"`/`"frozen"` in the keyword list, but not
  `"froze"` or `"freezing"`. Same pattern as the original `"crashing"`
  vs. `"crash"`/`"crashed"` finding.
- **Structural gaps** — not a missing word, but a limitation of exact
  substring matching itself:
  - `"locked myself out"` doesn't match `"locked out"` because a word
    is inserted mid-phrase.
  - `"broken"` is a Technical Support keyword, so "the item inside is
    **broken**" routes there even though the ticket is about a
    damaged physical product, not a technical fault — the same word
    is legitimately relevant to two different categories depending on
    context, which keyword matching can't disambiguate.
  - `"hung up on me"`, `"not what i ordered"`, `"double billed"` (as a
    standalone phrase) are new phrasings of concepts the rule already
    intends to cover, but chasing every possible real-world phrasing
    of every category is an unbounded exercise, not a fixable bug.

## Fix, deliberately scoped

Fixed only the word-form gaps, and only where the addition is a direct,
low-risk analogue of something already in the list:

- Added `"froze"`, `"freezing"` to Technical Support (matching the
  existing `"freeze"`/`"frozen"` pair).
- Added `"won't boot"`, `"wont boot"` (matching the existing
  `"won't load"`/`"wont load"` pair).
- Also closed the original, previously-deferred `"crashing"` gap
  (adding the word alongside existing `"crash"`/`"crashed"`) — that fix
  was deliberately not made in Phase 2 because a full pipeline re-run
  for one word wasn't worth it in isolation; bundling it into this
  already-necessary re-run made the marginal cost zero.

Re-ran the full cascade this requires — `label_tickets` →
`train_classifier` → `tune_classifier` → `register_model` — the same
discipline already established for labeling-rule changes, since every
downstream artifact depends on the labels.

**Did not** attempt to fix the structural gaps. `"locked myself out"`
would need phrase-proximity matching, not a keyword list; `"broken"`
would need context-aware disambiguation, not a simple word; and chasing
every new phrasing for `"hung up on me"`-style complaints has no natural
stopping point. Patching these with more keywords wouldn't fix the
underlying problem — no ground-truth labels exist to train against —
it would just make the illusion of accuracy stronger while hiding the
same limitation more effectively.

## Verification, including where the fix didn't fully work

- Full suite: 94/94 passing (2 new tests for the word-form fixes),
  `ruff check` clean.
- Category distribution shifted as expected: Technical Support grew
  from 38,829 to 42,804 tickets; several other categories shrank
  slightly too (a real, expected side effect of keyword-score
  competition, not a bug). Priority distribution was unaffected, since
  only category keywords changed.
- Re-tested the same battery after the fix. `"froze up"` and
  `"freezing"` now correctly predict Technical Support — both
  confirmed working in the trained classifier, not just the rule.
- `"won't boot up anymore"` still misses in the trained classifier,
  even though the rule itself now labels it correctly (confirmed
  directly: `assign_category` returns Technical Support). The gap here
  is different from the others — the trained model likely saw few or
  no real examples of this exact phrase in 789,547 training tickets, so
  fixing the rule didn't give the classifier enough signal to
  generalize to it. This distinction — a rule fix succeeding while the
  trained model still misses — is itself evidence for the underlying
  finding: the classifier's accuracy is bounded by what it was shown,
  not by genuine language understanding.
- The original `"crashing"` example ("the app keeps crashing every time
  I try to log in") *still* misclassifies, but for a new and
  interesting reason: that sentence also contains `"log in"`, an
  Account Access keyword, and now ties 1–1 against Technical Support's
  `"crashing"` match. Ties are broken by dictionary order, and Account
  Access is defined earlier than Technical Support, so it wins. Tested
  the same fix on a version of the sentence without the competing
  `"log in"` phrase ("the app keeps crashing after the latest update"):
  it correctly predicts Technical Support in both the rule and the
  trained model. The fix works — the original test sentence just
  happens to trigger a second, unrelated ambiguity.

## Residual risk

67% miss on natural phrasing (10/15, now roughly 9/15 with the same
battery updated to reflect the fixes) is not a number this project can
responsibly reduce much further without a fundamentally different
labeling approach — real human-annotated labels or an LLM labeler, both
ruled out earlier by budget, not by choice. The two fixes made here
close the cheapest, safest gaps; the honest position is that this
metric will not substantially improve without the underlying
weak-supervision approach changing, which is out of scope for this
project as built.
