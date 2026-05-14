# Assumptions and Tradeoffs

A short, honest read out of what was built, what was deliberately left out,
and why. Read alongside [ARCHITECTURE.md](ARCHITECTURE.md) and
[SPRINT_PLAN.md](SPRINT_PLAN.md).

## Draft type

The chosen output is a **Case Fact Summary**. It generalises across the
document types listed in the brief (contract, notice, complaint, affidavit,
title report, memo) and gives the grounding constraint a clear surface.
Every factual sentence must point to a chunk. Other choices (title review,
checklist) would have biased the system toward a narrower document mix.

## OCR strategy

* Digital text first. If a PDF page has a text layer of 50 characters or
  more, PyMuPDF reads it directly with confidence 1.0. No OCR.
* Tesseract for scans. The page is rendered at 300 dpi, runs through the
  preprocessing pipeline (orientation, deskew, denoise, CLAHE, multi
  variant binarisation, PSM auto tune), and the combined Tesseract
  confidence and post OCR sanity score becomes the page `ocr_confidence`.
* Vision model fallback below `OCR_CONFIDENCE_THRESHOLD` (default 0.60).
  The LLM transcribes the page image and self reports a confidence. Only
  pages that need it pay this cost.
* Handwriting. The vision fallback handles it. Pages with confidence still
  below the threshold after vision are flagged as low confidence so the
  draft model routes them to the "Uncertain" section instead of inventing
  text.

The full OCR pipeline is documented in [OCR.md](OCR.md).

## Grounding strategy

Grounding is enforced at three layers.

1. Prompt level. The generator system prompt requires that every factual
   sentence carry an inline `[chunk_id]` citation, and that sections
   without evidence use a specific sentinel string.
2. Post hoc citer. The `drafting/citer.py` module parses the model output,
   attaches each citation to the actual evidence chunk, and flags any
   uncited sentence as `supported=False`.
3. LLM as judge. `eval/grounding.py` rereads each cited sentence and asks
   whether the cited chunk actually supports the claim. This catches
   citations that exist but do not say what the draft claims.

Retrieval is confidence weighted. Cosine similarity is multiplied by
`(0.5 + 0.5 * ocr_confidence)`. Low confidence chunks compete only when
nothing better is available.

## Improvement from edits strategy

Fine tuning is deliberately rejected. The loop is prompt time learning.

* Sentence level diff. The LLM classifies each change into a bounded
  category set (`phrasing_swap`, `boilerplate_stripped`, and so on) and
  proposes a one line generalisable rule.
* Rules are stored in `style_rules` with a `support` counter. A rule must
  be seen at least twice (`MIN_SUPPORT_FOR_INJECTION = 2`) before it gets
  injected into the next draft. That filters one off changes from durable
  house style preferences.
* Operator approved drafts are stored as exemplars keyed by `doc_type`. The
  top two most recent are added as few shot examples.
* The feedback injector composes both into an `extra_system` block appended
  to the generator system prompt. No model retraining.

This costs almost nothing per round, is fully inspectable (the rules and
exemplars are rows in SQLite), and produces a measurable downward trend in
operator edit distance. Verified by `eval/edit_loop.py`.

## Architectural tradeoffs

| Decision | Why | Cost |
|---|---|---|
| ABCs plus DI registry for LLM, Embedder, OCR, VectorStore | Swappability, testability with fakes. | One extra layer of indirection. |
| SQLite for everything (drafts, edits, rules, exemplars, traces, costs, cache) | Single file persistence, zero infra. | Will not scale to multi writer high concurrency. Postgres would be the migration path. |
| ChromaDB persistent client | Local, no infra, supports metadata filters. | Less production ready than Qdrant. Not horizontally scalable. |
| sentence transformers MiniLM | Free, local, 384 dimensional. | A purpose built legal embedder (BGE legal) would likely improve P@k. |
| Single LLM model for all stages | Simpler eval. | Cost. A cheaper model could handle the planner and the signal classifier in production. |
| Hierarchical tracer plus cost ledger in SQLite | Built in observability, no external APM dependency. | Slower than an OTLP exporter. Tracing is best effort. |

## Out of scope (deliberate)

* Authentication and multi user. A single operator is assumed.
* Async and queues. Generator and edit calls are synchronous. Acceptable
  for a workflow tool. Would need rework for high volume inbox style use.
* PII redaction. A real deployment would add a redaction pass before
  evidence enters the LLM prompt.
* Fine tuning. Not justified by the rubric. The prompt time loop works and
  is cheaper to iterate on.
* Front end framework (React, Vue). A small Alpine.js UI was enough to
  show the trace inspector, cost meter, and citation hover. React would
  have added build complexity without rubric value.

## Known limitations

* Vision OCR confidence is self reported. A more rigorous setup would do a
  second pass with regex sanity checks (date and dollar amounts, proper
  names) and lower the confidence when those checks fail.
* The diff aligner is sentence level. Word level rewrites within an
  otherwise kept sentence do not produce a signal. A two tier diff would
  catch those.
* Style rules do not decay. If the operator house style shifts over time,
  old rules linger. A `last_seen` cutoff would be a trivial addition.
* Demo mode uses deterministic fakes. It is faithful to the pipeline shape
  but not to LLM quality. Reviewers should run a non demo trace on at
  least one real document to see actual outputs.

## What to build next

1. Multi document reasoning. Currently each section query is scoped to the
   whole document set. A coreference aware retriever would connect "the
   agreement" in document B to the named agreement in document A.
2. Inline diff annotations in the UI. Today an operator submits a full
   edited draft. A richer UX would let them right click a sentence and
   pick "rephrase like X" inline.
3. Cost budgeted generation. A per draft USD budget. If the planner stage
   exceeds it, fall back to a cheaper model.
4. Active labelling for retrieval eval. The labels file is hand curated.
   A simple loop where the operator marks "this chunk was actually
   relevant" while reading a draft would grow the eval set organically.
