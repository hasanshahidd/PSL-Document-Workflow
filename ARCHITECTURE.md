# Architecture

Five layer system. Every external dependency hides behind a provider
interface. Every cross module call is wrapped in a span. Every LLM token is
accounted for in a cost ledger.

```
                       FastAPI surface (api/main.py)
                                 |
   +-----------------------------+-----------------------------+
   |             |               |              |               |
   v             v               v              v               v
ingestion/    retrieval/      drafting/       edits/           eval/
   |             |               |              |               |
   +------+------+------+--------+------+-------+------+--------+
          v             v               v              v
        OCRProvider   EmbeddingP.    LLMProvider    VectorStore
        =====================  providers/  ==========================
                                 |
                                 | every call is wrapped by
                                 v
                       +-------------------------+
                       |  obs/ tracer, costs,    |
                       |  cache    +  logging    |
                       +------------+------------+
                                    v
                          storage/  (SQLite)
   tables: drafts, edits, style_rules, exemplars,
           traces, cost_ledger, llm_cache
```

## Layer 1. Providers (`providers/`)

Four ABCs in [providers/base.py](providers/base.py).

| ABC | Implementation | Swappable for |
|---|---|---|
| `LLMProvider` | [openai_llm.py](providers/openai_llm.py), [anthropic_llm.py](providers/anthropic_llm.py) | Mistral, vLLM, local Ollama |
| `EmbeddingProvider` | [st_embedder.py](providers/st_embedder.py) | BGE, OpenAI, Voyage |
| `OCRProvider` | [tesseract_ocr.py](providers/tesseract_ocr.py) | Textract, Azure OCR, paddleOCR |
| `VectorStore` | [chroma_store.py](providers/chroma_store.py) | Qdrant, pgvector, FAISS |

A registry ([providers/registry.py](providers/registry.py)) is the only
place that knows which concrete class to construct. `LLM_PROVIDER=auto`
resolves to OpenAI when `OPENAI_API_KEY` is set, otherwise Anthropic when
`ANTHROPIC_API_KEY` is set. Tests call `set_llm(fake)` to inject doubles
(see [tests/test_generator_with_fake_llm.py](tests/test_generator_with_fake_llm.py))
for a full end to end draft run powered entirely by fakes (zero network).

## Layer 2. Observability (`obs/`)

Three subsystems run underneath every call.

### `obs/tracer.py`. Hierarchical spans

Every meaningful operation opens a span with attributes. Spans nest. A
`generate_case_fact_summary` span contains `gather_evidence`, which contains
`retrieve`, which contains `vectorstore.query`. Each LLM round trip opens
its own `llm.call` span. Each span is flushed to the `traces` table on
close. `GET /traces/{draft_id}` returns the full tree. A span that catches
an exception is recorded with status `error` and the error text.

### `obs/costs.py`. Token and USD ledger

Each LLM provider records `(input_tokens, output_tokens, USD)` against the
active `trace_id` and `span_id`. Pricing is per model and configurable.
`GET /costs?trace_id=...` rolls up cost per draft. `GET /costs` gives the
grand total. The eval orchestrator can therefore tell you exactly how much
an experiment cost.

### `obs/cache.py`. Content addressed LLM cache

The provider hashes `(model, system, messages, max_tokens)` and checks the
`llm_cache` table before every network call. Cache hits skip both the
network call and the cost ledger. Re running eval on the same documents is
free.

### `logging_setup.py`. Structured JSON logs

Trace ID and span ID propagate through `ContextVar` instances. Every log
line is a single JSON object with `ts`, `level`, `logger`, `msg`,
`trace_id`, and `span_id`, ready for any log shipper.

## Layer 3. Domain modules

Each module is a thin orchestration layer over the providers.

* `ingestion/`. The PDF router decides digital versus scan per page, then
  hands scan pages to the OCRProvider. Below the OCR confidence floor the
  page goes through the LLM vision fallback. `struct_extract` calls the
  LLM in JSON mode to pull parties and doc_type.
* `retrieval/`. The chunker is paragraph aware and tags every chunk with
  its page number. The embedder is a sentence transformers wrapper. The
  vector store wraps ChromaDB. Retrieval is confidence weighted. Cosine
  similarity is multiplied by `(0.5 + 0.5 * ocr_confidence)` so the draft
  model never anchors on garbled text when better evidence exists.
* `drafting/`. The pipeline runs per section so each part of the summary
  gets its own focused evidence pool. The section list and writing
  instructions come from a `DraftTemplate` (see
  [drafting/templates.py](drafting/templates.py)). Two templates ship:
  Case Fact Summary for litigation style inputs (contract, notice,
  affidavit, ...) and Technical Document Summary for standards,
  benchmarks, manuals, RFCs, and similar. The right one is picked by
  `select_template(doc_type)` at the start of every draft.
  - `per_section.generate_section`. Plan retrieval queries, retrieve top
    K candidates via the bi encoder, rerank with a cross encoder
    (`retrieval/reranker.py`) to top N, then ask the LLM for ONLY that
    section's body using only those N chunks as allowed citations.
  - `citer.parse_draft`. After all sections are stitched, parse
    `[chunk_id]` citations, attach each to the matched evidence chunk,
    and flag any uncited sentence as `supported=False`.
  - `reviewer.review_and_annotate`. LLM as judge re reads every cited
    sentence against its actual chunk text. Sentences that fail flip to
    `supported=False`. Survivors get the verbatim supporting quote
    attached to the citation so the UI can show it on hover.

  Returns a `Draft` whose `draft_id` is also the `trace_id`.
* `edits/`. Sentence level diff. LLM signal classifier tags each change into
  a bounded category set. Support counted style rules. Per doc_type
  exemplar bank. Feedback injector composes both into the system prompt
  addendum for the next draft.
* `eval/`. Grounding (LLM as judge), retrieval P@k and MRR, edit loop
  improvement trend over N rounds, and a hallucination ablation that
  compares grounded versus unconstrained generation. The orchestrator
  writes `eval/results.md`.

## Layer 4. Storage (`storage/`, SQLite via WAL)

| Table | Purpose | Written by |
|---|---|---|
| `drafts` | Full Draft JSON for later edit submission. | `drafting/` |
| `edits` | Raw edited text plus diff plus signals. | `edits/` |
| `style_rules` | Learned imperatives with a `support` counter. | `edits/rules.py` |
| `exemplars` | Operator approved drafts per doc_type. | `edits/bank.py` |
| `traces` | One row per span. | `obs/tracer.py` |
| `cost_ledger` | One row per LLM call. | `obs/costs.py` |
| `llm_cache` | Hashed responses. | `obs/cache.py` |
| `ocr_cache` | Image hash keyed OCR results. | `ocr/cache.py` |

## Layer 5. API (`api/main.py`)

| Method | Path | Layer it exercises |
|---|---|---|
| POST | `/ingest` | ingestion plus retrieval |
| GET | `/documents` | ingestion |
| GET | `/documents/{id}` | ingestion |
| POST | `/retrieve` | retrieval |
| POST | `/drafts` | drafting |
| GET | `/drafts/{id}` | storage |
| POST | `/drafts/{id}/edits` | edits |
| GET | `/style-rules` | edits |
| GET | `/chunks/{id}` | retrieval |
| GET | `/traces/{trace_id}` | obs |
| GET | `/costs` | obs |
| GET | `/stats` | obs |
| GET | `/health` | meta |
| GET | `/demo-mode` | meta |

Domain errors (subclasses of `PSLError`) are caught by a single exception
handler in [api/main.py](api/main.py) and serialised as
`{"error": <class>, "detail": <str>}`.

## Engineering notes

* No global SDK clients in domain code. The only files that import
  `openai`, `anthropic`, `chromadb`, `sentence_transformers`, or
  `pytesseract` are inside `providers/` and `ocr/`. Domain modules touch
  ABCs only.
* Retries and backoff live in one place per provider, with exponential
  delay and jitter for rate limit or transient errors.
* Exception hierarchy ([errors.py](errors.py)) lets the API map any domain
  error to an HTTP status in one handler.
* Pure functions stay pure. The chunker, diff, and citer have no external
  dependencies and are exercised by unit tests without mocks.
* Tracing is best effort. A tracer failure is swallowed so it can never
  break the pipeline.
* Tests inject fake providers through `providers.registry`. The end to end
  draft flow runs against fakes in
  [tests/test_generator_with_fake_llm.py](tests/test_generator_with_fake_llm.py).

## How a single `POST /drafts` flows

```
api.create_draft
  +-> generate_case_fact_summary       (root span, trace_id = draft_id)
       +-> build_feedback_system       (reads style_rules + exemplars)
       +-> for each section in 5 sections:
       |     per_section.generate_section
       |       +-> llm.call            (planner queries)
       |       +-> retrieve            (bi encoder top K)
       |       +-> rerank              (cross encoder top N)
       |       +-> llm.call            (section body with allowed citations)
       +-> parse_draft                 (citer flags unsupported)
       +-> review_and_annotate         (self review pass)
             +-> llm.call x N          (one judge call per cited sentence)
  save_draft   > drafts table
  return Draft > response
```

After the response, `GET /traces/{draft_id}` returns every span above with
durations, attributes, and any errors. `GET /costs?trace_id=...` returns
the total USD for that draft.

## Rubric coverage

| Rubric area | What earns it |
|---|---|
| Document Processing (25) | PDF router, OCR pipeline with preprocessing and multi variant consensus, structured fields, per page OCR confidence, low confidence flagging, vision fallback. |
| Retrieval and Grounding (25) | Confidence weighted scoring, evidence map per draft, inline `[chunk_id]` citations, post hoc citer, hallucination ablation, inspectable via `/traces` and `/retrieve`. |
| Draft Quality (10) | Structured Case Fact Summary template with five sections plus an Uncertain section for low confidence inputs. |
| Improvement from Edits (25) | Classified diff signals, support counted style rules, per doc_type exemplar bank, feedback injector, measurable edit distance trend over N rounds. |
| Code Quality and System Design (10) | Provider ABCs, DI container, exception hierarchy, structured logging, retry and backoff, response cache, hierarchical tracing, tests with injected fakes. |
| Documentation (5) | README, SPRINT_PLAN, ARCHITECTURE, ASSUMPTIONS, OCR, eval/results, sample inputs and outputs, screenshots. |
