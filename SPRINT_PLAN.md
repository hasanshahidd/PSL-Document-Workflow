# Sprint Plan. AI Engineer Take Home Assessment

Project. Pearson Specter Litt. Document understanding, grounded drafting,
and improvement from edits.
Deadline. Friday, May 15, 2026 (end of day local time).

## 1. Assessment summary

Build an internal workflow that:

1. Ingests messy legal style documents (scanned PDFs, low resolution,
   handwritten notes, inconsistent formats).
2. Extracts usable text plus structured fields.
3. Retrieves relevant evidence from those documents.
4. Generates a grounded draft cited to source.
5. Improves over time by learning from operator edits.

Grading focus. Engineering quality, grounding, thoughtful design. Not
visual polish. Not legal correctness.

### Rubric (100 points)

| # | Area | Pts |
|---|---|---|
| 1 | Document Processing | 25 |
| 2 | Retrieval and Grounding | 25 |
| 3 | Draft Quality | 10 |
| 4 | Improvement from Edits | 25 |
| 5 | Code Quality and System Design | 10 |
| 6 | Documentation and Clarity | 5 |

## 2. Scope decisions (locked before coding)

| Decision | Choice | Reason |
|---|---|---|
| Draft type | Case fact summary | Generic, easy to evaluate grounding on. |
| Language and framework | Python 3.11 plus FastAPI plus Pydantic v2 | Standard, reviewer friendly. |
| OCR for digital PDFs | PyMuPDF (`fitz`) | Fast, high fidelity text plus layout. |
| OCR for scans and images | Tesseract via `pytesseract` plus a custom multi stage pipeline | Free, no API key, good enough. |
| OCR handwriting fallback | LLM vision pass on low confidence pages | Best handwriting reader without a paid OCR service. |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local, fast, 384 dimensional, free. |
| Vector store | ChromaDB (local persistent) | Zero infrastructure, supports metadata filters. |
| LLM | OpenAI or Anthropic via a provider registry. `LLM_PROVIDER=auto` resolves to whichever key is set. | Strong at grounded extraction with citations. |
| Edit learning | Diff classification, then style rules plus a few shot exemplar bank injected into the prompt. | A real improvement loop. No fine tuning. |
| Storage | SQLite (drafts, edits, exemplars, eval runs) plus JSON files for extracted documents. | Simple and inspectable. |

### Explicitly out of scope at MVP

* Authentication and multi user.
* Production grade UI (only stretch goal was a minimal Streamlit).
* Fine tuning or training a model.
* Horizontal scaling, queues, async workers.
* Docker (stretch).

## 3. Module responsibilities

* `ingestion/`. Accepts a file path. Returns a normalised
  `ProcessedDocument` (text per page, OCR confidence, structured fields,
  low confidence regions flagged).
* `retrieval/`. Chunking, embedding, indexing, top k retrieval with
  metadata (`doc_id`, `page`, `chunk_id`, `confidence`).
* `drafting/`. Assembles a prompt with retrieved evidence, generates a
  draft with inline citations, returns the draft plus an `evidence_map`
  (sentence index to chunk ids).
* `edits/`. Captures operator edits, runs sentence level diff, classifies
  each change, mines recurring patterns, persists style rules and the
  exemplar bank, and provides the feedback injector used by `drafting/`.
* `eval/`. Grounding score, retrieval P@k, edit loop improvement trend,
  hallucination rate.
* `api/`. FastAPI routes wiring everything together.

## 4. Risks and mitigations

| Risk | Mitigation |
|---|---|
| OCR fails on handwriting | Vision model fallback on low confidence pages. If still bad, the draft says "unclear" instead of guessing. Do not oversell handwriting support in the README. |
| Edit loop looks unconvincing | Use realistic simulated edits. Rewrite a phrase, strip boilerplate, add a missing section. Show the edit distance trend chart. This is the highest risk section by points. |
| LLM cost or latency burns time | Cache `generate(doc_set_hash, prompt_hash)` to disk. Prefer a smaller model for the planner and classifier. |
| Tesseract install pain on Windows | Document the exact `winget install` and installer steps in the README. Test on a clean shell. |
| Scope creep into the UI | Hard rule. No UI until evaluation and docs are committed. |
| Last hour push problems | Push to GitHub at the end of Day 1 already, even with WIP. Iterate via commits, not one big push. |

## 5. Rubric coverage map

| Rubric area | Pts | What earns it |
|---|---|---|
| 1. Document Processing | 25 | Industrial OCR pipeline. Orientation, multi binarisation consensus, PSM auto tune, deskew, denoise, CLAHE, legal vocabulary biasing, post OCR sanity scoring, image hash caching, vision fallback. |
| 2. Retrieval and Grounding | 25 | Chunking with metadata, ChromaDB index, evidence map per draft, inline citations, post hoc citer flags unsupported sentences, ablation showing hallucination drop. |
| 3. Draft Quality | 10 | Structured Case Fact Summary template with five sections, "Uncertain" section for low confidence, simulated examples committed. |
| 4. Improvement from Edits | 25 | Captured and categorised edits, support counted style rules, exemplar bank, feedback injector, measurable edit distance trend. |
| 5. Code Quality and System Design | 10 | Module boundaries match the rubric, Pydantic schemas, error handling on OCR and LLM calls, provider ABCs, DI registry, hierarchical tracing, cost ledger, response cache. |
| 6. Documentation and Clarity | 5 | README plus ARCHITECTURE plus ASSUMPTIONS plus OCR plus SUBMISSION plus eval/results plus sample inputs and outputs plus screenshots. |

## 6. Submission checklist

* [ ] Source code pushed to GitHub.
* [ ] `README.md` with setup and run instructions.
* [ ] `ARCHITECTURE.md` (short, with diagram).
* [ ] `ASSUMPTIONS.md` (tradeoffs, scope cuts).
* [ ] Sample inputs in `samples/inputs/`.
* [ ] Sample outputs in `samples/outputs/` (initial draft, edited, improved).
* [ ] `eval/results.md` with grounding percentage, retrieval P@k, edit loop trend.
* [ ] Invite `github.com/tsensei` as collaborator.
* [ ] Invite `github.com/abubakarsiddik31` as collaborator.
* [ ] Email `talha@ideabuilders.studio` with repo link plus intro.

## 7. Execution status (final)

| Sprint | Status | Evidence |
|---|---|---|
| 1. Foundation plus Ingestion | Done. | `ingestion/` (5 modules), 92 percent mean confidence on demo docs. |
| 2. Retrieval plus Drafting | Done. | `retrieval/`, `drafting/`, 100 percent grounding rate verified in UAT. |
| 3. Improvement from Edits | Done. | `edits/` (7 modules), learned rule crossed support 2 in the live UAT. |
| 4. Evaluation | Done. | `eval/` (5 modules plus orchestrator) produces `eval/results.md`. |
| 5. Docs, Ship, and UAT | Done. | UI, Dockerfile, CI, ASSUMPTIONS, six screenshots in `samples/screenshots/`. |

### Beyond the original plan (Sprint 5 plus additions)

* Single page Alpine.js UI with live trace Gantt, citation hover popover,
  numbered citation chips, and diff pill counters.
* `DEMO_MODE=true` env flag installs deterministic fake providers and
  pre seeds two synthetic documents. Reviewers can run the entire
  pipeline without an API key.
* Provider plus DI architecture. Four ABCs in `providers/` plus a
  registry. OpenAI, Anthropic, ChromaDB, sentence transformers, and
  Tesseract are imported in exactly one file each.
* OpenAI is a first class provider. `LLM_PROVIDER=auto` resolves to
  whichever key is set. Verified live with `gpt-4o-mini`. 100 percent
  grounding. $0.0004 per draft.
* Industrial OCR pipeline documented in [OCR.md](OCR.md). Orientation
  detection, deskew, denoise, CLAHE, multi binarisation consensus
  (Otsu, Sauvola, adaptive), PSM auto tune across four modes, legal
  vocabulary biasing, post OCR sanity scorer, image hash result cache.
* Hierarchical tracing (`obs/tracer.py`), USD cost ledger
  (`obs/costs.py`), content addressed LLM response cache
  (`obs/cache.py`), structured JSON logging with `trace_id` propagation.
* Confidence weighted retrieval scoring (`retrieval/retriever.py`).
* 21 pytest tests across 8 modules, all green. Fully hermetic via
  injected fakes.
* `eval/results.md` generated against real `gpt-4o-mini`. Grounding 100
  percent. Edit distance trend 0.407 to 0.286 over two rounds.
  Hallucination ablation. 0 of 14 unsupported (grounded) versus 15 of 21
  unsupported (baseline).
* GitHub Actions CI running pytest on every push.
* Dockerfile plus docker compose for one command deploy.
* PowerShell demo launcher (`scripts/demo.ps1`).
* Playwright UAT walkthrough with six reproducible screenshots embedded
  in the README.
* [SUBMISSION.md](SUBMISSION.md). A one page review time map of the
  repository.

Final tally. 60 plus Python files. About 3,700 lines of code. All syntax
clean. 21 of 21 tests passing. Real LLM evaluation committed.
