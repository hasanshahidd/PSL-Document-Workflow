# Submission Checklist

Pearson Specter Litt. AI Engineer take home.
Deadline. Friday, May 15, 2026 (end of day).

## What the brief required versus what is in the repo

### Required deliverables

| # | Item | Where it lives | Status |
|---|---|---|---|
| 1 | Source code | Repository root. | Done. 60 plus Python files across 11 packages. |
| 2 | README with setup and run | [README.md](README.md) | Done. OpenAI plus Anthropic plus demo mode setup, screenshots, eval numbers. |
| 3 | Short architecture overview | [ARCHITECTURE.md](ARCHITECTURE.md) | Done. Layered diagram, provider matrix, request flow trace. |
| 4 | Assumptions and tradeoffs | [ASSUMPTIONS.md](ASSUMPTIONS.md) | Done. Scope cuts, tech tradeoffs, known limits, what to build next. |
| 5 | Sample inputs and outputs | [samples/inputs/](samples/inputs/), [samples/outputs/](samples/outputs/) | Done. Two legal style inputs plus four markdown outputs (initial draft, edited, signals, post learning draft). |
| 6 | Evaluation approach and results | [eval/results.md](eval/results.md) | Done. Generated against real `gpt-4o-mini`. |

### Optional. All done.

| # | Item | Status |
|---|---|---|
| 7 | API endpoints | Done. FastAPI surface. `/ingest`, `/retrieve`, `/drafts`, `/drafts/{id}/edits`, `/style-rules`, `/traces/{id}`, `/costs`, `/stats`, `/health`, `/documents`, `/chunks/{id}`, `/demo-mode`. |
| 8 | Simple UI | Done. Single page Alpine.js app with live trace Gantt, citation hover, cost meter, learned rule list. |
| 9 | Tests | Done. 21 tests across 8 modules, all pass. Hermetic. Fake providers injected through the registry. |
| 10 | Docker setup | Done. [Dockerfile](Dockerfile) plus [docker-compose.yml](docker-compose.yml) with Tesseract bundled. |

## Rubric coverage (100 points)

### 1. Document Processing. 25 points

* Industrial OCR pipeline documented in [OCR.md](OCR.md). Orientation
  detection (Tesseract OSD), shadow removal, denoise, CLAHE, deskew,
  border trim, three variant binarisation consensus (Otsu, Sauvola,
  adaptive), PSM auto tune across four modes, legal vocabulary biasing
  via user words and user patterns, post OCR sanity scorer (regex plus
  structural cues), image hash result cache.
* Vision model fallback for pages whose combined
  `(tesseract_conf * sanity_score)` is below threshold.
* `ProcessedDocument` schema includes per page `ocr_confidence`, `source`
  (`digital`, `tesseract`, or `vision`), and an `OcrDiagnostics` block
  with the chosen variant, chosen PSM, applied preprocessing stages,
  sanity score, and rotation correction.

### 2. Retrieval and Grounding. 25 points

* Confidence weighted retrieval. Cosine similarity multiplied by
  `(0.5 + 0.5 * ocr_confidence)` so low OCR chunks lose to better evidence
  when both are available.
* Section aware multi query. One query per output section, deduplicated
  by `chunk_id`.
* Inline `[chunk_id]` citations on every factual sentence. The post hoc
  citer marks any uncited sentence as `supported=False`.
* Two inspection paths. `GET /chunks/{id}` returns the source text by
  chunk id. The UI shows it on citation hover.
* Hallucination ablation in the eval. The grounded run produced 0 of 14
  unsupported sentences. The baseline ablation produced 15 of 21.

### 3. Draft Quality. 10 points

* Case Fact Summary template with five sections. Parties, Timeline,
  Subject Matter, Material Facts, Uncertain.
* The "No supporting evidence found in the provided documents." sentinel
  keeps the model from inventing content when retrieval comes up empty.
* Live verification. See
  [samples/outputs/04_post_learning_draft.md](samples/outputs/04_post_learning_draft.md)
  and [eval/results.md](eval/results.md) round 2 (`grounding_rate 0.93`).

### 4. Improvement from Edits. 25 points

* Captured via `POST /drafts/{id}/edits` and stored with sentence level
  diff.
* The diff classifier tags each change into a bounded category set.
  `phrasing_swap`, `boilerplate_stripped`, `section_added`, `tone_shift`,
  `citation_corrected`, `fact_corrected`.
* Generalisable rules accumulate in a `style_rules` table with a
  `support` counter. Eligible for injection at `support >= 2`.
* Operator approved drafts stored as exemplars keyed by `doc_type`. The
  top N most recent are injected as few shot examples.
* The feedback injector composes rules plus exemplars into a system
  prompt addendum without changing any caller signatures.
* Measured improvement in `eval/results.md`. Edit distance dropped from
  0.407 to 0.286 in two rounds (30 percent reduction).

### 5. Code Quality and System Design. 10 points

* Provider ABCs plus DI registry. Swapping LLM providers is one new class
  plus one env var. Proven by working with both Anthropic and OpenAI (see
  screenshot 5, `gpt-4o-mini` live).
* Exception hierarchy (`PSLError`) with a single FastAPI handler.
* Retry plus backoff with jitter on rate limit and transient errors,
  isolated inside the provider.
* Structured JSON logging with trace id and span id propagation via
  `ContextVar`.
* Hierarchical tracing persisted to SQLite. Every span includes
  attributes, duration, and status.
* SQLite backed LLM response cache with content addressed keys. Verified.
  A second draft call for the same documents ran in 0.325 seconds with
  zero new cost. All cache hits.
* 21 hermetic tests covering chunker, citer, diff, rules, tracer, cost
  ledger, OCR pipeline, and a full draft pipeline against fake providers
  (zero network).

### 6. Documentation and Clarity. 5 points

* README, ARCHITECTURE, ASSUMPTIONS, OCR, SPRINT_PLAN, SUBMISSION (this
  file), eval/results.
* Six screenshots in [samples/screenshots/](samples/screenshots/)
  embedded in the README.
* Demo mode lets reviewers click through the full UI in 30 seconds with
  no API key.

## How a reviewer can verify everything

```powershell
# Option A. No API key. Full UI walk through.
.\scripts\demo.ps1
# Open http://localhost:8000. Click both seeded docs, Generate Draft, then submit edit.

# Option B. Real LLM.
# Put your OPENAI_API_KEY in .env, then:
uvicorn api.main:app --reload
# Same UI, real generations.

# Option C. Run the evaluator.
python -m eval.run --docs demo_licensing_agreement demo_notice_of_dispute --rounds 2 --out eval/results.md

# Tests.
pytest -q
```

## Submission checklist (manual steps left)

These three items require your GitHub auth and email account. They cannot
be automated from inside the codebase.

* [ ] `git init && git add . && git commit -m "PSL document workflow"` (if
  not already a repo).
* [ ] Create a GitHub repository and push.
* [ ] Invite [github.com/tsensei](https://github.com/tsensei) as
  collaborator.
* [ ] Invite
  [github.com/abubakarsiddik31](https://github.com/abubakarsiddik31) as
  collaborator.
* [ ] Email **talha@ideabuilders.studio** with the repository link and a
  short intro.

Suggested email body.

> Hi Talha,
>
> Submitting the AI Engineer take home. Repo: [your repo url]
>
> A quick orienting tour:
>
> * `README.md` has a 30 second demo (`scripts\demo.ps1`). No API key
>   needed.
> * `ARCHITECTURE.md` walks through the layered design.
> * `eval/results.md` has the grounding, edit loop, and ablation numbers
>   from a real `gpt-4o-mini` run.
> * The system supports both OpenAI and Anthropic via a provider
>   registry. `LLM_PROVIDER=auto` picks whichever key is set.
>
> Happy to walk through any part of it.
>
> [your name and brief intro]
