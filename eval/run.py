"""End-to-end evaluation orchestrator.

  python -m eval.run --docs <doc_id> [<doc_id> ...] \
                     --labels eval/labels.json \
                     --rounds 5 \
                     --out eval/results.md

Runs the four eval modules and writes a human-readable Markdown report
plus the underlying JSON.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from drafting.generator import generate_case_fact_summary
from eval.grounding import score_draft
from eval.retrieval import evaluate as eval_retrieval
from eval.edit_loop import run_loop
from eval.hallucination import run_ablation


def _md_table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    return "\n".join([head, sep, body])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", nargs="+", required=True, help="doc_ids to evaluate against")
    p.add_argument("--labels", type=Path, default=None, help="labels JSON for retrieval eval")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("eval/results.md"))
    p.add_argument("--skip-loop", action="store_true")
    p.add_argument("--skip-ablation", action="store_true")
    args = p.parse_args()

    results: dict = {"doc_ids": args.docs}

    # 1. Grounding
    print("[1/4] grounding score on a fresh draft ...")
    draft = generate_case_fact_summary(args.docs, use_feedback=False)
    results["grounding"] = score_draft(draft)
    results["grounding"].pop("verdicts", None)  # keep summary compact

    # 2. Retrieval P@k / MRR
    if args.labels and args.labels.exists():
        print("[2/4] retrieval P@k and MRR ...")
        retr = eval_retrieval(args.labels)
        retr.pop("per_query", None)
        results["retrieval"] = retr
    else:
        print("[2/4] retrieval eval skipped (no labels file)")
        results["retrieval"] = None

    # 3. Edit loop trend
    if not args.skip_loop:
        print(f"[3/4] edit-loop over {args.rounds} rounds ...")
        results["edit_loop"] = run_loop(args.docs, rounds=args.rounds)
    else:
        results["edit_loop"] = None

    # 4. Hallucination ablation
    if not args.skip_ablation:
        print("[4/4] hallucination ablation (grounded vs baseline) ...")
        results["hallucination"] = run_ablation(args.docs)
    else:
        results["hallucination"] = None

    # write JSON + markdown
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_lines = ["# Evaluation Results", "", f"Documents: `{', '.join(args.docs)}`", ""]

    g = results["grounding"]
    md_lines += [
        "## 1. Grounding (LLM-as-judge)",
        f"- cited sentences: **{g['n_cited']}**",
        f"- supported by cited evidence: **{g['n_supported']}**",
        f"- grounding score: **{g['grounding_score']:.2%}**",
        "",
    ]

    if results["retrieval"]:
        r = results["retrieval"]
        md_lines += [
            "## 2. Retrieval Quality",
            f"- queries: **{r['n_queries']}**, k={r['k']}",
            f"- Precision@{r['k']}: **{r['precision_at_k']:.2%}**",
            f"- MRR: **{r['mrr']:.3f}**",
            "",
        ]

    if results["edit_loop"]:
        el = results["edit_loop"]
        md_lines += [
            "## 3. Improvement-from-Edits Loop",
            f"- rounds: **{len(el['rounds'])}**",
            f"- first-round edit distance: **{el['first_distance']:.3f}**",
            f"- final-round edit distance: **{el['last_distance']:.3f}**",
            f"- absolute improvement: **{el['improvement']:.3f}**  "
            f"({'improved' if el['improved'] else 'no improvement'})",
            "",
            _md_table(el["rounds"], ["round", "edit_distance", "grounding_rate", "n_sentences"]),
            "",
        ]

    if results["hallucination"]:
        h = results["hallucination"]
        md_lines += [
            "## 4. Hallucination Ablation (grounded vs baseline)",
            "",
            _md_table(
                [
                    {"run": "grounded", **h["grounded"]},
                    {"run": "baseline", **h["baseline"]},
                ],
                ["run", "n_sentences", "n_unsupported", "grounding_rate"],
            ),
            "",
        ]

    args.out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"wrote {args.out} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
