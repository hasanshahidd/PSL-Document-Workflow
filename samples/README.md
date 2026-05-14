# Sample Inputs and Outputs

This folder contains the canonical demo run reviewers can read without
running anything. The texts under `inputs/` are also the documents that
DEMO_MODE seeds into the system at startup. See
[providers/demo_seed.py](../providers/demo_seed.py).

## Inputs

* [inputs/licensing_agreement.txt](inputs/licensing_agreement.txt). A
  software licensing contract between Acme Industries and Beta Holdings.
* [inputs/notice_of_dispute.txt](inputs/notice_of_dispute.txt). A follow up
  notice from counsel for Acme alleging non payment under the same
  agreement.

## Outputs

* [outputs/01_initial_draft.md](outputs/01_initial_draft.md). The first
  grounded Case Fact Summary produced from the two inputs.
* [outputs/02_operator_edited.md](outputs/02_operator_edited.md). The same
  draft after a simulated senior associate review (strips boilerplate,
  prefers active voice, adds a "Bottom line").
* [outputs/03_signals_extracted.json](outputs/03_signals_extracted.json).
  What the edit classifier extracted from the diff. Categorised changes
  and the reusable rules.
* [outputs/04_post_learning_draft.md](outputs/04_post_learning_draft.md).
  A fresh draft generated after the system learned from the edit. Notice
  the boilerplate is gone and the "Bottom line" line is already in place.
  The operator no longer has to ask for them.

The full evaluation report ([eval/results.md](../eval/results.md)) is
produced by `python -m eval.run` and contains grounding percentage,
retrieval P@k, edit loop trend, and the hallucination ablation.
