# Evaluation Results

Documents: `demo_licensing_agreement, demo_notice_of_dispute`

## 1. Grounding (LLM-as-judge)
- cited sentences: **8**
- supported by cited evidence: **8**
- grounding score: **100.00%**

## 3. Improvement-from-Edits Loop
- rounds: **2**
- first-round edit distance: **0.407**
- final-round edit distance: **0.286**
- absolute improvement: **0.122**  (improved)

| round | edit_distance | grounding_rate | n_sentences |
| --- | --- | --- | --- |
| 1 | 0.4074 | 0.7692 | 13 |
| 2 | 0.2857 | 0.9286 | 14 |

## 4. Hallucination Ablation (grounded vs baseline)

| run | n_sentences | n_unsupported | grounding_rate |
| --- | --- | --- | --- |
| grounded | 14 | 0 | 1.0 |
| baseline | 21 | 15 | 0.2857 |
