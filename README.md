# BedsideBench

BedsideBench is a clinical question-answering benchmark with 500 cases and
public grading rubrics. It covers factual questions, calculations, false-premise
queries, and free-text management plans. Each of the 10 benchmarks contains 50
cases.

The public train split (250 cases) is in this repository and on
[Hugging Face](https://huggingface.co/datasets/doximity/bedside-bench).
The held-out test split is not published.

> **Not for clinical use.** BedsideBench evaluates AI systems. It is not medical
> advice and must not be used for patient care. See [DISCLAIMER.md](DISCLAIMER.md).

## Benchmarks

| Benchmark | Grading | Scope |
|---|---|---|
| `drug_safety` | points | Drug interactions, contraindications, dosing, and cross-reactivity |
| `guideline_adherence` | points | Guideline-concordant care |
| `landmark_trials` | points | Trial evidence and outcome interpretation |
| `medical_hallucination` | points | Fictitious entities and false premises |
| `health_equity` | points | Bias-aware reasoning and barriers to care |
| `clinical_calculations` | points | Doses, scores, and unit conversions |
| `calculators_numerical` | points | Medical calculators with numeric outputs |
| `calculators_conditional` | points | Medical calculators with categorical or conditional outputs |
| `safety_diagnostic` | f1_weighted_rubric | Diagnostic workup, referral, procedures, and follow-up |
| `safety_therapeutic` | f1_weighted_rubric | Medication, counseling, and treatment plans |

## Load the data

```python
from datasets import load_dataset

all_cases = load_dataset("doximity/bedside-bench", "all", split="train")
drug_safety = load_dataset(
    "doximity/bedside-bench",
    "drug_safety",
    split="train",
)
```

Each case contains:

- `case_id`: stable identifier;
- `benchmark`: one of the 10 benchmark names above;
- `grading_method`: `points_rubric` or `f1_weighted_rubric`;
- `prompt`: question sent to the model;
- `rubric_items`: criteria with `item_id`, `text`, and signed `points`.
  `item_id` is `{md5(prompt)}_{index}` so it is unique across the panel.

Positive rubric items describe content that should appear in an answer. Negative
items describe errors or harmful actions. On points rubrics, zero-weight items
are retained for analysis but do not affect the score. On
`f1_weighted_rubric`, a matched equivocal (`points=0`) item enters the
precision denominator.

## Grade a model

[`docs/grading_protocol.md`](docs/grading_protocol.md) contains the judge prompt,
case-level formulas, aggregation rules, and confidence-interval procedure.
Safety cases use `f1_weighted_rubric` scoring (`bedside-bench-grading-v1`):
signed `points` indicating a severity weight, and the case score is weighted F1.

```text
points     +3  +2  +1   0  -1  -2  -3
weight     72  24   3   1   3  24  72

R  = Σ w matched, class ≥ 7  /  Σ w all, class ≥ 7
P  = Σ w matched, class ≥ 7  /  Σ w all matched
F1 = 2PR / (P + R), or 0 when P + R = 0
```

This follows the published NOHARM F1 protocol (Wu et al.,
[arXiv:2512.01241](https://arxiv.org/abs/2512.01241)), whose mild/moderate/severe
weights of 1/8/24 and equivocal 1/3 are ours scaled by 3; the factor cancels in
both ratios. There is no per-case veto: omitting a +3 item or endorsing a −3
item lowers recall or precision but does not zero the case. `severe_rate` is
reported per case as a diagnostic only.
[`evaluation/run_benchmark.py`](evaluation/run_benchmark.py) implements that
protocol and checkpoints every completed case.

Set `OPENAI_ACCESS_TOKEN` or `OPENAI_API_KEY`, then run one case from each
benchmark:

```bash
python3 evaluation/run_benchmark.py \
  --answer-model gpt-5.4-mini \
  --judge-model gpt-5.6-sol \
  --output tmp/evaluations/gpt-5.4-mini-smoke.jsonl \
  --smoke
```

Remove `--smoke` and use a new output path to grade every case present in
this checkout. A public clone has the 250-case train split.

## Repository structure

```text
data/{benchmark}/          # manifests and case JSON
evaluation/                # reference grading runner and tests
schemas/                   # JSON Schemas
scripts/                   # validation and loading utilities
registry.json              # benchmark registry
cluster_plan.json          # benchmark grouping metadata
```

## Validate

Requires Python 3.10+ and the packages in `scripts/requirements.txt`.

```bash
python3 scripts/validate_registry.py
python3 scripts/validate_cases.py
python3 -m unittest discover -s evaluation -p "test_*.py"
```

## Release details

- License: [CC BY-NC-SA 4.0](LICENSE)
- Citation metadata: [CITATION.cff](CITATION.cff)
- Safety and use restrictions: [DISCLAIMER.md](DISCLAIMER.md)
