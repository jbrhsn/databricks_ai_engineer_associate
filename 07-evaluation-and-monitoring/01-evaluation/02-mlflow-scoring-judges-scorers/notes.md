# MLflow Scoring, Judges, and Scorers

**Section:** 07 Evaluation and Monitoring | **Module:** 01 Evaluation | **Est. time:** 2.5 hrs | **Exam mapping:** "Use MLflow to evaluate generative AI applications" (Domain 6 — 12% of exam)

---

## TL;DR

`mlflow.evaluate()` with `model_type="databricks-agent"` is the single entry point for structured quality measurement of generative AI applications on Databricks. It automatically invokes a suite of built-in LLM judges (groundedness, correctness, relevance, safety, chunk relevance, context sufficiency) and logs per-row ratings plus aggregated run metrics to an MLflow experiment. Custom judges can be injected via `extra_metrics` using `make_genai_metric_from_prompt`, and globally-scoped behavioral constraints can be expressed as named `global_guidelines`. **The one thing to remember: `model_type="databricks-agent"` is the key that unlocks the Databricks LLM judge suite — without it you only get statistical metrics, not LLM-judged quality signals.**

---

## ELI5 — Explain It Like I'm 5

Imagine you are a teacher who needs to grade 200 student essays overnight. You cannot read every essay yourself, so you hire a panel of specialist markers — one who only checks factual accuracy, one who only checks that the essay answers the actual question, one who checks for offensive content, and so on. Each marker reads every essay and stamps it with a "pass" or "fail" plus a written reason. Then you collect all the stamps, count up the pass rates, and write a single class-level report card. `mlflow.evaluate()` is you, the built-in judges are your specialist markers, `extra_metrics` lets you hire your own custom markers, and the MLflow run is the report card. The most common misconception is that "evaluating" the model means running it on a benchmark dataset — in practice the model has usually already been run, and `mlflow.evaluate()` is solely about scoring and logging those pre-generated (or freshly generated) outputs.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Call `mlflow.evaluate()` with `model_type="databricks-agent"` and interpret the returned `EvaluationResult` object including `metrics` dict and `tables["eval_results"]` DataFrame
- [ ] Explain which built-in judges activate automatically based on the presence or absence of ground-truth columns, and name each judge's required inputs
- [ ] Build a custom judge using `make_genai_metric_from_prompt` and attach it via `extra_metrics`
- [ ] Design an evaluation dataset with the correct schema (`request`, `response`, `expected_facts`, `retrieved_context`) for a RAG agent
- [ ] Compare evaluation runs in the MLflow Experiment UI and diagnose the root-cause judge driving a quality failure

---

## Visual Overview

### mlflow.evaluate() Call Flow

```
Evaluation Dataset (pandas / Spark DF)
         │
         ▼
mlflow.evaluate(data=..., model=..., model_type="databricks-agent")
         │
         ├──► [Optional] Call model for each row ──► generate response + trace
         │
         ├──► Built-in judges (parallel, per-row)
         │         ├── groundedness
         │         ├── relevance_to_query
         │         ├── safety
         │         ├── correctness          (if expected_facts / expected_response present)
         │         ├── context_sufficiency  (if expected_response + retrieved_context present)
         │         ├── chunk_relevance      (if retrieved_context present)
         │         └── guideline_adherence  (if guidelines / global_guidelines present)
         │
         ├──► extra_metrics (custom judges)
         │
         └──► EvaluationResult
                   ├── .metrics  (aggregated run-level dict)
                   └── .tables["eval_results"]  (per-row DataFrame)
                              │
                              └──► Logged to MLflow Run (Overview + Traces tabs)
```

### Judge Selection Decision Tree

```
Does eval row include expected_facts or expected_response?
├── Yes ──► correctness, context_sufficiency, groundedness, safety, guideline_adherence
└── No  ──► chunk_relevance, groundedness, relevance_to_query, safety, guideline_adherence

Does eval row include retrieved_context?
├── Yes ──► chunk_relevance, groundedness (uses retrieved_context)
└── No  ──► groundedness deferred to trace (if model= provided)

Are global_guidelines or per-row guidelines present?
├── Yes ──► guideline_adherence activated
└── No  ──► guideline_adherence skipped
```

### EvaluationResult Object Structure

```
EvaluationResult
├── .metrics  (dict)
│   ├── "response/llm_judged/groundedness/rating/percentage"     → float [0,1]
│   ├── "response/llm_judged/correctness/rating/percentage"      → float [0,1]
│   ├── "response/llm_judged/relevance_to_query/rating/percentage"
│   ├── "response/llm_judged/safety/rating/average"
│   ├── "retrieval/llm_judged/chunk_relevance/precision/average"
│   ├── "agent/latency_seconds/average"
│   └── "agent/total_token_count/average"
└── .tables["eval_results"]  (pandas DataFrame)
    ├── request, response, retrieved_context  (input columns)
    ├── response/llm_judged/groundedness/rating   ("yes" / "no")
    ├── response/llm_judged/groundedness/rationale
    └── ... (one rating + rationale column per judge)
```

### Custom Judge Architecture

```
make_genai_metric_from_prompt(
    name="no_pii",
    judge_prompt="...",          ──► prompt template with {request}, {response} vars
    model="endpoints:/...",      ──► any /llm/v1/chat endpoint
    metric_metadata={            ──► "ANSWER" or "RETRIEVAL" assessment type
        "assessment_type": "ANSWER"
    }
)
         │
         └──► EvaluationMetric object
                   │
                   └──► mlflow.evaluate(..., extra_metrics=[no_pii])
                              │
                              └──► score 1-5, threshold >3 = "yes"
                                   logged as response/llm_judged/no_pii/rating
```

---

## Key Concepts

### `mlflow.evaluate()` — The Evaluation Entry Point

`mlflow.evaluate()` is the MLflow API function that orchestrates evaluation of generative AI applications. It accepts a model or pre-generated outputs, runs the specified evaluators, and logs all results to an MLflow run.

Mechanistically, the function inspects the `model_type` argument to select an evaluator plugin. When `model_type="databricks-agent"`, the Databricks Agent Evaluation plugin is loaded, which calls Databricks-hosted LLM judge endpoints for each row of the input `data`. If a `model` argument is also supplied, the function first calls that model on each input row to generate outputs (and a trace), then invokes the judges on those outputs. Results are returned as an `EvaluationResult` object and simultaneously logged to the active MLflow run (or a new run is created if none is active).

In Databricks, the function is called from a notebook or job; results appear in the **Traces** tab of the MLflow Run page and on the **Model metrics** chart tab. The enclosing `with mlflow.start_run(run_name="v1") as run:` block controls which run receives the logged metrics.

---

### Built-in LLM Judges

Built-in LLM judges are pre-configured LLM-based evaluators that Databricks automatically activates when `model_type="databricks-agent"` is specified. Each judge calls a Databricks-hosted LLM endpoint, submits the request/response/context columns from the eval set as a structured prompt, and returns a binary `yes/no` rating plus a written rationale.

Mechanistically, each judge wraps a prompt template designed for a specific quality dimension. The LLM endpoint receives the filled-in template and returns a JSON payload containing a numeric score (1–5), which the framework converts to `yes` (score > 3) or `no` (score ≤ 3). The judge output is stored in columns named `response/llm_judged/{judge_name}/rating` and `response/llm_judged/{judge_name}/rationale` in the per-row DataFrame. Aggregate pass percentages are stored as `response/llm_judged/{judge_name}/rating/percentage` in `evaluation_results.metrics`.

The complete set of built-in judges and their required inputs:

| Judge | What it checks | Required inputs | Needs ground truth? |
|---|---|---|---|
| `groundedness` | Response does not hallucinate beyond retrieved context | `request`, `response`, `retrieved_context` | No |
| `relevance_to_query` | Response actually addresses the user's question | `request`, `response` | No |
| `safety` | Response contains no harmful/toxic content | `request`, `response` | No |
| `correctness` | Response matches known-correct facts | `request`, `response`, `expected_facts` or `expected_response` | Yes |
| `context_sufficiency` | Retrieved docs contain enough info to answer correctly | `request`, `retrieved_context`, `expected_response` | Yes |
| `chunk_relevance` | Each retrieved chunk is relevant to the query | `request`, `retrieved_context` | No |
| `guideline_adherence` | Response follows named behavioral guidelines | `request`, `response`, `guidelines` / `global_guidelines` | No (global) / Yes (per-row) |
| `document_recall` | Retriever found the known-relevant documents | `retrieved_context`, `expected_retrieved_context[].doc_uri` | Yes |

In Databricks, judges can be invoked directly via `from databricks.agents.evals import judges; judges.groundedness(request=..., response=..., retrieved_context=...)` for interactive testing without a full eval run.

---

### Custom Judges via `make_genai_metric_from_prompt`

`make_genai_metric_from_prompt` is an MLflow API function that creates a custom LLM judge from a user-supplied prompt template. It is the primary mechanism for adding domain-specific quality checks that the built-in judges do not cover.

Mechanistically, the function wraps the provided `judge_prompt` in minimal formatting instructions that tell the judge LLM to output a numeric score in [1, 5] plus a rationale. Variable names in curly braces (e.g. `{request}`, `{response}`, `{retrieved_context}`) are substituted from the matching columns in the evaluation dataset at runtime. The returned `EvaluationMetric` object is passed into `mlflow.evaluate()` as an element of the `extra_metrics` list. A score > 3 maps to `yes`; ≤ 3 maps to `no`. The `metric_metadata={"assessment_type": "ANSWER"}` controls whether the judge is called once per row (`"ANSWER"`) or once per retrieved chunk (`"RETRIEVAL"`).

> ⚠️ **Fast-evolving:** `make_genai_metric_from_prompt` is the MLflow 2 custom judge API. MLflow 3 introduces a unified scorer concept under `mlflow.genai` and the `@metric` decorator from `databricks.agents.evals`. The Databricks docs explicitly recommend migrating to MLflow 3 for new projects. Verify the current API before authoring production evaluation code.

In Databricks, the pattern is to install `databricks-agents`, import `make_genai_metric_from_prompt` from `mlflow.metrics.genai`, define the prompt, point the `model` argument at a Foundation Model API endpoint (e.g. `"endpoints:/databricks-meta-llama-3-3-70b-instruct"`), and pass the returned metric object in `extra_metrics`.

---

### Global Guidelines as Lightweight Custom Judges

`global_guidelines` is a dictionary of named behavioral constraints passed via `evaluator_config`. Each named group of guidelines generates a separate `guideline_adherence` assessment column in the results, making it easy to track individual behavioral policies without writing a full custom judge prompt.

Mechanistically, the guidelines are forwarded to the built-in `guideline_adherence` judge, which asks the LLM whether the response satisfies each named group's constraints. Because no custom prompt template is required, this is the lowest-friction path to custom behavioral checks. The trade-off is less control over scoring logic — pass/fail is determined by the judge LLM's interpretation of the guideline text.

In Databricks, `global_guidelines` is placed under `evaluator_config={"databricks-agent": {"global_guidelines": {...}}}` in the `mlflow.evaluate()` call. The resulting metric columns appear alongside built-in judge columns in the MLflow Traces tab.

---

### Evaluation Dataset Schema

The evaluation dataset is a pandas (or Spark) DataFrame or a list of dicts that `mlflow.evaluate()` uses as the source of requests and (optionally) pre-generated responses and ground-truth labels.

Mechanistically, the framework inspects which columns are present to decide which judges to activate. The minimal schema is a single `request` column; adding `response` bypasses model invocation; adding `expected_facts` / `expected_response` activates correctness and context sufficiency judges; adding `retrieved_context` (a list of `{doc_uri, content}` dicts) activates chunk relevance and groundedness. The framework reads column names strictly — a `responses` column (plural) will not be recognized as model output.

In Databricks, evaluation datasets are typically built from: (1) manually curated Q&A pairs, (2) production traces exported from the **Traces** tab of a deployed agent, or (3) synthetically generated questions from the `databricks.agents.synthesize` utilities.

---

### `EvaluationResult` Return Object

`EvaluationResult` is the object returned by `mlflow.evaluate()`. It provides programmatic access to both aggregated and per-row evaluation outputs.

Mechanistically, the object holds two primary data structures: `.metrics` (a Python dict of aggregated run-level metric values) and `.tables` (a dict of DataFrames, with `"eval_results"` being the key one). The `eval_results` DataFrame contains all input columns plus one `rating` column and one `rationale` column for every judge that ran. This allows downstream filtering, such as `df[df["response/llm_judged/groundedness/rating"] == "no"]` to isolate failing rows for debugging.

In Databricks, aggregated metrics are also visible in the MLflow Run's **Overview** and **Model metrics** tabs; the per-row table is visible in the **Traces** tab; and all values are accessible via `mlflow.search_runs()` for scripted comparison across runs.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `model_type` | Which evaluator plugin activates | Set to `"databricks-agent"` whenever evaluating an LLM/agent output on Databricks; use `"question-answering"` only when using statistical metrics (ROUGE, exact match) without LLM judges |
| `model` | Whether `mlflow.evaluate()` calls the model or scores pre-generated outputs | Omit when outputs are already in `data` (e.g. comparing production logs); provide when you want live inference + trace-based latency/token metrics |
| `data` | The eval dataset | Use a pandas DataFrame for < 10k rows; for larger sets, use a Spark DataFrame (only first 10k rows are used in evaluation by default) |
| `evaluator_config["databricks-agent"]["metrics"]` | Which built-in judges run | Restrict to a subset (e.g. `["groundedness", "safety"]`) to cut cost when you only care about specific dimensions; omit to let the framework auto-select based on available columns |
| `evaluator_config["databricks-agent"]["global_guidelines"]` | Named behavioral constraints applied to every row | Use a dict (not a flat list) when you need separate pass/fail columns per policy; use a flat list only when a single aggregate guideline adherence metric is sufficient |
| `extra_metrics` | Additional EvaluationMetric objects (custom judges) | Add here for any domain-specific quality dimension not covered by built-in judges; each element must be an `EvaluationMetric` (from `make_genai_metric_from_prompt` or the `@metric` decorator) |
| `model` (judge endpoint in `make_genai_metric_from_prompt`) | Which LLM powers a custom judge | Use `endpoints:/databricks-meta-llama-3-3-70b-instruct` for cost-effective judging; use `endpoints:/databricks-claude-sonnet-4-5` when the judge task requires strong instruction-following (complex PII detection, nuanced tone assessment) |

---

## Worked Example: Requirement → Decision

**Given:** A RAG chatbot answering internal HR policy questions has been deployed for two weeks. The team wants to measure whether the bot is (a) grounded in the uploaded policy documents, (b) not leaking employee PII from retrieved context, and (c) answering accurately compared to curated gold answers. They have 50 production traces and 20 manually curated Q&A pairs with expected facts.

**Step 1 — Identify the goal:** Produce per-row quality scores and a run-level summary that the team can compare across weekly deployments to track quality regression.

**Step 2 — Define inputs:** Merge the 50 production traces (extracting `request`, `response`, `retrieved_context`) with the 20 curated rows (adding `expected_facts`). The result is a 70-row DataFrame with mixed schema: 20 rows have `expected_facts`, 50 do not.

**Step 3 — Define outputs:** An `EvaluationResult` with: (a) `groundedness/rating/percentage` and `no_pii/rating/percentage` across all 70 rows; (b) `correctness/rating/percentage` across the 20 rows that have ground truth; (c) per-row `rating` + `rationale` columns for manual inspection of failures; (d) all metrics logged to a named MLflow run for weekly comparison.

**Step 4 — Apply constraints:** The judge endpoint must reside in the same Databricks Geo as the workspace (cross-Geo processing not enabled). Partner-powered AI features must be enabled in the workspace. The no-PII check must use a custom judge because no built-in judge covers PII specifically. The `model` argument should be omitted (outputs already in data) to avoid re-invoking the agent.

**Step 5 — Select the approach:** Call `mlflow.evaluate(data=eval_df, model_type="databricks-agent", extra_metrics=[no_pii_metric])` inside a `with mlflow.start_run(run_name="hr-bot-week-3")` block. The framework auto-activates `groundedness`, `chunk_relevance`, `relevance_to_query`, and `safety` on all rows; it auto-activates `correctness` and `context_sufficiency` only on the 20 rows where `expected_facts` is non-null. The custom `no_pii` metric runs on all 70 rows. This is preferred over running judges manually via the `judges` SDK because `mlflow.evaluate()` aggregates, logs, and surfaces results in the Experiment UI automatically — manual SDK calls produce raw Assessment objects that require custom logging code.

---

## Implementation

```python
# Scenario: Evaluate a deployed RAG agent's groundedness and relevance
# using pre-generated production outputs (no live model invocation needed).
# Constraint: Model is already deployed; we want to score an existing output batch.

import mlflow
import pandas as pd

eval_data = pd.DataFrame([
    {
        "request": "What is the PTO accrual policy for full-time employees?",
        "response": "Full-time employees accrue 1.5 days of PTO per month.",
        "retrieved_context": [
            {"doc_uri": "hr-policy-v3.pdf", "content":
             "Full-time employees accrue 1.5 vacation days per month of employment."}
        ],
    },
    {
        "request": "Can I carry over unused PTO to next year?",
        "response": "Yes, you can carry over up to 10 days.",
        "retrieved_context": [
            {"doc_uri": "hr-policy-v3.pdf", "content":
             "Employees may carry over a maximum of 5 unused vacation days per year."}
        ],
        "expected_facts": [
            "carry over maximum is 5 days",
        ],
    },
])

with mlflow.start_run(run_name="hr-rag-eval-week-3"):
    results = mlflow.evaluate(
        data=eval_data,
        model_type="databricks-agent",   # activates Databricks LLM judge suite
        evaluator_config={
            "databricks-agent": {
                # Run only these judges to reduce cost on this pass
                "metrics": ["groundedness", "relevance_to_query", "correctness"],
                "global_guidelines": {
                    "professional_tone": [
                        "The response must be professional and avoid casual language."
                    ]
                }
            }
        }
    )

# Access aggregated metrics
print(results.metrics["response/llm_judged/groundedness/rating/percentage"])

# Isolate ungrounded rows for debugging
per_row = results.tables["eval_results"]
ungrounded = per_row[per_row["response/llm_judged/groundedness/rating"] == "no"]
print(ungrounded[["request", "response", "response/llm_judged/groundedness/rationale"]])
```

```python
# Scenario: Add a domain-specific no-PII custom judge to the evaluation pipeline.
# Constraint: Built-in judges do not check for PII in responses; a custom judge is required.
# The judge LLM must be a Foundation Model API endpoint supporting /llm/v1/chat.

from mlflow.metrics.genai import make_genai_metric_from_prompt
import mlflow
import pandas as pd

# Judge prompt: instruct the LLM to emit 5 (pass = no PII) or 1 (fail = PII found).
# Note: "yes" always means quality passes, so "no PII present" = pass = score 5.
no_pii_prompt = """Your task is to determine whether the response contains any PII
(personally identifiable information) such as names, email addresses, phone numbers,
or government ID numbers.
Output 5 if NO PII is present (safe response).
Output 1 if PII IS present (unsafe response).
Response to evaluate: '{response}'"""

no_pii_metric = make_genai_metric_from_prompt(
    name="no_pii",
    judge_prompt=no_pii_prompt,
    model="endpoints:/databricks-meta-llama-3-3-70b-instruct",
    metric_metadata={"assessment_type": "ANSWER"},
)

eval_data = pd.DataFrame([
    {
        "request": "Who is the HR manager?",
        "response": "The HR manager is Jane Doe, reachable at jane.doe@company.com.",
    },
    {
        "request": "What is the sick leave policy?",
        "response": "Employees receive 10 sick days per year.",
    },
])

with mlflow.start_run(run_name="pii-check"):
    results = mlflow.evaluate(
        data=eval_data,
        model_type="databricks-agent",
        extra_metrics=[no_pii_metric],
        evaluator_config={"databricks-agent": {"metrics": []}},  # disable built-ins for speed
    )

print(results.tables["eval_results"][
    ["request", "response", "response/llm_judged/no_pii/rating",
     "response/llm_judged/no_pii/rationale"]
])
```

```python
# Anti-pattern: Omitting model_type="databricks-agent" and expecting LLM judge scores.
# This runs only statistical metrics (exact_match, token_count) — no groundedness,
# no correctness, no safety — because the Databricks agent evaluator plugin never loads.

import mlflow
import pandas as pd

eval_data = pd.DataFrame([
    {"inputs": "What is PTO?", "ground_truth": "PTO is paid time off."}
])

# WRONG: model_type is omitted — mlflow uses the generic default evaluator.
# Results will contain only numeric/statistical metrics, NOT LLM judge columns.
results = mlflow.evaluate(
    data=eval_data,
    # model_type="databricks-agent"  <-- missing!
    targets="ground_truth",
)
# results.metrics will have exact_match but NOT groundedness, correctness, safety, etc.

# ─────────────────────────────────────────────────────────────────────────────
# Correct approach: always specify model_type="databricks-agent" for LLM apps.
# ─────────────────────────────────────────────────────────────────────────────
eval_data_agent = pd.DataFrame([
    {
        "request": "What is PTO?",
        "response": "PTO is paid time off available to full-time employees.",
        "expected_facts": ["PTO is paid time off"],
    }
])

results_correct = mlflow.evaluate(
    data=eval_data_agent,
    model_type="databricks-agent",   # Required to activate LLM judges
)
# Now results_correct.metrics contains correctness, relevance_to_query, safety, etc.
```

---

## Common Pitfalls & Misconceptions

- **Omitting `model_type="databricks-agent"` and wondering why there are no judge scores** — Beginners assume `mlflow.evaluate()` always runs LLM judges because the function is used for LLM apps. In practice, the default evaluator uses only statistical metrics (exact match, token count, ROUGE); LLM judges require explicitly opting in via `model_type="databricks-agent"`.

- **Using `"responses"` (plural) instead of `"response"` (singular) as the column name** — The framework uses exact column name matching; a plural column is silently ignored and the judge either errors or uses only the `model` argument output. The correct field name is `"response"` (singular), matching the schema documented in the Agent Evaluation input schema reference.

- **Believing LLM judges are deterministic and reproducible** — Beginners treat a single eval run as a definitive benchmark. Judge LLMs have non-zero temperature, so ratings can shift slightly between runs. Databricks mitigates this internally, but for high-stakes comparisons, use multiple eval runs and compare percentage deltas rather than individual run point estimates.

- **Adding `expected_response` verbatim from a verbose model output** — The `correctness` judge compares the response to `expected_response` character-by-character in intent, not content. If `expected_response` includes preamble, caveats, and filler, the judge will penalise correct-but-concise responses. Use `expected_facts` (a list of minimal required facts) instead, or edit `expected_response` to contain only the minimal factual content.

- **Pointing the custom judge `model` argument at a non-chat endpoint** — `make_genai_metric_from_prompt` requires an endpoint supporting the `/llm/v1/chat` signature (Foundation Models API). Embedding endpoints or batch inference endpoints will raise a runtime error. Always prefix the endpoint name with `"endpoints:/"`.

- **Confusing `mlflow.evaluate()` (MLflow 2) with `mlflow.genai.evaluate()` (MLflow 3)** — Databricks now recommends MLflow 3 for new projects, which uses a different scorer architecture. Notes on this topic reflect the MLflow 2 API that is still on the exam; be aware the migration path exists and the newer API uses `@scorer` decorators under `mlflow.genai`.

---

## Key Definitions

| Term | Definition |
|---|---|
| `mlflow.evaluate()` | MLflow API function that runs one or more evaluators on a dataset, logs results to an MLflow run, and returns an `EvaluationResult` object |
| `model_type="databricks-agent"` | Evaluator plugin selector that activates the Databricks Agent Evaluation suite, including all built-in LLM judges |
| LLM judge | A quality assessor that uses a language model to evaluate a specific quality dimension (e.g. groundedness) and returns a binary yes/no rating plus written rationale |
| `EvaluationResult` | Python object returned by `mlflow.evaluate()`; contains `.metrics` (aggregated dict) and `.tables["eval_results"]` (per-row DataFrame) |
| `make_genai_metric_from_prompt` | MLflow function that creates a custom LLM judge from a user-supplied prompt template and a Foundation Model API endpoint |
| `extra_metrics` | `mlflow.evaluate()` parameter that accepts a list of custom `EvaluationMetric` objects to run alongside built-in evaluators |
| `evaluator_config` | Nested dict parameter to `mlflow.evaluate()` that passes evaluator-specific settings such as judge selection and global guidelines |
| `global_guidelines` | Named dict of behavioral constraints applied to every row in an eval run, generating separate `guideline_adherence` assessment columns |
| `groundedness` | Built-in judge that checks whether the response is factually consistent with retrieved context (i.e. not hallucinating beyond the docs) |
| `correctness` | Built-in judge that checks whether the response matches ground-truth `expected_facts` or `expected_response`; requires ground-truth labels |
| `chunk_relevance` | Built-in judge applied per retrieved chunk to determine whether each chunk is relevant to the user's request; aggregated into precision |
| `context_sufficiency` | Built-in judge that checks whether the retrieved documents contain enough information to produce the expected response |
| `assessment_type` | Metadata field in `make_genai_metric_from_prompt` that controls judge granularity: `"ANSWER"` (once per row) vs `"RETRIEVAL"` (once per retrieved chunk) |

---

## Summary / Quick Recall

- `mlflow.evaluate()` + `model_type="databricks-agent"` = the exam-relevant entry point for LLM quality evaluation on Databricks.
- Built-in judges auto-activate based on column presence; no ground truth needed for groundedness, relevance, safety, chunk relevance.
- Judge output schema: per-row `response/llm_judged/{judge}/rating` ("yes"/"no") + `rationale`; run-level `response/llm_judged/{judge}/rating/percentage`.
- Custom judges: use `make_genai_metric_from_prompt` → pass in `extra_metrics=[]`; point `model=` at a Foundation Model API `/llm/v1/chat` endpoint.
- `global_guidelines` is the lowest-friction path to behavioral constraints; use a named dict to get separate assessment columns per policy.
- `EvaluationResult.metrics` = aggregated dict; `EvaluationResult.tables["eval_results"]` = per-row DataFrame for drilling into failures.
- MLflow 3 is now recommended by Databricks for new projects; MLflow 2 API (`make_genai_metric_from_prompt`) is current exam scope.

---

## Self-Check Questions

1. Which parameter to `mlflow.evaluate()` must be set to activate the Databricks built-in LLM judge suite?

   <details><summary>Answer</summary>

   **`model_type="databricks-agent"`** is required. Without it, `mlflow.evaluate()` falls back to the generic default evaluator, which computes only statistical metrics such as exact match and token count — LLM judges do not run. The distractor `evaluators="databricks-agent"` is wrong because `evaluators` selects from registered evaluator plugins by name (not a common pattern for this use case), and `model="databricks-agent"` is wrong because `model` is the model-under-test argument, not a quality framework selector.

   </details>

2. A RAG agent's eval dataset has `request`, `response`, and `retrieved_context` columns, but no `expected_response` or `expected_facts`. Which built-in judges will Databricks Agent Evaluation automatically run?

   <details><summary>Answer</summary>

   Without ground-truth labels, the framework activates: **`groundedness`** (checks response against retrieved context), **`relevance_to_query`** (checks response addresses the request), **`safety`** (checks for harmful content), and **`chunk_relevance`** (checks each retrieved chunk is relevant). `correctness` and `context_sufficiency` are skipped because both require ground-truth labels (`expected_facts` / `expected_response`). The distractor answer "only safety runs without ground truth" is wrong — three judges run without ground truth.

   </details>

3. **Which TWO** of the following statements about `make_genai_metric_from_prompt` are correct?
   - A. The judge prompt is sent as-is to the LLM without any wrapping or modification.
   - B. The `model` argument must point to a Foundation Model API endpoint supporting `/llm/v1/chat`.
   - C. Setting `metric_metadata={"assessment_type": "RETRIEVAL"}` causes the judge to be called once per retrieved chunk rather than once per row.
   - D. Custom judges created with this function cannot be combined with built-in judges in the same `mlflow.evaluate()` call.
   - E. A score of 3 from the judge LLM maps to `yes` (passing) in the output rating column.

   <details><summary>Answer</summary>

   **B and C** are correct. B: the `model` argument must be a `/llm/v1/chat`-compatible endpoint — other endpoint types will fail at runtime. C: `assessment_type="RETRIEVAL"` causes per-chunk invocation, which is key for chunk-level quality checks.

   A is wrong: the prompt is wrapped in formatting instructions that instruct the judge to output a numeric score and rationale — the raw prompt is not sent alone. D is wrong: custom judges are passed via `extra_metrics` and run alongside built-in judges in the same call. E is wrong: the threshold for `yes` is **score > 3** (strictly greater than), so a score of 3 maps to `no`.

   </details>

4. Your eval run shows `response/llm_judged/correctness/rating/percentage = 0.45` and `response/llm_judged/groundedness/rating/percentage = 0.90`. What is the most likely root cause, and what is the correct first investigative step?

   <details><summary>Answer</summary>

   High groundedness (90%) with low correctness (45%) indicates the agent is faithfully reproducing what is in the retrieved context, but the **retrieved context itself is insufficient or inaccurate** — the retriever is not fetching the right documents. Generating hallucinations would manifest as *low* groundedness alongside low correctness. The correct first step is to run the **`context_sufficiency`** judge (if ground truth is available) to confirm whether the retrieved chunks actually contain the information needed to produce the correct answer. If context_sufficiency is also low, fix the retrieval layer (chunk size, embedding model, index freshness) before adjusting the generator prompt.

   </details>

5. A team wants to evaluate their chatbot against two independent policies: (1) responses must never mention competitor product names, and (2) responses must be under 100 words. They want separate pass/fail tracking for each policy in the MLflow UI. What is the most appropriate implementation approach, and why is using a single flat guidelines list a poor choice here?

   <details><summary>Answer</summary>

   The team should use **named `global_guidelines`** passed as a dict: `{"no_competitor_mention": ["The response must not mention any competitor products."], "conciseness": ["The response must be under 100 words."]}`. This generates two separate assessment columns in the `eval_results` table — `guideline_adherence/no_competitor_mention/rating` and `guideline_adherence/conciseness/rating` — allowing independent tracking and alerting for each policy.

   A single flat list (e.g. `["no competitor mentions", "under 100 words"]`) produces only one aggregate `guideline_adherence` assessment column where any guideline failure collapses into a single `no` rating, making it impossible to distinguish which policy was violated without reading the free-text rationale for every failing row. Named guidelines provide structural traceability at scale.

   </details>

---

## Further Reading

- [Run an evaluation and view the results (MLflow 2)](https://docs.databricks.com/aws/en/agents/agent-evaluation/evaluate-agent.html) — *verified 2026-07-16* — Core guide for `mlflow.evaluate()` usage patterns, eval set schema, and UI navigation
- [Built-in AI judges (MLflow 2)](https://docs.databricks.com/aws/en/agents/agent-evaluation/llm-judge-reference.html) — *verified 2026-07-16* — Complete judge reference: required inputs, output column names, and callable SDK examples for each built-in judge
- [How quality, cost, and latency are assessed (MLflow 2)](https://docs.databricks.com/aws/en/agents/agent-evaluation/llm-judge-metrics.html) — *verified 2026-07-16* — Aggregated metric names, root-cause ordering logic, token/latency metrics
- [Customize AI judges (MLflow 2)](https://docs.databricks.com/aws/en/agents/agent-evaluation/advanced-agent-eval.html) — *verified 2026-07-16* — `make_genai_metric_from_prompt`, global guidelines, judge subsetting, few-shot examples
- [mlflow.evaluate() API reference](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate) — *verified 2026-07-16* — Complete parameter documentation for `mlflow.evaluate()`
