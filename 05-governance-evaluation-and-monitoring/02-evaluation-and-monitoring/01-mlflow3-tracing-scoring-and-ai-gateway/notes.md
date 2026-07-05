# MLflow3 Tracing, Scoring, and AI Gateway

**Section:** Governance, Evaluation, and Monitoring | **Module:** Evaluation and Monitoring | **Est. time:** 4 hrs
**Exam mapping:** Databricks GenAI Engineer Associate — Governance, Evaluation, and Monitoring domain (~15%); MLflow 3 tracing, evaluation with scorers/judges, and production monitoring are directly tested

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain traces and spans and instrument an agent with automatic and manual MLflow tracing
- Interpret span types (RETRIEVER, TOOL, CHAT_MODEL, …) and the RETRIEVER span's special schema
- Evaluate an agent with `mlflow.genai.evaluate()` using appropriate built-in and custom scorers
- Distinguish deterministic scorers from LLM-judge scorers and know which need ground truth
- Explain how LLM-as-a-judge works and align judges with human feedback
- Configure production monitoring with scheduled scorers and interpret drift/hallucination signals
- Connect evaluation to the CI/CD promotion gates from Section 04

## Core Concepts

### Traces and spans: the observability foundation

Everything in this chapter rests on **tracing**. When an agent handles a request, MLflow records a
**Trace** — the complete end-to-end record of that request — as a tree of **Spans**. A span is one
step: an LLM call, a retrieval, a tool execution. Each span captures its `inputs`, `outputs`,
`start`/`end` time, `status` (OK / ERROR), `attributes`, and any exceptions (as span events).

MLflow's spans are **OpenTelemetry-compatible**. Each span has a **type** (`from mlflow.entities import
SpanType`), and the built-in types are:

`CHAT_MODEL`, `CHAIN`, `AGENT`, `TOOL`, `EMBEDDING`, `RETRIEVER`, `PARSER`, `RERANKER`, `MEMORY`,
`UNKNOWN` (custom string types like `"ROUTER"` are allowed).

Two things trip people up:
- The LLM-call span type is **`CHAT_MODEL`**, not "LLM."
- The **`RETRIEVER` span has a specialized output schema**: a list of documents, each a dict with
  `page_content` (str) and optional `metadata` (with `doc_uri`, `chunk_id`). This schema is what
  powers the RAG scorers (`RetrievalGroundedness`, `RetrievalRelevance`) and the trace UI. If your
  retriever doesn't emit this shape, those scorers can't evaluate it.

### Automatic tracing (autolog)

For supported frameworks, tracing is one line. LangGraph (this course's framework) is a first-class
integration:

```python
import mlflow
mlflow.langgraph.autolog()   # every graph invocation now produces a full trace
# also: mlflow.langchain.autolog(), mlflow.openai.autolog(), mlflow.anthropic.autolog(), ...
```

Autolog automatically assigns span types (the LLM node becomes a `CHAT_MODEL` span, the retriever
becomes a `RETRIEVER` span, tools become `TOOL` spans). You can combine multiple autologs and manual
spans in one trace.

### Manual tracing

When you have custom code that isn't auto-instrumented, add spans yourself:

```python
import mlflow

# Decorator: one span per function
@mlflow.trace(span_type="TOOL", name="lookup_order")
def lookup_order(order_id: str) -> dict:
    return {"status": "shipped"}

# Context manager: for arbitrary blocks
with mlflow.start_span(name="post_process") as span:
    span.set_inputs({"raw": raw})
    result = clean(raw)
    span.set_outputs({"clean": result})
```

**Decorator-order gotcha:** for your own custom decorators, `@mlflow.trace` must be **outermost**. But
for web-framework route decorators (`@app.post(...)`), the framework decorator stays outermost and
`@mlflow.trace` goes **inner**.

### Tracing deployed agents

When you deploy with **`agents.deploy()`** (Section 04), **real-time tracing works automatically** — no
extra config. Traces flow to the agent's MLflow experiment for live viewing. Two things to know:

- **Git-folder gotcha:** if you deploy from a Databricks Git folder, real-time tracing won't work by
  default. Set a **non-Git-associated experiment** via `mlflow.set_experiment(...)` before deploying.
- **Long-term storage:** real-time traces are viewable immediately, but persisting them to Delta for
  historical analysis requires **Production Monitoring** (syncs ~every 15 min, no trace-size limit) or
  AI Gateway inference tables (which have size/sync limits). In production, use the lightweight
  `mlflow-tracing` package and enable async logging + sampling to avoid perf impact.

### Evaluation with `mlflow.genai.evaluate()`

Tracing tells you *what happened*; evaluation tells you *how good it was*. The core API is
`mlflow.genai.evaluate()`, which takes three things:

1. **Data** — a list of dicts with `inputs` and `expectations` (optionally pre-generated `outputs`/`trace`)
2. **A predict function** — generates outputs (omit if outputs are already present)
3. **Scorers** — the evaluation criteria

```python
import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Guidelines

dataset = [
    {"inputs": {"question": "What is AI Search?"},
     "expectations": {"expected_facts": ["AI Search is Databricks' vector search product"]}},
]

results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=my_agent_predict_fn,
    scorers=[
        Correctness(),
        RelevanceToQuery(),
        Guidelines(name="in_english", guidelines="The answer must be in English."),
    ],
)
```

### Scorers: deterministic vs LLM-judge

A **scorer** produces a score/feedback for a trace or output. There are two kinds — a key exam
distinction:

- **Deterministic / code-based (heuristic) scorers** — plain Python: exact match, format/regex checks,
  latency, token counts. Written with the `@scorer` decorator. Fast, free, reproducible. **Offline
  (development) only** for the general case.
- **LLM-judge scorers** — use an LLM to assess semantic quality (they understand paraphrases and
  nuance). Built-in or custom. Slower and costlier, but handle open-ended quality.

**Built-in scorers** you should recognize, and whether they need ground truth or a trace:

| Scorer | Measures | Needs ground truth? | Needs trace? |
|---|---|---|---|
| `RelevanceToQuery` | Does the response address the question? | No | No |
| `Correctness` | Are the expected facts supported? | **Yes** (`expected_facts`) | No |
| `Safety` | Free of harmful/toxic content? | No | No |
| `Guidelines` | Adheres to natural-language rules? | No* | No |
| `RetrievalGroundedness` | Response grounded in retrieved docs? (**anti-hallucination**) | No | **Yes** |
| `RetrievalRelevance` | Are retrieved docs relevant? | No | **Yes** |
| `RetrievalSufficiency` | Do docs contain all needed info? | Yes | **Yes** |

Naming to memorize: the "groundedness / faithfulness / hallucination" check is **`RetrievalGroundedness`**.
It (and `RetrievalRelevance`) require a trace containing a **RETRIEVER** span — you can't compute them
from a flat DataFrame of inputs/outputs.

**Custom scorers** use the `@scorer` decorator with keyword-only, all-optional params:

```python
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

@scorer
def response_not_too_long(*, outputs) -> Feedback:
    ok = len(outputs) < 2000
    return Feedback(value=ok, rationale=f"length={len(outputs)}")
```

Choose the judge LLM per scorer with the `<provider>:/<model>` format, e.g.
`Correctness(model="databricks:/databricks-gpt-5-mini")`.

### LLM-as-a-judge

An LLM judge is a scorer that (1) parses the trace to extract the fields it needs, (2) runs an LLM
assessment against instructions, and (3) returns a `Feedback` object (value + rationale) attached to
the trace. Build a custom one with `make_judge`:

```python
from mlflow.genai.judges import make_judge

judge = make_judge(
    name="issue_resolution",
    instructions="Given the user request {{ inputs }} and the agent answer {{ outputs }}, "
                 "decide whether the issue was resolved.",
    feedback_value_type=bool,
    model="databricks:/databricks-gpt-5-mini",
)
```

Template variables: `{{ inputs }}`, `{{ outputs }}`, `{{ expectations }}`, and `{{ trace }}` (a
whole-trace / "Agent-as-a-Judge" that can explore the trace autonomously — requires a `model`).

**Judge alignment** is how you make a judge agree with human experts. The flow:

1. Run the judge on ≥10 traces (15+ recommended).
2. Collect human feedback on those same traces, logged under the **exact same assessment name** as the
   judge (`mlflow.log_feedback(trace_id=..., name="issue_resolution", value=..., source=HUMAN)`).
3. `aligned = judge.align(traces, optimizer)` — default optimizer is **MemAlign** (also SIMBA, GEPA).
4. `aligned.register(experiment_id=...)`.

Alignment can cut false positives/negatives by 30–50%. **Judge reliability** is validated against human
judgment using metrics like **Cohen's Kappa**, accuracy, and F1 — a misaligned judge is worse than no
judge, so alignment matters.

### Production monitoring

Evaluation in development uses a fixed dataset; **production monitoring** runs scorers continuously on
a **sample of live traffic**, attaching results as `Feedback` to each trace. The lifecycle is
**register → start**:

```python
from mlflow.genai.scorers import Safety, ScorerSamplingConfig

sj = Safety().register(name="prod_safety")               # unique name per experiment
sj = sj.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))   # 100% of traffic
```

Rules and constraints for production monitoring:
- **≤20 scorers** per experiment.
- Custom scorers must be **`@scorer`-decorator functions** (not class-based), **defined in a Databricks
  notebook**, **self-contained** (all imports inline), with no import-requiring type hints.
- Sampling guidance: safety/security-critical scorers at `sample_rate=1.0`; expensive LLM judges at
  0.05–0.2 to control cost.
- Results appear on the experiment's **Traces** tab and monitoring dashboards after ~15–20 min, showing
  **quality trends over time** — which is how you spot **drift** and rising hallucination rates
  (`RetrievalGroundedness` trending down = grounding degrading).

## Deep Dive / Advanced Topics

### The tracing → evaluation → monitoring loop

These three aren't separate features; they're one continuous loop, and understanding that is the point
of this chapter:

1. **Tracing** captures what the agent did (the raw data).
2. **Evaluation** runs scorers on traces offline against a curated dataset — during development and in
   CI. Human experts label traces in the **Review App**, producing ground truth.
3. That human feedback **aligns the judges** so automated scoring matches expert judgment.
4. **Monitoring** runs the (now-aligned) scorers on live production traces continuously.
5. Production failures surface new cases → add them to the evaluation dataset → back to step 2.

The same scorers you pass to `mlflow.genai.evaluate()` in development are (largely) the ones you
`.register().start()` in production — consistency by design.

### Databricks Agent Evaluation and the two generations

"Agent Evaluation" is the Databricks product built on this. Note the generational split:
- **MLflow 3 (current):** `mlflow.genai.evaluate()` with scorers/judges; Agent Evaluation is folded
  into managed MLflow via `mlflow[databricks] >= 3.1`.
- **MLflow 2 (legacy):** `mlflow.evaluate(data=..., model_type="databricks-agent", ...)` with a
  `request`/`response`/`retrieved_context`/`expected_response` schema and `global_guidelines`. Still
  documented but Databricks recommends MLflow 3.

The **Review App** is where domain experts run labeling sessions to produce ground truth and feedback;
that feedback attaches to traces and feeds judge alignment. Judge outputs must **not** be used to
train/fine-tune LLMs, and EU workspaces use EU-hosted judge models.

### Evaluation as a CI/CD gate (tie-back to Section 04)

The deployment jobs from Section 04 use this machinery as an **automated promotion gate**: the
evaluation task runs `mlflow.genai.evaluate()` (with scorers like `RetrievalGroundedness`, `Safety`,
`Correctness`) on a new model version, surfaces metrics on the model-version page, and only a passing
result (plus human approval via UC governed tags) lets the model alias flip to `production`.

MLflow also supports pytest-based gating: mark tests with `@mlflow.test` (needs `mlflow >= 3.14`), call
`mlflow.genai.evaluate()`, and assert `result.passed` — a failing assertion blocks the PR. Pin
`MLFLOW_GENAI_JUDGE_DEFAULT_MODEL` so judges grade consistently across CI runs, and use aligned judges
to reduce flakiness. A **dataset evaluation** answers a measurement question (aggregate scores); a
**regression test** answers a binary gating question about a known failure mode.

## Worked Examples & Practice

### End-to-end: trace, evaluate, and monitor the docs agent

Continuing the docs agent, instrument it, evaluate it offline, then monitor it in production.

```python
import mlflow
from mlflow.genai.scorers import Correctness, RetrievalGroundedness, RelevanceToQuery, Safety

# 1. TRACE — turn on LangGraph autolog so every invocation is captured with proper span types
mlflow.langgraph.autolog()

# 2. EVALUATE — offline, against a curated dataset with ground-truth facts
eval_data = [
    {"inputs": {"question": "How do I enable Change Data Feed?"},
     "expectations": {"expected_facts": [
         "ALTER TABLE ... SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"]}},
    {"inputs": {"question": "What sync mode do storage-optimized endpoints support?"},
     "expectations": {"expected_facts": ["TRIGGERED only"]}},
]

results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=docs_agent_predict_fn,   # runs the agent, producing traces with RETRIEVER spans
    scorers=[
        Correctness(),             # needs the expected_facts above
        RetrievalGroundedness(),   # anti-hallucination; needs the RETRIEVER span from the trace
        RelevanceToQuery(),        # no ground truth needed
        Safety(),                  # no ground truth needed
    ],
)
print(results.metrics)   # aggregate pass rates per scorer

# 3. MONITOR — after deploying with agents.deploy(), schedule scorers on live traffic
from mlflow.genai.scorers import ScorerSamplingConfig

Safety().register(name="prod_safety").start(
    sampling_config=ScorerSamplingConfig(sample_rate=1.0))          # every request
RetrievalGroundedness().register(name="prod_grounded").start(
    sampling_config=ScorerSamplingConfig(sample_rate=0.2))          # 20% sample (cost control)
```

**Failure mode to observe:** try to run `RetrievalGroundedness` by passing a flat table of
`{question, answer}` pairs with **no predict function and no traces**. It fails/produces nothing,
because groundedness needs to see the *retrieved documents* — i.e., a trace containing a `RETRIEVER`
span. The fix is to evaluate with a `predict_fn` (so the agent actually runs and emits a trace) or to
supply pre-generated traces. This is the practical consequence of "RAG scorers need a trace."

## Common Pitfalls & Misconceptions

- **Pitfall:** Trying to compute `RetrievalGroundedness` / `RetrievalRelevance` from a plain DataFrame of inputs and outputs → **Why it happens:** other scorers work fine on flat data → **Fix:** RAG scorers need a **trace** with a `RETRIEVER` span (documents with `page_content`/`metadata`). Evaluate with a `predict_fn` so the agent runs and emits traces, or provide pre-generated traces.

- **Pitfall:** Deploying an agent from a Git folder and finding no real-time traces appear → **Why it happens:** `agents.deploy()` normally enables tracing automatically → **Fix:** from a Databricks Git folder, set a **non-Git-associated experiment** (`mlflow.set_experiment(...)`) before `agents.deploy()`; real-time tracing won't work with a Git-associated experiment by default.

- **Pitfall:** Using a class-based `Scorer` or a pandas-only code scorer for production monitoring → **Why it happens:** it works fine in offline evaluation → **Fix:** production monitoring supports only `@scorer`-decorator functions, defined in a notebook and self-contained (inline imports). Class-based scorers and trace/pandas-only code scorers are offline-only.

- **Pitfall:** Treating an LLM judge as trustworthy without checking it against humans → **Why it happens:** the judge produces confident-looking scores → **Fix:** validate judge reliability against human labels (Cohen's Kappa/accuracy/F1) and **align** the judge with human feedback (same assessment name, ≥10 traces, MemAlign). An unaligned judge can be systematically wrong.

- **Pitfall:** Confusing `Correctness` and `RelevanceToQuery` requirements → **Why it happens:** both sound like "is the answer good" → **Fix:** `Correctness` needs ground-truth `expected_facts` in the dataset; `RelevanceToQuery` needs no ground truth (it only checks the response addresses the question). Provide expectations for the scorers that require them.

## Key Definitions

| Term | Definition |
|---|---|
| Trace | The complete end-to-end record of one agent request, composed of a tree of spans |
| Span | A container for one step (LLM call, retrieval, tool); has inputs, outputs, timing, status, type; OpenTelemetry-compatible |
| Span type | The kind of step: `CHAT_MODEL`, `RETRIEVER`, `TOOL`, `CHAIN`, `AGENT`, `EMBEDDING`, `RERANKER`, etc. |
| Autolog | One-line automatic tracing for a framework (`mlflow.langgraph.autolog()`) |
| `@mlflow.trace` | Decorator that creates a span for a function; must be outermost for custom decorators, inner for web routes |
| Scorer | A criterion that scores a trace/output; deterministic (code) or LLM-judge |
| LLM-as-a-judge | A scorer that uses an LLM to assess semantic quality, returning a `Feedback` value + rationale |
| `RetrievalGroundedness` | The scorer measuring whether a response is grounded in retrieved docs (anti-hallucination); needs a trace |
| Judge alignment | Tuning a judge to match human feedback (same assessment name, ≥10 traces, MemAlign/SIMBA/GEPA optimizers) |
| Production monitoring | Scheduled scorers run on a sample of live traces; register → start; ≤20 scorers; `@scorer`-only custom scorers |

## Summary / Quick Recall

- **Trace = one request; span = one step.** LLM-call span type is **`CHAT_MODEL`**; the **`RETRIEVER`** span has a special document schema (`page_content` + `metadata.doc_uri`) that RAG scorers depend on
- **Autolog** (`mlflow.langgraph.autolog()`) is one-line tracing; **`@mlflow.trace`** / `start_span` for custom code (decorator outermost for custom, inner for web routes)
- **`agents.deploy()` gives real-time tracing automatically** — except from Git folders (use a non-Git experiment); long-term Delta storage needs Production Monitoring (~15-min sync)
- **`mlflow.genai.evaluate(data, predict_fn, scorers)`** is the evaluation API
- Scorers are **deterministic (code, `@scorer`)** or **LLM-judge**; `Correctness` needs `expected_facts`, `RelevanceToQuery`/`Safety` don't; **RAG scorers (`RetrievalGroundedness`, `RetrievalRelevance`) need a trace**
- **`RetrievalGroundedness`** = the anti-hallucination scorer
- **LLM judges** parse the trace, run an assessment, return `Feedback`; build with **`make_judge`**; **align** with human feedback (same name, ≥10 traces, MemAlign) and validate with Cohen's Kappa/F1
- **Production monitoring:** **register → start**; ≤20 scorers; custom scorers must be **`@scorer`**, notebook-defined, self-contained; sample critical scorers at 1.0, expensive judges at 0.05–0.2
- The same scorers gate CI/CD in **deployment jobs** (Section 04); tracing → evaluation → monitoring is one loop

## Self-Check Questions

1. You pass a DataFrame of `{question, agent_answer}` pairs to `mlflow.genai.evaluate()` with the `RetrievalGroundedness` scorer, and it produces nothing useful. What's wrong and how do you fix it?

<details>
<summary>Answer</summary>
`RetrievalGroundedness` (like `RetrievalRelevance`) requires a **trace** containing a `RETRIEVER` span — it must see the retrieved documents to judge whether the answer is grounded in them. A flat DataFrame has no such trace. Fix it by evaluating with a `predict_fn` so the agent actually runs and emits traces, or by supplying pre-generated traces in the dataset.
</details>

2. Which of these built-in scorers require ground-truth expectations in the dataset, and which don't: `Correctness`, `RelevanceToQuery`, `Safety`, `RetrievalSufficiency`?

<details>
<summary>Answer</summary>
`Correctness` needs ground truth (`expected_facts`), and `RetrievalSufficiency` needs ground truth (to know what "all needed info" is). `RelevanceToQuery` and `Safety` need no ground truth — they assess the response on its own (does it address the query; is it free of harmful content).
</details>

3. Your LLM judge scores answers as "resolved" far more often than your human experts would. How do you systematically fix this rather than rewriting the prompt by hand?

<details>
<summary>Answer</summary>
Use **judge alignment**. Run the judge on ≥10 (ideally 15+) traces, collect human feedback on those same traces logged under the **exact same assessment name** as the judge, then call `judge.align(traces, optimizer)` (default MemAlign) and `.register()`. Alignment tunes the judge toward human judgment and can cut false positives/negatives by 30–50%. Validate the result with Cohen's Kappa/accuracy/F1.
</details>

4. You want to run a custom code scorer continuously on 100% of production traffic. Your scorer is a class-based `Scorer` defined in an imported module. Will this work in production monitoring?

<details>
<summary>Answer</summary>
No. Production monitoring supports only `@scorer`-decorator functions (not class-based `Scorer`), and they must be defined directly in a Databricks notebook and be self-contained (all imports inline, no external module references, no import-requiring type hints). Rewrite it as a self-contained `@scorer` function in the notebook, then `.register(name=...).start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))`.
</details>

5. Explain how tracing, evaluation, human feedback, and monitoring form a single loop for improving an agent over time.

<details>
<summary>Answer</summary>
Tracing captures what the agent did (the raw data). Evaluation runs scorers on those traces offline against a curated dataset; human experts label traces in the Review App to create ground truth. That human feedback aligns the LLM judges so automated scoring matches expert judgment. Monitoring then runs the aligned scorers on a sample of live production traces, surfacing quality trends and drift. New production failures become new evaluation cases, and the loop repeats — the same scorers are used offline (evaluation/CI gates) and online (monitoring) for consistency.
</details>

## Further Reading

- [Databricks — MLflow Tracing](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/) — traces, spans, autolog, manual tracing (updated 2026-06-23)
- [Databricks — Evaluate and monitor AI agents](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/) — `mlflow.genai.evaluate`, scorers, judges (updated 2026-06-23)
- [Databricks — Scorers and LLM judges](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/scorers) — built-in scorer reference and custom scorers (updated 2026-06-17)
- [Databricks — Monitor GenAI apps in production](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/production-monitoring) — scheduled scorers, sampling, dashboards (updated 2026-06-23)
- [Databricks — Trace agents deployed on Databricks](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/prod-tracing) — real-time tracing, the Git-folder gotcha (updated 2026-07-01)
- [MLflow — GenAI tracing](https://mlflow.org/docs/latest/genai/tracing/) and [evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/) — the open-source reference for the `mlflow.genai` APIs
