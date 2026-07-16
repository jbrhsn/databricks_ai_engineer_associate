# SME Feedback Loops

**Section:** 07 — Evaluation and Monitoring | **Module:** 01 — Evaluation | **Est. time:** 2 hrs | **Exam mapping:** Domain 6 — Evaluation & Monitoring (12% of exam)

---

## TL;DR

SME (subject matter expert) feedback loops are the mechanism by which human domain knowledge is captured, structured, and re-injected into the evaluation and improvement cycle of a GenAI application. Databricks provides the MLflow Review App — a hosted UI where SMEs can label existing traces, rate responses, and define ground-truth expectations — with results stored as `Assessment` objects attached to MLflow Traces. These assessments can be promoted to evaluation datasets and consumed directly by `mlflow.evaluate()` or `mlflow.genai.evaluate()`, closing the loop from production failure to measured improvement. **The one thing to remember: SME feedback in Databricks is always stored as structured `Assessment` objects on MLflow Traces, not as raw comments in a spreadsheet — this is what makes it programmatically consumable for evaluation reruns.**

---

## ELI5 — Explain It Like I'm 5

Imagine you run a restaurant and your chef (the AI) prepares dishes. You could taste every dish yourself all day, but some dishes involve ingredients only a specialist pastry chef would know are wrong. So you invite that specialist to your kitchen for a few hours: they taste the dishes, write on sticky notes exactly what was wrong and what the correct version should taste like, and stick those notes onto the original order tickets. Later, you collect all those sticky notes and turn them into a proper recipe checklist — a list of "this dish should have these specific flavors." The next time you run a cooking test, you use that checklist to judge whether new dishes pass. The most common mistake people make is thinking the pastry chef's visit was just a one-time quality check — it was actually a data-collection event that permanently improves how you test every future dish.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Explain the difference between `Assessment` types `feedback` and `expectation` and when each is appropriate
- [ ] Configure a labeling session in the MLflow Review App and assign it to SME reviewers using `create_labeling_session()`
- [ ] Convert SME-provided expectations into an evaluation dataset consumable by `mlflow.evaluate()`
- [ ] Design an annotation workflow with rubric, labeler calibration, and inter-annotator agreement measurement
- [ ] Trace a failure through the iterative improvement loop: SME flag → root-cause diagnosis → fix → re-evaluate → delta measurement

---

## Visual Overview

### The SME Feedback Loop End-to-End

```
Production App ──► Inference Table ──► MLflow Traces
                                            │
                                            ▼
                                   Review App (labeling session)
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                      Assessment: feedback        Assessment: expectation
                      (thumbs up/down,            (expected_facts,
                       free-text comment)          expected_response)
                              │                           │
                              │                           ▼
                              │                  Evaluation Dataset
                              │                  (Delta table in UC)
                              │                           │
                              └─────────────┬─────────────┘
                                            ▼
                                   mlflow.evaluate()
                                            │
                                            ▼
                              Metrics delta ──► Fix retrieval /
                                               generation / prompt
```

### Assessment Type Decision Tree

```
Is the label a judgment of what the app produced?
├── Yes ──► Use Assessment type: feedback
│          (thumbs up/down, Likert score, free-text comment)
│          → stored on Trace, NOT synced to eval dataset
└── No  ──► Is it a definition of what the correct output should be?
            ├── Yes ──► Use Assessment type: expectation
            │          (expected_facts, expected_response, guidelines)
            │          → sync to evaluation dataset via sync_expectations()
            └── No  ──► Re-examine — all SME labels are one of these two
```

### Iterative Improvement Loop

```
Step 1: Flag          Step 2: Diagnose      Step 3: Fix
────────────          ────────────────      ───────────
SME marks trace  ──►  Is failure in:   ──►  Retrieval:  tune chunk size,
thumbs-down           • Retrieval?          re-index, swap embeddings
                      • Generation?         Generation: adjust temp,
                      • Prompt?             add guardrails
                      • Data quality?       Prompt: rewrite system prompt
                           │                Data: fix ingestion pipeline
                           ▼
Step 4: Re-evaluate        Step 5: Measure delta
───────────────────        ─────────────────────
Run mlflow.evaluate()  ──► Compare groundedness %,
on same eval dataset       correctness %, latency
                           vs previous MLflow run
```

### Review App Data Flow (MLflow 3)

```
Developer creates labeling session
          │
          ▼
create_labeling_session(name, assigned_users, label_schemas)
          │
          ├──► Copies traces into session (originals untouched)
          │
          ▼
SME opens Review App URL (needs Databricks account, CAN_EDIT on experiment)
          │
          ▼
SME provides labels per schema (categorical / text / numeric / thumbs)
          │
          ▼
Assessment objects written to Trace.info.assessments
          │
          ├──► type="feedback"   → mlflow.search_traces(run_id=session.mlflow_run_id)
          │
          └──► type="expectation" → session.sync_expectations() → eval dataset (Delta/UC)
```

---

## Key Concepts

### Human vs Automated Evaluation — When Each Is Necessary

**What is it?** Automated evaluation uses LLM judges and deterministic metrics to score GenAI outputs at scale; human evaluation uses domain expert judgment for cases that automated metrics cannot reliably assess.

**How does it work mechanistically?** LLM judges can measure properties like groundedness, relevance, and safety using a secondary model, but they are trained on general knowledge and cannot encode domain-specific correctness. When a medical, legal, or financial AI app produces an output that is fluent and plausible but factually wrong in a domain-specific way, an LLM judge will typically pass it while a domain expert catches it immediately. The decision to use human review rests on a signal-to-cost ratio: automated metrics are cheap but have bounded precision; human labels are expensive but encode expert knowledge.

**Where does it appear in Databricks?** Databricks explicitly positions the Review App as a complement to `mlflow.evaluate()` LLM judges — both are surfaced in the same MLflow Experiment UI. The `databricks-agent` evaluator activates built-in judges (groundedness, relevance, safety, correctness) automatically; SME labels augment these when domain precision is required.

### MLflow Assessment Model — Feedback vs Expectation

**What is it?** An `Assessment` is the atomic unit of human feedback in MLflow. It is an object stored in `Trace.info.assessments` and has one of two types: `feedback` (qualitative judgment of what was produced) or `expectation` (definition of what should have been produced).

**How does it work mechanistically?** When an SME submits a label in the Review App, MLflow creates an `Assessment` object and attaches it to the trace. The `feedback` type stores a rating or comment about the actual output — it is read-only context for analysis and cannot be directly fed into `mlflow.evaluate()` as ground truth. The `expectation` type stores the correct answer — `expected_facts`, `expected_response`, `guidelines` — which can be synced to an evaluation dataset and used as inputs to LLM judges like the Correctness judge that requires ground truth.

**Where does it appear in Databricks?** Assessments are queried programmatically via `mlflow.search_traces(run_id=session.mlflow_run_id)` and accessed at `trace.info.assessments`. Expectation-type assessments are promoted to the evaluation dataset Delta table via `LabelingSession.sync_expectations(to_dataset="catalog.schema.table")`.

### The Databricks Review App

**What is it?** The Review App is a hosted web UI provided by Databricks (via the `databricks-agents` SDK and MLflow) that lets SMEs view traces, answer structured labeling questions, and submit assessments — without needing full Databricks workspace access.

**How does it work mechanistically?** When a developer calls `agents.deploy()`, a review app is automatically created and its URL is printed in the notebook output. Alternatively, `review_app.get_review_app()` retrieves the app tied to the current MLflow experiment. Labeling sessions are created with `my_app.create_labeling_session()` or `mlflow.genai.labeling.create_labeling_session()` (MLflow 3). A session copies a selected set of traces into an MLflow Run; the SME reviews those copies so originals are never mutated. The SME needs a Databricks account provisioned at the account level and `CAN_EDIT` permission on the MLflow experiment, but does not need workspace login for chat-with-the-bot mode.

**Where does it appear in Databricks?** The review app URL is available via `review_app.get_review_app().url`. Feedback on inference-table traces is also written to `{catalog}.{schema}.{model_name}_payload_assessment_logs_view`. In MLflow 3, the API moves to `mlflow.genai.label_schemas.create_label_schema()` and `mlflow.genai.labeling.create_labeling_session()`.

### Labeling Session Design and Annotation Workflow

**What is it?** A labeling session is a scoped unit of review work: a finite set of traces assigned to specific SMEs, governed by defined label schemas (the questions SMEs answer), and captured as a single MLflow Run.

**How does it work mechanistically?** Label schemas define the input type (categorical, numeric scale, or free-text) and the assessment type (feedback or expectation). Before labeling begins, rubric calibration is essential: all annotators must independently label a small gold set (5–10 examples), then compare results. Disagreements are resolved through discussion to align on the rubric definition. Inter-annotator agreement (IAA) is measured using Cohen's kappa for two annotators or Fleiss' kappa for three or more; kappa ≥ 0.6 is the minimum threshold for usable annotation data. When annotators disagree on a production trace, options include: majority vote, adjudication by a senior SME, or flagging the trace as ambiguous and excluding it from the eval dataset.

**Where does it appear in Databricks?** Label schemas are defined with `review_app.create_label_schema(name, type, title, input, instruction)`. Built-in schemas for common cases are `review_app.label_schemas.GUIDELINES` and `review_app.label_schemas.EXPECTED_FACTS`. Sessions are created with `my_app.create_labeling_session(name, agent, assigned_users, label_schemas)`.

### Converting SME Labels to Evaluation Datasets

**What is it?** An evaluation dataset is a Delta table in Unity Catalog containing rows of `(inputs, expectations)` pairs that can be passed directly to `mlflow.evaluate()` for systematic quality measurement across versions.

**How does it work mechanistically?** The path from SME label to eval dataset is: (1) SME submits an `expectation`-type label (e.g. `expected_facts = ["The policy requires 30-day notice"]`); (2) the Assessment is written to the trace; (3) the developer calls `my_session.sync_expectations(to_dataset="catalog.schema.my_eval_table")`; (4) the table gains a row where `inputs` is the original request JSON and `expectations` contains the SME-provided ground truth. This table is then readable by `spark.read.table()` and passable to `mlflow.evaluate(data=evals, model_type="databricks-agent")`. The Correctness LLM judge uses `expected_response` or `expected_facts` from the expectations field as the gold standard.

**Where does it appear in Databricks?** Datasets are created with `datasets.create_dataset("catalog.schema.table")` and backed by Delta tables in Unity Catalog. The schema includes `dataset_record_id`, `inputs`, `expectations`, `source`, and audit columns (`created_by`, `last_updated_by`).

### Active Learning and Feedback Prioritization

**What is it?** Active learning is the practice of strategically selecting which production traces to send to SMEs for labeling, rather than sending a random sample, to maximize the information value of each annotation.

**How does it work mechanistically?** Two primary strategies apply: (1) *Uncertainty sampling* — send traces where the LLM judge gave a borderline score (e.g. groundedness between 0.4 and 0.6) because those are the cases where automated metrics are most likely to be wrong. (2) *Diversity sampling* — ensure the selected traces cover the full distribution of query types and failure modes, not just the most frequent patterns. In practice, a combined strategy works best: filter by judge score range, then cluster by query embedding and sample one trace per cluster.

**Where does it appear in Databricks?** Traces can be filtered programmatically using `mlflow.search_traces()` with SQL-style filters on tag values, or by querying the inference table directly. The selected subset is then passed to `my_session.add_traces(traces)`.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `label_schemas` (in `create_labeling_session`) | Which questions SMEs see and what input type they use; determines whether assessments are `feedback` or `expectation` | Use `expectation` schemas whenever the output will feed back into `mlflow.evaluate()`; use `feedback` schemas for qualitative signals that won't become ground truth |
| `assigned_users` (in `create_labeling_session`) | Which Databricks account users receive `CAN_EDIT` permission on the experiment and `QUERY` permission on the endpoint | Set to specific email addresses for accountability and IAA measurement; use `["users"]` only for broad open feedback, never for curating eval datasets |
| `type` in `create_label_schema` | Whether the label is `"feedback"` (judgment of actual output) or `"expectation"` (ground truth definition) | Set to `"expectation"` for any label that represents the correct answer; set to `"feedback"` for preference ratings and comments |
| `InputCategorical` vs `InputText` vs numeric | The response format the SME sees in the UI | Use `InputCategorical` for agreement/binary labels (kappa calculable); use `InputText` for expected_response values where free text is needed |
| `enable_comment` in `create_label_schema` | Whether a free-text rationale box appears alongside the categorical answer | Always set to `True` for safety-critical or novel-domain reviews; free-text rationale enables root-cause diagnosis beyond the binary label |
| `sync_expectations(to_dataset=...)` target table | Which Unity Catalog Delta table receives the expectation-type assessments from a completed labeling session | Use the same target table for all sessions evaluating the same application; table accumulates ground truth over time |

---

## Worked Example: Requirement → Decision

**Given:** A healthcare documentation AI generates clinical note summaries. Automated groundedness scores average 0.72, but the clinical team reports that several summaries contain plausible-but-incorrect drug dosage statements that the LLM judge does not catch. The team has two clinicians who can spend 2 hours per week reviewing traces.

**Step 1 — Identify the goal:** Capture clinician judgment on drug-dosage correctness in a structured form that can be used to retrain LLM judges and build a permanent evaluation dataset for regression testing.

**Step 2 — Define inputs:** 50 production traces per week from the inference table, pre-filtered to cases where the automated groundedness score is between 0.5 and 0.8 (uncertainty band), ensuring the clinicians review the marginal cases where the automated metric is weakest.

**Step 3 — Define outputs:** Two types of Assessment per trace: (a) `feedback` label `dosage_correct: Yes/No` with required comment; (b) `expectation` label `expected_facts` listing the specific dosage facts that should appear in the summary.

**Step 4 — Apply constraints:**
- Clinicians are not Databricks workspace users — they need account-level SCIM provisioning only
- Labels must be individually attributable (no anonymization) to support IAA measurement
- Expectation labels must be in the `expected_facts` reserved schema key so the Correctness judge can consume them

**Step 5 — Select the approach:**
Create a custom `dosage_correct` feedback schema (categorical) and use the built-in `EXPECTED_FACTS` expectation schema; create weekly labeling sessions of 50 traces using `create_labeling_session(assigned_users=[clinician1, clinician2], label_schemas=["dosage_correct", "expected_facts"])`; run IAA on the `dosage_correct` labels to confirm kappa ≥ 0.6; sync expectations to `catalog.clinical.dosage_eval_dataset` weekly; run `mlflow.evaluate()` against this growing dataset after each model iteration to track correctness improvement. This is preferred over having clinicians fill spreadsheets because the Assessment objects are directly consumable by `mlflow.evaluate()` with no ETL step, and the Delta table retains full audit history.

---

## Implementation

```python
# Scenario: Set up weekly SME labeling pipeline for a RAG app where automated
# judges miss domain-specific factual errors (clinicians review dosage correctness).
# Constraint: SMEs are not workspace users — account-level SCIM access only.

import mlflow
from mlflow.genai.label_schemas import create_label_schema, InputCategorical, InputText
from mlflow.genai.labeling import create_labeling_session

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/clinical-notes-eval")

# Step 1: Define label schemas (done once, reused across sessions)
dosage_feedback = create_label_schema(
    name="dosage_correct",
    type="feedback",          # judgment of actual output — will NOT sync to eval dataset
    title="Are all drug dosages in the summary clinically correct?",
    input=InputCategorical(options=["Yes", "No", "Partially correct"]),
    instruction="Select 'No' or 'Partially correct' if ANY dosage is wrong or missing.",
    enable_comment=True,      # clinician must explain what was wrong
    overwrite=True,
)

expected_facts_schema = create_label_schema(
    name="expected_dosage_facts",
    type="expectation",       # ground truth — WILL sync to eval dataset
    title="List the correct dosage facts that should appear in the summary.",
    input=InputText(),
    overwrite=True,
)

# Step 2: Pull uncertain traces from this week's inference run
# (pre-filter to groundedness 0.5–0.8 where judge is least confident)
traces = mlflow.search_traces(
    experiment_names=["/Shared/clinical-notes-eval"],
    filter_string="tags.groundedness_score >= '0.50' AND tags.groundedness_score <= '0.80'",
    max_results=50,
)

# Step 3: Create labeling session — SMEs automatically get CAN_EDIT on experiment
session = create_labeling_session(
    name="dosage_review_week_28",
    assigned_users=["dr.smith@hospital.org", "dr.jones@hospital.org"],
    label_schemas=[dosage_feedback.name, expected_facts_schema.name],
)
session.add_traces(traces)
print(f"Share with SMEs: {session.url}")
```

```python
# Scenario: After SMEs complete labeling, sync expectations to eval dataset
# and run mlflow.evaluate() to measure correctness delta vs last week's model.

import mlflow
import pandas as pd

EVAL_DATASET_TABLE = "clinical_catalog.dosage_eval.ground_truth_v1"

# Sync expectation-type labels from completed session to Delta table
session.sync_expectations(to_dataset=EVAL_DATASET_TABLE)

# Load the cumulative eval dataset
evals = spark.read.table(EVAL_DATASET_TABLE)

# Define the agent under test
@mlflow.trace(span_type="AGENT")
def clinical_notes_agent(request):
    # ... your RAG agent invocation here ...
    pass

# Run evaluation — Correctness judge uses expected_dosage_facts as ground truth
with mlflow.start_run(run_name="post_fix_eval_week_28") as run:
    results = mlflow.evaluate(
        model=clinical_notes_agent,
        data=evals,
        model_type="databricks-agent",
    )
    correctness_pct = results.metrics.get(
        "response/llm_judged/correctness/percentage", 0
    )
    print(f"Correctness (this run): {correctness_pct:.1%}")
    print(f"MLflow run ID: {run.info.run_id}")
```

```python
# Anti-pattern: Collecting SME feedback into a shared spreadsheet and manually
# converting it to an eval set before each test run.

# WHY IT BREAKS:
# 1. No audit trail — you cannot tell which clinician labeled which trace.
# 2. Manual ETL is error-prone — copy-paste corrupts expected_facts values.
# 3. Assessments are not attached to traces — you lose the link between the
#    label and the original inputs/retrieved context.
# 4. You cannot measure inter-annotator agreement without per-annotator records.
# 5. The spreadsheet is not versioned with the model — you cannot reproduce
#    which eval set was used for a given MLflow run.

# BROKEN APPROACH (do not do this):
sme_feedback = {
    "trace_id_001": {"correct": "No", "comment": "Dosage was 10mg not 100mg"},
    "trace_id_002": {"correct": "Yes", "comment": ""},
}
# Then manually builds a DataFrame and calls mlflow.evaluate() — loses provenance

# CORRECT APPROACH: Use Assessment objects via create_labeling_session()
# (see first snippet above). MLflow stores assessments on traces with:
# - trace_id linkage
# - user attribution (source.human.user_name)
# - timestamp (create_time)
# - type-safe schema (expectation vs feedback)
# sync_expectations() produces a Delta table with full audit columns automatically.
```

---

## Common Pitfalls & Misconceptions

- **Treating all SME labels as ground truth** — Beginners often sync both `feedback` and `expectation` labels to the eval dataset, assuming any human label is authoritative. Only `expectation`-type assessments encode correct answers; `feedback`-type assessments are qualitative judgments about a specific output and will cause LLM judges to misfire if used as ground truth because they may reference a particular response that won't appear in future runs.

- **Skipping labeler calibration and assuming annotators agree** — Teams often launch a labeling session with a rubric document and assume SMEs will interpret it consistently. Without running a calibration pass (independently labeling 5–10 gold examples and reconciling disagreements), kappa values routinely fall below 0.4, producing noisy labels that degrade the eval dataset quality rather than improving it.

- **Assigning all production traces to SMEs instead of prioritizing** — A natural instinct is to maximize coverage by sending every uncertain trace to reviewers. SME time is the bottleneck; sending low-information-density traces (those the LLM judge already scores with high confidence) wastes budget and burns out experts. The correct model is active learning: target the uncertainty band where automated metrics are weakest.

- **Conflating the review app URL with the labeling session URL** — The review app URL (tied to the MLflow experiment) is permanent; the labeling session URL changes with each new session. Beginners bookmark the session URL and share it with SMEs for future reviews, causing confusion when a new session is created. Always share `my_app.url` for bookmarking and `my_session.url` only for immediate navigation.

- **Assuming agents.deploy() feedback lives in MLflow** — Feedback collected via the inference table chat mode is written to `{catalog}.{schema}.{model_name}_payload_assessment_logs_view`, not directly to the MLflow Traces UI. Beginners look for it in the MLflow experiment and find nothing. The correct model is that inference-table feedback and labeling-session feedback are separate surfaces that can be joined by `trace_id`.

---

## Key Definitions

| Term | Definition |
|---|---|
| Assessment | An atomic MLflow object stored in `Trace.info.assessments` that captures a single human label; has type `feedback` (judgment of output) or `expectation` (ground truth definition) |
| Labeling Session | A finite, scoped MLflow Run containing copies of traces assigned to specific users for structured review using defined label schemas; originals are not mutated |
| Label Schema | A definition of the question type and response format (categorical, text, numeric) that SMEs see in the Review App; also specifies whether the result is `feedback` or `expectation` |
| Review App | A Databricks-hosted web UI backed by MLflow that allows SMEs to view traces and submit assessments without requiring Databricks workspace login |
| Inter-Annotator Agreement (IAA) | A statistical measure of labeling consistency across annotators; Cohen's kappa for two annotators, Fleiss' kappa for three or more; kappa ≥ 0.6 is the minimum threshold for annotation reliability |
| Evaluation Dataset | A Delta table in Unity Catalog containing `(inputs, expectations)` rows accumulated from synced expectation assessments; consumed by `mlflow.evaluate()` as ground truth |
| Feedback Assessment | An assessment of type `feedback` that records qualitative judgment (thumbs up/down, rating, comment) about an actual output; not usable as ground truth in eval reruns |
| Expectation Assessment | An assessment of type `expectation` that records the correct answer for a given input; syncable to an evaluation dataset and consumable by LLM judges like Correctness |
| `sync_expectations()` | A method on `LabelingSession` that promotes expectation-type assessments into a named Delta table evaluation dataset |
| Uncertainty Sampling | An active learning strategy that sends to SMEs only traces where the automated judge score falls in the uncertain mid-range (e.g. 0.4–0.6), maximizing information value per annotation |

---

## Summary / Quick Recall

- Assessments are the atomic unit of SME feedback in MLflow — they attach to traces and have two types: `feedback` (opinion) and `expectation` (ground truth).
- Only `expectation`-type labels can be synced to an evaluation dataset via `sync_expectations()` and used by `mlflow.evaluate()`.
- The Review App is deployed automatically with `agents.deploy()` and requires only account-level Databricks provisioning (not workspace login) for SMEs doing chat-with-the-bot.
- Labeling sessions copy traces — they never mutate the original trace — so session labels and production labels are always separated.
- Calibration and inter-annotator agreement (kappa ≥ 0.6) are non-negotiable prerequisites for creating a reliable eval dataset from SME labels.
- Active learning (uncertainty sampling + diversity sampling) is the correct strategy for selecting which traces to review; random sampling wastes SME time.
- The iterative loop is: SME flags failure → diagnose root cause (retrieval / generation / prompt) → fix → run `mlflow.evaluate()` against the same dataset → compare metrics delta across MLflow runs.

---

## Self-Check Questions

1. What is the key structural difference between an `Assessment` of type `feedback` and one of type `expectation` in MLflow?

   <details><summary>Answer</summary>

   A `feedback` assessment records a qualitative judgment about what the app *actually produced* — for example, a thumbs-down rating or a comment. It is read-only context and cannot be promoted to an evaluation dataset. An `expectation` assessment records the *correct answer that should have been produced* — for example, `expected_facts` or `expected_response`. Expectation assessments can be promoted to an evaluation dataset via `sync_expectations()` and consumed by `mlflow.evaluate()` as ground truth for the Correctness judge. The common distractor is thinking that any human label can be used as ground truth — only `expectation` assessments encode ground truth; `feedback` assessments encode preferences about a specific output that will not recur.

   </details>

2. Your team deployed a RAG agent using `agents.deploy()`. A product manager wants to let five domain experts review agent responses and flag incorrect answers. The experts do not have Databricks workspace accounts. What is the minimum setup required to enable their review?

   <details><summary>Answer</summary>

   The `agents.deploy()` call automatically creates a Review App and prints its URL. For the "chat with the bot" mode, domain experts do not need workspace access — they only need to be provisioned in the Databricks account (via SCIM or manual account-level registration). The developer then calls `agents.set_permissions(model_name=..., users=[list_of_emails], permission_level=agents.PermissionLevel.CAN_QUERY)` to grant query access to the model serving endpoint. The distractor is assuming workspace access is required — it is only required for labeling sessions, not for the chat-with-the-bot review mode.

   </details>

3. **Which TWO** of the following actions are required to make SME-provided ground truth consumable by `mlflow.evaluate()`?
   - A. Call `mlflow.log_param("expected_response", sme_answer)` on the session run
   - B. Define label schemas with `type="expectation"` for the ground truth fields
   - C. Call `session.sync_expectations(to_dataset="catalog.schema.eval_table")` after the session is complete
   - D. Set `enable_comment=True` on all label schemas so free-text rationales are captured
   - E. Tag each trace with `ground_truth=True` in the MLflow UI before labeling

   <details><summary>Answer</summary>

   **B and C** are correct. B is required because only assessments with `type="expectation"` are promoted to the evaluation dataset — `feedback`-type assessments are not synced. C is required because the sync does not happen automatically; the developer must call `sync_expectations()` after the labeling session is complete to write the expectation data to the Delta table. A is wrong because `mlflow.log_param` logs to the run metadata, not to the trace assessments, and is not readable by `mlflow.evaluate()`. D is wrong because `enable_comment=True` adds a rationale text box but does not affect whether the label is ground-truth-usable. E is wrong because tagging traces as `ground_truth=True` is not a mechanism in the MLflow assessment model.

   </details>

4. A team runs a labeling session and calculates Cohen's kappa of 0.28 between their two SME annotators on a binary "response is correct" label. What does this indicate, and what is the correct remediation?

   <details><summary>Answer</summary>

   A kappa of 0.28 indicates *poor* inter-annotator agreement (below the 0.4 threshold for "fair" agreement and well below the 0.6 minimum for reliable annotation data). This means the two SMEs are interpreting the rubric differently, and labels from this session should not be promoted to the evaluation dataset because they introduce noise rather than signal. The correct remediation is labeler calibration: both SMEs independently label a shared gold set of 10–15 representative examples, then meet to reconcile their disagreements, identify the ambiguous rubric criteria, and update the rubric definition. Only after kappa rises to ≥ 0.6 on a recalibration set should production labeling resume. The common distractor is averaging the two annotators' labels — this preserves the disagreement noise rather than resolving it.

   </details>

5. A team must decide between two approaches for selecting production traces for SME review: (A) send a random sample of 100 traces per week, or (B) send the 100 traces where automated groundedness scores fall between 0.45 and 0.65. Under what conditions does approach B provide more value, and when might approach A be preferable?

   <details><summary>Answer</summary>

   **Approach B (uncertainty sampling) provides more value** when the goal is to improve or calibrate LLM judges, because it focuses SME attention on the cases where the automated metric is least reliable — the SME label has maximum corrective impact. It also finds systematic blind spots in the automated metric more efficiently than random sampling. **Approach A (random sampling) is preferable** when the goal is to build a statistically representative evaluation dataset that reflects the true distribution of production queries — not just the hard cases. If the eval dataset consists only of borderline traces, the resulting quality metrics will be artificially pessimistic compared to the real production distribution. In practice, best-of-both is a stratified sample: uncertainty-sampled traces for calibrating judges, plus a separate random stratum for representative coverage.

   </details>

---

## Further Reading

- [Human feedback in MLflow](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/human-evaluation.html) — *verified 2026-07-16* — Overview of the assessment data model, feedback types, and three collection paths (developer, domain expert, end user)
- [Use the review app for human reviews (MLflow 2)](https://docs.databricks.com/aws/en/agents/agent-evaluation/review-app.html) — *verified 2026-07-16* — Full API reference for `create_labeling_session`, `sync_expectations`, label schemas, and dataset management
- [Collect feedback and expectations by labeling existing traces (MLflow 3)](https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/expert-feedback/label-existing-traces.html) — *verified 2026-07-16* — Step-by-step MLflow 3 workflow for `create_label_schema`, `create_labeling_session`, and converting to evaluation datasets
- [Run an evaluation and view the results (MLflow 2)](https://docs.databricks.com/aws/en/agents/agent-evaluation/evaluate-agent.html) — *verified 2026-07-16* — `mlflow.evaluate()` API, evaluation set schema (`expected_facts`, `expected_response`, `guidelines`), and comparing results across runs
