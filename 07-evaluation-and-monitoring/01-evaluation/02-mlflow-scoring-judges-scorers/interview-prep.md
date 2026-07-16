# MLflow Scoring, Judges, and Scorers — Interview Prep

**Section:** 07 Evaluation and Monitoring | **Role target:** Senior GenAI Engineer, ML Platform Engineer, Solutions Architect (AI/ML)

---

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What does `mlflow.evaluate()` do, and what does it return? | (1) Orchestrates evaluation: optionally calls the model, then runs evaluators. (2) Logs results to an MLflow run. (3) Returns an `EvaluationResult` with `.metrics` dict and `.tables["eval_results"]` DataFrame. | "It runs the model and checks accuracy" — misses the evaluator plugin concept, the logging, and the dual return structure. |
| What activates the Databricks LLM judge suite inside `mlflow.evaluate()`? | `model_type="databricks-agent"` loads the Databricks Agent Evaluation plugin. Without it, only statistical evaluators run. | "You need to import `databricks-agents`" — installation is a prerequisite but not the activation mechanism. |
| Explain the difference between `extra_metrics` and `evaluator_config`. | `extra_metrics` adds new judge objects (custom `EvaluationMetric` instances). `evaluator_config` configures existing evaluators — e.g., restricting which built-in judges run, injecting global guidelines. They serve orthogonal purposes. | Conflating the two — e.g., "you put custom judges in evaluator_config". |
| Which built-in judges require ground-truth labels, and which do not? | Require ground truth: `correctness`, `context_sufficiency`, `document_recall`. Do not require: `groundedness`, `relevance_to_query`, `safety`, `chunk_relevance`, `guideline_adherence` (with global guidelines). | Saying all judges need ground truth — a common gap that disqualifies you from discussing label-free monitoring use cases. |
| How does `make_genai_metric_from_prompt` convert a prompt into a binary yes/no rating? | LLM is asked to emit 1–5; MLflow wraps the prompt with formatting instructions. Score > 3 → `yes` (pass); ≤ 3 → `no` (fail). The `assessment_type` field controls per-row vs per-chunk invocation. | "The LLM outputs yes or no directly" — misses the numeric scoring and threshold mapping, which matters for calibration. |

---

## Applied / Scenario Questions

**Q:** Your RAG chatbot has been running in production for two weeks. Groundedness is 91%, but users are reporting incorrect factual answers. How do you diagnose the root cause using Databricks Agent Evaluation?

**Strong answer framework:**
- Start by noting that high groundedness + incorrect answers points to a **retrieval problem**, not a generation problem: the agent is faithfully using what it retrieved, but the retrieved context is wrong or incomplete.
- Add `expected_facts` columns to a sample of the failing rows and run `correctness` and `context_sufficiency` judges together. Low `context_sufficiency` confirms the retriever is not surfacing the right docs.
- Inspect the `context_sufficiency/rationale` column for specific explanations of what information is missing from retrieved context.
- Remediation: improve chunking strategy, re-index with a better embedding model, or add more source documents.
- Show tradeoff awareness: escalating from groundedness to context_sufficiency to correctness is a multi-layer diagnosis; each layer requires progressively more ground-truth curation effort.

---

**Q:** A colleague argues that you should use `evaluators="default"` with `model_type="question-answering"` for your GenAI chatbot evaluation because it is simpler. What is wrong with this approach?

**Strong answer framework:**
- `model_type="question-answering"` activates statistical metrics (exact match, ROUGE, toxicity via HuggingFace evaluate library) — these are token-level comparisons, not semantic quality checks.
- For conversational agents with open-ended answers, exact match and ROUGE are essentially useless — a correct answer worded differently from the reference will score near zero.
- The correct approach for GenAI applications is `model_type="databricks-agent"` which uses LLM judges for semantic quality assessment.
- Acknowledge the legitimate use of `"question-answering"`: it is appropriate for extractive QA tasks where the expected answer is a verbatim span from context, not for generative responses.

---

**Q:** You need to evaluate whether your agent's responses stay within the scope of your product documentation (no off-topic answers) and never mention competitor products. How would you implement this?

**Strong answer framework:**
- Use **named `global_guidelines`** in `evaluator_config`: `{"on_topic": ["The response must only answer questions about [Product]. Off-topic questions must be politely declined."], "no_competitor": ["The response must not mention competitor products by name."]}`.
- Named dict structure is important — it creates two separate assessment columns (`guideline_adherence/on_topic/rating`, `guideline_adherence/no_competitor/rating`) for independent tracking.
- Alternative: use `make_genai_metric_from_prompt` for more precise control (e.g., enumerating competitor names in the prompt, controlling the scoring scale).
- Trade-off: guidelines are simpler to maintain; custom prompt judges offer more precision at the cost of more authoring effort.

---

## System Design / Architecture Questions

**Q:** Design a weekly evaluation cadence for a deployed RAG agent that measures quality regression between releases. What components do you need and how do they connect?

**Approach:**

1. **Clarify requirements:** Establish what "quality regression" means — a > 5% drop in any key judge score? A new failure mode appearing? Define thresholds and responsible owner.

2. **Propose structure:**
   - **Eval dataset (Unity Catalog Delta table):** 50–100 curated rows with `request`, `expected_facts`, `retrieved_context` (optional); versioned as `catalog.schema.eval_set_v3`. Pull 30% from production traces weekly; keep 70% stable for longitudinal comparison.
   - **MLflow Experiment:** one experiment per application; one run per weekly eval named `"hr-bot-YYYY-MM-DD"`. Use consistent run naming for programmatic search.
   - **Evaluation job:** Databricks Workflow that runs `mlflow.evaluate()` on each new deployment candidate, logs results to the experiment, and appends aggregated metrics to a Delta table for dashboarding.
   - **Alert:** Lakeview dashboard with conditional alert on metric deltas > threshold; notify via Slack/PagerDuty via webhook.

3. **Justify choices and name tradeoffs:**
   - Running `mlflow.evaluate()` without the `model=` argument (scoring pre-generated responses) is faster and cheaper for CI gates; use `model=` only for full integration tests.
   - Running all 8 judges on every release is expensive; restrict to the 3–4 judges most critical for your use case using `evaluator_config["metrics"]`.
   - Synthetic eval dataset generation (`databricks.agents.synthesize`) can bootstrap coverage quickly but must be augmented with production traces to catch real-user failure modes.

---

## Vocabulary That Signals Expertise

Use these terms naturally — do not force them:

- **Assessment type (`ANSWER` vs `RETRIEVAL`)** — when discussing custom judge design; signals you understand per-chunk vs per-row judge invocation
- **Cohen's Kappa** — when discussing judge calibration quality; signals you know how judge accuracy is measured, not just assumed
- **Root-cause ordering** — when discussing how Databricks Agent Evaluation determines which judge is the primary failure cause (context_sufficiency → groundedness → correctness order)
- **Label-free evaluation** — the subset of judges that run without ground truth; signals you understand how to evaluate in production where curated labels are unavailable
- **`EvaluationResult.tables["eval_results"]`** — when describing how to drill into per-row failures; signals hands-on experience vs theoretical knowledge
- **`expected_facts` vs `expected_response`** — Databricks recommends `expected_facts` (minimal fact list) over `expected_response` (full answer string); signals familiarity with evaluation best practice documentation
- **Few-shot calibration** — passing `examples_df` in `evaluator_config` to improve judge accuracy for domain-specific ratings; signals awareness of judge tunability

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:

- **"The model grades itself"** — technically inaccurate; a *separate* judge model evaluates the agent's output; the agent and the judge are different systems
- **"BLEU/ROUGE score for chatbot quality"** — these are the pre-LLM-judge era metrics; using them for open-ended generative evaluation signals unfamiliarity with modern evaluation practice
- **"The eval set is just the training test split"** — evaluation datasets for production GenAI must reflect live user traffic and domain-specific quality criteria, not held-out training examples
- **"High groundedness means the answer is correct"** — groundedness and correctness are independent dimensions; a response can be grounded in (faithfully reproduce) wrong retrieved content
- **"Custom judges are too complex for production"** — `make_genai_metric_from_prompt` is a 5-parameter function; dismissing it signals unfamiliarity with the API

---

## STAR Answer Frame

**Situation:** I was the ML engineer responsible for a customer-facing knowledge-base chatbot. Two weeks after launch, the product team reported that 1 in 5 answers contained factually incorrect claims, even though our pre-launch eval showed 89% groundedness.

**Task:** Diagnose the quality regression and implement an evaluation pipeline that would catch similar regressions before production deployment.

**Action:** I exported 100 production traces from the MLflow Traces tab, manually labelled 40 of them with `expected_facts` lists (minimal facts required for a correct answer), and built an eval DataFrame combining these with the remaining 60 unlabelled traces. I then ran `mlflow.evaluate()` with `model_type="databricks-agent"` and discovered that `correctness` was only 48% on the labelled subset while `context_sufficiency` was 39%. Groundedness was still high — meaning the model was accurately reproducing what it retrieved, but the retriever was not surfacing the right documents. I re-indexed the knowledge base with a larger chunk size and re-ran the eval; correctness improved to 74%. I also added a custom `scope_adherence` judge using `make_genai_metric_from_prompt` to catch off-topic responses, and set up a weekly Databricks Workflow to run the eval automatically on each deployment.

**Result:** Quality regressions that previously reached production are now caught in CI. The correctness baseline is versioned in Unity Catalog alongside the eval dataset, and the team has a quantitative definition of "good enough to ship" for the first time.

---

## Red Flags Interviewers Watch For

- **Cannot name the difference between `model_type="databricks-agent"` and `model_type="question-answering"`** — fundamental to selecting the right evaluator for GenAI vs classical NLP tasks
- **Does not know which built-in judges require ground truth** — signals inability to design evaluation pipelines for production monitoring (where ground truth is usually unavailable)
- **Treats a single eval run's score as definitive** — LLM judge outputs have variability; experienced engineers track deltas and run multiple seeds for high-stakes comparisons
- **Conflates the agent model and the judge model** — these are architecturally separate systems; confusing them suggests surface-level familiarity with the evaluation framework
- **Cannot describe the schema of the eval dataset** — the exact column names (`request`, `response`, `retrieved_context`, `expected_facts`) are load-bearing; wrong column names cause silent judge deactivation
- **Has not accessed `evaluation_results.tables["eval_results"]`** — only looking at aggregated metrics misses the per-row rationales that make evaluation actionable
