# SME Feedback Loops — Interview Prep

**Section:** 07 — Evaluation and Monitoring | **Role target:** Senior ML/AI Engineer, GenAI Platform Engineer, MLOps Engineer

---

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is an Assessment in MLflow, and what are the two types? | `Assessment` is stored in `Trace.info.assessments`; type `feedback` = judgment of actual output; type `expectation` = ground truth definition; only expectations sync to eval datasets; both are attached to specific traces with attribution | Saying "it's just a thumbs up or down" — misses the `expectation` type and the Delta table sync pathway entirely |
| How does the Databricks Review App work mechanically? | Deployed automatically with `agents.deploy()` or manually via `review_app.get_review_app()`; labeling sessions copy traces (originals immutable); SMEs need account provisioning but not workspace login for chat mode; assessments are stored in MLflow Traces as Assessment objects | Saying SMEs need full Databricks workspace access — they don't for chat-with-the-bot; confusing the review app URL (permanent, per experiment) with the labeling session URL (per session) |
| What is the path from SME label to `mlflow.evaluate()` input? | Define `expectation`-type label schema → create labeling session → SME submits label → Assessment stored on trace → `sync_expectations(to_dataset=...)` → Delta table in Unity Catalog → `spark.read.table()` → `mlflow.evaluate(data=evals, model_type="databricks-agent")` | Stopping at "the SME labels the trace" — interviewers want to hear the full pipeline including `sync_expectations` and the Delta table |
| Why is inter-annotator agreement (IAA) necessary before building an eval dataset? | Labels from two annotators with kappa < 0.6 introduce noise that degrades eval dataset quality; disagreement signals rubric ambiguity, not annotator error; calibration (gold set + reconciliation) must precede production labeling | "We just use the majority vote" — doesn't fix the underlying rubric problem; kappa below 0.4 means the labels are nearly random |
| What is the difference between uncertainty sampling and random sampling in this context? | Uncertainty sampling targets traces where automated judge confidence is low (score 0.4–0.65) — maximizes information value per SME annotation; random sampling targets representativeness of the full distribution; use uncertainty sampling to calibrate judges, random sampling to build performance benchmarks | "We send all failing traces to the SME" — that's filtering, not sampling; does not help with the marginal cases where judges are weakest |

---

## Applied / Scenario Questions

**Q:** Your RAG agent has been in production for 4 weeks. Automated groundedness scores average 0.78, but account managers are escalating 3–5 incorrect responses per week involving competitor pricing data that the LLM judge does not flag. You have two domain experts who can spare 3 hours per week. Design the end-to-end SME feedback workflow.

**Strong answer framework:**
- **Schema design first:** Create two label schemas: (1) `pricing_correct` — `type="feedback"`, categorical (Yes / No / Partially), with `enable_comment=True` to capture what was wrong; (2) `expected_pricing_facts` — `type="expectation"`, InputText, to record the correct facts. The feedback schema captures weekly quality signal; the expectation schema builds the eval dataset.
- **Trace selection:** Filter inference table traces to groundedness 0.5–0.75 (uncertainty band for this agent) and keyword-match on competitor names in the request. This ensures SME time targets the cases that matter and that the judge misses.
- **Labeling session setup:** `create_labeling_session(name="pricing_review_wk_X", assigned_users=[expert1, expert2], label_schemas=["pricing_correct", "expected_pricing_facts"])`. Add the filtered traces. Share `session.url`.
- **IAA gate:** After the first session, calculate Cohen's kappa on `pricing_correct` labels where both experts reviewed the same trace. If kappa < 0.6, run a calibration session before proceeding.
- **Eval dataset sync:** `session.sync_expectations(to_dataset="catalog.sales.pricing_eval_v1")` after each weekly session. The Delta table accumulates ground truth over time.
- **Iterative loop:** After each model change, run `mlflow.evaluate(data=spark.read.table("...pricing_eval_v1"), model_type="databricks-agent")` and compare `correctness/percentage` delta across MLflow runs.
- **Show tradeoff awareness:** Acknowledge that `feedback` labels (pricing_correct) do not enter `mlflow.evaluate()` directly — they are used for trend analysis and to flag which traces to diagnose. Only `expectation` labels provide eval dataset signal.

---

**Q:** A colleague says: "We ran a labeling session, got 200 thumbs-down labels, and synced them to our eval dataset. Now our `mlflow.evaluate()` correctness scores are much lower than expected." What went wrong?

**Strong answer framework:**
- Identify the type mismatch: thumbs-down labels are `feedback`-type assessments — they record a reviewer's opinion about a specific output. They are not `expectation`-type assessments.
- `sync_expectations()` only syncs `expectation`-type labels. If the session only used a `feedback` schema (thumbs up/down), there is nothing to sync — the Delta table is likely empty or contains incorrect data.
- Even if feedback labels were incorrectly synced, an LLM Correctness judge needs `expected_facts` or `expected_response` to compare against — not a binary vote. A thumbs-down tells you something was wrong but not what the right answer was.
- Remediation: redesign the label schema to include `type="expectation"` for an `expected_response` or `expected_facts` field; re-run the session so SMEs provide both the judgment (feedback) and the correct answer (expectation).

---

## System Design / Architecture Questions

**Q:** Design a continuous SME feedback system for a 24/7 enterprise legal document AI. The system must ingest production failures, route selected traces to two specialist attorneys for weekly review, and ensure eval dataset quality does not degrade over time.

**Approach:**
1. **Clarify requirements:** Volume of traces (how many per day?), number of attorneys, their Databricks account status, review SLA, and which quality dimensions matter most (correctness of legal citations, appropriate disclaimers, clause accuracy).
2. **Propose structure:**
   - Inference table captures all production traces with MLflow Tracing enabled.
   - Weekly automated job: query inference table, filter to groundedness score 0.4–0.7 and legal-domain keyword presence, sample up to 60 traces, split 30 per attorney.
   - Label schemas: `legal_accuracy` (feedback, categorical), `expected_legal_facts` (expectation, InputText), `missing_citations` (feedback, categorical).
   - IAA monitoring job: for traces reviewed by both attorneys, compute kappa weekly and alert if below 0.6 — this is the canary for rubric drift.
   - `sync_expectations()` to `catalog.legal.eval_dataset_v1` after each session.
   - Monthly `mlflow.evaluate()` run on the cumulative dataset with correctness and groundedness metrics, logged to a dedicated MLflow experiment; dashboard via Delta table metrics extraction.
3. **Justify choices and name tradeoffs:**
   - Splitting traces between attorneys (vs. both reviewing all): halves SME burden but requires overlap subset for IAA calculation — trade coverage for agreement measurement on 10% overlap.
   - Delta table accumulation vs. rolling window: accumulation gives increasing statistical power over time; risk is eval dataset distribution drift if the app domain expands. Mitigation: tag each record with `app_version` so you can slice by version.
   - `feedback` labels are not synced to the eval table but are logged to a separate monitoring Delta table for trend analysis and root-cause triage — they have business value even though they don't feed `mlflow.evaluate()`.

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **Assessment** — when describing the specific MLflow object that stores a human label; signals you know the data model
- **Inter-annotator agreement / Cohen's kappa** — when discussing annotation reliability; signals you know evaluation methodology, not just tools
- **Expectation-type label** — contrasted with feedback-type; signals you understand which labels are eval-dataset-safe
- **Uncertainty sampling** — when discussing how to select traces for review; signals active learning awareness
- **`sync_expectations()`** — the specific API call that closes the loop from label to eval dataset; signals hands-on familiarity
- **Labeling session run ID** — knowing that labels are stored under an MLflow run tied to the session; signals understanding of the data lineage
- **Rubric calibration** — the process of aligning annotators on a gold set before production labeling; signals annotation methodology maturity
- **`_payload_assessment_logs_view`** — the inference table view where deployment-mode feedback lands; signals awareness of where data actually lives in production

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"Just collect thumbs up/down"** — ignores the expectation/feedback type distinction and the pipeline that makes human labels actionable
- **"The SME reviews the outputs"** — too vague; interviewers want to hear "labeling session," "label schema," "Assessment object," and "sync to eval dataset"
- **"We export to Excel and clean it up"** — signals no MLflow integration; raises data lineage and reproducibility red flags
- **"The review app is for chatting with the bot"** — conflates chat mode with the full labeling session capability; signals surface-level familiarity only
- **"Human evaluation is a QA step before release"** — frames it as an event rather than an ongoing system; immediately signals maturity gap
- **"IAA is a research thing, we don't need it in production"** — signals unfamiliarity with annotation reliability requirements; any evaluator who has shipped a labeled dataset will push back hard on this

---

## STAR Answer Frame

**Situation:** A customer support AI at a SaaS company was returning product pricing information that was 6–18 months out of date. Automated groundedness scores were 0.81 (good by automated standards), but account managers were manually correcting AI responses before sharing with customers — a signal the automated metric was not catching.

**Task:** I was responsible for designing an SME feedback system that would (a) quantify the pricing-accuracy failure rate with human ground truth, (b) create an eval dataset that would catch regressions when the knowledge base was updated, and (c) do this with only 4 hours per week of SME time from two product specialists.

**Action:** I designed two label schemas: a `pricing_correct` feedback schema (categorical, with mandatory comment) and an `expected_pricing_facts` expectation schema (freetext). I filtered inference table traces to the uncertainty band (groundedness 0.45–0.70) and keyword-matched on product SKU names to focus on the failure mode. I ran a 10-trace calibration session with both specialists before production labeling and measured kappa = 0.41 — below threshold. After a 30-minute alignment call to define "correct pricing" in terms of four specific data fields, we re-calibrated and reached kappa = 0.79. I then ran 3-week labeling sessions of 40 traces each, syncing expectations to a Delta eval table after each session. After the knowledge base update, I ran `mlflow.evaluate()` against the accumulated 120-trace dataset and confirmed correctness improved from 0.61 to 0.88.

**Result:** The eval dataset reduced pricing-related escalations by 74% over the following quarter because every knowledge base update was now tested against the SME-defined ground truth before deployment. The `pricing_correct` feedback labels, while not used in `mlflow.evaluate()`, revealed that 38% of the original failures were retrieval failures (wrong chunk returned) vs. generation failures — enabling the team to prioritize re-indexing the pricing catalog over prompt tuning.

---

## Red Flags Interviewers Watch For

- Candidate cannot distinguish `feedback` from `expectation` assessment types when asked follow-up questions about data flow into `mlflow.evaluate()`.
- Candidate describes SME review as "we share the agent link and collect feedback" with no mention of labeling sessions, schemas, or Assessment objects — signals they have used the chat UI but have not built an evaluation pipeline.
- Candidate cannot name a specific kappa threshold or explain why low kappa invalidates an eval dataset — signals no annotation methodology background.
- Candidate says "we retrain the model on SME feedback" — RAG agents in Databricks are typically not retrained on human labels; human labels feed the *evaluation* system, not the model directly. This confusion signals a classification/fine-tuning mental model being incorrectly applied to GenAI.
- Candidate has no answer for how to handle disagreement between annotators — a senior candidate should immediately mention calibration, adjudication by a senior SME, or exclusion of ambiguous examples.
- Candidate cannot explain what happens if you skip `sync_expectations()` — the labels exist as Assessment objects on traces but never enter the Delta eval table and cannot be used by `mlflow.evaluate()`.
