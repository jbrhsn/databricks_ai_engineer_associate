# Evaluation vs. Monitoring: The GenAI Application Lifecycle — Interview Prep

**Section:** 04 Application Development | **Role target:** Senior ML Engineer, AI Platform Engineer, MLOps Engineer, Solutions Architect (GenAI)

---

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is the difference between offline evaluation and online monitoring for a GenAI application? | (1) Offline eval runs before deployment on a fixed golden dataset, produces pass/fail scores that gate model registration; (2) online monitoring runs asynchronously after deployment on live inference table rows without requiring ground-truth labels; (3) offline catches regressions on known patterns, online catches distribution shift and novel failures | Saying "monitoring is just watching latency and error rates" — fails to mention LLM judge quality scores, which are the primary signal for GenAI applications |
| What is an inference table and why must it be enabled before production traffic is sent? | (1) AI Gateway-enabled Delta Table in Unity Catalog that auto-captures every request payload, response payload, `execution_duration_ms`, `status_code`, `served_entity_id`, and MLflow traces for agents; (2) data written only for requests received *after* enablement — retroactive logging is impossible; (3) prerequisite for any post-deployment quality diagnosis | Describing it as "a log file" — misses the Unity Catalog governance, Delta format queryability, and the critical retroactive impossibility |
| How does data profiling (formerly Lakehouse Monitoring) detect quality drift in an LLM application? | (1) `InferenceLog` or `TimeSeries` profile runs on a schedule using serverless compute; (2) computes rolling statistical profiles (mean, std, percentiles) over inference table columns per time window; (3) compares against prior window or a `baseline_table_name`; (4) fires alerts when drift metrics exceed thresholds; (5) `slicing_exprs` enables per-topic breakdown | Confusing data profiling with application-level logging or MLflow tracing — these are separate systems; data profiling is a scheduled batch job over the Delta table |
| How do you run an A/B test between two agent versions in Databricks Model Serving? | (1) Add both model versions as `ServedModelInput` entries in the endpoint; (2) assign percentage weights via `traffic_config` routes summing to 100; (3) both versions log to the same inference table, differentiated by `served_entity_id`; (4) require ≥ 1,000 requests per arm and p < 0.05 before declaring a winner | Saying "deploy two separate endpoints" — misses the `traffic_config` mechanism and the requirement to compare versions on the same traffic distribution simultaneously |
| Describe the Evaluation → Monitoring feedback loop for a production RAG application. | (1) Monitoring surfaces low-quality clusters in inference table; (2) LLM judge rationale column identifies retrieval miss vs. generation failure; (3) failing examples enriched into golden eval set; (4) offline `mlflow.evaluate()` gates next model version; (5) new version deployed with A/B split; (6) loop repeats continuously | Describing the loop as "monitor → retrain" — skips the offline eval gate step and the root-cause analysis step, which are what prevent fixing the wrong thing |

---

## Applied / Scenario Questions

**Q:** A production RAG agent's average groundedness score drops from 0.87 to 0.61 over 10 days. No code changes were made. The latency and error rate dashboards show nothing unusual. Your on-call page fires at midnight. Walk through your diagnostic process.

**Strong answer framework:**

1. **Confirm the signal is real:** Query the data profiling drift metric table to confirm the drop is statistically significant and not noise from a small sample size. Check whether the drop is uniform across all request types or concentrated in a specific slice.

2. **Isolate by topic/distribution:** Run a SQL query against the inference table grouping by extracted topic keywords and computing average groundedness per group per day. If the drop is concentrated in one topic cluster (e.g., requests mentioning a recently launched product), this is query distribution shift — not model degradation.

3. **Read the judge rationale:** The LLM judge rationale column in the request logs table (`trace` or scored metric table) describes *why* each request failed. "No relevant chunks found" → retrieval miss. "Contradicts retrieved context" → generation failure. These require different fixes.

4. **Cross-check retriever output:** If retrieval miss is suspected, sample 20 failing requests and call the retriever directly to confirm returned chunk relevance scores are low or zero for the new topic.

5. **Communicate triage:** Escalation-worthy if groundedness < 0.6 and affecting > 10% of traffic. Fix path: knowledge base augmentation (retrieval miss) or prompt adjustment / model update (generation failure), each validated offline before deployment.

**Trade-off to show:** Explain why "roll back to the previous model version" is rarely the right first response — the model version didn't change, so the previous version would behave identically on the new query distribution.

---

**Q:** Your team is preparing to ship a new version of a customer-facing RAG agent. The previous version has been in production for 6 months. Design the quality gate that must pass before the new version receives any production traffic, and describe the rollout strategy after passing.

**Strong answer framework:**

1. **Offline eval gate:** Run `mlflow.evaluate(model_type="databricks-agent")` against the current golden eval set. Gate criteria: `agent/percent_grounded ≥ 0.85` AND `agent/percent_correct ≥ 0.80`. Block registration if either threshold is not met. Surface per-row rationales for any failures and address them before re-evaluating.

2. **Regression check against production history:** Pull 200 requests from the last 30 days of the inference table that had high groundedness under the old version. Run the new version on these requests and compare scores. A new version should not regress on requests the old version handled well.

3. **A/B rollout plan:** Start at 95% champion / 5% challenger. Collect ≥ 1,000 requests per arm (at typical traffic this takes 1–3 days). Compute the two-sample test on groundedness scores across arms. If challenger is statistically significantly better or equivalent (non-inferior at p < 0.05), promote to 50/50 and repeat. Final promotion to 100% challenger (now new champion) after full significance confirmation.

4. **Monitoring hook:** Ensure data profiling is monitoring the `served_entity_id` slice for the challenger throughout the A/B test. If challenger groundedness drops below the lower bound of the champion's confidence interval within the first 200 requests, abort the rollout immediately.

---

## System Design Question

**Q:** Design a monitoring system for a RAG application that handles 10,000 queries/day and must detect quality regressions within 24 hours of onset.

**Approach:**

1. **Clarify requirements:**
   - What constitutes a "quality regression"? (e.g., groundedness drops below 0.75, or 3+ percentage points from trailing 7-day baseline)
   - What is the acceptable false-positive rate for alerts? (determines significance threshold)
   - What is the escalation path when an alert fires?
   - Is per-topic breakdowns required, or global score only?

2. **Proposed architecture:**

   ```
   Model Serving Endpoint
     └── AI Gateway inference table (100% logging, Unity Catalog Delta)
           └── Data Profiling (InferenceLog profile)
                 ├── Granularity: 1-hour windows (enables <24h detection)
                 ├── baseline_table_name: golden eval set
                 ├── slicing_exprs: ["request_topic", "user_segment"]
                 └── notification_settings: on_failure → PagerDuty + Slack
   ```

   At 10,000 queries/day = ~420/hour, hourly windows give ~420 samples per window — enough for statistically stable averages. Daily windows would give 10,000 samples but 24+ hour lag, violating the SLA.

3. **Justify choices and trade-offs:**
   - **100% logging vs. sampling:** At 10,000/day, storage cost is manageable (each row is ~1–5 KB JSON, so ~50 MB/day). Sampling at 100% preserves every request for retrospective debugging. Revisit at 100,000/day where sampling to 10–20% may be warranted.
   - **InferenceLog vs. TimeSeries profile:** InferenceLog is preferred because it auto-creates per-`model_id` (version) slices and includes model quality metric tracking — useful for A/B comparisons without additional configuration.
   - **CDF on the inference table:** Enable change data feed on the payload table so data profiling only processes newly appended rows per refresh, not the full historical table. Critical for cost control as the table grows past 1 million rows.
   - **Baseline table:** Point `baseline_table_name` to the golden eval set. This answers "how far has production drifted from our expected distribution" — not just "is today different from yesterday." Window-over-window drift catches short-term spikes; baseline drift catches long-term accumulation.
   - **Alert threshold:** Alert when groundedness drops ≥ 3 percentage points from the 7-day trailing average AND the sample size exceeds 100 requests in the window (to avoid alerting on noise from the first few requests of the day).

4. **What this system does NOT catch (honest trade-off):**
   - Requests where both retrieval and generation succeeded but the answer was factually wrong in a way the LLM judge missed (judge accuracy is ~90%, not 100%).
   - Latency regressions caused by the model serving infrastructure — add a separate p99 latency alert via Databricks SQL on `execution_duration_ms`.

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:

- **Inference table** — the specific Databricks artifact, not "request logs" or "a database"
- **AI Gateway-enabled inference table** — signals awareness of the current (non-legacy) path
- **Data profiling / InferenceLog profile** — current nomenclature replacing the deprecated "Lakehouse Monitoring" API
- **`slicing_exprs`** — the data profiling parameter for per-topic quality breakdowns; shows you know configuration-level details
- **`served_entity_id`** — the join key for separating A/B versions in the inference table
- **Query distribution shift** — signals you distinguish between input-side and model-side causes of quality drops
- **LLM judge rationale column** — signals you have worked with the per-row explanation output, not just aggregate scores
- **Feedback loop** — signals you understand the system as a continuous improvement cycle, not a one-time release
- **`baseline_table_name`** — shows you know how to anchor drift measurement to an expected reference distribution
- **Change data feed (CDF)** — signals awareness of the incremental processing optimization for large inference tables

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:

- **"Lakehouse Monitoring"** — the product has been renamed; use "data profiling" (Unity Catalog data quality monitoring)
- **"The model degraded"** as the first explanation for a groundedness drop — signals you jump to model-side diagnosis without checking query distribution shift first
- **"We just watch latency and error rate"** — signals you are applying traditional ML monitoring patterns to an LLM application where those signals are insufficient
- **"We'll turn on logging if something goes wrong"** — signals you don't understand that inference tables cannot be enabled retroactively
- **"I ran eval before deployment so we're covered"** — signals you treat offline eval as a one-time activity rather than one component of a continuous loop
- **"Our golden set has 500 questions so we're thorough"** — signals you conflate eval set size with production coverage; the question is whether the set covers the current production distribution, not its absolute size
- **"`quality_monitors` API"** — the deprecated API; use `data_quality` from `databricks.sdk` and the current `WorkspaceClient().data_quality.create_monitor()` path

---

## STAR Answer Frame

**Situation:** I was on-call for a customer-facing RAG agent at a financial services company. The agent answered questions about regulatory compliance documents. Three weeks after a major product launch, the team lead got a Slack message from a customer account manager: a client's legal team had found factually incorrect answers in several agent responses about the new product's compliance posture. No alerts had fired. Latency and error metrics were normal.

**Task:** I was responsible for diagnosing the root cause, containing the damage, and implementing a fix that would prevent recurrence — all without taking the agent offline, because it was integrated into the client's daily workflow.

**Action:**
1. Queried the inference table (`prod_catalog.compliance_agent.product_rag_payload_request_logs`) for requests containing the new product name over the prior 14 days. Found 847 requests, average groundedness score 0.38 (compared to 0.84 for all other request types). The LLM judge rationale column was consistent: "Retrieved chunks do not contain information about [new product]. Response appears to be model inference, not grounded in retrieved context."

2. Confirmed the root cause: the vector index had not been updated with the new product's compliance documentation at launch. Users started asking about it immediately; the retriever returned zero relevant chunks; the model generated plausible-sounding but ungrounded answers. This was a retrieval coverage gap, not model degradation.

3. Immediate containment: added a system prompt instruction to explicitly state "I cannot find compliance documentation for [new product] — please consult the compliance team directly" when retrieved context contains no relevant chunks (groundedness guard).

4. Permanent fix: ingested the new product's 43-page compliance document into the vector index, added 20 synthetic Q&A pairs derived from the document to the golden eval set, ran `mlflow.evaluate()` — groundedness for new-product queries improved from 0.38 to 0.91. Registered new model version. Deployed as 95/5 A/B split. After 1,200 requests per arm confirmed statistical significance (p = 0.003), promoted to full traffic.

5. Process improvement: added `request_topic` as a `slicing_expr` on the data profiling monitor so any future product launches would generate per-topic groundedness metrics automatically. Established a checklist item requiring knowledge base seeding to be completed before a product goes into the agent's scope.

**Result:** The client was informed of the root cause and fix within 4 hours. The updated agent went live 48 hours after detection. Groundedness for new-product queries remained above 0.89 for the following 6 weeks. More importantly, when the next product launched 3 months later, the monitoring slices showed stable quality from day one — the process change worked.

---

## Red Flags Interviewers Watch For

1. **Conflating system metrics with quality metrics.** A candidate who describes their monitoring stack as "Datadog for latency, PagerDuty for error rate" without mentioning LLM judge scores signals they have not dealt with the specific failure mode of silent hallucination in production.

2. **No mention of inference tables or equivalent logging.** Saying "we'd investigate the issue if it came up" without describing how the request-response history would be captured and queryable is a red flag. Any production GenAI system requires a request log; the candidate should know it needs to be enabled proactively.

3. **Treating offline eval and online monitoring as the same thing.** Describing `mlflow.evaluate()` as "production monitoring" confuses pre-deployment quality gates with post-deployment regression detection. They use different data sources, operate at different lifecycle stages, and catch different failure modes.

4. **No acknowledgment of query distribution shift.** A candidate who jumps directly from "groundedness dropped" to "we need to retrain the model" without mentioning the possibility of input distribution shift demonstrates a gaps in their mental model of what can cause quality degradation.

5. **No statistical rigor for A/B decisions.** Saying "we ran a quick A/B test and the new version looked better" without mentioning sample size, statistical significance, or the high variance of per-row LLM judge scores signals the candidate may make premature promotion decisions that introduce regressions rather than prevent them.
