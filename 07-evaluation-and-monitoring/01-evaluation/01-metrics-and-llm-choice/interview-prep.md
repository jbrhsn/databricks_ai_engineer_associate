# Evaluation Metrics and LLM Choice — Interview Prep

**Section:** 07 — Evaluation and Monitoring | **Role target:** Senior GenAI Engineer, MLOps Engineer, AI Platform Engineer, Solutions Architect

---

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is the difference between groundedness and correctness as evaluation metrics? | Groundedness = is every claim in the response traceable to retrieved context (no hallucination)? Correctness = does the response match the expected facts (factual accuracy)? High groundedness + low correctness means retrieval surfaced wrong content. Requires different inputs: groundedness needs retrieved\_context; correctness needs expected\_facts or expected\_response. | Saying "they're basically the same thing, both check if the answer is right." This collapses two different failure modes and signals you haven't debugged a real RAG failure. |
| What biases affect LLM-as-judge reliability and how do you mitigate them? | Verbosity bias (longer = higher score), position bias (first option in pairwise wins), self-preference bias (model prefers its own outputs). Mitigations: use different judge family from application model, calibrate against human labels, measure Cohen's Kappa, run pairwise in both orderings. | Listing "hallucination" as a judge bias, or saying "just use a better model as judge." Neither addresses the systematic directional biases. |
| When would you use ROUGE vs. an LLM-as-judge metric? | ROUGE: regression smoke tests on templated or summarization outputs where reference answers are reliable and verbatim fidelity matters. LLM-as-judge: open-ended generation, Q&A, RAG, agents — any task where correct answers can be paraphrased and no single reference captures all valid outputs. | "ROUGE is objective so it's always better" — ROUGE is objective but measures string similarity, not semantic accuracy. |
| What does `model_type="databricks-agent"` do in `mlflow.evaluate()`? | Activates Databricks Agent Evaluation backend, which runs the full suite of built-in LLM judges (groundedness, chunk\_relevance, context\_sufficiency, relevance\_to\_query, correctness, safety, guideline\_adherence) based on which columns are present in the evaluation dataset. Also computes token counts and latency metrics from traces. | "It just sets the evaluation mode." Needs to explain which judges run, under what conditions, and how the judge selection is data-driven. |
| What is context sufficiency and when is it the most important metric to check first? | Context sufficiency asks: "Did the retriever surface documents containing enough information to produce the expected response?" It is the first metric to check when correctness is low — if context is insufficient, no generator improvement will fix the problem. Requires ground truth (`expected_facts`). | Jumping to generator improvement (prompt engineering, model upgrade) when the retriever is failing. Context sufficiency is the causal gate before correctness in the Databricks judge priority ordering. |

---

## Applied / Scenario Questions

**Q:** Your RAG application has a groundedness score of 92% but users are reporting factually wrong answers at a rate of ~25%. What is your diagnostic plan and what metrics do you add to your evaluation pipeline?

**Strong answer framework:**
- Recognize that high groundedness + low perceived accuracy points to **retrieval failure, not generation failure** — the generator is faithfully reproducing what the retriever surfaced, but the retriever is surfacing incorrect or incomplete content.
- Add `context_sufficiency` (requires ground truth: `expected_facts`) to test whether retrieved chunks actually contain the facts needed to answer correctly.
- Add `correctness` with `expected_facts` (not `expected_response`) to quantify factual error rate against minimal required facts.
- Add `chunk_relevance` to measure retrieval precision — what fraction of retrieved chunks are actually relevant to the query?
- Show tradeoff awareness: context sufficiency requires ground truth annotation, which has a cost. If annotation budget is constrained, `chunk_relevance` is a reference-free proxy for retrieval quality that can be run immediately.

---

**Q:** A team asks you to choose between `databricks-meta-llama-3-3-70b-instruct` and `databricks-claude-sonnet-5` as the backbone for a customer support agent that handles product returns. The agent must respond in under 1 second and the team has a cost budget of $0.005 per query. Walk through your decision process.

**Strong answer framework:**
- Start with hard constraints: identify latency SLA (<1 second end-to-end) and cost cap ($0.005/query). Frontier models like Claude Sonnet 5 cost significantly more per token than Llama-3.3-70B.
- Consider task complexity: product returns is a structured, well-scoped task (policy lookup + response generation) — does not require frontier-level reasoning capability.
- Llama-3.3-70B on Databricks Foundation Model APIs is a strong mid-tier choice: strong instruction following, 128K context window (sufficient for policy documents), lower per-token cost.
- Acknowledge the decision requires empirical validation: run both models on a representative eval set with `correctness` and `relevance_to_query` judges; choose the cheaper model if quality is within acceptable margin.
- Show tradeoff awareness: Claude Sonnet 5 may be justified if the team finds edge cases (complex multi-policy returns, emotional de-escalation) where it consistently outperforms on the eval set.

---

## System Design / Architecture Questions

**Q:** Design an evaluation pipeline for a RAG application that must be run in CI/CD before every code merge and also monitored continuously in production. What components do you need, how do they connect, and what shared infrastructure enables comparison between dev and prod quality?

**Approach:**

1. **Clarify requirements:** Offline eval must complete in <15 minutes for CI viability. Production monitoring must not block inference latency. Same metrics must be comparable across both environments.

2. **Propose structure:**
   - **Offline (CI):** Curated eval set (50–200 rows) stored as a Delta table in Unity Catalog. `mlflow.evaluate()` called in a Databricks notebook job triggered by merge request, writing results to an MLflow experiment. Threshold gates (e.g., groundedness > 90%, correctness > 85%) block merges on failure.
   - **Production (online):** MLflow 3 production monitoring with scheduled scorers applied to live inference traces. Same judge definitions as offline. Aggregate metrics written to a Delta table for trend dashboarding.
   - **Shared infrastructure:** MLflow experiment acts as the single record of quality history. Both offline runs and online monitoring scorers log to the same experiment, making temporal comparison direct.

3. **Justify choices and name tradeoffs:** Storing the eval set in Delta (not flat files) enables versioned, auditable evaluation sets with data lineage. Using `expected_facts` (not `expected_response`) reduces annotation burden for the curated set. The main tradeoff is that ground-truth-dependent judges (correctness, context\_sufficiency) cannot run in production monitoring without a real-time oracle — reference-free judges (groundedness, chunk\_relevance, relevance\_to\_query) are used for production monitoring, with ground-truth judges reserved for offline eval.

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:

- **Causal judge ordering** — when explaining that context\_sufficiency is checked before correctness in Databricks' root-cause logic, because retrieval failure causes generation failure, not the reverse
- **Cohen's Kappa** — when discussing judge calibration quality; signals you know how inter-rater agreement is actually measured
- **Verbosity bias / position bias** — when discussing LLM-as-judge reliability; signals you know the failure modes of the approach you're recommending
- **Retrieval precision vs. retrieval recall** — chunk\_relevance measures precision (of retrieved chunks, how many are relevant?); document\_recall measures recall (of all relevant documents, how many did we retrieve?); using both correctly signals depth
- **Expected facts vs. expected response** — articulating why minimal facts are preferred over full reference answers signals practical annotation experience
- **Provisioned throughput vs. pay-per-token** — when discussing production model deployment on Databricks; signals you understand the infrastructure decision behind model selection

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:

- **"Just use BLEU/ROUGE for RAG evaluation"** — signals unfamiliarity with why string-overlap metrics fail on open-ended generation; this was reasonable in 2019 for machine translation, not for RAG in 2025+
- **"The judge is objective"** — LLM judges are not objective; they have documented systematic biases; saying this signals you've never calibrated a judge pipeline
- **"We measure accuracy"** — without qualifying what accuracy means (precision? F1? human agreement? which judge?) this is a vague non-answer that interviewers will probe until you're stuck
- **"We'll fine-tune the judge model"** — Databricks' terms of service explicitly prohibit using LLM judge outputs to train, improve, or fine-tune an LLM; saying this signals you haven't read the evaluation documentation
- **"Bigger model = better results always"** — signals you've never run a cost/latency/quality tradeoff analysis; production engineers know that 8B models frequently match 70B models on well-defined tasks

---

## STAR Answer Frame

**Situation:** Our team deployed a RAG-based customer support assistant for a financial services client. After three months in production, the client reported that approximately 20% of responses contained incorrect fee information, despite our pre-release evaluation showing 95% accuracy on our test set.

**Task:** I was responsible for diagnosing why production quality was diverging from our eval-set performance and redesigning the evaluation pipeline to catch this class of failure before it reached users.

**Action:** I ran a targeted analysis using the Databricks callable judges SDK (`from databricks.agents.evals import judges`) on 50 production traces where users had reported wrong fee information. I found that `chunk_relevance` scored 61% on those traces — meaning nearly 40% of retrieved chunks were irrelevant to the specific fee query — while `groundedness` was 94%. This confirmed the generator was faithfully reproducing retrieved content, but the retriever was surfacing outdated or wrong-category fee schedules. I then added `context_sufficiency` with a ground-truth eval set of 100 fee questions to our CI pipeline and set a merge-blocking threshold of 90%. I also worked with the product team to build `expected_facts` annotations for the 100 most common fee query types, which required about 8 hours of SME time rather than the 40+ hours a full reference-answer approach would have required.

**Result:** After fixing the retrieval configuration (re-chunking fee schedule documents by product category and adding metadata filters), `context_sufficiency` rose from 71% to 96% on the eval set. Production incorrect-fee-information reports dropped by ~75% in the following month. The new CI gate also caught two subsequent retrieval regressions in staging before they reached production.

---

## Red Flags Interviewers Watch For

- **Proposing to evaluate a RAG system with only a single metric** — signals you haven't thought through the pipeline stages; every experienced practitioner knows retrieval and generation must be evaluated separately.
- **Inability to distinguish between reference-based and reference-free evaluation** — should be immediate; if you need to think for more than a few seconds, it signals surface-level familiarity.
- **Claiming LLM-as-judge is "unbiased" or "objective"** — immediate red flag; documented biases (verbosity, position, self-preference) are well-known in the literature.
- **Not knowing what inputs each Databricks judge requires** — groundedness needs retrieved\_context; correctness needs expected\_facts or expected\_response; context\_sufficiency needs expected\_response and retrieved\_context. Confusing these inputs signals you've read about the judges but never run them.
- **Describing evaluation as something that runs "at the end" of a project** — signals waterfall thinking; production-grade GenAI evaluation is continuous, starting from first prototype through production monitoring.
- **Choosing a frontier model for every workload without discussing tradeoffs** — signals you optimize for capability headline numbers, not production requirements; interviewers for senior roles want to see constraint-driven model selection.
