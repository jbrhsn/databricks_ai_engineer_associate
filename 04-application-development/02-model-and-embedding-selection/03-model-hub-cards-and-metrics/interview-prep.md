# Reading Model Hub Cards and Evaluation Metrics — Interview Prep

**Section:** 04 Application Development | **Role target:** Senior ML Engineer, AI Platform Engineer, MLOps Engineer, Solutions Architect

---

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is a model card and what are its mandatory sections? | Structured README.md; YAML metadata block (license, datasets, model-index benchmarks) + prose sections; six core sections: Model Details, Uses (direct/downstream/out-of-scope), Bias/Risks/Limitations, Training Data, Evaluation Results, License; Mitchell et al. 2018 origin; HuggingFace Hub renders YAML as filter facets and structured benchmark widgets | "It's documentation for the model" — too vague; doesn't mention the YAML/prose dual structure, the machine-readable model-index block, or the governance value of the Limitations section |
| Explain the difference between MMLU 0-shot and 5-shot scores. | 0-shot: model sees only the question, no examples — measures raw knowledge encoding. 5-shot: model sees five solved examples before the question — simulates few-shot prompting and typically adds 3–8 points. Same benchmark, different evaluation protocol. When comparing models, always compare like-for-like (both 0-shot or both 5-shot); mixing them inflates the apparent advantage of one model | "5-shot is better" — misses the point; the difference is about what you are measuring (raw knowledge vs prompted knowledge) and whether your production prompt will include examples |
| What does pass@k measure in HumanEval, and what is the production-relevant value of k? | pass@k = probability at least one of k generated samples passes all unit tests; formula: `1 - C(n-c,k)/C(n,k)`; production-relevant k is **1** because latency-sensitive systems generate one response per request; k > 1 requires multiple generations + selection, multiplying cost and latency by k; papers often report pass@10 or pass@100 for ceiling comparisons | "Higher k is better for production" — gets the direction backwards; higher k improves the ceiling metric but is undeployable in real-time systems without explicit multi-sample infrastructure |
| Why is HELM's multi-dimensional output an advantage over MMLU's single accuracy score? | HELM measures accuracy + calibration (ECE) + robustness (accuracy under adversarial rephrasing) + fairness (accuracy gap across subgroups) + toxicity simultaneously across multiple scenarios; a model can score high on MMLU accuracy while being poorly calibrated, producing biased outputs, or generating toxic content — HELM exposes these; MMLU cannot. For regulated or safety-sensitive applications, calibration and fairness metrics are required, not optional | "HELM is too complicated to use" — this is a preference masquerading as a technical objection; the real answer acknowledges HELM's complexity but explains it is intentional and required for the use cases where it matters |
| What is the governance value of logging model selection rationale as MLflow tags? | Creates auditable, reproducible record of which model was chosen, which benchmarks informed the decision, what constraints (license, regulatory) were applied, and what alternatives were disqualified; enables compliance teams to reconstruct decision without tribal knowledge; supports model risk management (MRM) audit requirements; stored alongside the inference run, not in a separate system that can be archived or lost | "It's just good documentation practice" — true but undersells the specific value: MLflow tags are queryable programmatically via `mlflow.search_runs()`, versioned, and co-located with model artifacts — not a separate documentation system |

---

## Applied / Scenario Questions

**Q:** You are selecting a model for a customer-facing chatbot at a healthcare company. You have three candidate models with the following properties: Model A (MMLU 0.83, MT-Bench 8.2, Apache 2.0), Model B (MMLU 0.79, MT-Bench 8.7, research-only license), Model C (MMLU 0.81, MT-Bench 7.9, proprietary enterprise). Your company prohibits proprietary and research-only models in patient-facing systems. Walk through your selection process.

**Strong answer framework:**
- Start with the hard constraints (license), not the soft signals (benchmark scores). Model B (research-only) and Model C (proprietary) are immediately disqualified by policy — this is a binary gate, not a trade-off.
- Model A is the only eligible candidate. Document the disqualifications explicitly in the model selection record (MLflow tags).
- Note that Model A's lower MT-Bench (8.2 vs B's 8.7) represents a real quality gap for a chat application. Quantify whether this matters: run a small task-specific evaluation on representative healthcare dialogue samples to verify 8.2 translates acceptably to your domain.
- How to show tradeoff awareness: acknowledge that you are accepting a quality penalty to meet the licence constraint; propose mitigations (prompt engineering, fine-tuning on domain data) if the task-specific evaluation reveals the gap is operationally significant. Also read Model A's Limitations section for any healthcare-specific warnings — MMLU sub-category "medical genetics" or "clinical knowledge" is the relevant disaggregation.

---

**Q:** Your team deployed a model with MMLU 0.86 for a legal contract review application six weeks ago. Analysts are now reporting that the model frequently misclassifies indemnification clauses. How do you diagnose and respond?

**Strong answer framework:**
- Distinguish aggregate benchmark from sub-domain performance: MMLU 0.86 is an average across 57 subjects. Pull the sub-category breakdown — specifically "professional law" — from the model card's evaluation table. A model can score 0.92 on general subjects and 0.55 on professional law.
- Check the Limitations section of the model card. It may explicitly warn about degraded performance on legal reasoning or specific clause types that were under-represented in training data.
- Run HELM accuracy scenarios on legal QA benchmarks (e.g., LegalBench sub-tasks) to quantify the gap with an independent evaluation harness.
- Short-term fix: few-shot prompting with correctly-classified indemnification clause examples to inject domain knowledge at inference time.
- Long-term fix: domain fine-tuning on annotated contract review data, or switch to a model with documented strong performance on legal text; re-run the model selection process with legal-domain MMLU sub-category as the primary signal.
- Governance: log the incident in MLflow as a note on the original model selection run; update the selection rationale tags to reflect the discovered limitation; create a task-specific evaluation dataset for future model comparisons.

---

## System Design / Architecture Questions

**Q:** Design a model selection and governance pipeline for a regulated financial services firm that must select, document, and audit LLM choices across 20+ applications.

**Approach:**
1. **Clarify requirements:** Which regulations apply (SR 11-7 model risk management, GDPR data provenance)? What is the audit retention period? Does selection need approval sign-off, or is logging sufficient?
2. **Propose structure:**
   - Maintain a **model registry** in Unity Catalog where each registered model version has mandatory tags: `model_card_url`, `license`, `mmlu_5shot`, `humaneval_pass1` (if coding task), `mt_bench`, `helm_fairness_verdict`, `selection_rationale`, `disqualified_alternatives`.
   - Define a **selection checklist** as code (Python dataclass or Pydantic model) that validates required fields before a model is allowed into a production endpoint.
   - Run **task-specific evaluation** as a Databricks MLflow experiment for each application, producing a benchmark comparison run with the candidate models as nested runs.
   - Use **MLflow model aliases** (`champion`, `challenger`) to make the current production choice explicit and auditable.
   - Schedule **quarterly re-evaluation** runs that re-check benchmark scores against the Open LLM Leaderboard (in case newer models have been added that outperform the champion on task-relevant metrics).
3. **Justify choices and tradeoffs:** Unity Catalog for the registry (vs a spreadsheet) enables programmatic governance queries: `mlflow.search_model_versions(filter_string="tags.helm_fairness_verdict = 'PASS'")`. MLflow nested runs for evaluation keep comparison artefacts co-located with selection decisions. The tradeoff is engineering overhead for the initial pipeline build; the benefit is that each of 20 applications shares the same auditable infrastructure rather than 20 separate spreadsheets.

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **pass@1 vs pass@k** — when you use this you immediately signal you have read the HumanEval paper, not just the headline leaderboard number
- **benchmark contamination** — the risk that test set questions appeared in pre-training data, inflating benchmark scores; signals awareness of evaluation methodology limitations
- **Expected Calibration Error (ECE)** — the HELM metric for calibration; using this term shows you understand the difference between accuracy and confidence reliability
- **model-index YAML** — the machine-readable evaluation metadata block in a HuggingFace model card; shows you understand how Hub leaderboard data is structured
- **MMLU sub-category disaggregation** — "I would pull the professional law sub-category specifically" rather than citing the aggregate; signals domain-specific diligence
- **LLM-as-judge variance** — acknowledging that MT-Bench scores shift by judge model version; signals awareness of evaluation methodology rather than just the final score
- **provisioned throughput vs pay-per-token** — using Databricks-specific terminology correctly signals platform fluency

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"Bigger is better"** — conflating parameter count with task performance; shows no awareness of benchmark-task alignment or the existence of efficient small models that outperform large ones on specific tasks
- **"We checked the leaderboard"** — without specifying which leaderboard, which benchmark, which shot count, or whether the scores are self-reported; signals surface-level diligence
- **"The model scored 91%"** — citing a benchmark number without naming the benchmark; meaningless and signals lack of precision about what is being measured
- **"We'll just test it and see"** — skipping model card review entirely; misses licensing blockers, compliance signals, and documented limitations that would disqualify a model without expensive testing
- **"MT-Bench 8.6 is better than 8.1 from any paper"** — comparing MT-Bench scores across papers without verifying judge model; can lead to selecting an objectively weaker model because the paper used a more lenient judge

---

## STAR Answer Frame

**Situation:** I was the ML engineer responsible for selecting a foundation model for a regulatory reporting automation tool at a financial services company. We needed to extract specific numerical fields from 10-K filings and flag discrepancies. I had shortlisted four models based on MMLU scores from the Open LLM Leaderboard.

**Task:** I was responsible for the final model selection, the governance documentation, and ensuring the choice met our model risk management policy — which required a documented bias assessment and a reproducible audit trail.

**Action:** During model card review, I discovered that the top-ranked model by aggregate MMLU (0.86) had a Limitations section that explicitly stated "reduced reliability on tasks requiring sustained attention to long numerical sequences in financial statements." This matched our core use case almost exactly. I then disaggregated its MMLU scores and found the "college mathematics" and "professional accounting" sub-categories were 0.63 and 0.59 — well below the aggregate. The second-ranked model (MMLU 0.82 aggregate) had no such warning in its Limitations section and scored 0.78 and 0.74 on those sub-categories. I ran a 50-document task-specific evaluation, which confirmed Model 2 outperformed Model 1 by 19 percentage points on our actual extraction task despite having a lower aggregate MMLU score. I logged the full selection rationale, sub-category scores, limitations warnings, and evaluation results as MLflow run tags and registered the selected model in Unity Catalog with the governance metadata attached.

**Result:** The selected model passed model risk management review in the first submission (no rework cycles), which typically takes 2–3 rounds for AI tools. The audit trail enabled the compliance team to reproduce the selection decision in 20 minutes during the initial review. The system has been running in production for eight months with a 94% field extraction accuracy rate on held-out quarterly filings.

---

## Red Flags Interviewers Watch For

- **Citing a benchmark number without knowing the shot count** — "MMLU 0.85" with no mention of 0-shot vs 5-shot signals the candidate copied a number without understanding what it measures.
- **Not knowing that MT-Bench uses an LLM judge** — candidates who think MT-Bench is human-evaluated miss the entire methodological limitation of the benchmark (judge model version dependence, potential bias of the judge toward its own style).
- **Conflating HumanEval pass@k for different k values** — comparing models where one reports pass@1 and another reports pass@10 without flagging the incompatibility suggests the candidate has not read beyond the headline.
- **Ignoring the Limitations section** — if asked "how would you evaluate a model for a regulated use case" and the answer goes straight to MMLU/HumanEval without mentioning the Limitations or Bias sections of the model card, the candidate has missed the most governance-critical part of model selection.
- **No mention of license checking** — any model selection answer for a production deployment that does not include license verification signals that the candidate has not worked on a production system with real legal constraints.
- **"We trust the benchmark leaderboard"** — without any mention of benchmark contamination, self-reported vs independently-verified scores, or the gap between benchmark conditions and production distribution, this signals benchmark naivety that will cause production failures.
