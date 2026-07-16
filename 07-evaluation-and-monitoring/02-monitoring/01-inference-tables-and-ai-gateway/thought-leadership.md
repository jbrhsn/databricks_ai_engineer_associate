# Inference Tables and AI Gateway Monitoring — Thought Leadership

**Section:** 07 Evaluation and Monitoring | **Target audience:** Senior ML engineers, AI platform leads, tech leads | **Target publication:** LinkedIn long-form / personal blog

---

## Hook / Opening Thesis

Most teams find out their GenAI application has been giving wrong answers the same way restaurants find out about food poisoning: from the customers who got sick, not from the kitchen. Inference tables exist precisely to break that pattern — but the majority of production GenAI deployments I have seen either never enable them, or enable them and never query them.

---

## Key Claims

1. **Offline evaluation scores are a necessary but dangerously insufficient proxy for production quality.** The benchmark you ran before deployment reflects the distribution of questions your *evaluation team* wrote, not the distribution of questions your *users* ask at 2 AM on a Tuesday. These are never the same distribution.

2. **The gap between offline benchmark and live quality is not a testing failure — it is a data distribution failure.** No evaluation harness can simulate the long tail of real user behavior. Inference tables are the only mechanism that gives you access to that tail.

3. **Inference tables are the missing join key between your model's past and its future.** Every quality improvement loop — retraining, fine-tuning, prompt refinement, retrieval index updates — needs labeled examples from production. Without inference tables, you are generating synthetic data and hoping it generalizes.

4. **The cost of NOT monitoring is asymmetric.** A quality degradation that goes undetected for two weeks in a customer-facing application costs more in trust and escalations than the compute budget for a year of MLflow scorer evaluations.

5. **"We'll add monitoring later" is the most expensive phrase in GenAI engineering.** Every week without inference tables is a week of production data you can never recover. Unlike traditional software logs, inference payloads are not stored anywhere else — if you don't enable the table before traffic starts, that traffic is gone forever.

---

## Supporting Evidence & Examples

**On offline vs production distribution mismatch:** In a retrieval-augmented support bot I worked with, the offline evaluation dataset contained 300 carefully crafted questions from the product team. The inference table revealed that 40% of real user queries contained product version numbers, model identifiers, and error codes that appeared zero times in the evaluation set. The model's offline score was 0.81; its actual correctness rate on the version-number queries was 0.34.

**On the cost of delayed monitoring setup:** A team deploying a financial document summarizer enabled inference tables three weeks after go-live, after a compliance officer noticed inconsistent output formatting. Those three weeks of payloads — containing the exact requests that triggered the formatting failures — were unrecoverable. The debugging process that should have taken two days took six weeks of synthetic reproduction attempts.

**On sampling strategy:** At 50K daily requests, logging 100% at an LLM judge cost of $0.04/trace = $2,000/day is untenable. But sampling at 5% of requests and 20% of those for LLM judging gives you 500 evaluated traces/day at $20/day — statistically meaningful for detecting a 5-point quality drop with 95% confidence, and affordable. The `sampling_fraction` × `sample_rate` multiplication is the leverage point almost no one tunes deliberately.

**On the "we'll add it later" failure mode:** Databricks inference table delivery is at-least-once and approximately within one hour. But "at-least-once" means nothing if the table was never enabled. There is no retroactive logging. This is the architectural fact that changes the conversation from "should we monitor" to "when do we enable monitoring relative to deployment" — and the answer is always "before the first request."

---

## The Original Angle

Every article about GenAI monitoring focuses on *what to measure* — RAGAS scores, faithfulness, toxicity, latency. Almost none of them focuses on *when to start the clock* on data collection. The insight that changes teams' behavior is not "use MLflow scorers" — it is "the moment your endpoint receives its first request without inference tables enabled, you have permanently lost that data." Monitoring is not a post-deployment concern; it is a deployment prerequisite, the same way a black box recorder is installed before a plane's first flight, not after its first incident.

---

## Counterarguments to Address

**"We have application logs — those are good enough."** Application logs capture errors and latency. They do not capture the full request payload, the model's complete response, or the requester identity in a queryable Delta table that you can join with your evaluation datasets. Inference tables give you the same operational guarantees as application logs (at-least-once, best-effort within 1 hour) but with structured storage and Unity Catalog governance built in.

**"Storing every request is a privacy and compliance risk."** This is a legitimate concern that does not argue against inference tables — it argues for configuring them carefully. Unity Catalog ACLs control who can query the table. Sampling (`sampling_fraction`) reduces volume. PII masking at the application layer before the request reaches the endpoint is the correct pattern. The answer to "this data is sensitive" is governance, not "don't log."

**"MLflow production monitoring is Beta — it's not ready for production use."** Beta on Databricks means the feature is functional and supported but the API may change before GA. For non-critical quality monitoring (as opposed to safety guardrails, which belong in AI Gateway service policies), the Beta risk is acceptable. The alternative — no monitoring — has a higher expected cost than the risk of a minor API migration.

---

## Practical Takeaways for the Reader

- Enable inference tables at endpoint creation time, before the first request. Set a calendar reminder to check `auto_capture_config` status 24 hours after deployment.
- Start with `sampling_fraction=1.0` and reduce only when storage cost becomes meaningful. The cost of missing early traffic outweighs the storage savings.
- Build your first quality monitor using one of the Databricks starter notebooks for LLM inference table profiling — they handle the `from_json()` parsing and metric computation that most engineers get wrong on the first attempt.
- Register at least one MLflow safety scorer at `sample_rate=1.0` from day one. Safety is the one dimension where coverage matters more than cost.
- Treat the drift metrics table from Data Profiling as your early-warning system for distribution shift. A query input distribution that diverges from your training baseline is a leading indicator of quality problems, often appearing days before users start complaining.

---

## Call to Action

The next time you review a GenAI deployment checklist with your team, add one question before "deploy": "Is the inference table enabled, and do we have a query ready to check it 24 hours after go-live?" If the answer is no, fix that first. The model can always be improved; lost production data cannot be recovered.

---

## Further Reading / References

- [Monitor served models using AI Gateway-enabled inference tables](https://docs.databricks.com/aws/en/ai-gateway/inference-tables-serving-endpoints) — current schema, sampling configuration, and agent logging
- [Monitor GenAI apps in production](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/monitoring.html) — MLflow 3 production monitoring, scorer registration, sampling strategy
- [Data profiling](https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/data-profiling/) — inference log profile type, drift metrics, alerting

---
