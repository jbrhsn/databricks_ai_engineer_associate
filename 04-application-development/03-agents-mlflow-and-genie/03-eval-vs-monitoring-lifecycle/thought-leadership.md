# Evaluation vs. Monitoring — Thought Leadership

**Section:** 04 Application Development | **Target audience:** Senior ML engineers, AI platform leads, tech leads adopting GenAI in production | **Target publication:** LinkedIn long-form / personal engineering blog

---

## Hook / Opening Thesis

Your RAG application starts degrading on day 1 in production — you just won't notice for three weeks, because the signal you're watching (latency, error rate, 5xx count) is completely blind to the failure mode that actually kills LLM quality. Silent quality degradation is the defining production risk of GenAI that traditional MLOps monitoring was never designed to catch.

---

## Key Claims (5)

1. **HTTP 200 is not a quality signal.** A hallucinating RAG agent returns fast, successful responses. Every conventional monitoring alarm stays green while the model confidently tells users wrong things. The absence of 5xx errors is meaningless for LLM quality.

2. **Offline evaluation (`mlflow.evaluate()`) is a snapshot, not a guarantee.** A golden eval set of 50 curated questions covers the query distribution you anticipated at development time. Real users ask questions you never anticipated — about newly launched products, edge-case scenarios, and domain topics that weren't in scope when you wrote the eval set. Your golden set has a half-life measured in weeks.

3. **Inference tables are the prerequisite for everything.** You cannot diagnose a quality regression without the request-response history. Enabling AI Gateway-enabled inference tables after a problem surfaces means the data needed to understand the failure was never captured. This is the one configuration decision you cannot retroactively fix.

4. **Query distribution shift is the most common root cause, and it looks identical to model degradation from the outside.** When groundedness drops, the instinct is to retrain the model. In my experience, the more common explanation is that users started asking about topics absent from the knowledge base — a retrieval coverage gap, not a model failure. Distinguishing these requires querying the inference table; monitoring latency and error rates will never tell you.

5. **The feedback loop is the architecture.** Treating offline evaluation and online monitoring as separate, disconnected activities misses the point. The value is in the closed loop: monitoring alerts surface a query cluster that fails, those queries enrich the golden eval set, the next model version is gated against the enriched set, and deployment is validated before any user sees the new version. This loop — not the individual tools — is what makes GenAI applications maintainable at scale.

---

## Supporting Evidence & Examples

**The silent failure story.** A team at a B2B SaaS company deployed a RAG agent answering questions about their product catalog. Pre-deployment, `mlflow.evaluate()` showed 88% groundedness on 60 curated questions. They shipped. Three weeks later, the customer success team escalated: enterprise customers asking about a new product tier were getting plausible-sounding but factually wrong answers about pricing and feature availability.

Latency: perfectly normal. Error rate: 0.2% (background noise). The monitoring dashboard looked healthy.

When the team finally queried the inference table — which, fortunately, they had enabled from day one — they found a cluster of 1,400 requests over the previous 10 days, all containing the new product's name, all with LLM judge groundedness scores below 0.5 (scored retroactively after the incident). The judge rationale column was unambiguous: "No relevant chunks found. Response is not grounded in retrieved context." Retrieval miss, not model failure.

The fix: seed the vector index with the new product documentation, add 15 synthetic Q&A pairs for the new tier to the golden eval set, run offline eval, confirm groundedness > 0.85, deploy with a 95/5 A/B split, promote after 2,000 requests. Total time from detection to production fix: 4 days.

If inference tables had not been enabled, the team would have had no data to work from. The debugging process would have started from zero.

**Why the golden set has a half-life.** The query distribution of a real production application drifts continuously. New products launch. Seasonal events change what users care about. Regulatory changes introduce new question types. The eval set you wrote at development time is a snapshot of what you *expected* users to ask. Production data tells you what they *actually* ask. A monitoring system that compares live traffic against the golden eval set baseline immediately shows you when the two distributions diverge — that divergence is the early warning signal that your eval set needs updating.

---

## The Original Angle

Most GenAI production guidance focuses on one half of the problem: either evaluation (pre-release quality gates) or monitoring (production dashboards). The under-discussed insight is that these are not two separate problems — they are a single feedback loop, and breaking that loop is what causes silent quality degradation to persist for weeks.

The teams that catch regressions fast are the ones who close the loop: they use production monitoring not just to raise alarms but to automatically generate candidates for golden set enrichment. Their offline eval set is a living document, continuously updated by the production system that their model serves.

I have also observed that engineers who come from traditional ML backgrounds tend to reach for inference-centric metrics (latency, throughput, model version drift) when something goes wrong. Those metrics are correct for traditional ML. They are insufficient for LLM applications, where a model can be perfectly fast, perfectly available, and completely wrong. The monitoring stack has to be rebuilt around LLM judge scores as first-class metrics, not system metrics with LLM scores as an afterthought.

---

## Counterarguments to Address

**"LLM judge scoring in production is too expensive to run on every request."** This is a real constraint — and the solution is not to skip monitoring, it's to use sampling. AI Gateway inference tables support configurable `sampling_fraction`. Running LLM judges on 5–10% of traffic at a reasonable granularity (daily refresh) gives you a statistically representative quality signal at a fraction of the cost of full coverage. The cost of missing a quality regression for three weeks is almost always higher than the cost of daily judge scoring on a sample.

**"We have human reviewers who would catch quality issues."** Human review catches the issues users report. The characteristic of silent quality degradation is that users don't report it — they get a plausible-sounding answer and move on, or they quietly stop using the product. LLM judge monitoring catches what human escalations miss: the unreported 95% of failed interactions.

**"Our eval set is comprehensive — we wrote 200 questions."** Comprehensiveness at development time is not the same as coverage of the evolving production distribution. A 200-question eval set is an excellent foundation; it is not a monitoring system. The question to ask is not "how many questions do we have?" but "does our eval set include a question about the topic that caused the last quality incident?"

---

## Practical Takeaways for the Reader

- Enable AI Gateway inference tables on every production Model Serving endpoint before any traffic is sent — this is the one configuration decision you cannot make retroactively.
- Treat your golden eval set as a living document, not a one-time artifact. Schedule a monthly review where you query the inference table for the lowest-groundedness request clusters and add at least 10 new examples.
- Alert on LLM judge scores (groundedness, correctness) from data profiling, not on latency and error rate alone. The latter will never fire for the failure mode that matters most.
- When groundedness drops, query the inference table's request distribution *before* scheduling a retraining run — query-side drift is more common than model-side degradation, and they require completely different fixes.
- For A/B tests: collect at least 1,000 requests per arm before making a promotion decision. The variance in per-row LLM judge scores makes small samples unreliable.

---

## Call to Action

If you are deploying a GenAI application in the next 30 days, do one thing today: enable inference tables on your serving endpoint and create a data profiling monitor against that table. You don't need to do anything with it immediately. You need the data to exist when — not if — you need it.

What's your monitoring stack for LLM quality in production? I'd specifically like to hear from teams who have caught a quality regression before a user escalation — what was the signal that surfaced it?

---

## Further Reading / References

- [Agent Evaluation (MLflow 2)](https://docs.databricks.com/aws/en/agents/agent-evaluation/) — offline evaluation with LLM judges, eval set design
- [Monitor served models using AI Gateway-enabled inference tables](https://docs.databricks.com/aws/en/ai-gateway/inference-tables-serving-endpoints) — inference table schema and enablement
- [Data profiling overview](https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/data-profiling/) — Unity Catalog data profiling for drift detection and alerting
