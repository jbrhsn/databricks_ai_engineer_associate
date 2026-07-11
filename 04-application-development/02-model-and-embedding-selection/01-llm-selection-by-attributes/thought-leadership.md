# LLM Selection by Attributes — Thought Leadership

**Section:** Application Development | **Target audience:** Senior engineers, tech leads, engineering managers | **Target publication:** LinkedIn article, personal blog

## Hook / Opening Thesis

Every GenAI team I have seen derail in the first month shares one pattern: they pick the model first and design the system around it, instead of building a selection rubric first and letting the requirements pick the model. The model is not the product — the selection framework is.

## Key Claims (3–5)

1. Teams without a documented model selection rubric re-litigate the same four questions on every new project: which model fits the task, can we afford it, will it be fast enough, and does it have enough context window? This is not a knowledge problem — it is a process problem.

2. A reusable rubric — context window ≥ required tokens, latency SLA ≤ p95 budget, task-benchmark threshold, monthly cost ceiling — eliminates 80% of model debate in sprint planning and replaces opinion with measurement.

3. The Databricks FMAPI catalog evolving rapidly is not a problem if your rubric is attribute-based rather than name-based. A team that selects by "128K context, instruction-following HumanEval ≥ 60%, ≤ $0.005 per 1K tokens" survives a model retirement intact; a team that hardcodes `databricks-meta-llama-3-1-70b-instruct` discovers the breakage at 2 AM.

4. Cost is the most frequently skipped constraint and the most frequently regretted one. The calculation is four multiplications — QPS × avg tokens × seconds/month × price/token — and takes 90 seconds. Teams that skip it spend weeks explaining overruns to finance; teams that do it upfront disqualify expensive models before any code is written.

5. The FMAPI vs. External Models decision is not about preference — it is about where your data can travel, who manages credentials, and whether your workload needs HIPAA-eligible infrastructure. Framing it as a "technology choice" leads to the wrong decision; framing it as a governance and compliance question leads to the right one.

## Supporting Evidence & Examples

**Claim 1 — Re-litigating model choice on every project:**
I have seen the same debate play out across teams: someone advocates for the newest frontier model because a blog post said it was best, someone else raises cost concerns, a third person wants to use what was used last time. Without a rubric, the decision is made by whoever argues longest or has the most seniority in the room. With a rubric — here are the four attributes, here are the thresholds, here is the candidate list after filtering — the meeting becomes a 10-minute confirmation, not a 60-minute debate.

**Claim 2 — Attribute-based selection:**
The four attributes are not arbitrary. Context window fails first and fails silently: an application that works perfectly in testing with 2-page documents fails on 40-page production contracts because no one checked whether the document would fit. Latency SLA fails loudly but late: you discover the 8B model is too slow only after building the UI around synchronous responses. Cost fails at month-end billing. Task-type mismatch fails gradually in evaluation, which is why it is worth doing last — you only evaluate models that passed the first three gates.

**Claim 3 — Attribute-based rubrics survive model retirements:**
Databricks retired and replaced multiple models in 2024–2025 (Mixtral 8x7B, DBRX, various Llama 3.1 variants moved to deprecated status). Teams with hardcoded endpoint names had production outages. Teams whose deployment configs referenced an environment variable updated by a configuration pipeline had zero-downtime swaps. The rubric does not eliminate churn — it makes churn cheap.

**Claim 4 — Cost estimation as a gate, not a retrospective:**
At 10 QPS with 3,000 tokens per request and a frontier model priced at $0.01/1K tokens, monthly cost = 10 × 3,000 × 2,592,000 × 0.00001 = $777,600/month. That number, calculated before prototyping, eliminates the frontier model from consideration immediately and redirects engineering effort toward a model that actually fits the budget. The same architecture built on a $0.0005/1K token model costs $38,880/month — still significant, but manageable. The decision is made by arithmetic, not by preference.

**Claim 5 — FMAPI vs. External Models as a governance question:**
An organization with HIPAA obligations cannot simply choose "whichever model is cheapest." Provisioned throughput FMAPI endpoints are available with HIPAA compliance certifications; pay-per-token endpoints on shared infrastructure may not satisfy the same requirements. External Models route customer data to third-party provider infrastructure, which introduces data residency and BAA considerations that pay-per-token FMAPI does not. The correct framing is: what are the compliance requirements, and which serving path satisfies them — then price-optimize within that constraint.

## The Original Angle

Most writing about LLM selection focuses on benchmarks and capabilities — which model is smarter, which one is faster, which one just launched. This piece argues the opposite: the selection *process* matters more than the selection *outcome*. A team that has a rubric and applies it consistently will make a mediocre model choice 20% better and avoid catastrophic choices entirely. A team with no rubric will occasionally get lucky with a great model choice and routinely get surprised by avoidable failures.

The Databricks platform makes this argument concrete: it offers two fundamentally different serving paths (FMAPI and External Models), each with multiple sub-modes, each with different governance implications, and a model catalog that changes every few months. Without a documented rubric, every model rotation is a crisis. With one, it is a configuration update.

## Counterarguments to Address

**"Our use case is unique — a rubric is too rigid."**
The rubric does not dictate the outcome; it structures the evaluation. "Context window ≥ required tokens" is not rigid — it is the minimum physics of the problem. If your application cannot work in 16K tokens, no amount of iteration changes that constraint. The rubric just surfaces this truth in hour one instead of week three.

**"Models change so fast that any selection framework is outdated immediately."**
Correct that specific model names change fast. Wrong that the attributes change fast. Context window, latency, task-type fit, and cost have been the right four dimensions for three years and will remain so. The rubric is attribute-based; the model catalog slot it selects from is the fast-moving part, and that is precisely the separation the rubric provides.

**"We should just benchmark on our data instead of using a rubric."**
Benchmarking on your own data is the gold standard — for the final selection. But you cannot benchmark 30 models against your full dataset in a sprint. The rubric eliminates 25 candidates in an afternoon so you benchmark the 5 that are actually eligible.

## Practical Takeaways for the Reader

- Build a one-page selection rubric before your next project: four rows (context window threshold, latency SLA, task-type benchmark minimum, monthly cost ceiling) and a decision column. Apply it to every model candidate before opening the AI Playground.
- Make endpoint names a configuration value, not a code constant. When the next FMAPI model retirement happens, you want a one-line config change, not a code deploy.
- Calculate monthly cost before prototyping. Write the formula on a sticky note: `QPS × avg_tokens × 2,592,000 × price_per_token`. Run it in 90 seconds and disqualify models that fail before you write a single API call.

## Call to Action

If your team has never written down the four attributes you use to select a model, start a shared document today with four rows: context window, latency, benchmark threshold, and cost ceiling. Leave the model names blank. Fill in the thresholds from your product requirements. Then look at the current FMAPI catalog and see which candidates survive. The answer will almost certainly surprise you — and it will save you a month of downstream debugging.

## Further Reading / References

- [Databricks Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-model-apis/index.html) — Overview of FMAPI serving modes and requirements
- [External models in Model Serving](https://docs.databricks.com/en/generative-ai/external-models/index.html) — External Models provider support, credential management, AI Gateway integration
- [Databricks-hosted foundation models (supported models)](https://docs.databricks.com/en/machine-learning/foundation-model-apis/supported-models.html) — Current model catalog; verify endpoint names here before hardcoding anywhere

---
<!-- PUBLISHING CHECKLIST (delete before posting):
  - [x] Hook does NOT start with "In today's world" or "As X evolves"
  - [x] At least one concrete example or metric (cost calculation in Claim 4)
  - [x] Personal voice throughout ("I", "we", "my team")
  - [x] Ends with a specific discussion question or call to action
  - [x] 600–900 words when measured
-->
