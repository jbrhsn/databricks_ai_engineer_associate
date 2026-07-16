# Evaluation Metrics and LLM Choice — Thought Leadership

**Section:** 07 — Evaluation and Monitoring | **Target audience:** Senior ML engineers, AI platform leads, tech leads building production RAG and agent systems | **Target publication:** LinkedIn article / personal blog

---

## Hook / Opening Thesis

Most teams spend weeks agonizing over which LLM to use and ship the first evaluation metric they can instrument. That ordering is backwards — and it is why so many RAG applications in production look fine on the dashboard while silently feeding users wrong answers.

---

## Key Claims (3–5)

1. **Metric selection is a higher-leverage decision than model selection.** A well-designed evaluation signal will tell you exactly which model to use. An evaluation signal that measures the wrong thing will make a mediocre model look good and a great model look bad — then you'll spend weeks "improving" the wrong system.

2. **Groundedness and correctness are measuring fundamentally different failure modes**, and conflating them is the single most common evaluation mistake I see in production RAG deployments. High groundedness + low correctness always points to a retrieval problem, not a generation problem. Without this distinction, teams debug the generator for weeks and never fix the retriever.

3. **Reference-based metrics (BLEU, ROUGE) are not wrong — they are just scoped.** They are appropriate as regression smoke tests and CI gates on templated outputs. They are catastrophically wrong as primary quality signals for open-ended generation, where a correct paraphrase should score 100% but actually scores near 0%.

4. **LLM-as-judge is only as trustworthy as your calibration process.** Verbosity bias and position bias are real and systematic. Teams that skip Cohen's Kappa measurement against human labels are not running evaluation — they are running expensive autocomplete on their own outputs.

5. **The shift to MLflow 3's unified evaluation harness is not an upgrade, it is an architectural discipline.** Using the same scorer definitions for offline development evaluation and production monitoring is the only way to make "quality drift" a measurable, actionable metric rather than a post-hoc blame game.

---

## Supporting Evidence & Examples

**On metric mismatch:** I have seen teams deploy RAG applications where the eval pipeline used ROUGE-L against reference summaries. The model scored 0.62 on ROUGE-L. The team iterated for three sprints trying to raise it. Then someone ran the same responses through a groundedness judge — which scored 91%. The application was actually good at the task it was deployed for; ROUGE was measuring something orthogonal (phrasing fidelity) that the use case did not require. Three sprints lost to the wrong metric.

**On groundedness vs. correctness confusion:** In Databricks Agent Evaluation, the causal ordering of judges is documented explicitly: `context_sufficiency` is evaluated before `groundedness`, which is evaluated before `correctness`. This ordering reflects a real causal chain: if the retriever did not surface the right chunks, the generator cannot be grounded in the right information, and therefore the answer cannot be factually correct. Most teams skip `context_sufficiency` because it requires ground-truth labels — which is precisely why retrieval failures go undetected.

**On LLM judge bias:** Databricks measures judge quality using Cohen's Kappa against human raters and publishes these improvements on their engineering blog. Teams that do not run this calibration step cannot distinguish between "our application quality improved" and "the judge got more lenient." These are not the same thing.

**On model selection:** The Databricks Foundation Model APIs currently offer models ranging from Llama-3.1-8B (fast, cheap, 128K context) to GPT-5.6 Sol (frontier capability, 1.05M context). The right selection depends entirely on what your evaluation tells you is failing — if groundedness is low, no model upgrade will fix a retrieval problem. If context sufficiency is the bottleneck, you may need a model with a larger context window, not a smarter one.

---

## The Original Angle

The conventional wisdom is "pick the best model you can afford, then evaluate it." The insight I am arguing for is the reverse: **define your evaluation before you pick your model**, because your evaluation design will tell you which model tier you actually need. A team that starts with rigorous, layered metric design will discover that 80% of their use cases can be solved with a 70B model at one-fifth the cost of a frontier model — and they will have the evidence to make that call confidently instead of defaulting to the largest option out of anxiety.

This argument also has an organizational implication: evaluation is not a QA function that runs at the end of a sprint. It is an engineering discipline that shapes every architectural decision from day one.

---

## Counterarguments to Address

**"LLM judges are expensive and slow."** This is true for some judge configurations, but Databricks built-in judges are calibrated and run via managed endpoints with latency and cost metrics tracked per evaluation run. The callable SDK allows targeted single-judge calls for rapid debugging without a full evaluation run. The cost of running a groundedness judge on 100 samples is trivially small compared to the cost of one engineer-week debugging the wrong system component.

**"We can't build a curated eval set without annotation budget."** The `expected_facts` format in Databricks Agent Evaluation was designed precisely for this: instead of writing full reference answers, you write a two-to-four item list of minimal required facts. A subject matter expert can produce these in a fraction of the time required for full reference answers. If you genuinely have zero annotation capacity, reference-free judges (groundedness, relevance\_to\_query, safety) require no ground truth at all and still catch the highest-impact failure modes.

**"We're using RAGAS, not Databricks judges — isn't it the same thing?"** Conceptually similar, but the Databricks built-in judges (backed by partner-powered models calibrated against human raters) and the RAGAS open-source framework differ in calibration process, observability integration, and production monitoring continuity. For teams building on Databricks, the native judges integrate directly with MLflow tracing and experiment tracking — no additional infrastructure to maintain.

---

## Practical Takeaways for the Reader

- Before writing a single line of evaluation code, write down: "What failure mode am I most afraid of?" That answer determines your first metric.
- Separate your metrics by pipeline stage: retrieval quality (chunk\_relevance, context\_sufficiency) is a different problem from generation quality (groundedness, correctness). Never let a generation metric diagnose a retrieval failure.
- If you use LLM-as-judge, measure judge accuracy against at least 50 human-labeled samples and report Cohen's Kappa alongside your application quality metrics.
- Prefer `expected_facts` over `expected_response` for correctness evaluation — it makes your ground truth smaller, cheaper to collect, and more robust to valid paraphrases.
- Set up production monitoring with the same scorer definitions you use in development. Quality drift between your eval set and production traffic is the most important signal you can instrument, and you cannot measure it if you use different metrics in each environment.

---

## Call to Action

If your team currently uses a single quality metric for your RAG or agent application, run this experiment: take ten production responses that passed your current metric and run them through `chunk_relevance` and `groundedness` using the Databricks callable judges SDK. I would wager at least two of those ten "passing" responses will fail one of those judges. That gap is your blind spot — and it is fixable once you can see it.

---

## Further Reading / References

- [Built-in AI judges (MLflow 2) — Databricks](https://docs.databricks.com/aws/en/agents/agent-evaluation/llm-judge-reference) — Reference documentation for all Databricks built-in judges and their required inputs
- [How quality, cost, and latency are assessed by Agent Evaluation (MLflow 2) — Databricks](https://docs.databricks.com/aws/en/agents/agent-evaluation/llm-judge-metrics) — Metric name reference, aggregate formulas, and root-cause priority ordering
- [Evaluate and monitor AI agents — Databricks MLflow 3](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/) — Architecture of the unified evaluation and production monitoring harness
- [Databricks-hosted foundation models available in Foundation Model APIs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models) — Full model catalogue with capability descriptions and endpoint names
