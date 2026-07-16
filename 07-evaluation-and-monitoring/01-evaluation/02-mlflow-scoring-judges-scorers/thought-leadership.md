# MLflow Scoring, Judges, and Scorers — Thought Leadership

**Section:** 07 Evaluation and Monitoring | **Target audience:** Senior ML engineers, AI platform leads, GenAI practitioners | **Target publication:** LinkedIn article / personal engineering blog

---

## Hook / Opening Thesis

LLM-as-judge is not a replacement for human evaluation — it is a signal amplifier for what you already know matters. Teams that treat an 87% groundedness score as a shipping gate have not escaped the evaluation problem; they have outsourced it to a model they have not calibrated.

---

## Key Claims

1. **A judge score is only as meaningful as the dimension you chose to measure.** Databricks Agent Evaluation ships with eight built-in judges — but "groundedness" and "correctness" are not objective universal standards. They are prompt-based heuristics whose outputs vary with the underlying judge model, the temperature, and the phrasing of the question. Teams that treat these numbers as ground truth are building on sand.

2. **Low groundedness and high correctness is a real, underappreciated failure mode.** It means your agent is producing correct answers without using the retrieved context — likely because it is relying on parametric memory. In a regulated domain this is a compliance problem, not a quality win. Evaluation pipelines that do not track both dimensions together will miss it entirely.

3. **Custom judges are where evaluation actually earns its cost.** A PII-detection judge, a brand-tone judge, a factual-claim-scope judge — these are the checks that catch the failure modes that matter to your specific business. The built-in suite is a starting grid, not a finish line.

4. **LLM judge consistency is a property you must measure, not assume.** Databricks publishes Cohen's Kappa agreement scores against human raters for its built-in judges. Most teams never look at this number. If your judge agrees with humans 65% of the time, your 87% groundedness score has an error bar of ±15 points — which may be wider than the delta you are trying to detect between model versions.

5. **The evaluation dataset is the hardest artifact to maintain.** A 20-row manually curated eval set that is actually representative of production traffic is worth more than a 2,000-row synthetically generated set that is not. The discipline of building and versioning evaluation datasets is the real bottleneck in production GenAI quality work.

---

## Supporting Evidence & Examples

**Claim 1 — dimension selection:** In one project evaluating an internal policy Q&A bot, groundedness was consistently above 80%, but when we added a custom `scope_adherence` judge checking whether responses stayed within the domain of the HR handbook, pass rates dropped to 52%. The built-in judges had been telling us nothing about the failure mode that actually mattered to the legal team.

**Claim 2 — correctness without groundedness:** Databricks' own documentation shows the root-cause ordering: `context_sufficiency` before `groundedness` before `correctness`. If `context_sufficiency` fails but `correctness` passes, the correct inference is that the agent answered from parametric memory — a risk, not a feature, in any domain where source traceability matters.

**Claim 3 — custom judges at scale:** `make_genai_metric_from_prompt` takes roughly 15 minutes to write a first version of a custom judge. The cost of running a 100-row eval with a 70B parameter judge model at Foundation Models API rates is typically under $0.10. The ROI calculation is not complex.

**Claim 4 — judge consistency:** Databricks' blog post "Databricks announces significant improvements to the built-in LLM judges" (2024) explicitly discusses Cohen's Kappa as the quality metric for judge improvement. The implication: judge quality has *improved over time*, which means historical eval runs and current runs are not directly comparable without re-running old data.

**Claim 5 — eval set maintenance:** Production traces are the right source for eval data because they capture the actual distribution of user requests. The pattern of exporting a sample of production MLflow traces, tagging them with correctness labels, and committing them as versioned eval sets in Unity Catalog is operationally sound but almost never done in practice until a quality regression has already shipped.

---

## The Original Angle

Most writing on LLM evaluation focuses on which metric to use. The harder problem — and the one this piece addresses — is *what your metric score is evidence of*. An 87% groundedness score is evidence that 87% of the responses in this eval set, as judged by this LLM with this prompt, on this day, appeared to cite their retrieved context. It is not evidence that your production users are receiving grounded responses at 87% frequency. The gap between those two statements is the entire field of evaluation.

I have watched teams at three different organizations ship "high-quality" agents that failed in production not because their eval metrics were wrong, but because their eval sets were not representative, their judge model was miscalibrated for their domain, or they were measuring the wrong dimension entirely. The tooling (MLflow, Databricks Agent Evaluation) is genuinely excellent. The hard part is using it wisely.

---

## Counterarguments to Address

**"Human evaluation doesn't scale — LLM judges are a necessary compromise."** True, but the compromise requires acknowledging what you are trading away. LLM judges scale your ability to run *frequent, low-latency checks on the dimensions you have encoded in prompts*. They do not scale your coverage of all possible failure modes. Use LLM judges to maintain a cadence of quantitative signals; use periodic human review (even small-sample spot checks) to audit whether those signals are tracking the right things.

**"The built-in judges are calibrated by Databricks against human raters — that's good enough."** Databricks' calibration is against general-domain human raters using diverse, challenging academic datasets. If your application is a highly specialized domain (clinical guidelines, legal contracts, financial regulations), the judge's concept of "correctness" may not align with your domain experts' concept. Calibration is a starting point; domain-specific few-shot examples are the adjustment mechanism.

**"Evaluating every deployment is too expensive."** The cost of running 100 rows through `mlflow.evaluate()` with a 70B judge model is negligible compared to the cost of a production quality incident. The expensive part is not running the eval — it is maintaining the eval dataset and acting on the results. Start with 25–50 high-value representative questions and run them on every deployment. Add more questions when you encounter a production failure not covered by the existing set.

---

## Practical Takeaways for the Reader

- Before your next eval run, write down the specific failure mode you are trying to detect. Then check whether any of the eight built-in judges actually measures it. If not, write a custom judge before running the eval.
- Build your evaluation dataset from production traces, not synthetic generation. Export 50 traces from your first week of production, manually label 20 of them with `expected_facts`, and use the remaining 30 as unlabeled safety/groundedness/relevance checks.
- Run `results.tables["eval_results"][results.tables["eval_results"]["response/llm_judged/groundedness/rating"] == "no"]` on your next eval run and read the rationale column. The judge's free-text reasoning is often more actionable than the aggregated percentage.
- Version your evaluation datasets. A named MLflow Dataset (`mlflow.data.from_pandas(df, name="hr-rag-eval-v3")`) logged to your experiment provides traceability that a raw DataFrame does not.
- Track the delta between runs, not the absolute score. Whether you are at 76% or 87% groundedness matters less than whether it moved 5 points after your last retrieval change.

---

## Call to Action

The next time someone on your team shows a groundedness score as a shipping gate, ask two questions: "What does the judge LLM's prompt actually check?" and "Is our eval set representative of the traffic that will stress this failure mode?" If you cannot answer both questions confidently, you have tooling without methodology. The evaluation infrastructure is the easy part. Building the discipline to use it honestly is the work.

What failure mode in your current agent are you *not* measuring? Reply or comment — I am collecting patterns.

---

## Further Reading / References

- [Run an evaluation and view the results (MLflow 2)](https://docs.databricks.com/aws/en/agents/agent-evaluation/evaluate-agent.html) — Core `mlflow.evaluate()` API reference
- [Built-in AI judges (MLflow 2)](https://docs.databricks.com/aws/en/agents/agent-evaluation/llm-judge-reference.html) — Judge definitions, required inputs, and calibration methodology
- [How quality, cost, and latency are assessed (MLflow 2)](https://docs.databricks.com/aws/en/agents/agent-evaluation/llm-judge-metrics.html) — Aggregated metric schema and root-cause ordering
- [Customize AI judges (MLflow 2)](https://docs.databricks.com/aws/en/agents/agent-evaluation/advanced-agent-eval.html) — `make_genai_metric_from_prompt` and few-shot judge calibration

---
