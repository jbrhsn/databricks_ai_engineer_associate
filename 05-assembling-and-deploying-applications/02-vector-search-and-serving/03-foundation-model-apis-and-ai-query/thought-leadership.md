# Foundation Model APIs and ai_query — Thought Leadership

**Section:** 05-Assembling and Deploying Applications | **Target audience:** Senior engineers, ML platform leads, data engineering teams | **Target publication:** LinkedIn article, personal blog, conference talk

---

## Hook / Opening Thesis

Most teams building LLM applications manage two completely separate problems — model access and data access — and write twice as much glue code as they should. Databricks Foundation Model APIs collapse those two problems into one interface, and almost nobody is using the most powerful consequence of that design: SQL-native LLM inference at batch scale.

---

## Key Claims (3–5)

1. The OpenAI-compatible interface is not a marketing choice — it is the most important architectural decision Databricks made in the Model Serving layer, and its value compounds every time a new model family is added without breaking existing code.
2. `ai_query()` eliminates an entire class of orchestration complexity for batch enrichment pipelines; the teams still building Python loop-based batch inference are spending engineering hours on a solved problem.
3. The pay-per-token vs provisioned throughput choice is not a cost optimisation — it is a compliance and SLA architecture decision with cost as a secondary effect.
4. Unity AI Gateway (currently in Beta) signals that Databricks is treating AI governance the same way it treats data governance: as a first-class platform capability rather than an afterthought.
5. External model endpoints solve a real enterprise problem — centralised credential management across multiple LLM providers — that most organisations currently solve with bespoke secret-passing code that fails audits.

---

## Supporting Evidence & Examples

**Claim 1 — OpenAI-compatible interface compounds value:**
When my team evaluated switching from GPT-4 to Llama 3.3 70B for a classification pipeline, the code change was literally one string: the `model` parameter. No new SDK, no new auth flow, no new error handling. That portability is only possible because both endpoints speak the same OpenAI-compatible dialect through Databricks Model Serving. As Databricks adds new model families (and the July 2026 supported-models page lists over 50 endpoint names), this design means every model is immediately usable with zero integration cost.

**Claim 2 — `ai_query()` eliminates orchestration overhead:**
A common batch enrichment pattern I see in the wild: a Python notebook reads a Delta table into a Pandas DataFrame, loops through rows calling the OpenAI SDK, writes results back to Delta. This is serial, fragile, and hard to monitor. The equivalent `ai_query()` implementation is a single SQL `CREATE TABLE AS SELECT ai_query(...)` statement. Databricks handles parallelisation, retries, and progress monitoring automatically. The query profile view in the SQL editor shows completed and failed inferences in real time — a feature you would have to build yourself in the Python approach.

**Claim 3 — Provisioned throughput is a compliance decision:**
When a healthcare client asked whether their LLM pipeline could be HIPAA-eligible, the answer was not about model choice — it was about endpoint mode. Pay-per-token runs on shared infrastructure that does not carry HIPAA certification. Provisioned throughput does. Teams that default to pay-per-token for everything and then discover this compliance gap mid-audit face a production migration, not just a configuration change.

**Claim 4 — Unity AI Gateway as data governance parity:**
Databricks has spent years building Unity Catalog so that every data asset has lineage, access control, and auditing. Unity AI Gateway extends that philosophy to AI runtime traffic. Rate limits per team, per-principal usage attribution in system tables, guardrail policies on model endpoints — these are governance primitives, not monitoring dashboards. Teams that adopt Unity AI Gateway early will have audit trails that late adopters will have to reconstruct.

**Claim 5 — External model endpoints as credential governance:**
I have reviewed production architectures where OpenAI API keys were stored in three different secret managers, two CI/CD environment variables, and one plaintext config file. Databricks external model endpoints solve this with a single `{{secrets/scope/key}}` reference that is injected at request time. The application developer never sees the key; the security team controls rotation from one place.

---

## The Original Angle

The conventional framing of FMAPI is "easier access to LLMs." The more accurate framing is **"ML infrastructure parity with data infrastructure."** When a data engineer queries a Delta table, they do not worry about which storage system the bytes live on, whether the connection is rate-limited, or whether the query will be audited. FMAPI + Unity AI Gateway is Databricks' attempt to make querying a language model feel the same way — a governed, auditable, infrastructure-abstracted operation. Whether you reach that endpoint from a Python service, a SQL job, or a Databricks Workflow, the experience is intentionally identical. That convergence is the strategic bet, and understanding it changes how you design AI-enriched data products.

---

## Counterarguments to Address

**"Pay-per-token is always more expensive at scale."** Not necessarily true. For truly bursty workloads — a weekly batch job, a low-traffic RAG application, a development environment — pay-per-token is cheaper than provisioned throughput because you are not paying for idle model-unit hours. The crossover point depends on your request rate and model; run the numbers before provisioning.

**"`ai_query()` is a toy — real applications need Python."** For real-time, streaming, or agent-based use cases this is correct. But for batch enrichment — which represents the majority of enterprise ML workloads by volume — `ai_query()` in a Databricks SQL job is not a toy; it is often the right tool precisely because it integrates with existing SQL-based data pipelines without introducing a Python dependency.

**"Unity AI Gateway is just rate limiting."** It is also budget management, per-principal usage attribution for chargebacks, service-policy guardrails (content moderation at the infrastructure level), and inference table logging. Framing it as "just rate limiting" undersells its role as the audit and governance layer for AI traffic.

---

## Practical Takeaways for the Reader

- Before building a Python-based batch LLM pipeline, evaluate whether `ai_query()` in a Databricks SQL job satisfies the requirement — it likely does, and it will be easier to maintain and monitor.
- Explicitly decide on endpoint mode before your first production deployment: pay-per-token for exploratory and burst workloads, provisioned throughput for sustained-load and compliance-sensitive workloads.
- Store all external LLM provider API keys in Databricks Secrets and reference them through external model endpoints — never in code, notebooks, or environment variables in production.
- Review the supported-models page before coding endpoint names into production pipelines; the list evolves rapidly and endpoint names can change across model generations.
- Enable Unity AI Gateway and configure usage tracking before you need the audit trail, not after. Retroactive attribution is much harder than forward-looking instrumentation.

---

## Call to Action

If you have a batch LLM pipeline that runs on a Python compute cluster today, try rewriting it as a `CREATE TABLE AS SELECT ai_query(...)` statement on a Serverless SQL warehouse. Measure the lines of code, the failure handling, and the monitoring you get for free versus what you built. I would be interested to hear what the comparison looks like in your environment — reply below or connect on LinkedIn with your findings.

---

## Further Reading / References

- [Databricks Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-model-apis/index.html) — overview of FMAPI modes
- [Use `ai_query`](https://docs.databricks.com/en/large-language-models/ai-query.html) — syntax and best practices for SQL-native inference
- [AI governance with Unity AI Gateway](https://docs.databricks.com/en/ai-gateway/index.html) — rate limits, guardrails, and usage tracking (Beta, July 2026)
- [External models in Model Serving](https://docs.databricks.com/en/machine-learning/foundation-models/external-models/) — centralised credential management for third-party providers
- [Provisioned throughput Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-model-apis/deploy-prov-throughput-foundation-model-apis.html) — dedicated capacity configuration and compliance eligibility
