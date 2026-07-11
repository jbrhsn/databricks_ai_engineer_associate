# LLM Selection by Attributes — Interview Prep

**Section:** Application Development | **Role target:** Senior GenAI Engineer, ML Engineer, Solutions Architect (Databricks)

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is a context window and why does it matter for model selection? | Max total tokens (input + output) per request; hard model property, not configurable; determines how much retrieved context + history + output can fit; must be ≥ required document size + system prompt + output budget | Confusing `max_tokens` (output limit, configurable) with context window (total limit, fixed); saying "you can just increase max_tokens to get more context" |
| What is the difference between pay-per-token and provisioned throughput in Databricks FMAPI? | Pay-per-token: shared infrastructure, pre-configured endpoints, no latency guarantees, suitable for dev/PoC/low-volume prod; Provisioned throughput: dedicated GPU capacity, configurable `min_provisioned_throughput` (tokens/min), required for latency SLAs, fine-tuned models, HIPAA workloads; different cost structure | Saying pay-per-token is "for testing only" (it is usable for production) or that provisioned throughput is always cheaper (it depends on break-even QPS) |
| What are External Models in Databricks Model Serving and what problem do they solve? | Proxy endpoints for third-party providers (OpenAI, Anthropic, Cohere, Bedrock, Vertex AI); solves centralized credential management (API keys in Databricks Secrets, not in code); OpenAI-compatible interface; all requests route through Unity AI Gateway for rate limiting/budget controls | Thinking External Models bypass AI Gateway governance; thinking they use a different API format from FMAPI; not mentioning credential centralization as the primary value |
| How do you estimate monthly LLM API cost? | QPS × avg_tokens_per_request × seconds_per_month (2,592,000) × price_per_token; compute before prototyping; output tokens typically priced higher than input tokens; verify current pricing (changes frequently) | Estimating cost only after building the system; using a ballpark without working through the multiplication; forgetting to account for output tokens separately when providers price them differently |

## Applied / Scenario Questions

**Q:** Your team needs to build a document summarization service. Documents range from 5 to 120 pages. The service must respond within 15 seconds for interactive use. How do you approach model selection?

**Strong answer framework:**
- Start with context window: 120 pages ≈ 96,000 tokens; with system prompt (~500 tokens) and summary output (~1,500 tokens), minimum required window ≈ 98,000 tokens → only models with ≥128K context window are eligible
- Apply latency constraint: 15-second budget at ~30–50 tokens/second output means the model must generate 1,500 tokens within the budget; large reasoning models (slow TTFT) are likely eliminated; mid-size instruction-following models are likely candidates
- Task-type alignment: summarization is instruction-following + compression, not code generation or deep reasoning; a general-purpose instruction-following model with high MT-Bench scores is appropriate
- Cost estimate: compute QPS × 98K tokens × price/token for expected volume; verify against budget ceiling
- Serving mode: if p95 latency < 15 seconds is an SLA, use provisioned throughput; if this is a batch-friendly background service, pay-per-token may suffice
- Show trade-off awareness: "I would not default to the largest model — its TTFT may breach the 15-second budget even with sufficient context window. I would benchmark the top 2–3 candidates that clear context and latency gates on representative documents."

**Q:** After deploying to production, you receive alerts that 5% of requests are failing with a 400 error related to context length. How do you diagnose and fix this?

**Strong answer framework:**
- Diagnosis: the application is sending requests that exceed the model's context window; this indicates the production document distribution has longer-tail documents than testing covered
- Check: compute `len(tokenizer.encode(system_prompt + document + output_reserve))` for the failing requests; identify the p99 token length
- Fix options: (1) switch to a model with a larger context window if budget allows, (2) implement a document chunking or sliding-window summarization strategy if the task allows, (3) add a pre-flight token count check that routes long documents to a larger-context model
- Preventive: add a monitoring metric for prompt token length; set an alert at 90% of context window utilization

## System Design / Architecture Questions

**Q:** Design a production model serving architecture for a legal firm that needs to process 1,000 contracts per day with PII handling requirements, using Databricks.

**Approach:**
1. **Clarify requirements:** Are contracts processed in batch (overnight) or interactively (on demand)? Is HIPAA or other compliance certification required? What is the accuracy threshold for clause extraction? What is the monthly cost ceiling?
2. **Propose structure:**
   - Serving path: FMAPI provisioned throughput (HIPAA-eligible, dedicated infrastructure, no data sent to third-party providers)
   - Model: mid-tier instruction-following model with ≥128K context window (verify current FMAPI catalog)
   - Token budget: 1,000 contracts/day ÷ 86,400 seconds/day = 0.012 QPS; peak may be 10× if batched — set `min_provisioned_throughput` accordingly
   - Credential management: no external provider credentials needed for FMAPI; Unity Catalog permissions control which users can invoke endpoints
   - PII: implement pre/post-processing to redact or tokenize PII before sending to the model if required
3. **Justify choices and name trade-offs explicitly:**
   - FMAPI provisioned throughput over External Models: data stays within Databricks security perimeter; no third-party BAA needed; provisioned throughput supports HIPAA certifications
   - Provisioned throughput over pay-per-token: batch peak QPS may require guaranteed capacity; pay-per-token latency variance is unacceptable for time-sensitive workflows
   - Trade-off acknowledged: provisioned throughput has a higher cost floor; if the workload is truly low-volume and compliance only requires data residency (not HIPAA), pay-per-token within Databricks may suffice

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **Time-to-first-token (TTFT)** — when discussing latency for interactive applications; shows you distinguish TTFT from total generation time
- **Provisioned throughput / min_provisioned_throughput** — when discussing production serving; shows you know the configuration knob, not just the concept
- **Unity AI Gateway** — when discussing governance of foundation model requests; shows you know all requests (FMAPI and External) are governed through a single control plane
- **OpenAI-compatible interface** — when describing how Databricks endpoints work; signals you understand why the same client code works across endpoint types
- **Task-type alignment + benchmark proxy (HumanEval, MT-Bench, MMLU)** — when explaining how to screen models for a specific task; shows you go beyond "I'll try a few models and see"
- **Token budget decomposition** — when discussing context window planning; shows you can do the arithmetic (system_prompt + retrieval_context + output_budget = required_window)

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"GPT-4 is the best model"** — signals you are not evaluating against workload constraints and are unaware of the current model landscape on Databricks
- **"We'll just use the biggest model available"** — signals no cost awareness and no understanding that larger models have higher TTFT
- **"max_tokens controls the context window"** — a specific technical error that immediately signals shallow understanding of how LLM APIs work
- **"External Models are less secure / less governed"** — incorrect; External Models route through Unity AI Gateway just like FMAPI
- **"We'll figure out the model later"** — signals no understanding that model attributes (especially context window and cost) constrain architecture decisions made at design time

## STAR Answer Frame

**Situation:** Our team inherited a RAG-based contract review tool that had been built quickly for a demo. It was using a frontier model (high cost, large context window) on a pay-per-token endpoint. When the tool went to production processing 800 contracts per day, the monthly invoice came back at $45,000 — 15× over budget.

**Task:** I was responsible for diagnosing the cost overrun and proposing a re-architecture that brought monthly cost under $5,000 without degrading accuracy below the 94% clause extraction F1 we had measured in testing.

**Action:** I ran the cost formula: 800 contracts/day × 30 days × 35,000 avg tokens/contract × $0.06/1K tokens = $50,400/month. Confirmed the diagnosis. Then I applied the four-attribute rubric: context window needed ≥ 40K tokens (contracts were typically ≤ 30 pages); latency was flexible (batch processing, 4-hour window); task was structured extraction; cost ceiling was $5,000/month → price/token ceiling ≈ $0.0036/1K tokens. I shortlisted three FMAPI models that cleared context and cost gates, ran a 200-contract evaluation set, and found that a mid-tier instruction-following model achieved 93.8% F1 — just within the accuracy threshold. I also switched from pay-per-token to provisioned throughput to get predictable latency for the batch window and a slightly better effective price at this volume.

**Result:** Monthly cost dropped from $45,000 to $4,200 — a 91% reduction. Accuracy stayed at 93.8% F1, within the agreed threshold. The selection rubric we documented for this project became the team's standard for all subsequent model choices, eliminating the re-litigation pattern on the next three projects.

## Red Flags Interviewers Watch For

- Inability to explain the difference between `max_tokens` (output cap) and context window (total input+output limit) — this is a basic API literacy question
- No mention of cost estimation before or during model selection — signals the candidate has never shipped a production GenAI system under budget constraints
- Inability to distinguish provisioned throughput from pay-per-token and when each is appropriate — a core Databricks platform question
- Describing model selection as "try a few and pick the one that works" without mentioning any quantitative filtering criteria — acceptable for a junior role, a red flag at senior level
- Treating External Models as ungoverned or insecure — incorrect and signals they have not read the current documentation
- Hardcoding endpoint names in examples without noting they change — signals they have not operated FMAPI in production through a model retirement cycle
