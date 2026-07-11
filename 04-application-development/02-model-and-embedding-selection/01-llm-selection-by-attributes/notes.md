# LLM Selection by Attributes for Production GenAI Applications

**Section:** Application Development | **Module:** Model and Embedding Selection | **Est. time:** 2 hrs | **Exam mapping:** Domain 3 — Application Development (30%)

---

## TL;DR

Choosing the wrong LLM for a production workload — mismatched context window, wrong latency profile, or uncontrolled cost — is one of the most common and expensive mistakes in GenAI engineering. Every selection decision must be made against four concrete attributes: context window, inference speed/throughput, task-type alignment, and cost model. On Databricks, two serving paths exist: Foundation Model APIs (FMAPI) for Databricks-hosted models via pay-per-token or provisioned throughput, and External Models (AI Gateway) for third-party providers. **The one thing to remember: select a model by measuring required context tokens, latency SLA, task-benchmark fit, and monthly cost ceiling — in that order — before writing a single line of application code.**

---

## ELI5 — Explain It Like I'm 5

Imagine you need to hire a specialist for a job. You would not just pick the most famous expert in the world — you would ask: does this person have the right skills for the exact job, can they show up fast enough, and can I afford them for six months? Picking an LLM works exactly the same way. The "skills" map to task-type benchmarks, the "show up fast enough" maps to latency and throughput requirements, and "can I afford them" maps to your token cost budget. The most common mistake beginners make is picking the biggest, most famous model first and then discovering it is ten times too slow and ten times too expensive for their actual workload — the opposite order from correct.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Explain what context window size, inference speed, task-type alignment, and cost model mean and why each constrains model choice
- [ ] Compare Databricks Foundation Model APIs (pay-per-token vs provisioned throughput) and External Models (AI Gateway) and select the right serving path for a given workload
- [ ] Calculate a rough monthly token cost estimate from QPS, average tokens per request, and price per token
- [ ] Identify which FMAPI endpoint name to use for a given task type, given that the model catalog changes frequently
- [ ] Diagnose a model-selection mismatch from a symptom (truncated output, latency breach, cost overrun) and name the correct corrective action

---

## Visual Overview

### Model Attribute Trade-off Matrix

```
                HIGH QUALITY
                     ▲
                     │
      Large reasoning │  Large general
      models          │  purpose models
      (GPT-5.5 Pro,   │  (GPT-5.2, Gemini
      Claude Opus)    │   3.1 Pro)
                     │
LOW SPEED ◄──────────┼────────────────► HIGH SPEED
                     │
      Small/medium    │  Small fast
      reasoning       │  models
      (GPT OSS 20B)   │  (Claude Haiku,
                     │   GPT-5-nano)
                     │
                     ▼
                LOW QUALITY

   Cost roughly tracks quality × throughput demand.
   No model wins on all three axes simultaneously.
```

### Databricks Model Serving Options — Decision Tree

```
Does your model need to live OUTSIDE Databricks
(OpenAI, Anthropic, Cohere, Bedrock, Vertex)?
├── YES ──► External Models (AI Gateway)
│           • Centralized credential management
│           • Rate limiting & usage tracking via Unity AI Gateway
│           • Pay provider's per-token price
│
└── NO  ──► Foundation Model APIs (FMAPI)
            │
            Is this a PRODUCTION workload with throughput or
            latency guarantees, or a fine-tuned model?
            ├── YES ──► Provisioned Throughput endpoint
            │           • Set min_provisioned_throughput tokens/min
            │           • Reserved GPU capacity, predictable latency
            │           • Supports fine-tuned model weights
            │
            └── NO  ──► Pay-per-token endpoint
                        • Zero infrastructure setup
                        • Pre-configured endpoints in Serving UI
                        • Suitable for dev, PoC, low-volume prod
```

### Token Budget Calculation Flow

```
 Business requirement
 "Process N contracts/day"
          │
          ▼
 Estimate QPS
 QPS = N ÷ 86,400 (seconds/day)
          │
          ▼
 Measure avg tokens per request
 = system_prompt_tokens
   + retrieved_context_tokens
   + user_query_tokens
   + expected_output_tokens
          │
          ▼
 Monthly token volume
 = QPS × avg_tokens × 2,592,000 (seconds/month)
          │
          ▼
 Monthly cost
 = token_volume × price_per_token
 (check provider pricing page — fast-evolving)
          │
          ▼
 Does cost ≤ budget ceiling?
 ├── YES ──► proceed with model
 └── NO  ──► consider smaller model or
             increase context efficiency
```

---

## Key Concepts

### Context Window Size

Context window size is the maximum number of tokens — input plus output combined — that a model can process in a single request. It acts as a hard ceiling: any prompt that exceeds the window is truncated or rejected, making context window the first constraint to evaluate before all others.

Under the hood, transformer models process all tokens in the window simultaneously using self-attention, which scales quadratically in memory with sequence length. This means very large context windows require substantially more GPU memory per request and can increase latency even when only a small portion of the window is used. Providers therefore charge more per token for large-context variants, and throughput drops as context length grows.

In Databricks, context window limits are documented per model on the FMAPI supported-models page (fast-evolving — verify before use). The `max_tokens` parameter in any FMAPI or `ChatDatabricks` call controls the maximum *output* token count, not the total window; the total window is a hard model property. For a RAG application, the usable context budget is: `window_size − system_prompt_tokens − expected_output_tokens = available_retrieval_context_tokens`.

### Inference Speed and Throughput

Inference speed refers to how quickly the model produces a response, measured as time-to-first-token (TTFT) for interactive applications and tokens-per-second (TPS) for streaming. Throughput refers to how many concurrent requests a serving endpoint can sustain. These two metrics trade off: serving the same model at high concurrency increases throughput but may increase per-request latency.

Model parameter count is the dominant driver of TTFT: a 120B-parameter model requires roughly 15× more matrix multiplications than an 8B model for the same prompt, resulting in proportionally higher latency on equivalent hardware. Mixture-of-experts (MoE) architectures (e.g., Qwen3.5 122B A10B with 10B active parameters) partially break this rule by activating only a fraction of parameters per token. Serving infrastructure choices — GPU count, batching strategy, quantization — modulate but do not eliminate the size-latency relationship.

On Databricks, the distinction is between pay-per-token endpoints (shared infrastructure, no latency guarantees, suitable for low-volume production) and provisioned throughput endpoints (dedicated capacity, configurable `min_provisioned_throughput` in tokens/minute, appropriate for high-volume or SLA-bound production workloads). Provisioned throughput also supports HIPAA-compliant configurations.

### Task-Type Alignment

Task-type alignment means selecting a model whose training distribution and fine-tuning objective match the task the application will perform. Different task types demand different model capabilities: structured extraction benefits from instruction-following models, code generation requires code-specialized models (e.g., GPT-5.3 Codex), multi-step reasoning benefits from hybrid reasoning models (e.g., Claude Sonnet 4.6 with extended thinking), and high-volume classification benefits from small fast models.

Benchmark scores provide an objective signal for task fit. MMLU (Massive Multitask Language Understanding) signals breadth of factual knowledge. HumanEval measures code generation pass rates. MT-Bench measures multi-turn instruction-following quality. A model with high MMLU but low HumanEval is a poor choice for a code review bot regardless of its general reputation. Benchmarks are imperfect proxies, but they are the only quantitative signal available before running your own evaluation.

On Databricks, model descriptions on the FMAPI supported-models page describe intended task profiles ("excels at structured extraction," "optimized for dialogue use cases," "designed for agentic coding") — these descriptions, combined with benchmark data from provider model cards, inform the initial task-type screen. The AI Playground in the Databricks workspace allows interactive comparison across endpoints before committing to a serving strategy.

### Cost Model

The cost model for LLM API calls is pay-per-token: callers are billed per thousand (or per million) input tokens and per thousand output tokens, with output tokens typically priced higher than input tokens. This pricing structure means cost scales linearly with usage volume and is largely determined by prompt length plus output length, not by time elapsed.

Monthly cost estimation follows the formula: `monthly_cost = QPS × avg_tokens_per_request × seconds_per_month × price_per_token`. For example, at 1 QPS, 2,000 tokens per request, and $0.001 per token, monthly cost ≈ 1 × 2,000 × 2,592,000 × 0.000001 = $5,184/month. This calculation must be performed before model selection is finalized; teams that skip it frequently discover mid-month cost overruns.

On Databricks, External Models expose the underlying provider's per-token pricing (which varies by provider and model tier and changes frequently — always verify on the provider's pricing page). Databricks FMAPI pay-per-token pricing is listed in the Databricks pricing documentation. Provisioned throughput has a different cost structure (reserved capacity billing) that becomes more economical above a break-even QPS threshold relative to pay-per-token.

### Databricks Foundation Model API (FMAPI)

The Foundation Model API (FMAPI) is Databricks' unified serving layer for Databricks-hosted open and proprietary models, accessed through an OpenAI-compatible REST API. It eliminates the need to manage model deployment infrastructure; callers simply target a named endpoint and receive responses in OpenAI-format JSON.

FMAPI operates in three modes: pay-per-token (pre-configured endpoints, zero infrastructure setup, suitable for exploration and moderate production), provisioned throughput (dedicated GPU capacity, performance guarantees, required for fine-tuned models or HIPAA workloads), and AI Functions (optimized for batch inference via `ai_query()` in SQL or Spark). All three modes use the same API contract: OpenAI-compatible `POST /serving-endpoints/{name}/invocations` or the OpenAI Python client pointed at the Databricks workspace URL.

The endpoint catalog is fast-evolving. As of July 2026, hosted models include OpenAI GPT-5.x families (`databricks-gpt-5`, `databricks-gpt-5-mini`, `databricks-gpt-5-nano`, `databricks-gpt-5-2`, `databricks-gpt-5-5`, etc.), Google Gemini 2.5/3.x families, Anthropic Claude 4.x/5 families, Meta Llama 3.3/4 families, and open-weight models (GPT OSS 120B/20B, Gemma 3 12B, Qwen3.5). **Verify the current endpoint name list in the Databricks workspace Serving tab or the supported-models documentation before hardcoding any endpoint name — models are regularly added, retired, and renamed.**

### External Models (AI Gateway)

External Models in Model Serving allow Databricks to proxy requests to third-party LLM providers — OpenAI, Anthropic, Cohere, Amazon Bedrock, Google Cloud Vertex AI, AI21 Labs, and custom OpenAI-compatible providers — through a unified Databricks serving endpoint. The key differentiator is centralized credential management: provider API keys are stored once as Databricks secrets and are never exposed in application code or to end users.

When a request arrives at an external model endpoint, Databricks forwards it to the underlying provider, translating the request into the provider's native format if necessary, and returns the response in OpenAI-compatible format. Callers use the same OpenAI client pattern regardless of whether the backing model is GPT, Claude, or Cohere. As of July 2026, all foundation model requests (both FMAPI and External Models) are routed through Unity AI Gateway (Beta), which adds rate limiting, budget controls, and guardrails.

In the Databricks workspace, external model endpoints are created via the MLflow Deployments SDK (`mlflow.deployments.get_deploy_client("databricks").create_endpoint(...)`) or through the Serving UI. Each endpoint specifies the provider, model name, task type (`llm/v1/chat`, `llm/v1/completions`, `llm/v1/embeddings`), and provider-specific configuration (API key reference, base URL, deployment name). This configuration appears in the `external_model` field of the `served_entities` section.


---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `max_tokens` | Maximum number of output tokens the model will generate | Set to the 95th-percentile output length observed in testing; setting it too low causes truncated responses, too high wastes context budget |
| `temperature` | Randomness of token sampling (0 = deterministic, 2 = very random) | Use 0–0.2 for extraction/classification tasks requiring consistency; use 0.7–1.0 for creative or generative tasks; note: some models (e.g., Claude Sonnet 5) reject this parameter entirely |
| `top_p` | Nucleus sampling threshold — only tokens comprising the top P probability mass are considered | Leave at default (1.0) unless you have specific reason to reduce diversity; do not tune both `temperature` and `top_p` simultaneously |
| `min_provisioned_throughput` | Minimum guaranteed tokens/minute on a provisioned throughput endpoint | Set to `peak_QPS × avg_tokens_per_request × 60`; under-provisioning causes throttling during spikes, over-provisioning wastes reserved capacity |
| Pay-per-token endpoint (FMAPI) | Access mode for pre-configured shared endpoints | Use for development, PoC, and workloads with QPS < ~5; switch to provisioned throughput when latency SLAs tighten or QPS exceeds the break-even point |


---

## Worked Example: Requirement → Decision

**Given:** A compliance team needs to extract specific clause types (limitation-of-liability, indemnification, governing-law) from commercial contracts. Each contract is 15–80 pages. They estimate 500 contracts/day, a latency budget of 30 seconds per contract (batch process, not real-time), an accuracy requirement of >95% correct clause identification, and a monthly cost ceiling of $3,000.

**Step 1 — Identify the goal:** Accurately identify and extract specific legal clause types from long documents; accuracy is the primary success metric, not speed.

**Step 2 — Define inputs:** Each contract = ~40 pages average = ~32,000 tokens. System prompt = ~500 tokens. Output (extracted clauses) = ~1,000 tokens per contract. Total tokens per request = ~33,500.

**Step 3 — Define outputs:** Downstream system expects structured JSON with clause type, clause text, and page number.

**Step 4 — Apply constraints:**
- Context window ≥ 35,000 tokens (to accommodate the largest contracts with prompt overhead)
- Latency ≤ 30 seconds (batch, not interactive — almost any model qualifies)
- Accuracy: task is structured extraction from legal text → instruction-following + reasoning model preferred
- Cost ceiling: 500 contracts/day × 30 days × 33,500 tokens × price/token ≤ $3,000 → price/token ≤ $0.006/1K tokens

**Step 5 — Select the approach:** Use a mid-tier Databricks FMAPI model with a ≥128K token context window, strong instruction-following benchmarks, and pay-per-token pricing within the budget (e.g., `databricks-meta-llama-3-3-70b-instruct` or `databricks-gpt-5-mini` — verify current pricing). A large frontier reasoning model (e.g., GPT-5.5 Pro) would exceed the cost ceiling by ~10× for this volume. A small fast model (e.g., GPT-5-nano) may lack the instruction-following depth for >95% clause accuracy on complex legal language. The mid-tier choice balances all four constraints.

---

## Implementation

```python
# Scenario: Selecting a context-window-adequate FMAPI model for document analysis,
# where documents average 20K tokens and require a 128K-token context model.
# ChatDatabricks is used because it is the LangChain-native integration
# that allows model swaps without changing downstream chain code.

from langchain_databricks import ChatDatabricks
from langchain_core.messages import HumanMessage, SystemMessage

# Model chosen after checking: context window ≥ 128K, instruction-following task,
# cost within budget. Endpoint name verified in Databricks Serving UI (Jul 2026).
# WARNING: endpoint names change frequently — always verify before deploying.
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    max_tokens=1500,       # 95th-percentile expected output length
    temperature=0.0,       # deterministic output for structured extraction
)

system_prompt = (
    "You are a contract analysis assistant. "
    "Extract limitation-of-liability, indemnification, and governing-law clauses. "
    "Return JSON with keys: clause_type, clause_text, page_estimate."
)

def extract_clauses(contract_text: str) -> str:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=contract_text),
    ]
    response = llm.invoke(messages)
    return response.content

# Usage
result = extract_clauses(open("contract.txt").read())
print(result)
```

```python
# Scenario: Routing to an external provider (Anthropic) through AI Gateway
# when data residency requirements mandate a specific provider,
# while keeping application code provider-agnostic.

import mlflow.deployments
from openai import OpenAI

# Step 1: Create the external model endpoint (run once, not per request)
deploy_client = mlflow.deployments.get_deploy_client("databricks")
deploy_client.create_endpoint(
    name="anthropic-claude-extraction-endpoint",
    config={
        "served_entities": [{
            "external_model": {
                "name": "claude-3-5-sonnet-20241022",  # verify current model name
                "provider": "anthropic",
                "task": "llm/v1/chat",
                "anthropic_config": {
                    # Key stored in Databricks Secrets — never in code
                    "anthropic_api_key": "{{secrets/compliance_scope/anthropic_key}}"
                }
            }
        }]
    }
)

# Step 2: Query using OpenAI client — same code pattern as FMAPI
import os
client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints"
)

response = client.chat.completions.create(
    model="anthropic-claude-extraction-endpoint",  # endpoint name, not model name
    messages=[
        {"role": "system", "content": "Extract clauses from the contract below."},
        {"role": "user", "content": open("contract.txt").read()}
    ],
    max_tokens=1500,
    temperature=0.0,
)
print(response.choices[0].message.content)
```

```python
# Anti-pattern: Hardcoding a model endpoint without checking whether the context
# window can fit the actual document size, causing silent truncation on large inputs.

# WRONG — this will silently truncate contracts longer than the model's context window
llm = ChatDatabricks(
    endpoint="databricks-gpt-5-nano",  # context window may be insufficient
    max_tokens=4096,
)
# If a 50-page contract = ~40K tokens and the model window is smaller,
# the API returns a 400 error or silently truncates the input.
# The application "works" in testing with short documents and fails in production.

# Correct approach: always compute max expected input size first
MAX_CONTRACT_TOKENS = 40_000   # 80 pages × 500 tokens/page
SYSTEM_PROMPT_TOKENS = 500
OUTPUT_BUDGET_TOKENS = 1_500
REQUIRED_WINDOW = MAX_CONTRACT_TOKENS + SYSTEM_PROMPT_TOKENS + OUTPUT_BUDGET_TOKENS
# = 42,000 tokens → choose a model with context window ≥ 42K tokens

# Verified: databricks-meta-llama-3-3-70b-instruct has 128K context window (Jul 2026)
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    max_tokens=OUTPUT_BUDGET_TOKENS,
    temperature=0.0,
)
# Before deploying, add a guard:
def safe_invoke(llm, prompt_tokens: int, messages):
    if prompt_tokens > REQUIRED_WINDOW:
        raise ValueError(f"Prompt ({prompt_tokens} tokens) exceeds model window")
    return llm.invoke(messages)
```

---

## Common Pitfalls & Misconceptions

- **Picking the highest-benchmark model by default** — Beginners conflate "best model in benchmarks" with "best model for my task," assuming the top leaderboard model is always the correct choice. The correct mental model is that benchmark rank is a starting filter, not a final answer; cost, latency, and context window constraints frequently disqualify the top-ranked model before you reach evaluation.

- **Treating `max_tokens` as a context window control** — Newcomers set `max_tokens` to a large value believing it expands how much input the model can read. The correct mental model is that `max_tokens` controls only the *output* token limit; the input context window is a fixed property of the model and cannot be expanded by any API parameter.

- **Ignoring token cost until after prototyping** — Teams build and test with a powerful frontier model at low volume and are surprised when production cost is 20× the budget. The correct mental model is to compute the monthly cost estimate in Step 1 of design, before any code is written, because the cost difference between a large and small model can be 10–50×.

- **Hardcoding FMAPI endpoint names as permanent identifiers** — Engineers treat endpoint names like `databricks-meta-llama-3-1-70b-instruct` as stable constants in production configs. The correct mental model is that the FMAPI model catalog is explicitly fast-evolving — models are retired, renamed, and replaced on Databricks-controlled timelines; always maintain endpoint names as a configuration value that can be updated without a code deployment.

- **Confusing External Models with FMAPI** — Beginners assume "external" means "less governed" and bypass AI Gateway controls for external providers. The correct mental model is that External Models route through the same Unity AI Gateway as FMAPI, enabling the same rate limiting, budget caps, and audit logging — the distinction is where the model runs (Databricks infrastructure vs. provider infrastructure), not how governed it is.

---

## Key Definitions

| Term | Definition |
|---|---|
| Context window | The maximum total number of tokens (input + output) a model can process in a single request; a hard model property, not configurable via API |
| Time-to-first-token (TTFT) | The latency from sending a request to receiving the first generated token; the primary latency metric for interactive applications |
| Tokens per minute (TPM) | The throughput metric used to size provisioned throughput endpoints; equals QPS × avg_tokens_per_request × 60 |
| Foundation Model API (FMAPI) | Databricks' serving layer for Databricks-hosted models, providing an OpenAI-compatible endpoint with pay-per-token and provisioned-throughput access modes |
| External Models | Databricks Model Serving feature that proxies requests to third-party providers (OpenAI, Anthropic, Cohere, etc.) through a Databricks endpoint, with centralized credential management |
| AI Gateway (Unity AI Gateway) | Databricks control plane layer applied to all foundation model requests; provides rate limiting, budget enforcement, and guardrails |
| Provisioned throughput | An FMAPI access mode that reserves dedicated GPU capacity and guarantees a minimum tokens-per-minute throughput for a named endpoint |
| Pay-per-token | An FMAPI access mode using shared infrastructure, billed per token consumed, with no throughput guarantees; suitable for development and low-to-moderate production traffic |
| Task-type alignment | The degree to which a model's training and fine-tuning objective matches the intended application task (e.g., coding, extraction, reasoning, dialogue) |

---

## Summary / Quick Recall

- Evaluate model attributes in order: context window → latency/throughput → task-type fit → cost; failing the first constraint eliminates the model regardless of other attributes.
- FMAPI provides Databricks-hosted models via pay-per-token (shared, no guarantees) or provisioned throughput (dedicated, guaranteed TPM, required for fine-tuned models).
- External Models proxy third-party providers through a Databricks endpoint; all foundation model requests (FMAPI and External) route through Unity AI Gateway.
- Monthly cost = QPS × avg_tokens × seconds_per_month × price_per_token; compute this before prototyping.
- `max_tokens` controls only output length; context window size is a fixed model property.
- The FMAPI model catalog is fast-evolving — endpoint names are configuration values, not code constants.
- For structured extraction on long documents: prioritize context window ≥ required input size first, then task-type fit, then cost.

---

## Self-Check Questions

1. What does the `max_tokens` parameter control in a Databricks FMAPI or `ChatDatabricks` request?

   <details><summary>Answer</summary>

   `max_tokens` controls the maximum number of *output* tokens the model will generate in the response. It does not expand or control the input context window, which is a fixed property of the model. The most tempting wrong answer is that `max_tokens` sets the total context window size — this is incorrect; setting `max_tokens=8192` does not give a small model a larger context window.

   </details>

2. A team is building a real-time customer support chatbot on Databricks. Average conversation history is 2,000 tokens, the system prompt is 500 tokens, and the expected response is 300 tokens. They need p95 latency under 3 seconds and expect 50 concurrent users. Which FMAPI serving mode should they use, and why?

   <details><summary>Answer</summary>

   Provisioned throughput is the correct choice. The workload has a hard latency SLA (p95 < 3 seconds) and predictable concurrency (50 users), both of which require performance guarantees that pay-per-token (shared infrastructure, no latency guarantee) cannot provide. Pay-per-token is appropriate for development or workloads without latency SLAs; it would be the wrong choice here because shared infrastructure latency is variable and unguaranteed. The `min_provisioned_throughput` should be set to at least 50 × 2,800 tokens × 60 = 8,400,000 tokens/minute (with an appropriate safety margin).

   </details>

3. **Which TWO** of the following statements correctly describe External Models in Databricks Model Serving?
   - A. External model endpoints use a proprietary Databricks request format that differs from the OpenAI API
   - B. Provider API keys are stored as Databricks secrets and are not exposed in application code
   - C. External model requests bypass Unity AI Gateway and therefore cannot have rate limits applied
   - D. The same OpenAI client SDK can be used to query external model endpoints as FMAPI endpoints
   - E. External models can only be accessed via the MLflow Deployments SDK, not the REST API or OpenAI client

   <details><summary>Answer</summary>

   **B and D** are correct. B is correct because External Models specifically provide centralized credential management — provider keys are stored in Databricks Secrets and never appear in code. D is correct because external model endpoints expose an OpenAI-compatible interface, allowing the same OpenAI Python client to query them by pointing `base_url` at the Databricks workspace. A is wrong because the interface *is* OpenAI-compatible. C is wrong because as of July 2026, all foundation model requests (both FMAPI and External Models) route through Unity AI Gateway, enabling rate limits and guardrails. E is wrong because the OpenAI client, REST API, and MLflow Deployments SDK are all supported query methods.

   </details>

4. A contract extraction application processes 200 contracts per day, each approximately 30 pages (~24,000 tokens). The total token budget per request (including system prompt and output) is 26,000 tokens. An engineer proposes using a model with a 16,384-token context window to reduce cost. What will happen and what is the correct approach?

   <details><summary>Answer</summary>

   The 16K-token model will fail on every request (or silently truncate input) because 26,000 tokens exceeds its 16,384-token context window. The correct approach is to first establish the minimum required context window (26,000 tokens in this case), then filter the candidate model list to only models meeting that requirement, and *then* optimize for cost within that constrained set. Using a model with an insufficient context window to save money is a false economy — it produces incorrect or failed outputs. The correct remediation is to either select a model with ≥128K context window or reduce the retrieval context per request to fit within a smaller window.

   </details>

5. A team has two model options for a coding assistant: Model A is a general-purpose 70B model with MMLU score 85% and HumanEval pass@1 of 42%. Model B is a code-specialized model with MMLU score 72% and HumanEval pass@1 of 78%. Both have sufficient context windows and similar pricing. Which should they choose for the coding assistant, and what is the risk of the wrong choice?

   <details><summary>Answer</summary>

   Model B is correct for a coding assistant. HumanEval pass@1 directly measures the model's ability to generate correct code from natural language specifications, which is the primary task. Model A's higher MMLU score reflects broader factual knowledge breadth, not coding ability. The risk of choosing Model A (the "higher-overall-benchmark" model) is that 42% HumanEval vs 78% means roughly half as many code suggestions will be correct on the first attempt, directly degrading the product's usefulness. This illustrates why task-type benchmark alignment is more important than general benchmark ranking when the task is narrow and well-measured.

   </details>

---

## Further Reading

- [Databricks Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-model-apis/index.html) — *verified 2026-07-11* — Overview of FMAPI modes (pay-per-token, provisioned throughput, AI Functions), requirements, and usage guidance
- [Databricks-hosted foundation models available in Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-model-apis/supported-models.html) — *verified 2026-07-11* — **Fast-evolving:** current endpoint names, model families, and context windows; verify before use
- [Supported foundation models on Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/foundation-model-overview.html) — *verified 2026-07-11* — Region availability table for pay-per-token, AI Functions, and provisioned throughput endpoints
- [External models in Model Serving](https://docs.databricks.com/en/generative-ai/external-models/index.html) — *verified 2026-07-11* — External Models feature including supported providers (OpenAI, Anthropic, Cohere, Bedrock, Vertex AI), endpoint configuration, and credential management
- [Provisioned throughput Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-model-apis/deploy-prov-throughput-foundation-model-apis.html) — *verified 2026-07-11* — Step-by-step guide for deploying provisioned throughput endpoints, including `min_provisioned_throughput` configuration
