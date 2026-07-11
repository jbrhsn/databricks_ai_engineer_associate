# Foundation Model APIs and ai_query

**Section:** 05-Assembling and Deploying Applications | **Module:** 02-Vector Search and Serving | **Est. time:** 2.5 hrs | **Exam mapping:** Application Development — Model Serving and Deployment (30%)

---

## TL;DR

Databricks Foundation Model APIs (FMAPI) give you a single, OpenAI-compatible REST interface to query Databricks-hosted and external large language models without managing your own GPU infrastructure. Three endpoint modes — pay-per-token, provisioned throughput, and external models — map directly to different cost, latency, and compliance tradeoffs. The `ai_query()` SQL function brings that same capability into Databricks SQL and notebooks without writing any Python. **The one thing to remember: every FMAPI endpoint exposes the same OpenAI-compatible interface regardless of the underlying model or serving mode, so switching models requires changing only the `model` string.**

---

## ELI5 — Explain It Like I'm 5

Imagine a restaurant with a single, laminated menu that lists every dish from every cuisine — Italian, Japanese, Thai — but you order everything the same way and pay the same waiter. That menu is the OpenAI-compatible API, the waiter is Model Serving, and the kitchens are the actual model backends (Databricks-hosted, Anthropic, OpenAI). You don't need to learn a new ordering system for each kitchen. The most common misconception is that `ai_query()` is a completely separate system from the REST API — it is not; it is a SQL wrapper that calls the exact same Model Serving endpoints under the hood, just without leaving your SQL statement.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Distinguish between pay-per-token, provisioned throughput, and external model endpoint types and select the right one given a workload constraint
- [ ] Call a Foundation Model API endpoint from Python using the `openai` SDK with Databricks authentication
- [ ] Write `ai_query()` SQL expressions to call Model Serving endpoints from Databricks SQL and notebooks
- [ ] Explain the role of Unity AI Gateway (formerly AI Gateway) as a proxy layer providing rate limiting, usage tracking, and guardrails
- [ ] Compare cost and latency tradeoffs between pay-per-token and provisioned throughput for production versus exploratory workloads

---

## Visual Overview

### FMAPI Endpoint Type Decision Tree

```
What is your workload?
├── Exploration / proof-of-concept / low volume
│   └── Pay-per-token endpoint
│       (preconfigured, no setup, billed per input+output token)
│
├── Production / high throughput / SLA required / fine-tuned model
│   └── Provisioned throughput endpoint
│       (dedicated GPU capacity, tokens-per-second guarantee)
│
└── Need a specific external provider (OpenAI, Anthropic, Cohere…)
    └── External model endpoint
        (API key stored in Databricks Secrets, unified interface)
```

### Request Flow Through Unity AI Gateway

```
Python / SQL Client
        │
        ▼
  Model Serving Endpoint
        │
        ▼
  Unity AI Gateway  ──►  Rate limit check
        │               Usage tracking
        │               Guardrail evaluation
        ▼
  Backend model
  (Databricks-hosted │ External provider)
        │
        ▼
  OpenAI-format JSON response
  ──► back to caller
```

### OpenAI-Compatible Endpoint Routes

```
POST /serving-endpoints/{endpoint_name}/invocations
        │
        ├── /chat/completions  ──► multi-turn chat (messages[])
        ├── /completions       ──► single-turn text completion
        └── /embeddings        ──► dense vector output
```

### pay-per-token vs Provisioned Throughput Comparison

```
┌─────────────────────┬──────────────────────────────┬──────────────────────────────┐
│ Dimension           │ Pay-per-token                │ Provisioned Throughput       │
├─────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ Setup               │ Zero — endpoint pre-exists   │ Must create endpoint + alloc │
│ Billing             │ Per input + output token     │ Per model-unit-hour          │
│ Latency guarantee   │ None (shared capacity)       │ Yes (dedicated capacity)     │
│ Fine-tuned models   │ No                           │ Yes                          │
│ Compliance (HIPAA)  │ No                           │ Yes                          │
│ Best for            │ Dev, PoC, low-volume prod    │ High-volume prod, SLA apps   │
└─────────────────────┴──────────────────────────────┴──────────────────────────────┘
```

---

## Key Concepts

### Foundation Model API (FMAPI) — What, How, Where

Foundation Model APIs is a Databricks Designated Service that provides preconfigured and custom serving endpoints for accessing state-of-the-art models without managing GPU deployments. Under the hood, the request is routed through Model Serving, which maps the OpenAI-compatible request body to the backend model's native inference API, normalises the response into OpenAI format, and streams tokens back to the caller. The normalisation layer is what makes every endpoint look identical from client code — `choices[0].message.content` is always where the answer lives. In Databricks you access FMAPI endpoints from the **Serving** tab in the workspace sidebar, via the REST API (`POST /api/2.0/serving-endpoints/{name}/invocations`), or via the `openai` SDK pointing `base_url` at your workspace URL.

> ⚠️ Fast-evolving: The list of supported Databricks-hosted models changes frequently as new model families are added and older models are retired. Always verify current model names at [Supported foundation models](https://docs.databricks.com/en/machine-learning/foundation-model-apis/supported-models.html) before coding endpoint names into production pipelines.

### Pay-per-Token Endpoint Mode

Pay-per-token endpoints are preconfigured serving endpoints available immediately in every supported workspace. You pay for the exact number of input and output tokens consumed per request. The infrastructure is shared across Databricks customers, so there is no throughput guarantee and burst capacity is not reserved. Under the hood Databricks allocates inference requests to a shared GPU fleet; cold-start latency may occur during off-peak hours. In the Databricks UI these endpoints appear at the top of the **Serving > Endpoints** list under the heading "Foundation Model APIs." You reference them by their endpoint name (e.g., `databricks-meta-llama-3-3-70b-instruct`) as the `model` parameter in any OpenAI-compatible call.

### Provisioned Throughput Endpoint Mode

Provisioned throughput allocates dedicated GPU inference capacity — measured in tokens per second — to a specific endpoint. You specify `min_provisioned_throughput` and `max_provisioned_throughput` in tokens per second; Databricks auto-scales between those bounds. Because capacity is reserved, you get consistent low-latency responses regardless of platform load. This mode also supports fine-tuned model variants registered in Unity Catalog (`system.ai`) and is eligible for HIPAA compliance certification. To create a provisioned throughput endpoint you retrieve the `throughput_chunk_size` from the optimisation info API, multiply by integer multiples to set your bounds, then POST to `/api/2.0/serving-endpoints`.

### External Model Endpoints

External model endpoints let you proxy requests to third-party LLM providers (OpenAI, Anthropic, Cohere, Amazon Bedrock, Google Cloud Vertex AI, AI21 Labs, or any OpenAI-compatible custom endpoint) through a Databricks Model Serving endpoint. The API key or credential is stored once in Databricks Secrets and never appears in code; Model Serving injects it at request time. Externally hosted model responses are normalised to OpenAI format, so the calling code does not change. You create an external endpoint via the `served_entities[].external_model` configuration block in the endpoint creation API, specifying `provider`, `name` (the provider-side model name), `task` (`llm/v1/chat`, `llm/v1/completions`, or `llm/v1/embeddings`), and the provider-specific config block.

### OpenAI-Compatible REST Interface

Model Serving exposes a unified REST interface under `/serving-endpoints/{endpoint_name}/invocations` that is compatible with the OpenAI API. The three primary routes are `/chat/completions` (multi-turn conversation using a `messages` array), `/completions` (single-turn text completion using a `prompt` string), and `/embeddings` (dense vector generation using an `input` field). Authentication uses a Databricks personal access token or OAuth M2M token as a `Bearer` token in the `Authorization` header. Because the interface is OpenAI-compatible, any library or tool that supports the OpenAI API can call FMAPI with no code changes beyond `base_url` and `api_key`.

### Calling FMAPI from Python with the `openai` SDK

The `databricks-openai` package wraps the OpenAI client with Databricks-specific authentication auto-configuration. You install it with `pip install -U databricks-openai`, then instantiate the client by pointing `base_url` at `https://<workspace-host>/serving-endpoints` and setting `api_key` to a Databricks personal access token (or relying on environment-variable auto-detection). The `model` parameter receives the FMAPI endpoint name, not a model family name. Under the hood the client sends a standard OpenAI-format JSON body; Model Serving deserialises it, routes to the backend, and returns an OpenAI-format response. For production, Databricks recommends OAuth M2M tokens (service principals) over personal access tokens.

### `ai_query()` SQL Function

`ai_query()` is a general-purpose AI Function that lets you call any supported Model Serving endpoint directly from a SQL `SELECT` statement or Python `selectExpr`. The function signature is `ai_query(endpoint, request [, modelParameters => ...] [, returnType => ...] [, failOnError => ...])`. Under the hood `ai_query` batches rows, sends them to the endpoint over the same REST invocation path used by Python callers, and reassembles results into the DataFrame. It runs on Serverless compute or Databricks Runtime 18.2+; it is not available on Pro or Classic SQL warehouses. When Unity AI Gateway is enabled, `ai_query` calls to Databricks-hosted endpoints are automatically routed through the gateway for usage tracking. For batch inference at scale, Databricks recommends passing the full dataset in a single query and setting `failOnError => false` so successful rows are preserved even if some rows fail.

### Unity AI Gateway

> ⚠️ Fast-evolving: Unity AI Gateway is currently in Beta (as of July 2026). Features, configuration methods, and pricing may change. Verify current capabilities before architectural decisions.

Unity AI Gateway is the Databricks governance layer for AI runtime traffic. It sits between clients and Model Serving backends, applying rate limits (token or request limits per principal per time window), usage tracking (requests, token counts, and latency in `system` tables), budget caps, and service policies (guardrails) that inspect request and response content. The previous per-endpoint "AI Gateway" (configured on individual serving endpoints) remains available; Unity AI Gateway is the newer account-level control plane built on Unity Catalog. All `ai_query` calls to Databricks-hosted endpoints are automatically routed through it when the Beta is enabled. In the UI, Unity AI Gateway appears in the workspace sidebar.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| `min_provisioned_throughput` | Minimum tokens/second guaranteed; sets the floor for auto-scaling | Set to the 10th-percentile throughput of your workload so you avoid paying for idle capacity during quiet periods |
| `max_provisioned_throughput` | Maximum tokens/second; sets the ceiling; billing is per model-unit-hour at this ceiling | Set to the 95th-percentile peak throughput; over-provisioning wastes money, under-provisioning causes queuing |
| `model` (OpenAI SDK / `ai_query`) | Which endpoint receives the request | Use the Databricks endpoint name (e.g., `databricks-meta-llama-3-3-70b-instruct`) not the raw model family name; for `ai_query` use any endpoint name visible in Serving UI |
| `failOnError` (`ai_query`) | Whether a row-level error aborts the entire query | Set to `false` for large batch jobs; set to `true` (default) only when every row must succeed or the pipeline should halt |
| `returnType` (`ai_query`) | Casts the model's string output to a Spark SQL type (e.g., `BOOLEAN`, `STRUCT`) | Use when the model returns structured JSON and downstream transformations need typed columns; omit for raw string output |
| `temperature` | Sampling randomness (0 = deterministic, 1 = creative) | Use 0 for extraction, classification, and eval tasks; use 0.7–1.0 for generation and brainstorming |
| `max_tokens` | Maximum output tokens per request | Set to 2× the expected output length; do not leave unlimited — runaway completions increase latency and cost |
| `task` (external model endpoint) | Route type: `llm/v1/chat`, `llm/v1/completions`, or `llm/v1/embeddings` | Match to the model's capability: use `llm/v1/chat` for instruction-tuned models; use `llm/v1/embeddings` for embedding models |

---

## Worked Example: Requirement → Decision

**Given:** A data engineering team at a financial services firm needs to classify 50 million customer support tickets per day as "complaint", "query", or "compliment". Latency per ticket can be up to 5 seconds. The firm has strict HIPAA data residency requirements and runs existing pipelines as Databricks SQL jobs on Serverless compute. The team wants to avoid managing GPU infrastructure.

**Step 1 — Identify the goal:** Automated batch classification of text at scale with HIPAA compliance and no GPU ops burden.

**Step 2 — Define inputs:** A Delta table `catalog.schema.tickets` with columns `ticket_id STRING` and `text STRING`. The SQL job runs nightly via a Databricks Workflow.

**Step 3 — Define outputs:** A new Delta table `catalog.schema.classified_tickets` with `ticket_id STRING` and `classification STRING` (`"complaint"`, `"query"`, `"compliment"`).

**Step 4 — Apply constraints:**
- Must be HIPAA-eligible → pay-per-token endpoints are NOT eligible; provisioned throughput endpoints ARE.
- Must run from SQL without custom Python → `ai_query()` is the right interface.
- 50M tickets/day at ≤5 s each → need meaningful tokens/second throughput; a small provisioned endpoint (e.g., 2–4 model units) is likely sufficient for nightly batch.
- No GPU ops → provisioned throughput is managed by Databricks; no cluster to maintain.

**Step 5 — Select the approach:** Create a provisioned throughput endpoint for an instruction-tuned model (e.g., `databricks-meta-llama-3-3-70b-instruct`) registered in `system.ai`, with `min_provisioned_throughput` set to 1 chunk and `max_provisioned_throughput` set to 4 chunks. Then call `ai_query()` from the nightly SQL job with `failOnError => false` and `returnType => 'STRING'`. This approach satisfies HIPAA (provisioned throughput is HIPAA-eligible), avoids GPU ops (Databricks manages the endpoint), and uses SQL-native syntax that integrates cleanly with the existing Workflow. The alternative — a Python notebook with the OpenAI SDK — would require a compute cluster and introduce operational overhead without adding capability.

---

## Implementation

```python
# Scenario: Call a Databricks-hosted LLM from Python to generate summaries of
# customer feedback in a proof-of-concept pipeline where cost is more important
# than throughput guarantees (pay-per-token mode, no endpoint provisioning needed).

import os
from openai import OpenAI

# Use databricks-openai package for auto-configured auth, or set manually:
client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],          # PAT or OAuth M2M token
    base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints",
)

feedback_text = "The onboarding process was confusing and took too long."

response = client.chat.completions.create(
    model="databricks-meta-llama-3-3-70b-instruct",  # pay-per-token endpoint name
    messages=[
        {"role": "system", "content": "You are a concise customer insight analyst."},
        {"role": "user", "content": f"Summarise this feedback in one sentence: {feedback_text}"},
    ],
    max_tokens=60,
    temperature=0.2,
)

summary = response.choices[0].message.content
print(summary)
```

```python
# Scenario: Embed product descriptions for semantic search in a RAG pipeline
# using the BGE embedding model available on Databricks pay-per-token.

import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints",
)

texts = [
    "Lightweight hiking boots with Gore-Tex waterproofing",
    "Trail running shoes for rocky terrain",
]

response = client.embeddings.create(
    model="databricks-bge-large-en",   # BGE embedding model endpoint
    input=texts,
)

vectors = [item.embedding for item in response.data]
print(f"Embedding dimension: {len(vectors[0])}")  # 1024 for bge-large-en
```

```sql
-- Scenario: Classify 10 million support tickets as complaint/query/compliment
-- using ai_query() in a Databricks SQL batch job. failOnError=false preserves
-- successful rows when a minority of rows produce model errors.

CREATE OR REPLACE TABLE catalog.schema.classified_tickets AS
SELECT
  ticket_id,
  text,
  ai_query(
    'my-llama-provisioned-endpoint',        -- provisioned throughput endpoint name
    CONCAT(
      'Classify the following support ticket as exactly one of: ',
      '"complaint", "query", or "compliment". ',
      'Reply with only the single label word. Ticket: ', text
    ),
    modelParameters => named_struct('max_tokens', 10, 'temperature', 0.0),
    failOnError => false
  ) AS classification
FROM catalog.schema.tickets;
```

```python
# Anti-pattern: Using a raw model family name instead of the Databricks endpoint name,
# and pointing base_url at the generic OpenAI API — this will fail with a 404 or auth error
# because Databricks endpoints have a different base URL and naming convention.

# WRONG — this hits OpenAI's servers, not Databricks, and leaks no Databricks token:
from openai import OpenAI
bad_client = OpenAI(api_key="sk-openai-key")   # OpenAI key, not Databricks token
bad_response = bad_client.chat.completions.create(
    model="llama-3.3-70b-instruct",  # not a valid OpenAI model; also wrong endpoint
    messages=[{"role": "user", "content": "Hello"}],
)

# Correct approach: point base_url at your Databricks workspace and use the
# Databricks endpoint name (prefixed with 'databricks-' for hosted models):
from openai import OpenAI
import os
good_client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints",
)
good_response = good_client.chat.completions.create(
    model="databricks-meta-llama-3-3-70b-instruct",  # correct endpoint name
    messages=[{"role": "user", "content": "Hello"}],
)
```

```python
# Scenario: Create an external model endpoint for Anthropic Claude so the
# data science team can access it through Databricks Model Serving with
# centralised credential management and AI Gateway rate limiting.

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

client.create_endpoint(
    name="anthropic-claude-chat",
    config={
        "served_entities": [
            {
                "name": "claude-entity",
                "external_model": {
                    "name": "claude-3-5-sonnet-20241022",
                    "provider": "anthropic",
                    "task": "llm/v1/chat",
                    "anthropic_config": {
                        # Key stored in Databricks Secrets — never in plaintext
                        "anthropic_api_key": "{{secrets/my_scope/anthropic_key}}"
                    },
                },
            }
        ]
    },
)
# After creation, query it identically to any FMAPI endpoint:
# client = OpenAI(api_key=DATABRICKS_TOKEN, base_url=f"{HOST}/serving-endpoints")
# client.chat.completions.create(model="anthropic-claude-chat", messages=[...])
```

---

## Common Pitfalls & Misconceptions

- **Hardcoding model family names as `model` values** — Beginners copy model names from blog posts (e.g., `"llama-3.3-70b"`) without realising that the `model` field in FMAPI calls must be the Databricks *endpoint* name, not the underlying model's canonical name. The correct value for a pay-per-token Llama 3.3 70B call is `"databricks-meta-llama-3-3-70b-instruct"`.

- **Using `ai_query()` on Pro or Classic SQL warehouses** — Because `ai_query()` requires serverless compute (or DBR 18.2+ on classic compute), beginners often hit a "function not supported" error on legacy warehouse types. The fix is to switch the SQL warehouse to Serverless or run the query from a Serverless job.

- **Assuming pay-per-token is always cheaper** — Pay-per-token has no upfront cost but variable per-request pricing. At sustained high throughput (thousands of requests per minute), provisioned throughput becomes cheaper per token because you are paying a flat hourly model-unit rate. Beginners default to pay-per-token for everything; the correct mental model is: use pay-per-token for development and bursty low-volume workloads, and evaluate provisioned throughput for sustained production loads.

- **Treating external model endpoints as a passthrough without governance** — Engineers sometimes configure external model endpoints expecting the same governance they have on Databricks-hosted endpoints. The key insight is that governance (rate limiting, guardrails, usage tracking via Unity AI Gateway) applies to *endpoint-level* configuration, not to the provider. You must explicitly configure AI Gateway features on the external model endpoint to get them.

- **Splitting large batches manually before calling `ai_query()`** — Some engineers split a 10M-row table into 100 small chunks and call `ai_query()` in a loop, believing this speeds up processing. This actually reduces throughput because `ai_query()` is designed to parallelise internally when given the full dataset in one query. The correct pattern is to submit the entire table in a single `SELECT ... ai_query(...)` statement.

- **Confusing Unity AI Gateway (Beta) with the legacy per-endpoint AI Gateway** — The workspace sidebar now shows "AI Governance (Unity AI Gateway)" which is account-level and built on Unity Catalog; it is distinct from the per-endpoint AI Gateway configured via the `ai_gateway` field on individual serving endpoints. Both exist simultaneously during the transition period; be explicit about which you are configuring.

---

## Key Definitions

| Term | Definition |
|---|---|
| Foundation Model API (FMAPI) | A Databricks Designated Service that provides OpenAI-compatible serving endpoints for accessing both Databricks-hosted and external foundation models without managing GPU infrastructure |
| Pay-per-token mode | An FMAPI endpoint mode using shared Databricks-managed GPU capacity, billed per input and output token, with no throughput guarantees; recommended for development and low-volume workloads |
| Provisioned throughput mode | An FMAPI endpoint mode that allocates dedicated GPU capacity measured in tokens per second, provides performance guarantees, supports fine-tuned models, and is eligible for HIPAA compliance |
| External model endpoint | A Model Serving endpoint that proxies requests to third-party LLM providers (OpenAI, Anthropic, Cohere, etc.) via Databricks, with credentials stored in Databricks Secrets and responses normalised to OpenAI format |
| `ai_query()` | A Databricks SQL function that calls any Model Serving endpoint from within a SQL `SELECT` statement, enabling batch LLM inference without writing Python code |
| Unity AI Gateway | A Beta account-level governance layer built on Unity Catalog that routes AI traffic through rate limits, budget caps, usage tracking, and service-policy guardrails |
| Model unit | The unit of dedicated inference capacity in provisioned throughput; each model has a `throughput_chunk_size` in tokens/second; you provision in integer multiples of this chunk |
| OpenAI-compatible interface | A REST API that accepts and returns JSON in the same format as the OpenAI API (`/chat/completions`, `/completions`, `/embeddings`), allowing any OpenAI-SDK-compatible client to call Databricks endpoints unchanged except for `base_url` and `api_key` |
| `databricks-openai` package | A Python package (`pip install databricks-openai`) that provides an OpenAI client pre-configured for Databricks authentication, simplifying FMAPI calls |
| `failOnError` | An `ai_query()` parameter (default `true`) that when set to `false` allows the query to complete with error messages in failed rows rather than aborting the entire batch |

---

## Summary / Quick Recall

- FMAPI has three modes: **pay-per-token** (shared, no setup, dev-friendly), **provisioned throughput** (dedicated, SLA, HIPAA-eligible, fine-tune-capable), **external models** (third-party providers via Databricks proxy).
- All FMAPI endpoints expose the same **OpenAI-compatible REST interface**; only `base_url` and the `model` string (endpoint name) differ from OpenAI calls.
- The `model` parameter must be the **Databricks endpoint name** (e.g., `databricks-meta-llama-3-3-70b-instruct`), not the raw model family name.
- `ai_query(endpoint, request)` enables **SQL-native LLM inference**; requires Serverless compute or DBR 18.2+; not available on Pro/Classic warehouses.
- For large batch jobs, pass the **full dataset in one query** — `ai_query()` handles parallelisation internally; set `failOnError => false`.
- **Unity AI Gateway** (Beta, July 2026) provides account-level rate limits, usage tracking, and guardrails; it automatically wraps `ai_query()` calls to Databricks-hosted endpoints when enabled.
- Provisioned throughput is configured via `min_provisioned_throughput` / `max_provisioned_throughput` in tokens/second, queried from the model optimisation info API.

---

## Self-Check Questions

1. What must you specify as the `model` parameter when calling a Foundation Model API endpoint via the OpenAI SDK?

   <details><summary>Answer</summary>

   You must specify the **Databricks endpoint name** — for example, `"databricks-meta-llama-3-3-70b-instruct"` — not the underlying model's canonical name (e.g., `"llama-3.3-70b"`). The `model` field in FMAPI calls maps to a Model Serving endpoint, not directly to a model family. The most tempting wrong answer is using the bare model family name; this fails because Databricks routing resolves the endpoint name, not the model name.

   </details>

2. A data engineer runs the following query on a Pro SQL warehouse and receives an error:

   ```sql
   SELECT ai_query('databricks-gpt-oss-120b', review_text) AS summary
   FROM catalog.schema.reviews;
   ```

   What is the most likely cause, and how should it be fixed?

   <details><summary>Answer</summary>

   `ai_query()` is not supported on Pro or Classic SQL warehouses; it requires **Serverless compute** (or Databricks Runtime 18.2+ on compute clusters). The fix is to change the SQL warehouse type to **Serverless**, or to run the query from a Serverless Databricks Workflow job or a notebook with Serverless compute attached. A common wrong guess is that the endpoint name is incorrect — but the error here is the warehouse type, not the model reference.

   </details>

3. **Which TWO** of the following statements correctly describe when to choose provisioned throughput over pay-per-token for a Foundation Model API endpoint?
   - A. The workload requires HIPAA-eligible data handling
   - B. The application is a developer proof-of-concept with intermittent traffic
   - C. The deployment must serve a fine-tuned model variant registered in Unity Catalog
   - D. The team wants to avoid specifying `min_provisioned_throughput` and `max_provisioned_throughput`
   - E. The endpoint is for one-off batch jobs that run monthly

   <details><summary>Answer</summary>

   **A and C** are correct. Provisioned throughput endpoints are the only mode that qualifies for HIPAA compliance certifications (A), and they are the only mode that supports serving fine-tuned model variants registered in Unity Catalog (C). B is wrong — proofs-of-concept with intermittent traffic are exactly the pay-per-token use case. D is wrong — provisioned throughput *requires* specifying those bounds; pay-per-token has no such configuration. E is wrong — monthly batch jobs benefit from pay-per-token's no-upfront cost model since dedicated capacity would sit idle between runs.

   </details>

4. A team has an `ai_query()` batch job that processes 5 million rows nightly. They observe that manually splitting the data into 50 batches of 100k rows and running them sequentially completes faster than a single query. What explains this observation, and is the team's approach correct?

   <details><summary>Answer</summary>

   The team's observation is likely incorrect as a general principle. `ai_query()` is designed to handle parallelisation, retries, and scaling automatically when given the full dataset in a single query. Manually splitting into sequential batches *reduces* throughput by preventing the function from using its full internal parallelism. The correct approach is to submit the entire 5M-row dataset in one query. If the team is seeing better performance with manual batching, it is likely an artifact of warm-up behaviour or warehouse session limits on their specific warehouse size — the recommended fix is to increase the Serverless warehouse size or verify the endpoint's throughput capacity, not to split manually.

   </details>

5. A team must proxy requests to both OpenAI GPT models and Anthropic Claude models through Databricks, with centralised API key management, rate limiting, and per-team usage attribution. What is the minimum architectural combination that satisfies all three requirements?

   <details><summary>Answer</summary>

   The correct combination is: (1) **two external model endpoints** — one configured with `provider: "openai"` and one with `provider: "anthropic"`, each storing API keys in Databricks Secrets rather than plaintext — and (2) **Unity AI Gateway** (or the per-endpoint AI Gateway) configured on each endpoint with rate limits and usage tracking enabled. The external model endpoints satisfy centralised credential management and the unified interface. AI Gateway provides rate limiting and the usage tracking needed for per-team attribution via system tables. Using only external model endpoints without configuring AI Gateway satisfies credential centralisation but misses rate limiting and usage attribution. Using Unity AI Gateway on Databricks-hosted endpoints instead of external model endpoints does not solve the requirement to route traffic to OpenAI and Anthropic.

   </details>

---

## Further Reading

- [Databricks Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-model-apis/index.html) — *verified 2026-07-11* — Overview, modes, and requirements for FMAPI
- [Databricks-hosted foundation models available in Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-model-apis/supported-models.html) — *verified 2026-07-11* — Current list of supported model endpoint names by family
- [Use foundation models](https://docs.databricks.com/en/machine-learning/model-serving/score-foundation-models.html) — *verified 2026-07-11* — Querying options (OpenAI SDK, REST, MLflow, `ai_query`)
- [Use `ai_query`](https://docs.databricks.com/en/large-language-models/ai-query.html) — *verified 2026-07-11* — Syntax, parameters, supported models, and best practices for the SQL function
- [Provisioned throughput Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-model-apis/deploy-prov-throughput-foundation-model-apis.html) — *verified 2026-07-11* — Deploying and configuring provisioned throughput endpoints
- [External models in Model Serving](https://docs.databricks.com/en/machine-learning/foundation-models/external-models/) — *verified 2026-07-11* — Configuring OpenAI, Anthropic, Cohere, Bedrock, and custom provider endpoints
- [AI governance with Unity AI Gateway](https://docs.databricks.com/en/ai-gateway/index.html) — *verified 2026-07-11* — Rate limits, usage tracking, guardrails, and budget management (Beta)
- [Enrich data using AI Functions](https://docs.databricks.com/en/large-language-models/ai-functions.html) — *verified 2026-07-11* — Context for `ai_query` within the broader AI Functions family
