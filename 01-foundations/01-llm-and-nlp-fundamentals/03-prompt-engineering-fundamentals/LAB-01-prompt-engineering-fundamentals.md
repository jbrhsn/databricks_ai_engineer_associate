# LAB-01: Prompt Engineering Fundamentals

**Lab:** LAB-01 | **Section:** Foundations | **Module:** LLM & NLP Fundamentals | **Est. time:** 1.5 hrs

## Objective

Iterate a single extraction task from a fragile zero-shot prompt to a reusable, parameterized LangChain `ChatPromptTemplate` that drives a Databricks Foundation Model chat endpoint to return schema-valid, deterministic JSON.

## Prerequisites

- Completed the prompt-engineering-fundamentals notes (zero-shot vs few-shot, system/role messages, temperature, structured output concepts).
- A Databricks workspace with **Foundation Model APIs** enabled and pay-per-token access in a supported region.
- A serverless notebook or an attached all-purpose cluster running Databricks Runtime 14.3 LTS ML or above.
- Ability to install PyPI packages: `databricks-openai`, `databricks-langchain`.
- Authentication: run inside a Databricks notebook (auth is automatic) **or** a Databricks Personal Access Token (PAT) / service-principal token if running from outside the workspace.

> ⚠️ Fast-evolving: verify against current official docs before relying on this. Foundation Model API endpoint/model-service names, the `databricks-openai` client surface, and structured-output support change frequently.

## Setup

Install the query clients, restart Python so the new packages load, then build an OpenAI-compatible client that points at your workspace serving endpoints. Inside a notebook, the token and workspace host are read from the notebook context via `dbutils`; you never hardcode a secret.

```python
# Setup: install the Databricks-authored OpenAI client and the LangChain integration.
# Constraint: we need one client that works for both raw OpenAI-style calls (Steps 1-4)
# and LangChain composition (Step 5), against a Databricks Foundation Model endpoint.
%pip install -qU databricks-openai databricks-langchain openai langchain-core
dbutils.library.restartPython()
```

```python
# Setup: construct the OpenAI-compatible client.
# Preferred path INSIDE a Databricks notebook — auth is injected automatically.
from databricks_openai import DatabricksOpenAI

client = DatabricksOpenAI()

# Pick a current general-purpose chat endpoint. Verify the exact name in the
# workspace Serving UI or the supported-models doc — the catalog drifts.
CHAT_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# --- Portable fallback for running OUTSIDE a notebook (PAT auth) ---
# Uncomment and use this block instead if you are not inside a Databricks notebook.
# import openai
# ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
# DATABRICKS_HOST  = ctx.apiUrl().get()          # e.g. https://<workspace>.cloud.databricks.com
# DATABRICKS_TOKEN = ctx.apiToken().get()        # notebook-scoped PAT
# client = openai.OpenAI(
#     api_key=DATABRICKS_TOKEN,
#     base_url=f"{DATABRICKS_HOST}/serving-endpoints",
# )

# Smoke test: confirm the endpoint answers before iterating on prompts.
resp = client.chat.completions.create(
    model=CHAT_ENDPOINT,
    messages=[{"role": "user", "content": "Reply with the single word: ready"}],
    max_tokens=5,
)
print(resp.choices[0].message.content)
```

> ⚠️ Fast-evolving: verify against current official docs before relying on this. `databricks-meta-llama-3-3-70b-instruct` is used as a stable general-purpose example; newer Claude / GPT / Gemini model-service names (e.g. `databricks-claude-sonnet-4-5`) may be preferred in your workspace. Structured outputs (Step 3) are supported on chat models served via Foundation Model APIs; Anthropic Claude models accept only `json_schema`, not `json_object`.

## Steps

The running task: extract structured fields from a free-text customer support message — `category`, `sentiment`, `priority`, and `product`.

### Step 1 — Baseline zero-shot prompt

Send a single user message with no role priming, no examples, and no format contract. This is the fragile baseline: the model decides the format, the field names, and the wording, so downstream parsing is unreliable.

```python
# Step 1: establish a zero-shot baseline so later improvements are measurable.
# Constraint: no system message, no examples — observe how loose the output is.
TICKET = (
    "Hi, the CloudSync Pro app keeps crashing every time I try to upload a file. "
    "This is really frustrating, I have a deadline tomorrow. Please help ASAP!"
)

baseline = client.chat.completions.create(
    model=CHAT_ENDPOINT,
    messages=[
        {"role": "user",
         "content": f"Extract category, sentiment, priority, and product from this ticket:\n{TICKET}"}
    ],
    max_tokens=256,
)
print(baseline.choices[0].message.content)
# Typical result: prose or a loosely-formatted list — not machine-parseable, keys vary run to run.
```

### Step 2 — Add a system role + few-shot examples

Prime the model with a system message that fixes its persona and the exact output contract, then give 2–3 worked examples (few-shot). The examples teach the label vocabulary (`billing`/`technical`/`account`, `positive`/`neutral`/`negative`, `low`/`medium`/`high`) and the flat `key: value` shape, sharply reducing variance versus Step 1.

```python
# Step 2: add a system role + few-shot examples to constrain vocabulary and shape.
# Constraint: still plain text output — we are isolating the effect of role + examples
# before layering structured-output enforcement in Step 3.
SYSTEM = (
    "You are a support-ticket triage assistant. Classify each ticket and reply with "
    "exactly four lines: 'category:', 'sentiment:', 'priority:', 'product:'. "
    "category is one of [billing, technical, account]. "
    "sentiment is one of [positive, neutral, negative]. "
    "priority is one of [low, medium, high]. product is the product name or 'unknown'."
)

FEW_SHOT = [
    {"role": "user", "content": "My invoice charged me twice this month, can you refund one?"},
    {"role": "assistant", "content": "category: billing\nsentiment: negative\npriority: medium\nproduct: unknown"},
    {"role": "user", "content": "Love the new DataViz dashboard, just wanted to say thanks!"},
    {"role": "assistant", "content": "category: account\nsentiment: positive\npriority: low\nproduct: DataViz"},
]

few_shot_resp = client.chat.completions.create(
    model=CHAT_ENDPOINT,
    messages=[{"role": "system", "content": SYSTEM}] + FEW_SHOT +
             [{"role": "user", "content": TICKET}],
    max_tokens=128,
)
print(few_shot_resp.choices[0].message.content)
# Compare against Step 1: consistent labels, consistent four-line shape, no stray prose.
```

### Step 3 — Enforce structured JSON output

Text is still brittle to parse. Pass a `response_format` of type `json_schema` with `strict: True` so the endpoint is contractually bound to return JSON matching your schema. Then parse with `json.loads` — no regex, no string surgery.

```python
# Step 3: enforce a JSON schema so output is guaranteed machine-parseable.
# Constraint: FMA supports a SUBSET of JSON schema — keep it flat, use enums, avoid
# anyOf/oneOf/$ref/pattern. This is the right choice when output feeds a pipeline.
import json

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "ticket_triage",
        "schema": {
            "type": "object",
            "properties": {
                "category":  {"type": "string", "enum": ["billing", "technical", "account"]},
                "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                "priority":  {"type": "string", "enum": ["low", "medium", "high"]},
                "product":   {"type": "string"},
            },
            "required": ["category", "sentiment", "priority", "product"],
        },
        "strict": True,
    },
}

structured_resp = client.chat.completions.create(
    model=CHAT_ENDPOINT,
    messages=[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": TICKET}],
    response_format=response_format,
    max_tokens=128,
)
raw = structured_resp.choices[0].message.content
parsed = json.loads(raw)          # succeeds because the schema is enforced
print(parsed)
print(type(parsed), parsed["category"], parsed["priority"])
```

### Step 4 — Sweep temperature and max_tokens

`temperature` controls sampling randomness; `max_tokens` caps output length. Run the same schema-constrained request at `temperature=0.0` (greedy, near-deterministic) versus `temperature=0.9` (diverse) and observe that extraction tasks want low temperature for repeatability, while a too-small `max_tokens` truncates JSON into invalid output.

```python
# Step 4: quantify the effect of temperature/max_tokens on a structured task.
# Constraint: extraction must be repeatable, so we expect temp=0.0 to be stable.
def triage(temperature, max_tokens=128):
    r = client.chat.completions.create(
        model=CHAT_ENDPOINT,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": TICKET}],
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content

cold = [triage(0.0) for _ in range(3)]   # expect 3 identical strings
hot  = [triage(0.9) for _ in range(3)]   # may vary in product/sentiment wording

print("temp=0.0 identical?", len(set(cold)) == 1)
print("temp=0.0:", cold[0])
print("temp=0.9 distinct count:", len(set(hot)))

# Anti-pattern: starving the model of output tokens truncates JSON -> json.loads raises.
# Uncomment to observe the failure, then note why it breaks.
# broken = triage(0.0, max_tokens=8)
# json.loads(broken)   # raises json.JSONDecodeError — the closing brace never arrived.
#
# Corrected version: give enough headroom for the full object (128 is ample here).
fixed = triage(0.0, max_tokens=128)
json.loads(fixed)      # parses cleanly
print("fixed parses OK")
```

### Step 5 — Refactor into a reusable LangChain ChatPromptTemplate

Hardcoded message lists do not scale. Move the system contract and few-shot examples into a `ChatPromptTemplate` with a `{ticket}` variable, bind it to a `ChatDatabricks` model, and pipe them into a chain (`prompt | llm`). Now any teammate can call the same governed prompt with a new ticket and no copy-paste.

```python
# Step 5: package the prompt as a parameterized, reusable template.
# Constraint: same endpoint, temp=0 for determinism; the template is the unit of reuse.
from databricks_langchain import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate

llm = ChatDatabricks(
    endpoint=CHAT_ENDPOINT,   # inside a notebook, auth is automatic
    temperature=0.0,
    max_tokens=128,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user",   "My invoice charged me twice this month, can you refund one?"),
    ("assistant", "category: billing\nsentiment: negative\npriority: medium\nproduct: unknown"),
    ("user",   "Love the new DataViz dashboard, just wanted to say thanks!"),
    ("assistant", "category: account\nsentiment: positive\npriority: low\nproduct: DataViz"),
    ("user",   "{ticket}"),   # <-- the only thing that changes per call
])

chain = prompt | llm

result = chain.invoke({"ticket": TICKET})
print(result.content)

# Reuse on a fresh ticket with zero prompt edits:
result2 = chain.invoke({"ticket": "I forgot my password and cannot log into my account."})
print(result2.content)
```

## Validation

The lab succeeds when (1) the Step 3 output parses as valid JSON with all four expected keys and enum-valid values, and (2) the Step 4 `temperature=0.0` runs are byte-for-byte identical.

```python
# Validation: assert schema conformance and determinism. All asserts must pass silently.
import json

# (1) Step 3 output is valid JSON with the expected keys and enum values.
parsed = json.loads(structured_resp.choices[0].message.content)
assert set(parsed.keys()) == {"category", "sentiment", "priority", "product"}, parsed
assert parsed["category"]  in {"billing", "technical", "account"}, parsed
assert parsed["sentiment"] in {"positive", "neutral", "negative"}, parsed
assert parsed["priority"]  in {"low", "medium", "high"}, parsed
assert isinstance(parsed["product"], str) and parsed["product"], parsed

# (2) temperature=0.0 is repeatable across runs.
runs = [triage(0.0) for _ in range(3)]
assert len(set(runs)) == 1, f"temp=0 not deterministic: {runs}"

print("VALIDATION PASSED")
```

**Expected output:** the extraction JSON prints as `{'category': 'technical', 'sentiment': 'negative', 'priority': 'high', 'product': 'CloudSync Pro'}` (exact `product` string may vary slightly), and the console shows `VALIDATION PASSED`. If the first assert fails, your `response_format` schema was not honored (check the endpoint supports structured outputs); if the determinism assert fails, confirm `temperature=0.0` and that the endpoint isn't a high-variance reasoning model.

## Teardown

- No persistent resources are created by this lab — there are **no** serving endpoints, Vector Search indexes, tables, or registered models to delete. Foundation Model API pay-per-token endpoints are shared and managed by Databricks.
- Do not leave tokens in notebook cells or output. If you used the PAT fallback, clear the variables (`del DATABRICKS_TOKEN`) and, for a real PAT, revoke it under **Settings → Developer → Access tokens** when finished.
- Detach the notebook from compute (**Detach** in the cluster selector) or let a serverless session auto-terminate to stop incurring compute charges.
- Optionally clear notebook outputs (**Edit → Clear → Clear all cell outputs**) so extracted text and any tokens are not persisted in the notebook file.

## Reflection Questions

1. **What breaks if you change X?** If you drop the system message from Step 2 (or raise `temperature` to 0.9 for the Step 3 extraction), what specifically degrades — label consistency, JSON validity, or determinism — and why does an extraction task punish high temperature more than a brainstorming task would?
2. **What would you do differently in production?** How would you version and govern this prompt (e.g. register it with MLflow so changes are tracked), attach an evaluation harness to catch regressions when the endpoint model is upgraded, and add guardrails (schema validation, AI Gateway rate limits/PII filtering) before this feeds a downstream system?
3. **How does this connect to the next topic?** The `ChatPromptTemplate | ChatDatabricks` chain you built is the exact seam where retrieved context is injected in a RAG pipeline — how would you extend this template to accept a `{context}` variable, and how might the choice of model family (Llama vs Claude vs GPT) change your structured-output strategy given that Claude accepts only `json_schema`?
