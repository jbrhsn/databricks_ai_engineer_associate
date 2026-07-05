# Matching Models, Tasks, and Problem Decomposition

**Section:** Foundations | **Module:** GenAI and Prompting Fundamentals | **Est. time:** 1.5 hrs
**Exam mapping:** Databricks GenAI Engineer Associate — Foundations domain (~15%); model selection and problem decomposition reasoning appear in Application Development scenarios

## Learning Objectives

By the end of this chapter, you will be able to:

- Classify a given NLP task into the correct category (classification, extraction, generation, summarization, transformation)
- Evaluate a model selection decision using the four key trade-off dimensions: capability, latency, cost, and openness
- Explain the trade-offs between open and closed models and when each is appropriate
- Decompose a complex multi-step user request into a sequence of focused, independently testable LLM sub-tasks
- Identify when a task should not use an LLM at all

## Core Concepts

### LLM Task Taxonomy

Before selecting a model or designing a pipeline, you need to name the task precisely. LLM tasks
fall into a small set of categories, and the category determines appropriate models, prompting
strategies, and evaluation metrics.

| Category                       | What it does                                                 | Examples                                                                                     |
| ------------------------------ | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| **Classification**       | Assigns input to one of a fixed set of labels                | Sentiment analysis, intent detection, content moderation, topic routing                      |
| **Extraction**           | Identifies and pulls structured data from unstructured text  | Named entity recognition, key-value extraction, form parsing                                 |
| **Generation**           | Produces new text from a prompt                              | Answering questions, writing code, drafting emails, chatbot responses                        |
| **Summarization**        | Condenses a longer text while preserving meaning             | Meeting notes → summary, long document → abstract                                          |
| **Transformation**       | Reformats or translates text without adding/removing meaning | Translation, format conversion (prose → JSON), style rewriting                              |
| **Reasoning / planning** | Multi-step logic, math, or decision-making                   | SQL generation from natural language, step-by-step problem solving, tool selection in agents |

**Why it matters:** The category determines how you evaluate quality. Classification is measured
with precision/recall/F1. Generation is evaluated with human preference or LLM-as-judge.
Extraction is measured with exact-match or schema compliance. Knowing the category prevents
applying the wrong evaluation metric.

### The Four Model Selection Dimensions

No model is universally best. Selection is always a trade-off across four dimensions:

**1. Capability** — Can the model do the task well?

- Measured by benchmark scores (MMLU, HumanEval, MATH) and real-world testing on your data
- Larger models generally perform better on reasoning and knowledge tasks
- Specialized fine-tuned models often outperform larger general models on their specific domain

**2. Latency** — How fast does it respond?

- Smaller models (7B–13B) typically generate 50–100 tokens/sec; larger models (70B+) generate 20–40 tokens/sec
- Latency matters for interactive applications (chatbots, copilots) where >3 sec feels slow
- Batch processing tolerates high latency; real-time UI does not

**3. Cost** — How much does it cost per call?

- API pricing is per token (input + output); larger models cost more per token
- At scale, the cost difference between an 8B and a 70B model can be 10x or more
- For tasks the smaller model handles adequately, there is no reason to pay for the larger one

**4. Openness** — Who controls the model weights?

- **Open / open-weight:** Weights are publicly available (Llama, Mistral, DBRX); you can deploy on your own infrastructure, fine-tune, and avoid vendor lock-in
- **Closed / proprietary:** Weights are not accessible; you call via API (GPT-4o, Claude, Gemini); subject to vendor pricing changes, availability, and data privacy terms

### Open vs. Closed Models — Decision Framework

| Dimension          | Open / Open-weight                           | Closed / Proprietary                                  |
| ------------------ | -------------------------------------------- | ----------------------------------------------------- |
| Data privacy       | Full control — data never leaves your infra | Data sent to vendor; subject to their privacy policy  |
| Cost at scale      | Lower (you pay compute, not per-token API)   | Higher at scale; predictable per-token pricing        |
| Fine-tuning        | Full access; can train on proprietary data   | Limited or no access; prompt-based adaptation only    |
| Capability ceiling | Llama 3.1 405B approaches GPT-4 class        | State-of-the-art; closed models often lead benchmarks |
| Deployment burden  | You manage infrastructure, scaling, uptime   | Vendor manages everything; lower ops burden           |
| Vendor lock-in     | None — weights are portable                 | High — switching requires recoding API calls         |

**On Databricks specifically:** Databricks makes it easy to deploy open-weight models via
Model Serving (import from Hugging Face or Unity Catalog) while maintaining Unity Catalog
governance over who can access the endpoint. This is a key exam topic.

### Model Size and Task Fit

A rough practical guide:

| Model size         | Best for                                                                 | Avoid for                                                                       |
| ------------------ | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| **7B–13B**  | Classification, extraction, format conversion, simple Q&A                | Complex multi-step reasoning, code generation from scratch, creative generation |
| **30B–70B** | General chat, complex Q&A, summarization, code assistance                | Real-time <1-second latency requirements; extreme cost sensitivity              |
| **70B+**     | Complex reasoning, agent planning, code generation, long-form generation | Any use case where a 30B model is measurably sufficient                         |

**The right answer in most real-world scenarios:** Test the smallest model first. If it meets
your accuracy threshold and latency SLA, stop there. Scale up only when you have evidence the
smaller model is insufficient.

### Problem Decomposition

**Problem decomposition** is the practice of breaking a complex user request into a sequence
of simpler, focused sub-tasks — each solved by a separate LLM call (or sometimes a non-LLM step).

**Why decompose?**

- Complex tasks in a single prompt produce lower quality than a chain of focused prompts
- Each step can be independently tested, debugged, and replaced
- Different steps may be best served by different models (e.g., a fast cheap model for
  classification, a capable model for generation)
- Failures are easier to isolate and diagnose

**Decomposition patterns:**

**Sequential chain:** Each step feeds into the next

```
User query
  → Step 1: Classify query type (fast 8B model, temperature=0)
  → Step 2: Retrieve relevant documents (vector search, no LLM)
  → Step 3: Generate answer from query + documents (70B model)
  → Step 4: Format answer for UI (fast 8B model)
```

**Parallel fan-out:** Multiple steps run concurrently, then merge

```
Long document
  → Summarize section 1 (parallel)
  → Summarize section 2 (parallel)
  → Summarize section 3 (parallel)
  → Merge summaries into final summary
```

**Conditional routing:** Classify first, then route to a specialized step

```
User query
  → Classify: "technical question" | "billing question" | "general question"
  → If technical: send to technical specialist prompt + code model
  → If billing: send to billing specialist prompt + structured output
  → If general: send to general assistant prompt
```

**Rule: Each sub-task should be testable in isolation.** If you cannot evaluate step 2's output
without running steps 1 and 3, your decomposition has a problem.

### When NOT to Use an LLM

The exam tests this judgment: not every task benefits from an LLM.

Use a **non-LLM approach** when:

- The task is a **deterministic rule** (e.g., "extract all email addresses" → use a regex)
- The task is **pure math** (e.g., "calculate compound interest" → use Python)
- The task has a **structured data format** (e.g., "parse a CSV" → use pandas)
- **Latency < 100ms is required** — LLM inference cannot reliably achieve this
- **Cost per call is prohibitive** for a high-volume low-value task

Use an LLM when:

- The task requires **language understanding** (ambiguity, context, inference)
- The rules are **too complex or numerous** to write explicitly
- The input is **unstructured text** that no regex or parser can handle reliably

## Deep Dive / Advanced Topics

### Routing Classifiers — A Practical Pattern

In production GenAI systems, a common pattern is the **routing classifier**: a fast, cheap, small
model that classifies incoming queries and routes them to the appropriate specialized handler.

```python
def route_query(query: str) -> str:
    """Return 'rag', 'direct', or 'reject' for a given user query."""
    response = cheap_client.chat.completions.create(
        model="databricks-meta-llama-3-1-8b-instruct",  # fast, cheap
        messages=[
            {"role": "system", "content": """Classify this query.
Return exactly one word: 'rag' (needs document retrieval), 'direct' (model can answer from knowledge), or 'reject' (off-topic or unsafe)."""},
            {"role": "user", "content": query}
        ],
        temperature=0,
        max_tokens=5
    )
    return response.choices[0].message.content.strip().lower()
```

This pattern lets you:

- Avoid expensive retrieval for queries the model can answer directly
- Block off-topic or unsafe queries before incurring generation cost
- Route specialized queries to a domain-specific model

### Model Cascading (Escalating Complexity)

**Model cascading** tries a cheap small model first; if it expresses low confidence, escalates
to a larger model. Requires the small model to produce a confidence signal (e.g., include a
`"confidence": "HIGH" | "LOW"` field in structured output).

```python
def cascade_answer(query: str) -> str:
    # Try the cheap model first
    result = call_model_with_confidence(query, model="llama-3-1-8b")
    if result.confidence == "HIGH":
        return result.answer
    # Escalate to the capable model
    return call_model(query, model="llama-3-1-70b")
```

This can reduce average cost significantly on mixed-difficulty query distributions.

## Worked Examples & Practice

### Decomposing a Complex Request End-to-End

**User request:** "Summarize our Q3 sales call recordings, identify the top three customer objections, and draft email responses for each objection."

**Naive approach (single prompt, will underperform):**

```python
# One prompt trying to do all three tasks — fragile, hard to test
messages = [{"role": "user", "content": f"Here are the transcripts: {all_transcripts}\nSummarize them, find objections, and draft emails."}]
```

**Decomposed approach:**

```
Step 1: Summarize each transcript independently (parallel calls, 8B model)
  Input: one transcript per call
  Output: 3–5 bullet summary per transcript
  Model: llama-3-1-8b-instruct (fast, cheap — straightforward summarization)

Step 2: Aggregate summaries and extract objections (single call, 70B model)
  Input: all summaries concatenated
  Output: JSON list of top 3 objection types with supporting evidence
  Model: llama-3-1-70b-instruct (needs reasoning across multiple summaries)

Step 3: Draft an email response for each objection (3 parallel calls, 70B model)
  Input: one objection + context per call
  Output: professional email draft
  Model: llama-3-1-70b-instruct (needs quality writing)
```

**Why this is better:**

- Each step has a clear, testable output format
- Steps 1 and 3 run in parallel — total latency is step2 + max(step1_latencies) + max(step3_latencies)
- If the objection extraction (step 2) is wrong, you fix only that prompt — not the whole pipeline
- Different models are used based on actual task difficulty, not one expensive model for everything

### Model Selection Worked Scenario

**Scenario:** A Databricks customer wants to build an automated support ticket classifier that
assigns each incoming ticket to one of 20 predefined categories (P1 severity routing). Expected
volume: 10,000 tickets/day. Latency requirement: <2 seconds. Budget: minimize.

**Task analysis:**

- Type: **Classification** (fixed label set, 20 categories)
- Latency: <2 seconds — achievable with any API-hosted model, but tighter with smaller ones
- Volume: 10,000/day = ~0.12 tickets/second peak; not extreme
- Data sensitivity: support tickets may contain customer PII → data privacy matters

**Model decision:**

1. Start with `llama-3-1-8b-instruct` on Databricks Foundation Model API — fast, cheap, handles classification well
2. Build a few-shot prompt with 2–3 examples per category (or a subset of the most confusing categories)
3. Test on a sample of historical tickets; measure accuracy vs. human labels
4. If accuracy is insufficient, escalate to `llama-3-1-70b-instruct` or fine-tune the 8B model on labeled data
5. Do **not** start with GPT-4o — closed model, data privacy concern, higher cost, no evidence it is needed

**Expected outcome:** 8B model with few-shot prompt typically achieves 80–90%+ on well-defined
classification tasks. Fine-tuning on labeled tickets often pushes it to 95%+.

## Common Pitfalls & Misconceptions

- **Pitfall:** Defaulting to the largest, most capable model for every task → **Why it happens:** More capable = safer choice feels intuitive → **Fix:** Start with the smallest model that plausibly handles the task; test before escalating. Overusing large models is expensive and often unnecessary
- **Pitfall:** Treating "use an LLM" as the solution before defining the task type → **Why it happens:** Excitement about LLMs leads to skipping the "what is the actual task?" question → **Fix:** Name the task category first (classification, extraction, generation, etc.); then decide if an LLM is the right tool vs. a regex, a lookup table, or a traditional ML model
- **Pitfall:** Designing a single-prompt pipeline for a multi-step task because it is simpler → **Why it happens:** Fewer prompts = simpler code, which feels like less work → **Fix:** Accept the upfront complexity of decomposition; it pays back in debuggability, testability, and quality. A single complex prompt is often harder to maintain than a three-step chain
- **Pitfall:** Choosing a closed model for a use case with strict data residency or privacy requirements → **Why it happens:** Closed models are often higher quality; the privacy implications are easy to overlook in development → **Fix:** Confirm data governance requirements before selecting a model provider. If data cannot leave your cloud region or organization, use an open-weight model deployed on your own Databricks Model Serving endpoint
- **Pitfall:** Decomposing tasks without defining the output format of each step → **Why it happens:** Decomposition is focused on the logic, not the interface between steps → **Fix:** Before building a pipeline, define the exact JSON schema or text format for each step's output. Each step's output is the next step's input — mismatches cause silent failures

## Key Definitions

| Term                  | Definition                                                                                                                        |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Classification        | An LLM task that assigns input text to one of a predefined set of labels                                                          |
| Extraction            | An LLM task that identifies and pulls structured data (entities, key-value pairs) from unstructured text                          |
| Generation            | An LLM task that produces new text — answers, code, summaries, creative content                                                  |
| Problem decomposition | Breaking a complex user request into a sequence of simpler, independently testable LLM sub-tasks                                  |
| Routing classifier    | A small, fast model that classifies incoming queries and routes them to appropriate handlers before any expensive generation step |
| Model cascading       | A pattern that tries a cheap, small model first and escalates to a larger model only when the small model signals low confidence  |
| Open-weight model     | An LLM whose weights are publicly released (e.g., Llama, Mistral, DBRX); can be deployed on private infrastructure and fine-tuned |
| Closed model          | An LLM whose weights are not publicly released; accessed only via vendor API (e.g., GPT-4o, Claude, Gemini)                       |

## Summary / Quick Recall

- Five task categories: classification, extraction, generation, summarization, transformation (+ reasoning)
- Four selection dimensions: capability, latency, cost, openness — always a trade-off
- Open-weight: full data control, fine-tunable, no vendor lock-in; closed: higher capability ceiling, less infra burden
- Start with the smallest model that plausibly works; escalate only with evidence
- Decompose complex tasks: each step should be testable in isolation, have a defined output format, use the appropriate model size
- Not every task needs an LLM — use regex/Python/pandas when the task is deterministic, structured, or latency-critical
- Routing classifiers (fast cheap model decides where to send each query) are a key production pattern

## Self-Check Questions

1. A user asks an LLM application: "Extract all dates mentioned in this contract and return them as a list." What task category is this, and what output format should you specify?

<details>
<summary>Answer</summary>
This is an **extraction** task — identifying specific structured data (dates) from unstructured text. The appropriate output format is a JSON array of date strings, e.g., `["2026-01-15", "2026-07-01"]` in a standardized format. Specify the exact format in the system prompt and use JSON mode + Pydantic validation to enforce it.
</details>

2. Your company handles medical records. A colleague proposes using GPT-4o for a clinical note summarization feature because it achieves the best benchmark scores. What concern should you raise?

<details>
<summary>Answer</summary>
Data privacy and compliance. Medical records are protected under regulations like HIPAA (US) or GDPR (EU). Sending them to OpenAI's API means patient data leaves your organization's infrastructure and is processed by a vendor whose data handling policies must be verified for compliance. The better approach: use an open-weight model (e.g., Llama 3.1 70B) deployed on your own Databricks Model Serving endpoint, keeping data within your governed infrastructure. Only if the vendor's data processing agreement explicitly covers the applicable regulations should a closed model be considered.
</details>

3. A user request is: "Read our 200-page product spec, identify all API endpoints, test each one, and write a migration guide for version 2." Why is a single LLM prompt a poor approach, and how would you decompose it?

<details>
<summary>Answer</summary>
A single prompt is poor because: (1) 200 pages likely exceeds the context window; (2) the tasks are heterogeneous — extraction, programmatic testing (not an LLM task at all), and generation; (3) failures in one step are impossible to isolate. Decomposition: Step 1: Chunk the spec and extract API endpoints from each chunk (extraction, parallel, small model). Step 2: Aggregate and deduplicate endpoints (deterministic Python code — no LLM needed). Step 3: Test each endpoint (Python/HTTP calls — no LLM). Step 4: Generate migration guide from endpoint list + test results (generation, large model).
</details>

4. You have a support ticket classifier running at 85% accuracy with a 70B model. A colleague suggests switching to an 8B model to cut costs. How would you approach this decision?

<details>
<summary>Answer</summary>
Run the 8B model on the same evaluation set and measure its accuracy. If 8B achieves ≥85% (or an acceptable accuracy for the business), switch — you get the cost reduction for free. If 8B underperforms, consider fine-tuning the 8B model on labeled tickets, which often brings a small model to near-70B performance on a specific task at a fraction of the inference cost. Only keep the 70B model if fine-tuning also fails to meet the threshold.
</details>

5. Explain the routing classifier pattern and why it is valuable in a production RAG system.

<details>
<summary>Answer</summary>
A routing classifier is a fast, cheap (small) model placed at the front of a pipeline that classifies each incoming query and routes it to the appropriate handler. In a RAG system, it might classify queries as: "needs retrieval" (send to the full RAG pipeline), "direct answer" (model can answer from knowledge — skip retrieval), or "off-topic/unsafe" (reject). The value: expensive retrieval (vector search + embedding + generation) is skipped for queries that don't need it, reducing cost and latency for a significant fraction of real-world queries. Routing adds minimal latency (~50ms) but can cut pipeline cost by 30–50% depending on query distribution.
</details>

## Further Reading

- [Databricks — Supported models on Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-models/supported-models.html) — current list of available models with context window sizes and pricing
- [LMSYS Chatbot Arena leaderboard](https://chat.lmsys.org/) — human preference rankings of current LLMs; useful for calibrating model capability intuitions
- [Databricks — Model Serving for custom models](https://docs.databricks.com/en/machine-learning/model-serving/index.html) — how to deploy open-weight models on your own Databricks endpoint
- [LangChain — LCEL (LangChain Expression Language)](https://python.langchain.com/docs/concepts/lcel/) — the primary framework for building decomposed LLM chains covered in Section 03
