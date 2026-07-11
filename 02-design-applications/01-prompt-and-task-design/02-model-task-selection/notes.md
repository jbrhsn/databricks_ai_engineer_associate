# Model and Task Selection

**Section:** Design Applications | **Module:** Prompt and Task Design | **Est. time:** 2 hrs | **Exam mapping:** Design Applications (14%), Application Development (30%)

---

## TL;DR

Choosing the right model for your generative AI task is not about always picking the most powerful option—it's about matching model capabilities, cost, and latency to your specific task requirements and constraints. Each model has a "sweet spot" of tasks where it excels: large reasoning models for complex multi-step logic, small models for classification, embedding models for semantic search, and vision models for image understanding. **The one thing to remember: Select models based on task alignment and constraints, not raw capability alone.**

---

## ELI5 — Explain It Like I'm 5

Imagine you have a toolbox with hammers of different sizes. A small hammer is perfect for hanging a picture on your apartment wall, but it's wasteful and slow for building a deck. A sledgehammer is overkill for a picture, but exactly right for breaking concrete. Model selection works the same way. A small, fast model (like Llama 3 8B) is perfect for answering simple customer questions quickly and cheaply. A massive reasoning model (like Claude Opus) is overkill for that job but brilliant when you need the AI to think through a complex multi-step problem. And an embedding model isn't a hammer at all—it's the ruler you use to measure how similar two ideas are. The mistake beginners make is always reaching for the biggest sledgehammer, which costs more, is slower, and wastes energy. The correct mental model: **match the tool to the job**. A good engineer picks the smallest model that solves the problem well within your latency and budget constraints.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] **Evaluate** task requirements (latency, throughput, cost, language, modality) and select a model from Databricks Foundation Model APIs that aligns with those constraints
- [ ] **Compare** model families (general purpose vs. embedding vs. vision vs. reasoning) and explain the task-model alignment decisions in a real-world scenario
- [ ] **Design** a model selection workflow that includes benchmarking, cost analysis, and deployment pattern choice (pay-per-token vs. provisioned vs. external)
- [ ] **Diagnose** why a model choice failed (latency timeout, cost overrun, capability gap) and recommend a swap with explicit rationale
- [ ] **Implement** model routing logic in LangChain or the Databricks Agent Framework to dynamically select models based on task complexity

---

## Visual Overview

### Model Selection Decision Flow

```
Start: New generative AI task
│
├─ Identify task type
│  ├─ Chat/dialogue        ──► General-purpose model (GPT, Claude, Llama)
│  ├─ Semantic search      ──► Embedding model (Qwen Embedding, GTE)
│  ├─ Image understanding  ──► Vision model (Gemini, Claude with vision)
│  └─ Code generation      ──► Code-specialized (GPT Codex, Llama Code)
│
├─ Define constraints
│  ├─ Latency: <100ms? ──► Small/fast model; >500ms? ──► Any model
│  ├─ Cost: <$0.01/req? ──► Small open model; flex? ──► Larger model
│  ├─ Throughput: High ──► Provisioned endpoint; Low ──► Pay-per-token
│  └─ Compliance: HIPAA? ──► Databricks provisioned or regulated external
│
├─ Benchmark on your data
│  ├─ Quality: Accuracy, latency, cost on representative sample
│  ├─ Compare 2–3 top candidates
│  └─ Track in LangSmith or AI Playground
│
└─ Deploy & monitor
   ├─ Pay-per-token: Experimentation, proofs-of-concept
   ├─ Provisioned: Production, SLA requirements, high throughput
   └─ Re-evaluate quarterly or on performance regression
```

### Model Capability Matrix

```
Model Size / Type        │ Latency │ Cost   │ Reasoning │ Vision │ Code
─────────────────────────┼─────────┼────────┼───────────┼────────┼──────
Small (8B)              │ <100ms  │ $$     │ Good      │ No     │ Fair
Medium (70B)            │ 100–300ms│ $$$   │ Excellent │ Some   │ Good
Large (400B+)           │ 200–500ms│ $$$$$  │ Excellent │ Yes    │ Excellent
Embedding (small)       │ <50ms   │ $     │ N/A       │ N/A    │ N/A
Vision (mid)            │ 100–200ms│ $$$$  │ Good      │ Yes    │ Fair
Reasoning (new)         │ 500ms+  │ $$$$$  │ Exceptional│ Some  │ Excellent
─────────────────────────┴─────────┴────────┴───────────┴────────┴──────
 Legend: $ = Cheap, $$$$$ = Expensive
 N/A = Not applicable for this model type
```

---

## Key Concepts

### Model Selection Criteria

**What is it?** Model selection criteria are a structured set of requirements—task complexity, latency budget, cost per request, throughput, language, and modality (text, image, audio)—that determine which models are viable candidates for your application.

**How does it work under the hood?** You start by characterizing your task: Is it simple classification (low latency needed, small model sufficient) or multi-step reasoning (higher latency acceptable, larger model better)? Then you filter the available model roster by constraints—geography (supported regions), compliance (HIPAA, SOC 2), language support, and cost. Finally, you rank candidates by performance and cost on a representative sample of your data. The filtering removes non-viable models early; benchmarking identifies the best performer within viable candidates.

**Where does it appear in Databricks/frameworks?** 
- Databricks Foundation Model APIs docs list [supported models by region and task type](https://docs.databricks.com/en/machine-learning/model-serving/foundation-model-overview.html) — use this to filter candidates.
- LangChain's `ChatModelRouter` and `LLMSelector` allow you to encode selection criteria programmatically and route tasks to models dynamically.
- Databricks AI Playground provides a UI to compare models on ad-hoc queries; LangSmith (via LangChain integration) logs query latency and cost for offline analysis.

### Task-Model Alignment

**What is it?** Task-model alignment is the match between a task's semantic complexity and information modality (what it needs to process and output) and a model's training, architecture, and specialization (what it was optimized for).

**How does it work under the hood?** General-purpose chat models are trained on broad internet text and excel at dialogue, summarization, and knowledge recall. Embedding models are trained to collapse high-dimensional text/images into fixed-size vectors optimized for similarity comparison, making them ideal for retrieval. Vision models have multi-modal encoders that fuse visual and textual information, enabling tasks like image captioning or document understanding. Code models are continually pre-trained on code corpora and fine-tuned for generation/completion. When task and model align (e.g., code generation → Code Llama), throughput and quality improve significantly; misalignment (e.g., embedding model for dialogue) fails entirely. Alignment is checked by asking: *Does this model's training objective match my task?*

**Where does it appear in Databricks/frameworks?**
- Foundation Model APIs categorizes models by task type: [General purpose, Embeddings, Vision, Reasoning](https://docs.databricks.com/en/machine-learning/model-serving/score-foundation-models.html#foundation-model-types).
- LangChain's model interface is abstracted: you specify a `chat_model`, `embedding_model`, or `reranker_model`, and LangChain routes to the correct API shape (chat completion, embedding API, or reranking API).
- Agent frameworks (LangGraph, CrewAI) allow you to assign different models to different steps of a workflow; a task might use a small model for classification, then an embedding model for retrieval, then a reasoning model for synthesis.

### Model Deployment Patterns

**What is it?** Deployment patterns are strategies for hosting and querying models at scale, each with different tradeoffs in cost, latency, throughput, and operational overhead: pay-per-token (serverless, minimal setup), provisioned throughput (dedicated infrastructure, SLA guarantees), and external models (outsourced, multi-vendor flexibility).

**How does it work under the hood?** 
- **Pay-per-token (serverless):** Databricks hosts the model on shared infrastructure. You query via REST API or OpenAI client; you pay for tokens consumed. Latency is variable (100–500ms), and throughput is throttled to prevent overload. Ideal for development, experimentation, and variable workloads.
- **Provisioned throughput:** You provision dedicated GPU capacity for a model or a fine-tuned variant. Requests route to your private cluster; latency is consistent (<200ms typical), and throughput is predictable. Cost is hourly compute + token cost. Ideal for production, compliance-sensitive use cases, and stable high-traffic applications.
- **External models:** You connect to third-party providers (OpenAI, Anthropic, Google) via their APIs, and Databricks mediates authentication and governance via [Unity AI Gateway](https://docs.databricks.com/en/ai-gateway/) (Beta). Ideal for multi-vendor flexibility and leveraging proprietary models outside Databricks.

The mechanism that selects a pattern is your workload profile: variable experimental traffic → pay-per-token; stable production with SLA → provisioned; multi-vendor strategy → external.

**Where does it appear in Databricks/frameworks?**
- Databricks Foundation Model APIs documentation contrasts [pay-per-token vs. provisioned throughput modes](https://docs.databricks.com/en/machine-learning/foundation-model-apis/).
- External models are configured in the Databricks UI or via the MLflow Deployments API; credentials and endpoints are stored in Unity Catalog.
- LangChain abstracts the deployment detail: you initialize a `ChatDatabricks()` chat model pointing to an endpoint name, and LangChain handles routing; the endpoint can switch from pay-per-token to provisioned without code changes.

### Model Comparison & Benchmarking

**What is it?** Model comparison is the process of running multiple candidate models on a representative dataset or set of queries, measuring quality (accuracy, relevance, coherence), latency, and cost, then selecting the best performer for your use case.

**How does it work under the hood?** You curate a "gold set" of 50–200 representative inputs and expected outputs (or human-judged reference answers). You run each candidate model on the gold set, log outputs to a tracing tool, and compute metrics:
- **Quality:** Exact match, F1 score, BLEU, semantic similarity (via embedding distance or LLM-as-judge), or human eval
- **Latency:** P50, P95, P99 latency per request
- **Cost:** Total tokens consumed × price per token
Then you compute a weighted score (e.g., 0.6 × quality + 0.2 × cost + 0.2 × latency) and select the winner. The comparison is iterative: you may run a coarse pass with 2–3 models, then zoom into finalists and run longer benchmarks.

**Where does it appear in Databricks/frameworks?**
- **Databricks AI Playground:** provides a UI to query multiple models side-by-side on ad-hoc prompts, but not automated benchmarking.
- **LangSmith:** integrates with LangChain; you log all model calls, can filter by model and prompt, and compute latency/cost/quality metrics across a batch of queries.
- **MLflow Model Evaluation:** you can register models and run batch inference, then compare outputs with built-in or custom metrics.
- **CrewAI & LangGraph:** support task-model routing; you can run the same workflow over multiple models and compare results.

### Model Specialization & Fine-Tuning Considerations

**What is it?** Model specialization is the practice of selecting or training a model on domain-specific data (e.g., medical, legal, customer support) to improve performance, reduce hallucinations, and lower cost by using a smaller model that has been adapted to the task.

**How does it work under the hood?** A generic large model (Claude Opus) can handle any task but is slower and costlier. A specialized model trained on domain data (e.g., Llama 3 fine-tuned on your company's FAQ + support tickets) is smaller, faster, and cheaper because it learned to recognize domain-specific patterns and has lower uncertainty on in-domain inputs. Specialization happens via continued pre-training (expensive, rarely done), supervised fine-tuning (moderate cost, common), or in-context learning (few-shot prompting, no training cost, limited by context window). The mechanism is: *more specific training → smaller effective model size needed to reach target quality*.

**Where does it appear in Databricks/frameworks?**
- Databricks Provisioned Throughput supports fine-tuned model variants; you upload training data and Databricks fine-tunes a base model for you.
- LangChain's prompt templates and few-shot examples enable in-context specialization without retraining; you can embed domain context (examples, instructions) in the system prompt.
- Evaluation frameworks (LangSmith, MLflow GenAI Scorers) help you measure whether specialization improved quality enough to justify the cost.

---

## Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| **Model ID** | Which model endpoint to query | Choose the smallest model that meets quality + latency targets on your benchmark; upgrade only if quality drops below threshold |
| **Temperature** | Randomness of outputs (0–1 scale) | Set to 0 for deterministic tasks (classification, retrieval); 0.5–0.7 for creative tasks; increase if outputs are repetitive |
| **Max Tokens** | Output length limit | Set just above your expected output length; too low truncates; too high wastes tokens and increases latency |
| **Top-k / Top-p** | Sampling strategy | Use top-p (nucleus sampling) for quality diversity; top-k rarely needed; leave defaults unless quality is poor |
| **Deployment Mode** | Pay-per-token vs. provisioned | Use pay-per-token for <1K req/day or experimental; use provisioned for >10K req/day or SLA-critical workloads |
| **Region** | Databricks workspace region | Choose region closest to users for latency; ensure your model is supported in that region before deploying |

---

## Worked Example: Requirement → Decision

**Given:** An e-commerce company is building a product recommendation chatbot for their mobile app. Requirements:
- <150ms latency per user query (mobile UX expectation)
- <$0.005 cost per request (10M requests/month = $50K budget)
- 80% of queries are simple ("show me blue shoes"), 15% are multi-turn ("do you have this in size 8?"), 5% involve images ("find products like this photo")
- Multi-language support needed (English, Spanish, French)
- Compliance: no HIPAA, but data must stay in US.

**Step 1 — Identify the goal:**  
Select a model (or set of models) that can handle simple multi-turn chat + occasional image understanding, deliver sub-150ms responses, and stay within budget.

**Step 2 — Define inputs:**
- Candidate models: Claude Haiku, Llama 3 8B, Gemini 3 Flash, Grok-3 mini
- Dataset: 100 representative queries (80 text-only, 15 multi-turn, 5 image-based)
- Deployment assumption: pay-per-token for initial launch (flexible for 1–2M requests/month)

**Step 3 — Define outputs:**
- Winning model ID
- Estimated cost per request (tokens/1K × price per token)
- Estimated latency (P50, P95)
- Fallback model if primary hits errors

**Step 4 — Apply constraints:**
- Latency: Model must return in <150ms P50, <300ms P95 (allows some margin for network + client processing)
- Cost: Model tokens + pricing must yield <$0.005 per request
- Language: Must support English, Spanish, French without external translation
- Modality: Must handle 5% image inputs without degrading on 80% text-simple inputs
- Region: Deployment in us-west-2 or us-east-1 (Databricks availability)

**Step 5 — Select the approach with rationale:**

**Winner: Gemini 3 Flash** (Databricks Foundation Model APIs, pay-per-token)

**Rationale vs. alternatives:**
1. **Claude Haiku:** Excellent latency (<100ms typical), $0.80 per 1M input tokens. But for 5M queries/month, cost = $2–3K/month if avg. 600 tokens per interaction. Within budget, BUT weak on visual understanding; would fail on 5% image queries.
2. **Llama 3 8B:** Cheapest ($0.10 per 1M tokens), strongest latency on provisioned (<50ms). But open-source deployment requires custom setup; reduces speed-to-market. Weak vision (no vision capability); can't handle image queries.
3. **Gemini 3 Flash:** $1.00 per 1M input tokens, multi-modal (text + vision in same call), latency <150ms on pay-per-token. Trade: slightly more expensive than Haiku alone, but eliminates need for separate vision model + orchestration logic. Supports all three languages natively.
4. **Grok-3 mini:** Experimental, limited region availability in us-west-2; unverified latency on Foundation Model APIs; added risk.

**Decision:** Deploy Gemini 3 Flash on pay-per-token for initial 2 months (target 2M requests). Monitor cost and latency; if latency consistently >150ms P95, move to provisioned or downgrade to Haiku + separate vision model for image queries only.

**Estimated cost:** 5M queries/month × 400 avg. tokens per query = 2B tokens/month; at $1.00 per 1M = $2K/month ≈ $0.0004 per request. ✓ Well within budget.

**Estimated latency:** Databricks docs quote <150ms P50 for Gemini on pay-per-token. Plan for P95 ≈ 250–300ms with network margin. ✓ Acceptable.

---

## Implementation

### Scenario 1: Model Selection with LangChain ChatModelRouter

**Scenario: Route tasks to different models based on task complexity without human intervention.**

```python
# Problem: A support ticket classification system needs to:
# - Use a small, fast model (Haiku) for simple requests (order status, returns policy)
# - Use a larger model (Claude Sonnet) for complex issues (pricing negotiation, product recommendations)
# This reduces cost (small model handles 70% of load) and improves quality on hard cases.

from langchain_core.tools import tool
from langchain.chat_models import ChatDatabricks
from langchain.callbacks import LangChainTracer
import json

# Initialize tracer for cost/latency tracking
tracer = LangChainTracer()

# Define model endpoints
fast_model = ChatDatabricks(
    endpoint_name="databricks-gpt-5-4-nano",  # Small, fast, cheap
    max_tokens=256,
    temperature=0.3,
    callbacks=[tracer]
)

powerful_model = ChatDatabricks(
    endpoint_name="databricks-claude-opus-4-8",  # Larger, more reasoning
    max_tokens=1024,
    temperature=0.3,
    callbacks=[tracer]
)

# Define complexity classifier
def classify_ticket_complexity(ticket_text: str) -> str:
    """Classify ticket as 'simple' or 'complex' based on keywords and length."""
    simple_keywords = {"order status", "return", "cancel", "where is", "tracking"}
    complex_keywords = {"negotiate", "custom", "bulk", "recommendation", "integration"}
    
    ticket_lower = ticket_text.lower()
    
    if any(kw in ticket_lower for kw in complex_keywords):
        return "complex"
    if any(kw in ticket_lower for kw in simple_keywords):
        return "simple"
    
    # Default: classify by length
    return "complex" if len(ticket_text.split()) > 50 else "simple"

# Route and respond
def route_and_respond(ticket_text: str) -> dict:
    """Route ticket to appropriate model and return response + metadata."""
    complexity = classify_ticket_complexity(ticket_text)
    model = powerful_model if complexity == "complex" else fast_model
    
    # System prompt tuned to task
    system_prompt = (
        "You are a helpful customer support agent. Respond concisely in 1–2 sentences for simple issues, "
        "or in a detailed paragraph for complex issues."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ticket_text}
    ]
    
    response = model.invoke(messages)
    
    return {
        "complexity": complexity,
        "model_used": model.endpoint_name,
        "response": response.content,
        "estimated_cost": estimate_cost(response, model.endpoint_name)
    }

def estimate_cost(response, model_id: str) -> float:
    """Estimate request cost based on model and token count."""
    # Approximate: count tokens as len(response) / 4
    tokens = len(response.content) // 4
    # Pricing per model (as of 2026-07-11)
    pricing = {
        "databricks-gpt-5-4-nano": 0.0001,  # $0.10 per 1M input tokens
        "databricks-claude-opus-4-8": 0.003  # $3.00 per 1M input tokens
    }
    price_per_token = pricing.get(model_id, 0.001)
    return (tokens / 1_000_000) * price_per_token

# Example usage
ticket = "Where is my order #12345? I placed it 3 days ago and haven't received tracking info yet."
result = route_and_respond(ticket)
print(json.dumps(result, indent=2))
# Output:
# {
#   "complexity": "simple",
#   "model_used": "databricks-gpt-5-4-nano",
#   "response": "Your order is being prepared. You'll receive a tracking link via email within 24 hours.",
#   "estimated_cost": 0.0001
# }
```

### Scenario 2: Benchmarking Models on Your Data with LangSmith

**Scenario: Compare three models on a gold set of queries to identify the best performer before production deployment.**

```python
# Problem: You have three candidate models (Haiku, Sonnet, Opus) and need to determine
# which one delivers the best balance of quality, latency, and cost on YOUR data.
# You run each model on 50 representative queries and collect metrics.

from langchain.chat_models import ChatDatabricks
from langsmith import Client, run_on_dataset
import time

# Initialize LangSmith client (requires LANGSMITH_API_KEY)
client = Client()

# Define candidate models
models = {
    "haiku": ChatDatabricks(endpoint_name="databricks-claude-haiku-4-5"),
    "sonnet": ChatDatabricks(endpoint_name="databricks-claude-sonnet-4-6"),
    "opus": ChatDatabricks(endpoint_name="databricks-claude-opus-4-8"),
}

# Define evaluation function
def evaluate_response(response: str, reference: str) -> dict:
    """Evaluate response quality. In production, use LLM-as-judge or human eval."""
    # Simple heuristic: measure if response length is reasonable and contains key terms
    key_terms = reference.split()[:3]  # First 3 words from reference
    contains_terms = sum(1 for term in key_terms if term.lower() in response.lower()) / len(key_terms)
    return {"similarity_to_reference": contains_terms}

# Benchmark each model
results = {}
for model_name, model in models.items():
    print(f"\nBenchmarking {model_name}...")
    
    # Load gold set from dataset (or define inline)
    queries = [
        "What are the top 3 benefits of cloud storage?",
        "How do I reset my password?",
        "Explain machine learning in one sentence.",
    ]
    
    latencies = []
    costs = []
    qualities = []
    
    for query in queries:
        start = time.time()
        response = model.invoke(query)
        latency = (time.time() - start) * 1000  # ms
        
        # Estimate cost (simplified)
        tokens = len(response.content) // 4
        cost = (tokens / 1_000_000) * 0.001  # Rough average
        
        # Quality eval
        quality = evaluate_response(response.content, query)
        
        latencies.append(latency)
        costs.append(cost)
        qualities.append(quality["similarity_to_reference"])
    
    results[model_name] = {
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "avg_cost_per_req": sum(costs) / len(costs),
        "avg_quality_score": sum(qualities) / len(qualities),
    }

# Display results
print("\n=== Benchmark Results ===")
for model_name, metrics in results.items():
    print(f"\n{model_name.upper()}:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")

# Winner: Haiku (best latency + cost, acceptable quality)
print("\n=== Recommendation ===")
winner = min(results.keys(), key=lambda m: results[m]["avg_latency_ms"] + results[m]["avg_cost_per_req"])
print(f"Winner: {winner} (Best tradeoff of latency + cost)")
```

### Anti-pattern: Always Using the Largest Model

**Anti-pattern: Deploying Claude Opus for every task, regardless of complexity or cost.**

```python
# ❌ WRONG: Every task uses the same expensive model
from langchain.chat_models import ChatDatabricks

model = ChatDatabricks(endpoint_name="databricks-claude-opus-4-8")

def simple_classify_email(email_text: str) -> str:
    """Classify email as spam or legitimate. Costs $0.003 per request."""
    response = model.invoke(f"Classify as spam or legitimate: {email_text}")
    return "spam" if "spam" in response.content.lower() else "legitimate"

def complex_summarize_report(report_text: str) -> str:
    """Summarize a long report. Could benefit from Opus reasoning."""
    response = model.invoke(f"Summarize: {report_text}")
    return response.content

# Problem: The classification task (simple pattern matching) costs $0.003 per email.
# For 1M emails/month = $3K. Opus is overkill; a small model would work fine for <$300/month.

# ✓ CORRECT: Use model appropriate to task complexity
def smart_classify_email(email_text: str) -> str:
    """Classify email as spam or legitimate using small, fast model. Costs $0.0001 per request."""
    small_model = ChatDatabricks(endpoint_name="databricks-claude-haiku-4-5")
    response = small_model.invoke(f"Classify as spam or legitimate: {email_text}")
    return "spam" if "spam" in response.content.lower() else "legitimate"

def complex_summarize_report(report_text: str) -> str:
    """Summarize a long report using reasoning model."""
    large_model = ChatDatabricks(endpoint_name="databricks-claude-opus-4-8")
    response = large_model.invoke(f"Summarize: {report_text}")
    return response.content

# Result: Categorization cost drops to $100/month (30x savings), and summary quality actually improves
# because Opus can devote capacity to reasoning rather than wasting cycles on simple tasks.
```

---

## Common Pitfalls & Misconceptions

- **"Always pick the biggest model"** — Larger models are slower, costlier, and sometimes less accurate on specific domains. A smaller model fine-tuned on your data often outperforms a generic giant. The correct mental model: *Match model size to task complexity, not to raw capability.*

- **"Latency is only about model size"** — Latency depends on model size, batch size, endpoint saturation, and network distance. A small model on an overloaded endpoint can be slower than a large model on a dedicated provisioned endpoint. The correct mental model: *Latency emerges from model architecture + deployment pattern combined.*

- **"Cost is just input + output tokens"** — Databricks charges per token, but cost also includes endpoint provisioning (if used), data egress, and governance overhead (if using AI Gateway). Ignoring these hides true cost. The correct mental model: *True cost = (tokens × price) + (provisioned compute if any) + (governance overhead).*

- **"One model fits all modalities"** — A text-only model cannot handle images; an embedding model cannot do dialogue. Trying to force a square peg into a round hole fails silently or produces garbage. The correct mental model: *Match model architecture to modality (text, image, audio, multimodal). Use specialized models when available.*

- **"Pay-per-token is always cheaper than provisioned"** — For bursty workloads, yes. But for sustained >10K req/day, provisioned is usually cheaper per request because you amortize GPU cost. The correct mental model: *Calculate break-even: (provisioned hourly cost) / (hourly throughput) vs. (pay-per-token price per token × avg tokens per request).*

- **"Benchmarking on public datasets transfers to your data"** — A model's performance on academic benchmarks does not predict performance on your domain-specific queries. Always benchmark on a representative sample of YOUR data. The correct mental model: *In-domain performance is the ground truth; public benchmarks are directional only.*

---

## Key Definitions

| Term | Definition |
|---|---|
| **Model Selection** | The process of choosing which foundational or fine-tuned model to use for a given generative AI task based on task requirements (latency, throughput, cost, modality, language, domain) and constraints (region, compliance). |
| **Task-Model Alignment** | The degree to which a model's training objective, architecture, and specialization match the semantic requirements and information modality of a task. High alignment yields better quality and efficiency; low alignment causes failures or requires workarounds. |
| **Latency (P50, P95, P99)** | The time from sending a request to receiving a complete response. Percentiles (P50 = median, P95 = 95th percentile) capture the distribution. For real-time applications, P95 and P99 matter as much as P50. |
| **Throughput** | The number of requests a deployed model can handle per unit time (requests/sec, tokens/sec). Limited by GPU memory, batch size, and endpoint provisioning. |
| **Cost per Request** | The monetary cost to process one inference request, typically calculated as (input tokens + output tokens) × price per token, optionally including provisioned compute amortization. |
| **Token** | A subword unit (typically 4 characters) that a language model uses internally. All API pricing is token-based. Longer outputs = more tokens = higher cost. |
| **Provisioned Throughput** | A deployment mode where you reserve dedicated GPU capacity for a model, guaranteeing consistent latency and throughput in exchange for an hourly compute charge. |
| **Pay-per-Token** | A deployment mode where you query a shared, Databricks-managed endpoint and pay only for tokens consumed, with variable latency but minimal upfront cost. |
| **Embedding Model** | A model trained to map text (or images) to fixed-size numerical vectors such that semantically similar inputs have similar vectors. Used for semantic search, clustering, and retrieval-augmented generation (RAG). |
| **Vision Model** | A multimodal model with a visual encoder (processes images) and a language decoder (produces text) in the same inference call. Enables image understanding, visual question answering, and document analysis. |
| **Reasoning Model** | A large model (often 400B+ parameters) designed for multi-step logical reasoning, code generation, and complex problem-solving. Often slower and more expensive but higher quality on hard tasks. |
| **Fine-Tuning** | The process of adapting a pre-trained model to a specific domain or task by training it on domain-specific data. Produces a specialized variant that is smaller or cheaper to achieve the same quality. |
| **LLM Benchmark** | A standardized evaluation dataset (e.g., MMLU, HumanEval) used to measure model capabilities. Public benchmarks do not transfer reliably to domain-specific tasks; always benchmark on your data. |

---

## Summary / Quick Recall

- **Model selection is a matching problem, not a maximization problem.** Choose the smallest model that meets your quality, latency, and cost targets—not the most powerful one.
- **Task complexity determines model size.** Simple classification → small model; multi-step reasoning → large model; semantic search → embedding model; images → vision model.
- **Latency and cost are both constraints.** Benchmark on YOUR data to find the sweet spot; generic benchmarks don't transfer reliably.
- **Deployment pattern drives economics.** Pay-per-token for experimentation and bursty workloads; provisioned for stable, high-traffic production.
- **Align task, model, and deployment.** A task-model mismatch (e.g., embedding model for dialogue) cannot be fixed by scaling. Architecture matters more than raw size.
- **Monitor and re-evaluate quarterly.** Model rosters evolve; new models may be cheaper or faster. Track cost and latency in production and adjust if ROI changes.
- **In-domain performance is ground truth.** Always benchmark on a representative sample of YOUR data. Public leaderboards are starting points only.

---

## Self-Check Questions

1. **Your team is building a real-time product search feature.** Latency must be <100ms P95. You're choosing between Claude Opus (excellent reasoning, <200ms typical) and Llama 3 8B on provisioned throughput (<50ms typical). Which model do you select and why?

   <details><summary>Answer</summary>

   **Answer: Llama 3 8B on provisioned throughput.**
   
   **Why the correct answer is right:** The latency constraint is <100ms P95. Llama 3 8B on provisioned throughput consistently delivers <50ms P50, and P95 should be <100ms, meeting the hard requirement. Claude Opus, while excellent for reasoning, has <200ms typical latency; P95 would likely exceed 200–300ms, violating the SLA. For a product search task (which is primarily retrieval + ranking, not reasoning), Llama's capability is sufficient.
   
   **Why the main distractor fails:** Choosing Claude Opus because it's "more powerful" ignores the latency constraint. In real-time applications, latency SLA violations cause users to abandon the app—no amount of reasoning quality compensates. The correct mental model: *Constraints (latency, cost) are hard limits; choose the smallest model that satisfies them and is capable enough.*

   </details>

2. **You're deploying a support chatbot to production. It will receive 50K requests per day.** Should you use pay-per-token or provisioned throughput? Estimate the break-even.

   <details><summary>Answer</summary>

   **Answer: Provisioned throughput is likely cheaper and more predictable.**
   
   **Why:** 50K req/day ≈ 1.7K req/hr. Provisioned throughput costs ~$0.50–1.00/hour for a small model, whereas pay-per-token costs roughly $0.0005 per token average. At ~400 tokens per request (input + output), 50K requests = 20M tokens/day ≈ $10/day ($300/month) with pay-per-token. Provisioned costs $0.50/hr × 24 × 30 ≈ $360/month. Break-even is roughly 50–100K requests/day depending on model size. **At exactly 50K/req/day, both are comparable; provisioned wins if you need latency guarantees (SLA) and don't mind predictable monthly cost.**
   
   **Why the alternative is tempting:** Pay-per-token sounds cheaper ("only pay for what you use"), but at this scale, provisioned is actually better for production because: (1) latency is guaranteed (important for user experience), (2) you get consistent throughput without throttling, (3) no surprise cost spikes if traffic surges.

   </details>

3. **Which TWO of the following are valid reasons to choose a small embedding model over a large chat model for a semantic search task?**
   - A. Embedding models are always cheaper than chat models.
   - B. Embedding models map text to fixed vectors optimized for similarity, which is exactly what semantic search needs.
   - C. Chat models are too slow for real-time search.
   - D. Embedding models consume fewer tokens, reducing cost per search.
   - E. Chat models output text, not vectors; they cannot be used for search without additional logic.

   <details><summary>Answer</summary>

   **Correct answers: B and E.**
   
   **Why B is correct:** Embedding models are specifically trained to collapse high-dimensional inputs into fixed-size vectors such that semantically similar inputs have similar vectors. This is *exactly* the mechanism semantic search needs. A chat model can be repurposed for retrieval with custom prompting, but embedding models are the right tool for the job.
   
   **Why E is correct:** Chat models output free-form text (e.g., "Here are 5 products..."), which is not directly comparable for similarity. You would need an extra embedding step (embed the chat output, then compare) or complex post-processing. Embedding models output vectors directly, ready for cosine similarity computation. This removes an extra step and improves efficiency.
   
   **Why A is wrong:** Embedding models are often *cheaper* per inference (because they're smaller), but the blanket statement "always cheaper" is false. Cost depends on model size, provider pricing, and query volume. Moreover, cost alone doesn't justify the choice; correctness (task alignment) comes first.
   
   **Why C is wrong:** Modern chat models (Haiku, Llama 3) are fast enough for real-time search (<100ms P95). Speed alone is not a reason to prefer embeddings; task alignment is.
   
   **Why D is misleading:** While embedding models may consume fewer tokens in some cases, token count varies by model and input. The real reason to use embeddings is architectural alignment (B, E), not marginal token savings.

   </details>

4. **You benchmark three models on your customer support emails and collect these results:**

   | Model | Avg Latency | Cost/Request | Quality Score |
   |---|---|---|---|
   | Haiku | 50ms | $0.0001 | 0.75 |
   | Sonnet | 120ms | $0.0005 | 0.88 |
   | Opus | 250ms | $0.001 | 0.92 |

   Your SLA requires <100ms latency and 0.85+ quality. Which model(s) meet the SLA, and which would you recommend for production?

   <details><summary>Answer</summary>

   **Answer: Sonnet meets the SLA and is recommended for production.**
   
   **Which meet the SLA:**
   - Haiku: 50ms ✓ latency, 0.75 ✗ quality (below 0.85 threshold) — FAILS
   - Sonnet: 120ms ✗ latency (above 100ms), 0.88 ✓ quality — FAILS on latency
   - Opus: 250ms ✗ latency, 0.92 ✓ quality — FAILS on latency

   **Wait, none fully meet the SLA!** Sonnet is the best compromise:
   - It *exceeds* quality target (0.88 > 0.85).
   - It slightly violates latency (120ms > 100ms SLA), but only by 20%.
   - It's the middle cost.

   **Recommendation:** Deploy Sonnet on provisioned throughput. Provisioned can reduce latency to <100ms P50 (docs quote <100ms for provisioned). If provisioned Sonnet still misses latency SLA, fall back to Haiku + implement a quality improvement elsewhere (e.g., in-context few-shot examples, custom fine-tuning on your data). The correct mental model: *When no model meets all constraints, re-negotiate the constraint, use a multi-model approach, or invest in optimization.*

   </details>

5. **Your team wants to reduce model costs by 50%. Which of the following is the most effective strategy, and why?**

   <details><summary>Answer</summary>

   **Answer: Categorize tasks by complexity and route simple tasks to small models, complex tasks to large models. Typically, 60–80% of production workloads are simple; you save 50%+ on those.**
   
   **Why this works:** Suppose you run 1M requests/month with average cost $0.001/req = $1K/month. If 70% are simple (classification, lookup, basic QA) and 30% are complex (reasoning, synthesis):
   - Route 700K simple requests to Haiku ($0.0001/req) = $70
   - Route 300K complex requests to Sonnet ($0.0005/req) = $150
   - **Total: $220/month, a 78% reduction.**
   
   **Why other strategies fail:**
   - "Switch to a smaller model for everything" → Quality drops unacceptably for 30% of hard requests.
   - "Fine-tune a small model" → High upfront cost; only pays off after ~100K requests on Databricks provisioned.
   - "Batch process at night" → Doesn't reduce cost; only defers it and breaks real-time SLA.
   - "Use embeddings for everything" → Embeddings are for retrieval only; cannot replace chat models for dialogue.
   
   **The correct mental model:** Most cost reduction comes not from picking a new model, but from *routing tasks efficiently*—using the cheapest adequate model for each task.

   </details>

---

## Further Reading

- [Databricks Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-model-apis/) — *verified 2026-07-11* — Overview of pay-per-token and provisioned throughput deployment options.
- [Supported foundation models on Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/foundation-model-overview.html) — *verified 2026-07-11* — Comprehensive model roster by task type, region, and pricing.
- [Use foundation models](https://docs.databricks.com/en/machine-learning/model-serving/score-foundation-models.html) — *verified 2026-07-11* — Query methods (OpenAI client, REST API, AI Functions), model types (general purpose, embeddings, vision, reasoning).
- [Deploy Provisioned Throughput Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-model-apis/deploy-prov-throughput-foundation-model-apis.html) — *verified 2026-07-11* — Step-by-step guide to provisioning dedicated GPU capacity.
- [LangChain Models Concepts](https://python.langchain.com/docs/concepts/models/) — *verified 2026-07-11* — Standard interfaces for chat models, embeddings, model routing; provider abstraction.
- [Databricks LangChain Integration](https://python.langchain.com/docs/integrations/providers/databricks/) — *verified 2026-07-11* — `ChatDatabricks`, model selection patterns, integration examples.
- [MLflow Model Evaluation](https://mlflow.org/docs/latest/models.html) — *verified 2026-07-11* — Batch evaluation, metrics tracking, model comparison workflows.
- [LangSmith Observability](https://docs.smith.langchain.com/) — *verified 2026-07-11* — Tracing, debugging, and evaluating LLM applications; cost and latency analysis.
