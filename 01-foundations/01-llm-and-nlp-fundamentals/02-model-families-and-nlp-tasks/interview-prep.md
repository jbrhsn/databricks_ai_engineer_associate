# Model Families & NLP Tasks — Interview Prep

**Section:** Foundations | **Role target:** GenAI Engineer / ML Engineer

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What are the three transformer architecture families and what is each good at? | Encoder-only (bidirectional attention → understanding: classification, NER, embeddings); decoder-only (causal attention → generation: chat, RAG answers, agents); encoder-decoder (encode source then decode target via cross-attention → translation, abstractive summarization). | Naming only "BERT vs GPT" without explaining attention direction or why it dictates task fit. |
| Difference between foundation, fine-tuned, and instruction-tuned models? | Foundation = broad pre-training (next-token); fine-tuned = further trained on task/domain data; instruction-tuned = SFT + preference alignment to follow prompts (the `-Instruct`/chat suffix). | Treating them as synonyms, or claiming a raw base model will reliably follow chat instructions. |
| Open vs proprietary models — what actually differs? | Open (Llama/Gemma/GPT-OSS/Qwen) = downloadable weights, self-host, full fine-tune, in-perimeter data residency; proprietary (Claude/GPT/Gemini) = API-only, frontier quality, vendor-controlled residency. | Saying "open just means free" and ignoring fine-tunability and data residency. |
| Why is a chat LLM the wrong tool for high-volume text classification? | Understanding task at batch scale; per-token billing + interactive concurrency limits + brittle free-text parsing. Right tool = small model + batch inference (AI Functions / `ai_classify`). | "It works though" — ignoring cost, throughput, and output-contract reliability. |
| What are Databricks' Model Serving options for foundation models? | FMAPI pay-per-token, FMAPI provisioned throughput, external models (third-party proxy), AI Functions (batch); all governed via Unity AI Gateway. | Collapsing pay-per-token = "toy" and provisioned throughput = "prod" instead of a workload-profile decision. |
| When do you pick an embedding model over a generative model? | Retrieval/semantic search/clustering need fixed-length vectors → encoder-only embedding model on `llm/v1/embeddings` + Vector Search. Generation needs a decoder LLM. | Calling a chat endpoint to "get embeddings." |

## Applied / Scenario Questions

**Q:** You're designing a RAG assistant on Databricks that must (a) retrieve relevant policy docs and (b) generate grounded answers. Which model types and serving routes for each stage, and why?

**Strong answer framework:**
- Retrieval stage → encoder-only **embedding** model (e.g. `databricks-gte-large-en`) on the `llm/v1/embeddings` route, vectors stored in Databricks Vector Search.
- Generation stage → instruction-tuned **decoder-only** LLM on FMAPI (pay-per-token for prototype; provisioned throughput if SLA/high volume).
- Show tradeoff awareness: name that the embedding model is small/cheap and the generative model is where quality and cost concentrate; mention Unity AI Gateway for rate limits/guardrails across both.

**Q:** A pipeline sends 500k tickets/day through a 70B chat endpoint just to classify them. It's slow and expensive. What do you change?

**Strong answer framework:**
- Diagnose it as a task/serving mismatch (understanding task + batch volume on a generation model + interactive endpoint).
- Move to **AI Functions batch inference** (`ai_classify`) over the Delta table with a small model; structured labels, one pass.
- Quantify the win (per-token → per-batch, structured output removes regex fragility). Note you'd keep the chat LLM only for genuinely generative sub-tasks.

## System Design / Architecture Questions (if applicable)

**Q:** Design a multi-task NLP platform on Databricks serving classification, semantic search, summarization, and an agent — for multiple teams under central governance.

**Approach:**
1. **Clarify requirements** — per-task volume, latency SLAs, data-residency/compliance needs, which teams, and whether any specific vendor model is contractually required.
2. **Propose structure** — classification via AI Functions batch (small model); search via encoder-only embedding endpoint + Vector Search; summarization/agent via instruction-tuned decoder LLM (pay-per-token → provisioned throughput as volume/SLA grows); any contracted third-party model via an external model endpoint. Front everything with Unity AI Gateway for rate limits, budgets, usage tracking, and guardrails. For the agent, use LangGraph as the orchestration layer (stateful graph nodes), LangChain components inside nodes, and reserve CrewAI only as a comparison point.
3. **Justify choices and name tradeoffs** — smallest-capable-model per task for cost; provisioned throughput only where SLA/custom-weights/compliance demand it; external endpoints centralize credentials; MLflow flavors (`transformers`, `sentence_transformers`) let you bring custom/fine-tuned or encoder-decoder models onto serving. Call out that governance and serving-mode choices are coupled.

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **Causal vs bidirectional attention** — explains *why* decoder-only generates and encoder-only understands.
- **Autoregressive generation** — the one-token-at-a-time mechanism of decoder models.
- **Instruction-tuned / SFT + preference alignment** — precise reason chat models follow prompts.
- **`llm/v1/chat | completions | embeddings`** — the task contract of a serving endpoint.
- **Provisioned throughput** — reserved optimized capacity for SLA/custom-weights/compliance.
- **AI Functions batch inference / `ai_query` / `ai_classify`** — the scalable path for understanding tasks over tables.
- **MLflow flavor** — how a model is packaged for serving (`transformers`, `sentence_transformers`, `python_function`).
- **Unity AI Gateway** — the governance/routing control plane.

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- "Just use GPT/Claude for everything" — signals no cost/latency/task-fit reasoning.
- "Embeddings are a type of chat output" — conflates generation with retrieval vectors.
- "Open models are just the free ones" — misses fine-tunability and data residency.
- "Provisioned throughput is for production, pay-per-token is for testing" — false dichotomy.
- "Fine-tuned and instruction-tuned are the same" — imprecise; loses the SFT/alignment distinction.
- "DBRX is the default Databricks model" — outdated; verify the current supported-model list (models are frequently added/retired).

## STAR Answer Frame

**Situation:** A support-analytics pipeline routed ~500k tickets/day through a frontier chat LLM interactive endpoint to assign categories; cost was climbing and throughput was capped.
**Task:** I owned cost and latency for the enrichment layer without dropping classification accuracy.
**Action:** I re-framed the work as an *understanding* task at *batch* volume, replaced the interactive chat calls with AI Functions batch inference (`ai_classify`) over the Delta table using a small model, and kept the decoder LLM only for the genuinely generative summary panel — all fronted by Unity AI Gateway for rate limits and usage tracking.
**Result:** Classification moved from per-token interactive calls to a single scalable batch pass, cut enrichment cost by well over an order of magnitude, removed the brittle regex label-parsing step, and freed interactive capacity for the real-time summary feature.

## Red Flags Interviewers Watch For

- Defaulting to the biggest/newest model for every task without asking about volume or latency.
- Unable to explain *why* attention direction dictates task fit (memorized "BERT vs GPT" only).
- Treating open/proprietary purely as a price question.
- Not knowing the three-plus Databricks serving modes or when each applies.
- Proposing a chat endpoint for retrieval/embeddings.
- Naming specific model versions with false confidence instead of flagging that the supported-model list is fast-evolving and should be verified.
