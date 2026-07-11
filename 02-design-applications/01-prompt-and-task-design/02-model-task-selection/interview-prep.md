# Model and Task Selection — Interview Prep

**Section:** Design Applications | **Role target:** Senior Engineer, Solutions Architect, ML Platform Lead, Staff Engineer

---

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| **What is task-model alignment and why does it matter more than raw model capability?** | (1) Task-model alignment is the degree to which a model's training objective and architecture fit the semantic requirement and modality of a task. (2) Misalignment (e.g., using a chat model for semantic search) fails silently or requires workarounds; alignment is a 10x lever on cost and quality. (3) An embedding model is cheaper and better for search than Opus because it was built for that exact job. Example: Text classification → use small classifier model or fine-tuned small chat model, not GPT-4, even though GPT-4 is "more powerful." | Saying "big models are always better" or "capability is the only thing that matters." Beginners confuse model size with suitability. Also missing the hidden cost/quality win of specialization. |
| **Describe the tradeoff between pay-per-token and provisioned throughput deployments. When should you use each?** | Pay-per-token: shared infrastructure, variable latency (100–500ms), minimal setup, ideal for experimentation and variable workloads <10K req/day. Provisioned: dedicated GPU, consistent latency (<100ms P50), fixed hourly cost, ideal for production and stable >50K req/day. Breakeven: calculate (hourly_cost × 730) / daily_requests vs. pay-per-token cost. Include consideration of SLA requirements (latency guarantees matter for user-facing features). | Only mentioning cost difference ("provisioned is cheaper") without addressing latency consistency or SLA guarantees. Or picking pay-per-token for everything "because it's on-demand" without calculating true breakeven. |
| **How would you design a model selection workflow for a new production task?** | (1) Define constraints (latency P50/P95, cost per request, throughput, region, compliance). (2) Identify candidate models matching task type (embeddings for search, chat for dialogue, vision for images). (3) Create a gold set (50–200 representative examples). (4) Benchmark each candidate on quality, latency, cost. (5) Rank by weighted score (e.g., 0.5×quality + 0.3×cost + 0.2×latency). (6) Select winner and deploy with monitoring (LangSmith, MLflow). Re-benchmark quarterly. | Missing the "golden set" step or benchmarking on public datasets only (not your data). Also missing the iterative/monitoring piece—acting as if model selection is a one-time decision. |
| **What are common model selection mistakes you've seen or heard about?** | (1) Always picking the largest model (wastes cost, ignores constraints). (2) Benchmarking on public datasets (MMLU scores don't predict your domain performance). (3) Assuming pay-per-token is always cheaper (breaks even at ~50K req/day). (4) Not specializing (fine-tuning or in-context learning can half the model size needed). (5) Ignoring latency SLA (fast incorrect answer is worse than slow correct one in some cases, but vice versa in others—context matters). | Being too generic ("you should try multiple models") without specific tradeoffs. Or dismissing fine-tuning as "too expensive" when it pays for itself in weeks at scale. |

---

## Applied / Scenario Questions

**Q: You're building a real-time chat support feature. 100K requests/day expected. Latency SLA <150ms P95. Budget $500/month.** Can you recommend a model and deployment strategy? Walk through your reasoning.

**Strong answer framework:**
1. **Identify constraints:** 100K req/day = ~4K req/hr; <150ms P95 latency (hard); $500/month budget. This rules out some models and deployment modes.
2. **Task type:** Chat support = general-purpose model. Embeddings won't work (not dialogue). Vision not needed (assuming text-only).
3. **Cost math:** Assume 400 tokens/request average (inputs + outputs). 100K req/day = 40M tokens/day ≈ 1.2B tokens/month. At pay-per-token $0.0001/token (rough middle ground), cost = $120/month. At provisioned, a mid-size model is ~$0.50/hr × 730 ≈ $365/month (within budget). **Pay-per-token is cheaper initially, but:**
4. **Latency risk:** Pay-per-token typical latency is 100–200ms; P95 could hit 300ms+. Provisioned guarantee is <100ms P50, P95 <200ms (acceptable). At 100K req/day, latency consistency (SLA) > saving $50/month.
5. **Recommendation:** Use Llama 3 8B or Claude Haiku on Databricks provisioned throughput (within $500 budget). Benchmark both on 50 sample support queries before committing. Fallback: if latency still misses, use pay-per-token (accept variable latency) + add retry logic on client.

---

**Q: Your team is overspending on LLM inference. Monthly bill is $50K.** You're asked to cut it 50% without sacrificing quality. What's your approach?

**Strong answer framework:**
1. **Gather data:** Analyze request distribution: what % are simple (classification, lookup, basic Q&A) vs. complex (reasoning, multi-step, open-ended)? Real workloads: typically 70–80% simple.
2. **Segment & route:** Route simple tasks (e.g., "which plan is best for me?") to Haiku ($0.0001/token). Route complex tasks (e.g., "help me debug this") to Sonnet ($0.0003/token). If 75% is simple: 75% cost × 0.1 + 25% cost × 0.3 ≈ 0.35× original = 65% cost, or 35% savings.
3. **Benchmark:** Validate that Haiku quality is acceptable on simple tasks (often it is; it's designed for 80% of real-world use cases).
4. **Fine-tune if needed:** If 20% of simple tasks still miss quality threshold, fine-tune Haiku on 500–1K examples ($2K investment). Likely pays for itself in 1–2 weeks.
5. **Quick wins:** Look for common queries where in-context few-shot examples improve quality on smaller models (no retraining cost).
6. **Result:** Target 50% cost reduction likely achievable through segmentation + minor fine-tuning. Typical: $50K → $20–25K/month.

---

**Q: You're designing an internal system to help engineers choose models for their projects.** What information would you collect from them, and what decision tree would you build?

**Strong answer framework:**
1. **Information to collect:**
   - Task description (chat, search, classification, generation, code, vision, embedding, etc.)
   - Latency SLA (if any)
   - Daily request volume
   - Cost budget (if any)
   - Language requirements
   - Modality (text-only, images, audio, etc.)
   - Data available for benchmarking (size, quality)
   - Compliance requirements (region, HIPAA, etc.)

2. **Decision tree:**
   ```
   Is task text-only?
     No → Use vision model (Gemini, Claude with vision, etc.)
     Yes → Continue
   
   Is task semantic search / similarity?
     Yes → Use embedding model (Qwen Embedding, GTE)
     No → Continue
   
   What's the daily request volume?
     < 10K req/day → Start with pay-per-token
     > 50K req/day → Consider provisioned throughput
   
   What's the expected latency SLA?
     < 100ms P95 → Small model (Haiku, Llama 3 8B) + provisioned
     < 500ms P95 → Medium model (Sonnet, Llama 3 70B) + pay-per-token OK
     None / flexible → Optimize for cost first
   
   Do you have >500 in-domain examples?
     Yes → Consider fine-tuning a small model (cost savings 30-50%)
     No → Use generic model; benchmark on public datasets first
   
   Recommend: [Model], [Deployment Mode], [Estimated Cost], [Expected Latency], [Quality]
   ```

3. **Monitoring:** After deployment, track cost and latency for 2 weeks; re-benchmark quarterly; adjust if new models appear or workload changes.

---

## System Design / Architecture Questions

**Q: Design a model selection and routing system for a large organization with 50+ AI features running in production.** How would you architect it to enable easy model swaps, cost tracking, and benchmarking?

**Approach:**
1. **Model Registry (Unity Catalog / MLflow):** Catalog all approved models (Databricks Foundation Models, external models via AI Gateway). Each model entry includes: model ID, pricing, supported regions, latency SLA, capabilities (embeddings, vision, etc.), and current production deployments.

2. **Task Classification Layer:** Incoming request includes a task type (or inferred from metadata). Route to a model selector that queries the registry.

3. **Model Selector Logic (LangChain / LangGraph):** Implements decision rules: *If (task_type == "search" AND budget < $X) then use embedding model, else use chat model.*

4. **Routing:** Selected model endpoint is queried (via OpenAI client, REST API, or LangChain ChatDatabricks). Response is cached if applicable.

5. **Observability:** All requests logged to LangSmith and cost tracking database (task type, model used, latency, tokens, cost, quality proxy). Dashboards show cost per task type, model, team.

6. **Quarterly Re-evaluation:** Report on cost per task, latency distribution, quality metrics. Identify candidates for model swaps (new model cheaper or faster) or fine-tuning.

7. **Feedback Loop:** Product teams flag quality issues; data is collected and used for benchmarking in next cycle.

**Key design decisions:**
- Centralize model approvals (security + cost control).
- Use LangChain/LangGraph abstraction so model swaps require only config changes, not code.
- Log all cost + latency data; make it accessible for optimization.
- Treat model selection as a continuous process, not one-time decision.

---

## Vocabulary That Signals Expertise

Use these terms naturally—don't force them:
- **"Task-model alignment"** — When you immediately recognize that an embedding model is the right choice for retrieval, not a chat model. Signals you think architecturally, not just by capability.
- **"Latency SLA vs. cost tradeoff"** — Acknowledging that constraints are often contradictory and you're solving for the most important one first.
- **"In-domain benchmarking"** — When you insist on validating models on your actual data, not leaderboards. Red flag if someone says "MMLU score" as the main criterion.
- **"Token accounting"** — Calculating (input tokens + output tokens) × price to estimate true cost per request. Shows rigor.
- **"Provisioned throughput break-even"** — Mentioning the calculation: hourly cost × 730 / daily requests. Signals you've done the math, not guessing.
- **"Fine-tuning ROI"** — "We spent $2K, broke even in 3 weeks" signals you're serious about optimization, not just picking bigger models.
- **"Throughput vs. latency"** — Understanding that they're related but distinct (one endpoint can have high throughput but variable latency; provisioned trades predictability for dedicated capacity).
- **"Model specialization"** — Distinguishing between general-purpose and specialized models, and knowing when each is appropriate.

---

## Vocabulary That Signals Weakness

Avoid these—they suggest shallow thinking:
- **"Just use GPT-4 / Opus / the best model."** — Ignores constraints, cost, and task alignment. Red flag for production naiveté.
- **"MMLU scores predict real-world performance."** — Confusing benchmarks with reality. Beginners make this mistake; experts know better.
- **"Fine-tuning is too expensive."** — True for massive models trained from scratch; false for adapting pre-trained models. Shows limited cost-benefit analysis.
- **"We'll use pay-per-token forever."** — Not true at scale. Shows lack of operational thinking.
- **"All embedding models are the same."** — They're not. Qwen vs. GTE vs. BGE have different quality/latency/cost profiles for different languages/domains.
- **"Model X has higher accuracy on the benchmark, so we should use it."** — Ignoring latency, cost, and your specific data. Premature optimization on the wrong dimension.

---

## STAR Answer Frame

**Situation:** Your organization had a chatbot consuming $100K/month on Claude Opus for all requests.  
**Task:** You were asked to reduce costs by 50% without sacrificing quality.  
**Action:** (1) Analyzed request logs; found 75% were simple classification or lookup queries. (2) Built a router using LangChain that classified incoming requests as "simple" or "complex." (3) Benchmarked Haiku on the simple 25% and found 98% quality match. (4) Deployed: simple requests → Haiku ($0.0001/token), complex → Opus ($0.003/token). (5) Set up LangSmith tracking for cost + latency + quality metrics.  
**Result:** Cost dropped to $35K/month (65% reduction). Latency improved (Haiku P50 50ms vs. Opus 150ms). Quality on simple tasks stayed the same; quality on complex tasks improved because Opus could focus on hard reasoning. Deployed with zero downtime using feature flags.

---

## Red Flags Interviewers Watch For

1. **Can't explain why a model choice failed.** If they pick a model and it underperforms, do they understand *why*? (Latency? Cost? Wrong modality? Hallucinations?) Or do they just say "that model was bad"?

2. **Defensive about model selection.** If you push back ("have you benchmarked?"), do they double down on a decision, or are they curious and willing to test? Production AI requires humility and iteration.

3. **Ignoring constraints.** If they ignore budget or latency SLA ("well, bigger is better"), they'll cause production headaches.

4. **No monitoring/observability plan.** Can they articulate how they'd measure whether the model choice is working in production? Or is it "ship it and hope"?

5. **Confusing capability with suitability.** Saying "Claude is the smartest, so we should use it for everything." Intelligent people know smartness ≠ fitness for task.

6. **Never heard of fine-tuning or in-context learning as cost levers.** Production AI engineers know multiple ways to get quality at lower cost; beginners think "upgrade the model" is the only lever.
