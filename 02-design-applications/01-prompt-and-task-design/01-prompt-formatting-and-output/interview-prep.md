# Prompt Formatting and Output Specification — Interview Prep

## Core Concepts

### Main Question: Explain Prompt Formatting and Output Specification

**What you should cover:**
1. **Definition:** Prompt formatting is the structure and content of instructions sent to an LLM (system message, role, examples, constraints). Output specification is the contract defining what format and schema the model must return (JSON, Pydantic, fields, types).
2. **Why it matters:** Explicit prompts with clear examples produce consistent, high-quality outputs. Output schemas enforce validation, catch errors early, and enable downstream automation.
3. **Core mechanisms:**
   - Prompts work via pattern matching. Vague prompts activate fewer patterns; explicit prompts + examples activate precise patterns.
   - Output validation catches occasional model failures (null fields, type mismatches) automatically and triggers retries.
   - System messages frame reasoning; few-shot examples demonstrate format; constraints guide token efficiency.
4. **Production context:** In real systems, unvalidated output fails silently and corrupts data. Validated output with retries fails loudly and recovers. Schema-first design is non-negotiable.

**Why interviewers ask this:** Separates engineers who ship prototypes from those who build production systems. Clarity on prompts and output specification signals maturity in LLM application development.

---

## Scenario Questions

### Scenario 1: Multi-Step Classification Pipeline

**Setup:** You're designing a customer support system that classifies tickets into categories (bug, feature_request, question, billing) and extracts priority (low, normal, high). Currently, manual classification takes 4 hours per day. You want to automate it.

**Questions:**
- What output schema would you define, and why?
- Would you use few-shot examples? What would they look like?
- How would you handle cases where the model misclassifies a ticket?
- What constraints would you add to the prompt, and why?

**What they want to hear:**
- **Schema:** Pydantic model with `category` (enum of 4 options), `priority` (enum of 3 options), and `confidence` (float 0–1). Possibly add `reasoning` (string) for audit trail.
- **Examples:** 2–3 representative tickets showing the full input–output format. Include one edge case (e.g., ambiguous ticket that could be billing or bug).
- **Error handling:** Use structured output with retries. If validation fails, the LLM retries with error feedback. Log failures that still escape retries for human review.
- **Constraints:** Max 100 tokens per response (tickets are short, summaries should be brief). Set temperature 0.2–0.3 (deterministic, not creative). Use a model with native structured output support if available.

**Red flags (things not to say):**
- "We'll ask the model to return JSON and parse it with regex." (Fragile, no schema validation.)
- "Increase the model's temperature for variety." (Increases errors, not accuracy.)
- "Add a 500-token prompt listing all edge cases." (Overwhelming; use examples instead.)

---

### Scenario 2: Extracting Structured Data From Unstructured Text

**Setup:** Your company ingests product reviews from multiple sources (web, emails, social media). You need to extract: product ID, rating (1–5), sentiment (positive/negative/mixed), and key complaint (if any). Reviews range from 10 to 500 tokens.

**Questions:**
- How would you design the output schema?
- Would ProviderStrategy or ToolStrategy be better for this use case? Why?
- What would you do if the model returns a malformed rating (e.g., 6 instead of 1–5)?
- How would you test whether your prompt is working?

**What they want to hear:**
- **Schema:** Pydantic with `product_id` (string, validated against known product list), `rating` (int, range 1–5), `sentiment` (Literal), `complaint` (optional string, null if no complaint).
- **Strategy:** ProviderStrategy if using OpenAI/Claude/Gemini/Grok (faster, cheaper, native support). Fall back to ToolStrategy for other models.
- **Malformed data:** Use OutputFixingParser or handle_errors=True in ToolStrategy. The validation error is sent back to the LLM with instructions to fix it. Usually succeeds on retry.
- **Testing:** (a) Sample 10–20 reviews covering edge cases (long reviews, sarcasm, missing ratings). (b) Check output validation: count how many fail schema validation vs. pass. (c) Review failure cases—are they retrying successfully? (d) Compare model outputs to human labels if possible (ground truth check). (e) Use LangSmith to trace each request and spot patterns.

**Red flags:**
- "We'll manually review 10% of outputs to catch errors." (Too slow; automate validation instead.)
- "We'll use the same prompt for all reviews." (Fails on long reviews or edge cases; add examples.)
- "If validation fails once, we skip that review." (Loses data; retry instead.)

---

## System Design Question

**Design a robust LLM-powered data extraction pipeline for invoice processing.**

**Requirements:**
- Extract: vendor name, invoice number, invoice date, line items (product, quantity, unit price), total amount.
- Invoices vary in format (PDF, email, scanned images). Some are in English, some in Spanish.
- Must handle ~1,000 invoices/day with <5% manual review rate.
- Need audit trail (what the LLM extracted, confidence scores, any retries).

**Design considerations:**
1. **Input preprocessing:** OCR for images/PDFs; text cleaning.
2. **Prompt design:** 
   - System message: "You are an invoice analyst."
   - Few-shot examples: 2–3 representative invoices in each language.
   - Output schema: Pydantic model with typed fields, optional fields for missing data.
   - Constraints: Temperature 0.1 (deterministic), max 200 tokens.
3. **Output strategy:** 
   - Use ProviderStrategy with GPT-4 or Claude (native structured output).
   - Pair with OutputFixingParser to retry on validation failure.
4. **Error handling:**
   - Log all validation failures + error messages for analysis.
   - Manual review queue for items that fail after 2 retries.
   - Metrics dashboard: extraction rate, retry rate, human review rate.
5. **Observability:**
   - LangSmith tracing for all requests.
   - Track latency, cost, token usage per invoice.
   - Periodically compare model outputs to human labels to catch drift.

**Architecture sketch:**
```
Invoices (PDF/email/image)
    │
    ├─► OCR + Text Cleaning
    │
    ├─► Prompt Template (system + examples + schema)
    │
    ├─► LLM (GPT-4 with ProviderStrategy)
    │
    ├─► OutputFixingParser (retry on failure)
    │
    ├─► Validation Check ──► Pass ──► Storage (database)
    │                    └──► Fail (after 2 retries) ──► Manual Review Queue
    │
    ├─► Tracing (LangSmith)
    │
    └─► Metrics (extraction rate, retry rate, cost)
```

**What they want to hear:**
- Recognition that LLMs are not 100% reliable; build for failures, not perfection.
- Schema-first thinking: define output contract before choosing prompt.
- Provider-aware optimization: choose model + strategy based on requirements (cost, latency, accuracy).
- Observability: you can't improve what you don't measure. Track extraction rates, retries, human review.
- Language handling: few-shot examples in Spanish cover that case without separate pipelines.

**Red flags:**
- "We'll just ask the LLM to return JSON." (No schema, no validation, no retries.)
- "One example is enough for all invoices." (Edge cases won't be covered.)
- "We'll use a cheaper model to save cost." (Accuracy matters; wrong extractions cost money in downstream processing.)

---

## Vocabulary & Definitions (Strength/Weakness Check)

**Can you explain these terms correctly?**

| Term | What you should know | Strength sign | Weakness sign |
|---|---|---|---|
| **ProviderStrategy** | Uses model's native structured output API (e.g., OpenAI, Claude). Faster, fewer tokens, higher reliability. | "ProviderStrategy when available, ToolStrategy as fallback" | "Same as ToolStrategy, just a detail." |
| **ToolStrategy** | Uses tool calling to enforce structured output. Works with all models, but slower and higher token usage. | "Tool calling adds overhead; use ProviderStrategy if possible" | "Always use ToolStrategy." |
| **Output Validation** | Checking that model output matches schema (type, fields, ranges). Required in production. | "Catches errors early, enables automatic retries" | "Optional optimization." |
| **Few-Shot Prompting** | Including 1–3 labeled examples in the prompt. Dramatically improves accuracy without fine-tuning. | "1–3 examples covering edge cases; diversity matters" | "The more examples, the better." |
| **System Message** | The initial message that defines the LLM's role and framing. Shapes all subsequent reasoning. | "Brief, role-focused, sets expectations" | "Just background info; not critical." |
| **Token Budget** | Max tokens allowed for output. Controls cost and prevents runaway generation. | "Set upfront, pair with prompt constraints" | "Not a design decision; leave at default." |
| **OutputParser** | Decodes raw LLM output into structured objects and validates against schema. | "Catches type mismatches, integrates with retry logic" | "Just converts text to JSON." |
| **Pydantic Model** | Python class with type hints used to define and validate data schemas. | "Enables type safety, automatic validation, clear contracts" | "Just a nice way to write classes." |

---

## STAR Example

**Situation:** I was designing a product classification system for an e-commerce platform. The LLM needed to categorize products into 15 categories and assign confidence scores. The initial approach used loose prompts and manual JSON parsing—it failed on 18% of products due to inconsistent output.

**Task:** I needed to make the system robust enough for production (target: <2% failure rate) without slowing down inference or exploding token usage.

**Action:** I:
1. Defined a Pydantic model with the output schema: `category` (Literal of 15 enum values), `confidence` (float 0–1), `reasoning` (optional string).
2. Added 3 few-shot examples covering edge cases (ambiguous products, rare categories, multi-category items).
3. Switched from manual JSON parsing to LangChain's ProviderStrategy (with OpenAI, which supports native structured output).
4. Wrapped the response with `OutputFixingParser` to retry on validation failures.
5. Instrumented everything with LangSmith tracing to monitor extraction rates and retry patterns.

**Result:** Failure rate dropped from 18% to 0.3%. Retries accounted for the 0.3%—those got flagged for human review. Latency increased by ~50ms per request due to retries, but accuracy was worth it. Within a month, the model was classifying 50,000+ products/day with <1% requiring human intervention. I also discovered that two of the fifteen categories were confusing the model (via retry logs), so I added a clearer example for those categories. This data-driven iteration improved accuracy further.

**What they want to hear:**
- Problem-solving approach: recognize the issue, design a solution, measure the result.
- Schema-first thinking: output contract defined before prompt iteration.
- Production maturity: validation, retries, tracing, human escalation.
- Iteration: use data to improve, not guesses.

---

## Red Flags (Things Not to Say)

- **"LLMs always return correct JSON."** ← Naive. They occasionally deviate. Validate everything.
- **"We'll add a 1,000-token prompt with every possible edge case."** ← Backfires. Use 2–3 examples instead.
- **"Temperature should always be 0.7 for variety."** ← Dangerous for extraction tasks. Use 0.1–0.3 for determinism.
- **"We don't need error handling; the model is good enough."** ← Famous last words. Every system needs retries.
- **"Schema validation is an optimization we can add later."** ← Will cost you weeks debugging when it breaks in production.
- **"ProviderStrategy and ToolStrategy are basically the same."** ← No. ProviderStrategy is faster and cheaper when available.
- **"We'll just ask the model to be careful / be precise / think step-by-step."** ← Wishful thinking. Explicit format + examples work.

---

## Closing Thoughts

Prompt formatting and output specification are the bridge between vague AI demos and production systems. Interviewers care about this because it separates folks who've shipped real systems from those who haven't. You don't need to be a prompt engineering PhD—just understand the core principles: explicit prompts work better, schemas enforce correctness, retries handle edge cases, and data drives iteration. Master these, and you'll stand out.
