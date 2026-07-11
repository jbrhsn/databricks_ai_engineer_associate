# Prompt Augmentation and Guardrails — Interview Prep

**Section:** 04 Application Development | **Role target:** Senior AI/ML Engineer, Solutions Architect, GenAI Platform Engineer

---

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is the difference between an input guardrail and an output guardrail? | Input guardrails run *before* the LLM call and validate/sanitise the user message (PII masking, prompt injection detection, topic filtering). Output guardrails run *after* the LLM call and validate the model's response (schema check, toxicity score, groundedness). They guard different threat surfaces and both are required. | Saying "guardrails check whether the response is safe" — this describes only the output side and leaves the input unprotected, missing the injection attack vector. |
| How does `ChatPromptTemplate.from_messages()` prevent prompt injection compared to f-string concatenation? | Typed template slots (`{question}`) are filled by name-keyed dict values at invoke time. The template class handles escaping; the user's text is inserted into a specific role-labelled slot, not concatenated into the full prompt string. An attacker cannot close one message and open another via a typed slot. F-string concatenation has no role structure — it is a raw string, and adversarial instructions in the user input become part of the instruction stream. | Saying "ChatPromptTemplate is just syntax sugar for f-strings" — this misses the structural safety property of role-typed messages. |
| What does Databricks AI Gateway Safety filtering use under the hood, and what are its limitations? | Meta Llama Guard 2-8b (an 8B safety classifier trained on Llama 3) applied symmetrically to requests and responses. Covers 11 harm categories. Limitation: not available in regions without Foundation Model APIs pay-per-token support; not supported on custom model endpoints; adds ~100–300ms latency; not applied to function-call intermediate responses (only the final output). | Saying "it uses a GPT model" or "it is rule-based regex" — both are wrong. It is a dedicated Llama Guard classifier. |
| Explain the mechanism of `with_fallbacks()` in LangChain. | Returns a `FallbackChain` wrapper. On invocation, tries the primary chain. If the primary raises an exception in `exceptions_to_handle`, tries each fallback in order. Does not retry — that requires `tenacity` or similar. The `exceptions_to_handle` parameter should list specific provider errors, not bare `Exception`. | Confusing fallbacks with retries — `with_fallbacks()` switches models; `tenacity` retries the same call. Both serve different reliability goals. |
| What is the difference between Databricks AI Gateway (serving endpoints) and Unity AI Gateway? | AI Gateway for serving endpoints is the existing workspace-level feature (UI: Serving → endpoint → AI Gateway section). Unity AI Gateway is the newer (Beta, Jul 2026) enterprise control plane that governs Model Services, MCP Services, and Model Provider Services through service policies attached to Unity Catalog securables. The serving-endpoint version is labelled "legacy" in current docs; Unity AI Gateway is the strategic direction. | Treating them as the same product — they have different configuration paths, different governance scopes, and different maturity levels. |

---

## Applied / Scenario Questions

**Q:** You are asked to add guardrails to an existing LangChain LCEL RAG chain. The chain currently reads: `retriever | format_docs | prompt | llm | StrOutputParser()`. What steps do you take and where do you insert each guardrail?

**Strong answer framework:**
- Insert a `RunnableLambda(sanitise_input)` at the very start of the chain (before the retriever), so PII and injection patterns are removed before any data hits the retrieval layer or the prompt template.
- Restructure the chain to `RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})` so the retriever receives the sanitised query and the prompt template receives both context and question by name.
- Wrap the `llm` with `llm.with_fallbacks([fallback_llm], exceptions_to_handle=(RateLimitError,))`.
- Append a `RunnableLambda(validate_output_schema)` or use `llm.with_structured_output(MySchema)` after the LLM step.
- Separately, enable AI Gateway PII masking on the serving endpoint as a backstop for direct API callers.
- Demonstrate tradeoff awareness: synchronous output validation adds latency; asynchronous MLflow faithfulness evaluation adds zero latency and is better for groundedness monitoring in high-throughput paths.

---

**Q:** An automated test shows your chatbot occasionally returns PII that was in the user's own message ("Your phone number is 555-123-4567 — confirmed"). AI Gateway PII masking is enabled. What is the most likely root cause?

**Strong answer framework:**
- AI Gateway masking applies to the request *after* it leaves your application layer. If the user's phone number is echoed in the response, either: (a) the response PII masking is not enabled (only input masking is configured — check `guardrails.output`), or (b) the LLM is referencing the phone number from conversation history that was injected via `MessagesPlaceholder` and not re-masked before injection.
- Demonstrate tradeoff awareness: the fix is twofold — enable output PII masking in AI Gateway *and* apply `mask_pii()` to each message in the conversation history before injecting it into the `MessagesPlaceholder` variable.

---

## System Design / Architecture Question

**Q:** Design the guardrail architecture for a healthcare patient portal chatbot. The bot answers questions about appointment scheduling, medication refills, and billing. Requirements: HIPAA compliance, PII must never appear in logs, the primary model is provisioned throughput (DBRX), and the system must remain available even during primary model outages.

**Approach:**
1. **Clarify requirements:** Confirm PHI categories in scope (names, DOB, MRN, SSN, insurance IDs), logging pipeline (MLflow, LangSmith, Databricks payload logging), and acceptable latency budget per call (e.g. 5s P95).
2. **Propose structure:**
   - *Input layer (application)*: `mask_phi(input)` using Presidio with a custom healthcare entity recogniser (extend base Presidio with MRN and insurance ID patterns) → `validate_intent(input)` to reject non-healthcare topics → typed `ChatPromptTemplate` slot injection.
   - *Prompt layer*: hardened system prompt with explicit PHI handling instructions and refusal clause for off-topic questions.
   - *Infrastructure layer*: AI Gateway PII masking (Block mode) and Safety filtering enabled on the DBRX provisioned throughput endpoint. Note: Safety filter is not supported on provisioned throughput per the documentation — validate in workspace, may require external model endpoint instead.
   - *Fallback layer*: `primary_llm.with_fallbacks([claude_haiku_endpoint])` in the LCEL chain; AI Gateway Fallbacks (for the external model fallback endpoint if DBRX is external).
   - *Output layer*: `with_structured_output(AppointmentResponse)` schema validation; asynchronous `mlflow.evaluate()` faithfulness batch job nightly on sampled traces.
   - *Audit layer*: Databricks AI Gateway payload logging to Unity Catalog Delta tables with column-level masking on PHI columns.
3. **Justify choices and name tradeoffs explicitly:**
   - Application-layer masking vs. Gateway-only: Gateway does not protect LangSmith/MLflow trace logs — both layers required.
   - Synchronous schema validation vs. async groundedness: schema validation is lightweight (microseconds); faithfulness scoring via LLM judge is 1–3s extra latency — make it async.
   - Provisioned throughput DBRX vs. external model endpoint: AI Guardrails are supported on provisioned throughput per AI Gateway docs (supported column) but Safety filter requires Foundation Model API pay-per-token region support — worth verifying before architecture sign-off.

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:
- **Typed template slot** — when explaining why `ChatPromptTemplate` is safer than f-strings; signals you understand the injection mechanism
- **Fail-closed semantics** — when describing Unity AI Gateway service policy evaluation; a misconfigured policy DENYs, not ALLOWs — signals operational maturity
- **Defense-in-depth** — when describing layered guardrails (application + Gateway); signals security thinking
- **ON CALL / ON RESULT evaluation phases** — Unity AI Gateway service policy terminology; signals you have read current docs, not just legacy docs
- **`with_structured_output()`** — LangChain method that enforces schema at the model API level; signals you know the production pattern, not just `StrOutputParser`
- **Presidio** — the Microsoft library Databricks uses for PII detection; signals vendor-stack knowledge
- **Llama Guard 2-8b** — the specific model behind Databricks AI Gateway safety filter; signals depth beyond marketing copy

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:
- **"Just use a content filter"** — too vague; does not distinguish input vs. output, application vs. infrastructure layer, or detection method
- **"The system prompt prevents jailbreaks"** — signals you have not read any prompt injection research; system prompts are a mitigating control, not a security boundary
- **"LangChain handles guardrails automatically"** — no framework does this out of the box; guardrails are always the developer's responsibility to wire up
- **"AI Gateway replaces application-layer guardrails"** — misunderstands the threat surface; Gateway only protects the endpoint wire, not traces or intermediate variables
- **"We use GPT-4 so hallucination is not a problem"** — signals no experience with production failures; all current LLMs hallucinate under some conditions

---

## STAR Answer Frame

**Situation:** Our team shipped a RAG-based internal knowledge chatbot for a legal team. Three weeks after launch, a paralegal reported that the bot had included a colleague's personal email address (from a retrieved document) in its response to an unrelated question.

**Task:** I was responsible for the guardrail architecture. The incident exposed that we had no PII detection on the output path — we had assumed the retrieved documents were clean.

**Action:** I added three controls: (1) Application-layer regex PII masking (`mask_pii()`) applied to the user question *and* to each retrieved document's `page_content` before injection into the prompt. (2) Enabled AI Gateway output PII masking (Mask mode) on the serving endpoint as a backstop for any PII that passed the application filter. (3) Added an `mlflow.evaluate()` nightly batch job on the previous day's traces, flagging responses containing email-pattern strings using a regex scorer, so future misses would be caught within 24 hours rather than three weeks.

**Result:** In the 90 days following the fix, the monitoring job flagged two near-misses (both caught and masked at the Gateway layer before reaching users), and zero user-reported PII incidents. The application-layer mask eliminated PII from LangSmith traces, which had been the secondary exposure we discovered during the investigation.

---

## Red Flags Interviewers Watch For

- Describing guardrails only as "AI safety" concerns rather than reliability and compliance engineering — signals awareness of only the marketing narrative, not the operational reality
- Proposing to apply guardrails only on the output without explaining why the input is also a threat surface — misses prompt injection entirely
- Inability to explain the difference between `with_fallbacks()` and retry logic — these are complementary but distinct mechanisms, and conflating them signals inexperience with production LLM systems
- Not knowing that Databricks AI Gateway Safety filter is not supported on custom model serving endpoints — signals the candidate has not read the current documentation limitations table
- Saying the hardened system prompt is "the main defence against adversarial inputs" — this is the primary indicator that the candidate has not shipped a production GenAI system with real users
