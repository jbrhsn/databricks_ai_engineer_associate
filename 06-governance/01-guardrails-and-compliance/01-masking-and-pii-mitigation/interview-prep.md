# Masking and PII Mitigation — Interview Prep

**Section:** 06 Governance | **Role target:** Senior Data Engineer, ML Engineer, Solutions Architect (GenAI/Governance focus)

---

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is the difference between masking, pseudonymization, and anonymization? | • Masking hides visible values but doesn't prevent re-identification<br>• Pseudonymization replaces identifiers with reversible substitutes (still regulated data)<br>• Anonymization makes re-identification no longer reasonably possible<br>• GDPR treats pseudonymized data as personal data | Saying "they all hide PII" without explaining that pseudonymized data is still regulated and anonymization requires proving re-identification is infeasible |
| Why can't you rely solely on AI Gateway PII filtering for compliance? | • Gateway only protects serving boundary, not upstream artifacts<br>• PII may already be in logs, traces, vector indexes, embeddings<br>• Defence in depth requires application-layer controls first<br>• Gateway is a backstop, not a replacement for early transformation | Treating endpoint filtering as comprehensive protection without acknowledging that data may already be leaked upstream |
| When should you use regex versus NER for PII detection? | • Regex for rigid formats (SSN, email, phone) — fast, deterministic<br>• NER for contextual entities (names, locations) — understands surrounding language<br>• Hybrid approach for production: regex + NER/Presidio<br>• Regex alone misses context-dependent identifiers | Claiming regex is sufficient for all PII types, ignoring that names and contextual entities don't follow fixed patterns |
| What does "data minimization" mean in the context of RAG pipelines? | • Only embed fields necessary for retrieval quality<br>• Remove unnecessary sensitive fields before chunking<br>• Reduces compliance scope and leakage risk<br>• GDPR principle: collect/process only what's needed for stated purpose | Vague answer about "using less data" without connecting to chunking decisions, embedding scope, and compliance reduction |

---

## Behavioral Questions

**Q:** Tell me about a time you had to implement PII protection in a production ML or LLM system. What was your approach and what challenges did you face?

**Strong answer framework:**
- **Situation:** Describe the system (RAG, chatbot, analytics pipeline) and the sensitive data types (PHI, customer PII, financial data)
- **Task:** Explain the compliance requirement (GDPR, HIPAA, internal policy) and the technical constraint (performance, accuracy, auditability)
- **Action:** Walk through your layered approach:
  - Detection method selection (regex + NER/Presidio, why hybrid)
  - Transformation timing (before chunking/indexing, not after)
  - Governance layer (Unity Catalog raw vs masked tables, grants)
  - Endpoint controls (AI Gateway as backstop)
  - Monitoring and audit trail
- **Result:** Quantify outcome (compliance audit passed, zero PII leaks in production, retrieval quality maintained at X%, reduced compliance scope by Y%)
- **Reflection:** What you'd do differently (earlier detection, better testing, automated validation)

**Red flags to avoid:**
- Claiming you "just masked everything" without explaining detection strategy or timing
- No mention of defence in depth or multiple control layers
- Treating masking as a one-time preprocessing step rather than a pipeline concern
- Not distinguishing between different transformation types (mask vs tokenize vs anonymize)

---

## Applied / Scenario Questions

**Q:** You're building a healthcare chatbot on Databricks that answers patient questions using case notes stored in Delta tables. The notes contain patient names, SSNs, phone numbers, and diagnoses. The legal team says you must comply with HIPAA minimum necessary standard. How do you architect the data pipeline and serving layer?

**Strong answer framework:**
1. **Ingestion layer:**
   - Create two tables: `patient_notes_raw` (restricted) and `patient_notes_masked` (curated)
   - Apply detection (regex for SSN/phone, Presidio for names) during ETL
   - Tokenize patient IDs for continuity, redact unnecessary direct identifiers
   - Write only masked content to serving table

2. **Governance layer:**
   - Use Unity Catalog to enforce access: raw table restricted to compliance team, masked table accessible to app
   - Create governed view with additional field-level masking if needed
   - Document data lineage and transformation logic

3. **Retrieval layer:**
   - Chunk and embed only from masked table
   - Exclude fields not needed for answering questions (data minimization)
   - Vector index contains no raw identifiers

4. **Serving layer:**
   - Application-level sanitization of live user input before logging
   - AI Gateway PII filtering on endpoint as backstop
   - System prompt instructs model never to request or reveal identifiers

5. **Monitoring:**
   - Inference tables with restricted access
   - Audit logs showing which governed sources were accessed
   - Periodic validation that no raw PII leaked into derived artifacts

**Trade-off awareness:**
- More aggressive masking improves privacy but may reduce retrieval precision
- Tokenization preserves continuity but requires secure key management
- Early transformation prevents leakage but requires careful ETL design

---

**Q:** Your RAG system is returning accurate answers, but a security audit found patient email addresses in the vector index. The team says they masked emails in the final output, so they thought it was safe. What went wrong and how do you fix it?

**Strong answer framework:**
1. **Root cause:** Masking happened after chunking and embedding, so raw emails were already stored in the vector index
2. **Why this matters:** Vector index is a derived artifact that persists sensitive data; even if output is masked, the index can be queried or leaked
3. **Immediate fix:**
   - Rebuild vector index from scratch using pre-masked source data
   - Verify no other derived artifacts (logs, traces, cached prompts) contain raw emails
4. **Systemic fix:**
   - Move detection and masking to ingestion/preprocessing stage
   - Create separate raw and curated tables in Unity Catalog
   - Index only from curated table
   - Add validation step to confirm no PII in embeddings
5. **Prevention:**
   - Establish "mask before chunking" as architectural principle
   - Add automated PII detection tests on chunk source tables
   - Document data flow showing where transformations occur

**What this reveals about your thinking:**
- You understand timing matters more than just "having masking somewhere"
- You know vector indexes are persistent storage, not ephemeral
- You think in terms of defence in depth and validation

---

**Q:** A data scientist wants to use a pseudonymized patient dataset for training a custom medical NER model. They argue it's safe because direct identifiers are replaced with tokens. What questions do you ask and what guidance do you give?

**Strong answer framework:**
1. **Questions to ask:**
   - Can the tokens be reversed through a mapping key? (If yes, still regulated)
   - What other fields are in the dataset? (Quasi-identifiers can enable re-identification)
   - What's the linkage risk with other available datasets?
   - What's the legal basis and purpose limitation for this use?
   - Is the training use consistent with original data collection purpose?

2. **Guidance:**
   - Pseudonymized data is still personal data under GDPR and still PHI under HIPAA if linkage is possible
   - Need same governance controls as raw data: access restrictions, audit trail, purpose limitation
   - Consider whether true anonymization is feasible (k-anonymity, differential privacy techniques)
   - If training requires identifiable data, document legal basis and minimize retention
   - Ensure model artifacts (weights, embeddings) don't leak training data

3. **Trade-off discussion:**
   - Stronger anonymization may reduce model quality if context is lost
   - Synthetic data generation might be safer alternative
   - Federated learning could keep data in governed environment

**Red flags in weak answers:**
- Accepting "it's pseudonymized so it's fine" without probing linkage risk
- Not mentioning GDPR/HIPAA implications
- Ignoring purpose limitation and legal basis questions

---

## System Design / Architecture Questions

**Q:** Design a production-grade PII mitigation architecture for a Databricks-hosted customer support assistant that uses RAG over support tickets, knowledge base articles, and CRM data. The system must handle 10K requests/day, comply with GDPR, and maintain audit trails.

**Approach:**

1. **Clarify requirements:**
   - What PII types? (Names, emails, phone, account IDs, payment info)
   - What's the retrieval quality vs privacy trade-off tolerance?
   - Who needs access to raw vs masked data?
   - What's the audit retention requirement?
   - What's the acceptable latency for detection/masking?

2. **Propose structure:**

   ```
   ┌─────────────────────────────────────────────────────┐
   │ Ingestion Layer                                     │
   │ • Support tickets, KB articles, CRM → Delta tables  │
   │ • Detection: Presidio + custom regex for account IDs│
   │ • Transformation: tokenize account IDs, redact PII  │
   │ • Write to: raw (restricted) + masked (serving)     │
   └─────────────────────────────────────────────────────┘
                          ↓
   ┌─────────────────────────────────────────────────────┐
   │ Governance Layer (Unity Catalog)                    │
   │ • catalog.support.tickets_raw (compliance team only)│
   │ • catalog.support.tickets_masked (app + analysts)   │
   │ • Governed views with additional field masking      │
   │ • Service policies for AI securables                │
   └─────────────────────────────────────────────────────┘
                          ↓
   ┌─────────────────────────────────────────────────────┐
   │ Retrieval Layer                                     │
   │ • Chunk only from masked tables                     │
   │ • Embed with data minimization (exclude unnecessary)│
   │ • Vector Search index on curated content            │
   └─────────────────────────────────────────────────────┘
                          ↓
   ┌─────────────────────────────────────────────────────┐
   │ Application Layer                                   │
   │ • Sanitize live user input before logging           │
   │ • Build prompts from masked chunks                  │
   │ • System prompt: never request/reveal identifiers   │
   └─────────────────────────────────────────────────────┘
                          ↓
   ┌─────────────────────────────────────────────────────┐
   │ Serving Layer                                       │
   │ • AI Gateway PII filtering on endpoint (backstop)   │
   │ • Model serving with inference table logging        │
   │ • Restricted access to inference tables             │
   └─────────────────────────────────────────────────────┘
                          ↓
   ┌─────────────────────────────────────────────────────┐
   │ Monitoring & Audit                                  │
   │ • Inference tables (restricted, PII-sanitized)      │
   │ • Unity Catalog audit logs                          │
   │ • Periodic validation: no PII in derived artifacts  │
   │ • Compliance reporting dashboard                    │
   └─────────────────────────────────────────────────────┘
   ```

3. **Justify choices and name tradeoffs:**
   - **Two-tier storage:** Separates raw (compliance) from masked (serving) so governance is platform-enforced, not app-enforced
   - **Early transformation:** Masking before chunking prevents PII from spreading into embeddings, indexes, traces
   - **Hybrid detection:** Presidio for packaged recognizers + custom regex for domain-specific IDs balances coverage and precision
   - **AI Gateway as backstop:** Centralized endpoint control protects all callers, but doesn't replace application-layer controls
   - **Tokenization for account IDs:** Preserves continuity for support workflows while hiding raw identifiers
   - **Trade-offs:**
     - More aggressive masking → better privacy, potentially lower retrieval precision
     - Presidio adds latency vs pure regex → acceptable for batch ingestion, may need optimization for real-time
     - Separate tables → storage overhead, but cleaner governance and audit trail

**What strong candidates demonstrate:**
- Defence in depth thinking (multiple layers, not one silver bullet)
- Understanding that timing matters (mask before chunking/indexing)
- Platform governance (Unity Catalog) vs application logic separation
- Concrete Databricks components (Delta, Unity Catalog, Vector Search, AI Gateway, inference tables)
- Trade-off awareness and quantification

---

## Troubleshooting Scenarios

**Scenario 1:** After deploying PII masking, users complain that the chatbot can no longer answer questions about specific customer cases because all account IDs are replaced with `[ACCOUNT_ID]`. How do you diagnose and fix this?

**Diagnosis:**
- Redaction removed information needed for retrieval and reasoning
- Model can't distinguish between different customers when all IDs are identical placeholders
- Over-aggressive masking broke business functionality

**Fix:**
1. **Short-term:** Use tokenization instead of redaction for account IDs — replace with consistent pseudonyms like `ACCT_A1B2C3` so model can still reason about distinct entities
2. **Medium-term:** Evaluate whether account IDs are truly PII in this context (if they're not linkable to individuals, may not need masking)
3. **Long-term:** Implement semantic placeholders that preserve entity type and uniqueness: `[ACCOUNT_ID_1]`, `[ACCOUNT_ID_2]`

**Key insight:** Masking strategy must balance privacy and functionality; sometimes tokenization or pseudonymization is better than redaction.

---

**Scenario 2:** Compliance audit finds that inference tables contain raw customer emails in the `input` field, even though AI Gateway PII filtering is enabled. What happened?

**Diagnosis:**
- Inference tables log the request payload before AI Gateway filtering runs
- Gateway filters the request to the model, but logging happens at a different point in the pipeline
- Application didn't sanitize input before sending to endpoint

**Fix:**
1. **Immediate:** Restrict access to inference tables, purge or mask historical data
2. **Systemic:** Add application-layer sanitization before endpoint call
3. **Configuration:** Verify inference table logging scope — exclude or sanitize sensitive fields
4. **Validation:** Add automated tests that check inference tables for PII patterns

**Key insight:** Understand the order of operations in the serving pipeline; logging may happen before gateway filtering.

---

**Scenario 3:** A data engineer reports that PII detection is flagging too many false positives (common words mistaken for names), causing important context to be redacted. How do you tune the system?

**Diagnosis:**
- NER model or Presidio recognizer has low precision for this domain
- Score threshold too low, or recognizer not calibrated for domain-specific language
- Need to balance recall (catching real PII) vs precision (avoiding false positives)

**Fix:**
1. **Tune score threshold:** Raise minimum confidence for entity detection (Presidio `score` parameter)
2. **Add domain context:** Fine-tune NER model on domain-specific examples, or add custom recognizers
3. **Whitelist common terms:** Exclude known non-sensitive terms that match patterns
4. **Hybrid approach:** Use high-precision regex for critical identifiers (SSN, email), lower threshold for contextual entities
5. **Human review loop:** Sample flagged entities, measure precision/recall, iterate

**Key insight:** PII detection is a precision/recall trade-off; production systems need tuning and validation, not just out-of-box defaults.

---

## Code Review Questions

**Q:** Review this PII masking implementation. What are the problems?

```python
def mask_pii(text: str) -> str:
    # Mask emails and SSNs
    text = text.replace("@", "[AT]")
    text = re.sub(r"\d{3}-\d{2}-\d{4}", "[SSN]", text)
    return text

# Use in RAG pipeline
chunks = chunk_documents(raw_documents)
embeddings = embed_chunks(chunks)
index.add(embeddings)

# Mask before showing to user
def get_answer(question):
    results = index.search(question)
    masked_results = [mask_pii(r) for r in results]
    return llm.generate(masked_results, question)
```

**Problems:**
1. **Email masking is broken:** Replacing `@` with `[AT]` doesn't hide the email, just changes format; still identifiable
2. **Masking happens too late:** Raw PII is already in chunks, embeddings, and index before masking
3. **No phone number detection:** Only handles SSN and (broken) email
4. **No contextual entity detection:** Misses names, addresses, other PII
5. **Regex is too narrow:** SSN pattern only matches exact `XXX-XX-XXXX` format, misses variations
6. **No validation:** No check that masking actually worked or that index is PII-free

**Correct approach:**
```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def mask_pii_correct(text: str) -> str:
    results = analyzer.analyze(
        text=text,
        entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"],
        language="en"
    )
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text

# Mask BEFORE chunking and indexing
masked_documents = [mask_pii_correct(doc) for doc in raw_documents]
chunks = chunk_documents(masked_documents)
embeddings = embed_chunks(chunks)
index.add(embeddings)

# Now retrieval and generation work with already-masked content
def get_answer(question):
    sanitized_question = mask_pii_correct(question)
    results = index.search(sanitized_question)
    return llm.generate(results, sanitized_question)
```

---

## Trade-off Discussions

**Q:** When should you choose to block PII at the AI Gateway versus mask it?

**Framework:**
- **Block when:**
  - The PII is unnecessary for the task (user accidentally included SSN in question about product features)
  - Regulatory requirement is strict no-processing (certain PHI in non-healthcare contexts)
  - Model should never see the value under any circumstance
  - Example: Block SSN in a product recommendation chatbot

- **Mask when:**
  - Sentence structure matters for understanding (need to know "a person called" even if name is hidden)
  - Model needs to reason about entity types (distinguish between email vs phone vs name)
  - Downstream processing requires placeholder awareness
  - Example: Mask patient name in medical Q&A but preserve `[PATIENT]` so model knows context

- **Trade-offs:**
  - Blocking is safer but may break legitimate use cases
  - Masking preserves more context but requires careful placeholder design
  - Both are backstops; prefer early transformation over endpoint-only controls

**Strong answer demonstrates:**
- Understanding that block vs mask is a context-dependent decision
- Awareness that endpoint controls are last line of defence
- Ability to map business requirements to technical controls

---

**Q:** Your team debates whether to use Unity Catalog masking views or application-layer masking. What factors drive the decision?

**Framework:**

| Factor | Unity Catalog Views | Application-Layer Masking |
|---|---|---|
| **Governance enforcement** | Platform-enforced, can't be bypassed | Depends on app code discipline |
| **Centralization** | Single source of truth for all consumers | Each app implements own logic |
| **Flexibility** | Less flexible, SQL-based transformations | More flexible, arbitrary Python logic |
| **Performance** | Query-time overhead for complex masking | One-time preprocessing cost |
| **Auditability** | Built-in Unity Catalog audit logs | Requires custom logging |
| **Use case fit** | Best for structured data, BI, multiple consumers | Best for unstructured text, LLM preprocessing |

**Recommendation:** Use both in defence in depth:
- Unity Catalog for structured data governance (raw vs masked tables, governed views)
- Application-layer for LLM-specific preprocessing (chunking, prompt assembly, live input sanitization)
- Never rely on application code alone for primary governance boundary

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:

- **Defence in depth** — when explaining layered PII controls (ingestion, governance, application, endpoint)
- **Data minimization** — when discussing which fields to include in chunks or embeddings
- **Quasi-identifiers** — when explaining re-identification risk in pseudonymized datasets
- **Linkage attack** — when discussing why pseudonymization alone isn't sufficient
- **Purpose limitation** — when evaluating whether data use is consistent with collection purpose (GDPR)
- **Minimum necessary** — when applying HIPAA principles to LLM data access
- **Governed securable** — when referring to Unity Catalog objects (tables, views, functions, models)
- **Inference table** — Databricks-specific term for logged serving requests/responses
- **Service policy** — Unity Catalog mechanism for governing AI-related securables
- **Presidio recognizer** — when discussing packaged PII detection components

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:

- **"Just encrypt it"** — Encryption protects data at rest/in transit, but doesn't prevent authorized users from seeing PII; confuses access control with visibility control
- **"Anonymized means hidden"** — Anonymization is about re-identification risk, not just hiding values; signals misunderstanding of GDPR/compliance definitions
- **"The model won't remember PII"** — Ignores that PII can leak through logs, traces, embeddings, outputs, not just model weights
- **"Masking at the end is fine"** — Signals unawareness that derived artifacts (indexes, embeddings) may already contain raw PII
- **"Regex catches everything"** — Overconfidence in pattern matching; misses contextual entities and format variations
- **"We pseudonymized so it's not personal data anymore"** — Fundamental misunderstanding of GDPR; pseudonymized data is still regulated
- **"Just use AI Gateway and you're compliant"** — Treats endpoint filtering as comprehensive solution; ignores upstream leakage and defence in depth

---

## STAR Answer Frame

**Situation:** Our healthcare client needed a Databricks-hosted patient-support chatbot that answered benefits questions using historical case notes and policy documents. The notes contained patient names, SSNs, phone numbers, diagnoses, and treatment details. We were subject to HIPAA minimum necessary requirements and had to pass a compliance audit before production launch.

**Task:** I was responsible for designing and implementing the end-to-end PII mitigation architecture, ensuring no PHI leaked into vector indexes, model prompts, traces, or inference logs, while maintaining retrieval quality sufficient for accurate answers.

**Action:**
1. **Detection strategy:** Implemented hybrid detection using regex for rigid identifiers (SSN, phone, member IDs) and Presidio for contextual entities (patient names, provider names). Validated on 10K sample notes, achieving 98% recall and 94% precision.

2. **Transformation timing:** Applied masking during ingestion, before chunking and embedding. Created two-tier storage in Unity Catalog: `patient_notes_raw` (restricted to compliance team) and `patient_notes_masked` (accessible to app). Tokenized patient IDs for case continuity, redacted unnecessary direct identifiers.

3. **Governance layer:** Used Unity Catalog grants to enforce access boundaries. Created governed views with additional field-level masking for analytics consumers. Documented data lineage showing transformation points.

4. **Retrieval layer:** Built vector index only from masked table, excluding fields not needed for answering questions (data minimization). Reduced embedding scope by 40% while maintaining retrieval quality.

5. **Serving layer:** Added application-layer sanitization of live user input before logging. Configured AI Gateway PII filtering on endpoint as backstop. System prompt instructed model never to request or reveal identifiers.

6. **Validation:** Implemented automated tests checking for PII patterns in chunks, embeddings, and inference tables. Set up periodic audits of derived artifacts.

**Result:**
- Passed HIPAA compliance audit with zero findings
- Zero PII leaks detected in 6 months of production (50K+ requests)
- Maintained 92% answer accuracy vs baseline (pre-masking)
- Reduced compliance scope by 60% by keeping raw PHI out of derived artifacts
- Audit trail provided clear evidence of defence-in-depth controls

**Reflection:** If I were to do it again, I'd invest more upfront in domain-specific NER fine-tuning to reduce false positives, and I'd implement automated PII detection in CI/CD to catch regressions earlier.

---

## Red Flags Interviewers Watch For

**Specific to PII mitigation and governance:**

1. **Treating masking as a one-time preprocessing step** — Shows lack of understanding that PII can enter at multiple points (ingestion, live input, retrieval, outputs)

2. **No mention of timing** — Not explaining when masking happens relative to chunking, indexing, logging signals shallow understanding of data flow

3. **Confusing masking with anonymization** — Using terms interchangeably shows lack of compliance knowledge

4. **Over-reliance on single control layer** — Depending only on AI Gateway or only on application code, without defence in depth

5. **No validation strategy** — Not explaining how you'd verify masking worked or detect PII leakage

6. **Ignoring derived artifacts** — Focusing only on source tables or final outputs, missing that embeddings, indexes, traces, logs all need protection

7. **Vague about Databricks components** — Can't name specific tools (Unity Catalog, AI Gateway, inference tables, Vector Search) or how they fit together

8. **No trade-off awareness** — Claiming perfect solution with no downsides; not acknowledging privacy vs utility tension

9. **Treating pseudonymization as safe** — Not understanding that reversible transformations still create regulated data

10. **No mention of compliance frameworks** — Not connecting technical controls to GDPR, HIPAA, or other regulatory requirements

**What interviewers want to see:**
- Systematic thinking about data flow and leakage points
- Defence in depth mindset with multiple control layers
- Concrete Databricks platform knowledge
- Understanding of compliance implications
- Practical experience with detection/masking trade-offs
- Validation and monitoring awareness
