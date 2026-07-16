# Masking and PII Mitigation for LLM Applications on Databricks

**Section:** 06 Governance | **Module:** 01 Guardrails and Compliance | **Est. time:** 2.5 hrs | **Exam mapping:** Governance (8%)

---

## TL;DR

PII mitigation in LLM systems is the discipline of detecting sensitive data before it reaches prompts, storage, logs, retrieval indexes, or model outputs, then applying the right transformation so the application still works without exposing regulated information. On Databricks, this is not one feature but a layered control stack: application-layer detection and masking, Unity Catalog governance on stored data, and AI Gateway guardrails on model traffic. For the exam and for production, the key distinction is that masking is not the same as anonymization, and compliance depends on whether data can still be linked back to a person. **The one thing to remember: treat PII protection as a defence-in-depth pipeline — detect early, transform appropriately, govern access centrally, and never assume one guardrail layer is enough.**

---

## ELI5 — Explain It Like I'm 5

Imagine a mailroom in a hospital where every letter must be copied and sent to different teams, but some letters contain a patient's name, phone number, and insurance number. Before the copies go anywhere, one worker scans for sensitive details, another covers some parts with black marker, and a supervisor decides which teams are allowed to see the original versus only the covered version. That is how PII mitigation works in an AI app: one step finds the private details, another step hides or replaces them, and another step controls who can still access the real version. The common mistake is thinking the black marker alone solves everything, but if the original letter was already copied into logs or sent to the wrong desk, the damage already happened. Good governance means protecting the data before, during, and after the AI call.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Explain the difference between PII detection, masking, pseudonymization, tokenization, and anonymization in exam and production terms
- [ ] Compare regex, NER-based detection, and Presidio-based detection for different LLM application risk profiles
- [ ] Design a Databricks-native PII mitigation pipeline spanning ingestion, retrieval, prompting, serving, and monitoring
- [ ] Apply Unity Catalog masking and access-governance concepts to sensitive structured data used by GenAI applications
- [ ] Configure AI Gateway PII filtering decisions conceptually and choose when to block versus mask
- [ ] Map GDPR and HIPAA requirements to concrete engineering controls for LLM applications on Databricks
- [ ] Diagnose common masking failures such as leaking PII into vector indexes, traces, or model outputs

---

## Visual Overview

### End-to-End PII Mitigation Pipeline

```
Raw source data
    │
    ├──► Ingestion-time detection
    │        ├── Regex for fixed patterns
    │        ├── NER for contextual entities
    │        └── Presidio for policy-driven detection
    │
    ├──► Transformation decision
    │        ├── Redact
    │        ├── Mask
    │        ├── Tokenize
    │        ├── Pseudonymize
    │        └── Drop field entirely
    │
    ├──► Governed storage in Unity Catalog
    │        ├── Raw restricted table
    │        └── Masked serving table / view
    │
    ├──► Retrieval / prompt assembly
    │        ├── Only approved fields enter chunks
    │        └── Prompt-time masking as backstop
    │
    ├──► AI Gateway request / response filtering
    │        ├── Block unsafe payloads
    │        └── Mask detected PII
    │
    └──► Monitoring and audit
             ├── Inference tables
             ├── Access logs
             └── Compliance evidence
```

### Detection Method Decision Tree

```
Need to detect sensitive data
│
├── Is the pattern rigid and well-defined?
│     ├── Yes ──► Regex
│     └── No
│
├── Does meaning depend on surrounding words?
│     ├── Yes ──► NER / contextual model
│     └── No
│
├── Need packaged recognizers, policy controls,
│   and multiple entity types in one framework?
│     ├── Yes ──► Presidio
│     └── No ──► Custom hybrid pipeline
│
└── Production best practice
      └── Combine regex + contextual detection,
          then validate with governance controls
```

### Raw vs Masked Data Access in Unity Catalog

```
                    ┌──────────────────────────────┐
                    │ catalog.schema.patient_raw   │
                    │ name | dob | ssn | diagnosis │
                    └──────────────────────────────┘
                               │
                 restricted grants / row-column policy
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │ catalog.schema.patient_safe  │
                    │ patient_token | age_band     │
                    │ masked_contact | diagnosis   │
                    └──────────────────────────────┘
                               │
                               ▼
                    RAG indexing / BI / LLM app
```

### Where PII Can Leak If You Mask Too Late

```
User input with PII
    │
    ├──► App logs request body        ◄── leak
    ├──► Trace captures prompt        ◄── leak
    ├──► Retriever stores raw chunk   ◄── leak
    ├──► LLM sees original value      ◄── leak
    └──► Output echoes value back     ◄── leak

Correct pattern:
User input ──► detect + transform first ──► logs / traces / model / output
```

---

## Key Concepts

### PII in LLM Applications

**What it is:** Personally identifiable information (PII) is data that identifies, relates to, or can reasonably be linked to a specific person. In LLM systems, PII risk is broader than a database column because the same information can appear in prompts, retrieved chunks, chat history, traces, inference tables, and generated outputs.

**How it works under the hood:** LLM applications move text through multiple stages: ingestion, chunking, embedding, retrieval, prompt assembly, model inference, and logging. If PII enters any of those stages in raw form, it can be copied into derived artifacts such as vector indexes, cached prompts, or monitoring tables. That means the privacy boundary is not just the source table; it is every intermediate representation the system creates. The engineering task is therefore to identify where sensitive values can propagate and to apply controls before those propagation points.

**Where it appears:** On Databricks, PII may live in Delta tables governed by Unity Catalog, in documents processed for RAG pipelines, in model serving request/response payloads, and in inference tables or application traces. Exam questions often test whether you understand that protecting only the final output is insufficient if the prompt or retrieval corpus already contains raw PII.

---

### PII Detection Methods: Regex, NER, and Presidio

**What it is:** PII detection is the process of identifying sensitive entities in text or structured fields so they can be blocked, masked, or transformed before downstream use. The three exam-relevant approaches are regex for fixed patterns, named entity recognition (NER) for contextual entities, and Presidio for a production-oriented detection framework that combines recognizers and policy logic.

**How it works under the hood:** Regex matches character patterns such as email addresses, SSNs, or phone numbers using deterministic rules. It is fast and transparent, but it only catches what the pattern explicitly describes, so it misses contextual entities like a person's name unless the format is constrained. NER models infer entity spans from surrounding language, which lets them detect names, locations, or organizations even when the exact string format varies, but they can produce false positives and false negatives depending on context and domain shift. Presidio acts as an orchestration layer: it runs multiple recognizers, scores candidate entities, resolves overlaps, and returns structured findings that can then be anonymized or redacted according to policy.

**Where it appears:** Regex is commonly implemented in PySpark transformations or Python preprocessing functions before chunking or prompt assembly. NER can be applied in notebook pipelines or custom preprocessing services. Databricks AI Gateway documentation states that PII detection uses Microsoft Presidio under the hood for supported entity filtering on serving endpoints, making Presidio the most directly exam-relevant packaged detector in the Databricks ecosystem.

---

### Masking, Redaction, Tokenization, Pseudonymization, and Anonymization

**What it is:** These are different transformation strategies for sensitive data, and the distinctions matter. Masking or redaction hides the visible value; tokenization replaces it with a surrogate token mapped elsewhere; pseudonymization replaces identifiers with reversible substitutes under controlled linkage; anonymization irreversibly removes the ability to link data back to a person.

**How it works under the hood:** Redaction usually replaces the detected span with a placeholder such as `[EMAIL]` or blacked-out text, preserving sentence structure but removing the original value. Tokenization stores the original value in a secure mapping system and substitutes a token like `cust_tok_9182`, allowing later re-identification only through the token vault. Pseudonymization similarly preserves analytical usefulness by replacing direct identifiers with consistent aliases, but because a linkage key exists, the data is still personal data under GDPR. True anonymization requires that re-identification is no longer reasonably possible, which is much harder than simply masking a few fields because combinations of quasi-identifiers can still reveal identity.

**Where it appears:** In Databricks pipelines, redaction and masking are often applied during ETL or prompt preprocessing, tokenization may be implemented through external secure services before data lands in Delta, and pseudonymized datasets are commonly exposed through governed views for analytics or RAG. Exam questions often hinge on the fact that pseudonymized data is still regulated, while anonymized data is intended to fall outside many personal-data obligations if done correctly.

---

### Unity Catalog Data Masking and Central Governance

**What it is:** Unity Catalog is Databricks' centralized governance layer for data and AI assets, and it is the control plane used to manage access to sensitive tables, views, functions, and policies. In the context of masking, the key idea is that sensitive raw data should remain tightly restricted while downstream consumers access masked or transformed representations through governed objects.

**How it works under the hood:** Unity Catalog enforces permissions at the catalog, schema, table, view, function, and other securable levels. Rather than giving every application direct access to raw tables, teams create curated views or transformation pipelines that expose only the fields and formats appropriate for the use case. This separates storage of the original data from consumption of the masked version, which is critical because masking logic embedded only in application code is easy to bypass. Governance becomes durable when the platform, not just the app, controls who can query raw versus transformed data.

**Where it appears:** In Databricks, this shows up as Unity Catalog grants, governed schemas, secure views, and service policies for AI-related securables. For exam purposes, remember that Unity Catalog is the place to centralize access control and data-governance boundaries, while application code handles context-specific preprocessing before LLM calls.

---

### AI Gateway PII Filtering

**What it is:** AI Gateway PII filtering is an infrastructure-layer guardrail on model serving traffic that detects sensitive entities in requests and responses and then either blocks or masks them according to configuration. It is a backstop control, not a substitute for application-layer privacy engineering.

**How it works under the hood:** Requests sent to a governed serving endpoint pass through the gateway before reaching the model. The gateway runs PII detection on the payload, identifies supported entity types, and then applies the configured action: block the request/response entirely or mask the detected spans before forwarding. Because this happens at the endpoint boundary, it protects all callers that use that endpoint, including notebooks, SDK clients, and applications that might not share the same preprocessing code. However, it cannot retroactively protect data that was already logged or embedded upstream before the request reached the gateway.

**Where it appears:** In Databricks model serving and AI Gateway documentation, PII detection is configured as part of endpoint guardrails. Exam questions may contrast AI Gateway with application-layer masking; the correct answer is usually defence in depth, with gateway controls as centralized enforcement and app-layer controls as earlier-stage prevention.

> ⚠️ Fast-evolving: AI Gateway and Unity AI Gateway capabilities are changing quickly. Verify the current official Databricks documentation before relying on exact UI labels or policy attachment flows.

---

### Compliance Frameworks: GDPR and HIPAA in Practice

**What it is:** GDPR and HIPAA are compliance frameworks that impose requirements on how personal or protected health information is collected, processed, stored, shared, and audited. For LLM applications, they matter because prompts, retrieval corpora, and outputs can all become regulated processing activities.

**How it works under the hood:** GDPR focuses on lawful basis, purpose limitation, data minimization, security, access rights, and accountability for personal data. That means an LLM system should only process the minimum necessary personal data, should not repurpose it casually for training or analytics, and should maintain governance evidence such as access controls and auditability. HIPAA focuses on protected health information (PHI), minimum necessary use, safeguards, and controlled disclosure in healthcare contexts. In practice, both frameworks push engineers toward early detection, least-privilege access, masking or de-identification where possible, and strong separation between raw regulated data and downstream AI artifacts.

**Where it appears:** On Databricks, compliance-aligned controls include Unity Catalog permissions, masked views, governed model serving, inference logging with restricted access, and documented data flows. For the exam, know the engineering translation: GDPR and HIPAA are not just legal labels; they drive concrete design choices such as block vs mask, raw vs curated tables, and whether re-identification remains possible.

---

### Why Timing Matters: Mask Before Chunking, Indexing, and Logging

**What it is:** The timing of PII mitigation determines whether sensitive data is merely hidden from one output or actually prevented from spreading through the system. The safest pattern is to transform sensitive data before it is chunked, embedded, logged, or sent to the model.

**How it works under the hood:** If raw documents are chunked and embedded first, the vector index may already contain sensitive text even if later prompts display only masked snippets. If prompts are logged before masking, traces and inference tables may retain the original values. If the model sees raw PII, it can reproduce or transform it in outputs, creating additional leakage paths. Early transformation reduces the number of downstream artifacts that need remediation and narrows the compliance scope of derived assets.

**Where it appears:** In Databricks RAG pipelines, this means applying detection and transformation during ingestion or preprocessing before writing Delta tables used for chunking and indexing. In serving pipelines, it means sanitizing inputs before application logs and using AI Gateway as a second line of defence at the endpoint.

---

### Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| Detection method (`regex`, `NER`, `Presidio`, hybrid) | How sensitive entities are identified in text | Use regex for rigid formats, NER for context-dependent entities, Presidio or hybrid pipelines when you need multiple recognizers and policy-driven production handling |
| Regex strictness | How narrowly or broadly a pattern matches | Start strict for high-risk identifiers like SSNs to reduce false positives; widen only when recall gaps are proven with test samples |
| Entity score threshold | Minimum confidence required to treat a detected span as PII | Raise the threshold when false positives break business workflows; lower it when missing a sensitive entity is more costly than over-masking |
| Transformation action (`block`, `mask`, `redact`, `tokenize`, `pseudonymize`) | What happens after detection | Block when the model must never process the value, mask/redact when sentence structure still matters, tokenize/pseudonymize when downstream joins or longitudinal analysis are required |
| Placeholder format (for example `[EMAIL]`, `[PATIENT_NAME]`) | How masked entities are represented in transformed text | Use semantic placeholders when the model still needs to reason about the type of missing value; use generic placeholders only when entity type itself is sensitive |
| Raw-to-curated table split | Whether raw and transformed data are stored separately | Always separate raw restricted data from masked serving data when multiple consumers exist; do not rely on one shared table with informal conventions |
| AI Gateway PII action | Whether detected PII at the endpoint is blocked or masked | Choose **Block** for highly regulated or unnecessary personal data; choose **Mask** when the model needs the surrounding sentence but not the actual identifier |
| Re-identification key custody | Who can reverse a token or pseudonym | Keep linkage material outside general analytics and app access paths; if many teams can reverse it, you have not meaningfully reduced risk |
| Logging scope | Which request/response fields are retained for monitoring | Exclude or sanitize prompt fields containing user text unless there is a documented need and restricted access path |
| Chunking source selection | Which fields are allowed into embeddings and retrieval chunks | Include only fields needed for retrieval quality; if a field is not needed for answering questions, do not embed it |

### Worked Example: Requirement → Decision

**Given:** A healthcare provider wants to build a Databricks-hosted assistant that answers patient-benefit questions using policy documents and prior case notes. The assistant must help support agents quickly, but it must not expose patient names, phone numbers, member IDs, or diagnoses to the model unless absolutely necessary. The organization is subject to HIPAA and also serves EU residents, so GDPR-style minimization and auditability matter too.

**Step 1 — Identify the goal:** Enable accurate support answers from governed enterprise data while minimizing PHI/PII exposure across ingestion, retrieval, prompting, and serving.

**Step 2 — Define inputs:** Structured Delta tables with patient and claims metadata, unstructured case notes, policy PDFs, live agent questions, and chat history captured during support sessions.

**Step 3 — Define outputs:** A grounded answer for the support agent, plus auditable records showing which governed sources were used and evidence that sensitive identifiers were not unnecessarily exposed.

**Step 4 — Apply constraints:**
- PHI and personal data should follow minimum-necessary handling
- Raw identifiers must not be embedded into vector indexes if retrieval does not require them
- Some case continuity is needed, so fully dropping all identifiers would break workflows
- Multiple clients may call the serving endpoint, so centralized endpoint controls are required
- Compliance review requires clear separation between raw data access and masked application data

**Step 5 — Select the approach:** Build a two-tier design. First, preprocess source data into a restricted raw layer and a curated masked layer in Unity Catalog; tokenize or pseudonymize patient/member identifiers needed for continuity, redact unnecessary direct identifiers from notes before chunking, and index only the curated layer for RAG. Second, add application-layer detection plus AI Gateway PII filtering on the serving endpoint so live prompts and outputs are sanitized even if a caller submits raw identifiers. This is better than relying only on endpoint masking because endpoint controls cannot remove PII that was already embedded into the retrieval corpus or captured in upstream traces.

---

## Implementation

```python
# Scenario: Prepare support-case notes for a Databricks RAG pipeline so that direct identifiers
# are removed before chunking and embedding, while preserving enough structure for retrieval.

import re
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

EMAIL_RE = r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
PHONE_RE = r"\b(?:\+?\d{1,2}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"
SSN_RE = r"\b\d{3}-\d{2}-\d{4}\b"
MEMBER_ID_RE = r"\bMEM-\d{6,10}\b"

@F.udf(returnType=StringType())
def mask_case_note(text: str) -> str:
    # Replace direct identifiers before the text is chunked or embedded.
    if text is None:
        return None

    masked = re.sub(EMAIL_RE, "[EMAIL]", text, flags=re.IGNORECASE)
    masked = re.sub(PHONE_RE, "[PHONE]", masked)
    masked = re.sub(SSN_RE, "[SSN]", masked)
    masked = re.sub(MEMBER_ID_RE, "[MEMBER_ID]", masked)
    return masked

raw_notes_df = spark.table("main.governance.patient_case_notes_raw")

masked_notes_df = (
    raw_notes_df
    .withColumn("note_text_masked", mask_case_note(F.col("note_text")))
    .select(
        "case_id",
        "patient_token",
        "created_at",
        "note_text_masked"
    )
)

# Write only the curated masked text to the table used by downstream chunking/indexing jobs.
masked_notes_df.write.mode("overwrite").saveAsTable(
    "main.governance.patient_case_notes_masked"
)
```

---

```python
# Scenario: Detect contextual entities in free-text escalation messages before sending them to
# a Databricks serving endpoint, using Presidio-style analysis plus masking as an app-layer guardrail.

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def sanitize_prompt(user_text: str) -> str:
    results = analyzer.analyze(
        text=user_text,
        entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN"],
        language="en",
    )
    anonymized = anonymizer.anonymize(text=user_text, analyzer_results=results)
    return anonymized.text

incoming = (
    "Patient John Smith called from 555-123-4567 and said his SSN is 123-45-6789. "
    "Explain his deductible status."
)

safe_prompt = sanitize_prompt(incoming)
print(safe_prompt)
# Example output:
# "Patient <PERSON> called from <PHONE_NUMBER> and said his SSN is <US_SSN>. Explain his deductible status."
```

---

```sql
-- Scenario: Expose a governed, masked view in Unity Catalog so analysts and GenAI pipelines
-- can query useful customer-support data without direct access to raw identifiers.

CREATE OR REPLACE VIEW main.governance.support_cases_serving AS
SELECT
  case_id,
  patient_token,
  date_trunc('DAY', created_at) AS created_day,
  regexp_replace(email, '(^.).*(@.*$)', '$1***$2') AS masked_email,
  regexp_replace(phone, '(\\d{3})\\d{3}(\\d{4})', '$1***$2') AS masked_phone,
  diagnosis_group,
  note_text_masked
FROM main.governance.support_cases_raw;

-- Grant downstream consumers access to the masked view, not the raw table.
GRANT SELECT ON VIEW main.governance.support_cases_serving TO `support_ai_app`;

-- Keep raw-table access tightly restricted.
REVOKE SELECT ON TABLE main.governance.support_cases_raw FROM `support_ai_app`;
```

---

```python
# Anti-pattern: masking only after retrieval and prompt assembly, which leaves raw PII in the
# vector index, traces, and intermediate prompt state even if the final displayed answer looks safe.

def build_prompt_wrong(retrieved_chunks: list[str], user_question: str) -> str:
    context = "\n\n".join(retrieved_chunks)  # retrieved_chunks may already contain raw PII
    prompt = f"Context:\n{context}\n\nQuestion:\n{user_question}"
    return prompt.replace("123-45-6789", "[SSN]")  # too late and too narrow

# Correct approach:
# 1. Detect and transform sensitive text before writing chunk source tables
# 2. Build embeddings only from curated masked content
# 3. Sanitize live user input before logging or prompt assembly
# 4. Use AI Gateway PII filtering as a centralized backstop at the endpoint

def build_prompt_correct(masked_chunks: list[str], sanitized_question: str) -> str:
    context = "\n\n".join(masked_chunks)
    return f"Context:\n{context}\n\nQuestion:\n{sanitized_question}"

# What breaks in the anti-pattern:
# - Raw identifiers may already be stored in the vector index
# - Prompt traces may capture the original value
# - Simple string replacement misses alternate formats and entity types
# - Compliance scope expands because derived artifacts now contain regulated data
```

---

```python
# Scenario: Call a Databricks model serving endpoint through an application-layer privacy filter,
# preserving entity type information while preventing raw identifiers from reaching the model.

import requests

def mask_for_llm(text: str) -> str:
    text = re.sub(EMAIL_RE, "[EMAIL]", text, flags=re.IGNORECASE)
    text = re.sub(PHONE_RE, "[PHONE]", text)
    text = re.sub(SSN_RE, "[SSN]", text)
    return text

question = "My email is alice@example.com and my SSN is 123-45-6789. Can I update my benefits?"
sanitized_question = mask_for_llm(question)

payload = {
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a benefits assistant. Never ask for or reveal personal identifiers. "
                "Answer using policy guidance only."
            ),
        },
        {
            "role": "user",
            "content": sanitized_question,
        },
    ],
    "max_tokens": 300,
    "temperature": 0.0,
}

response = requests.post(
    "https://<workspace-host>/serving-endpoints/benefits-assistant/invocations",
    headers={
        "Authorization": "Bearer <token>",
        "Content-Type": "application/json",
    },
    json=payload,
    timeout=30,
)

print(response.status_code)
print(response.text)
```

---

## Common Pitfalls & Misconceptions

- **Thinking masking and anonymization are the same** — Beginners see a hidden value like `[EMAIL]` and assume the privacy problem is solved permanently. The correct mental model is that masking hides presentation, while anonymization is about whether re-identification is still reasonably possible across the whole dataset and process.

- **Relying only on regex for all PII detection** — Regex feels attractive because it is simple, fast, and easy to explain, so teams overextend it to names, addresses, and contextual identifiers it cannot reliably capture. The correct mental model is that regex is one detector class for rigid formats, not a universal privacy solution.

- **Masking after chunking or indexing** — Teams focus on what the user sees in the final answer and forget that embeddings, vector indexes, and traces may already contain raw sensitive text. The correct mental model is that the first irreversible spread point matters most, so transform sensitive data before derived artifacts are created.

- **Assuming AI Gateway replaces application-layer controls** — Centralized endpoint filtering sounds comprehensive, so beginners assume it covers the entire pipeline. The correct mental model is that gateway controls protect the serving boundary, while application-layer controls protect logs, traces, preprocessing steps, and any path before the endpoint.

- **Using pseudonymized data as if it were no longer regulated** — Because direct identifiers are replaced with tokens, teams sometimes treat the dataset as anonymous. The correct mental model is that if a linkage key exists or re-identification remains feasible, the data is still personal data and still needs governance.

- **Embedding unnecessary sensitive fields for retrieval quality** — Engineers sometimes include every available field in chunks "just in case" it improves recall. The correct mental model is that retrieval quality should be optimized with the minimum necessary content, because every extra sensitive field increases leakage risk and compliance burden.

---

## Key Definitions

| Term | Definition |
|---|---|
| PII | Information that identifies, relates to, or can reasonably be linked to a specific individual; in LLM systems this includes data in prompts, retrieved text, logs, and outputs, not just database columns |
| PHI | Protected health information regulated in healthcare contexts; effectively health-related personal data whose handling is subject to HIPAA safeguards and minimum-necessary principles |
| Regex-based detection | Deterministic identification of sensitive text using explicit character-pattern rules such as email or SSN formats |
| Named Entity Recognition (NER) | A contextual modeling approach that labels spans such as person names, locations, or organizations based on surrounding language rather than fixed string patterns |
| Presidio | A Microsoft privacy-detection and anonymization framework that combines recognizers, scoring, and transformation utilities for sensitive-entity handling |
| Redaction | Removal or replacement of sensitive content so the original value is no longer visible in the transformed text |
| Tokenization | Replacement of a sensitive value with a surrogate token whose original meaning can be restored only through a separate secure mapping system |
| Pseudonymization | Replacement of direct identifiers with consistent substitutes while retaining a controlled path to re-identification; still regulated personal data under GDPR |
| Anonymization | Irreversible transformation that makes re-identification no longer reasonably possible, considering the full dataset and available linkage paths |
| Unity Catalog | Databricks' centralized governance layer for data and AI assets, used to enforce permissions and expose governed raw versus curated data objects |
| AI Gateway PII filtering | Endpoint-level Databricks guardrail that detects supported sensitive entities in requests and responses and then blocks or masks them according to policy |
| Data minimization | The principle of collecting and processing only the data necessary for the stated purpose, a core design implication of GDPR and a practical best practice for LLM systems |

---

## Summary / Quick Recall

- PII mitigation in GenAI is a pipeline problem, not a single masking function.
- Regex is best for rigid formats; NER handles context; Presidio packages multi-entity detection and anonymization workflows.
- Masking hides values, tokenization and pseudonymization preserve controlled linkage, and anonymization aims to remove linkage entirely.
- Pseudonymized data is still regulated personal data; do not treat it as anonymous.
- Apply transformations before chunking, embedding, logging, and prompt assembly whenever possible.
- Unity Catalog governs who can access raw versus curated data; do not rely only on application code for privacy boundaries.
- AI Gateway PII filtering is a centralized backstop for serving traffic, not a replacement for earlier-stage controls.
- GDPR and HIPAA translate into engineering choices such as minimum necessary data use, least privilege, auditability, and controlled disclosure.

---

## Self-Check Questions

1. What is the most important difference between pseudonymization and anonymization?

   <details><summary>Answer</summary>

   Pseudonymization replaces identifiers with substitutes but preserves a path to re-identification, usually through a linkage key or controlled mapping, so the data remains regulated personal data. Anonymization aims to make re-identification no longer reasonably possible, which is a much stronger and harder-to-achieve condition. The tempting wrong answer is that both simply "hide names"; that confuses visible masking with the deeper question of whether identity can still be reconstructed.

   </details>

2. A team is building a RAG system over customer-support notes. They can answer questions accurately without customer email addresses, but the raw notes currently include them. What is the best application design choice?

   <details><summary>Answer</summary>

   Remove or mask the email addresses before the notes are chunked and embedded, then build the vector index from the curated text. This follows data minimization and prevents sensitive values from spreading into derived artifacts such as embeddings and retrieval chunks. The main distractor is to keep raw notes in the index and rely on output masking later; that fails because the model, traces, and index may already contain the original emails even if the final answer hides them.

   </details>

3. **Which TWO** controls best represent defence in depth for PII mitigation in a Databricks LLM application?
   - A. Apply application-layer sanitization before logging and prompt assembly
   - B. Depend only on the model's instruction-following ability not to repeat PII
   - C. Use AI Gateway PII filtering on the serving endpoint
   - D. Store raw and masked data in the same shared table and rely on naming conventions
   - E. Increase model temperature so it paraphrases identifiers instead of copying them

   <details><summary>Answer</summary>

   **A and C** are correct. Application-layer sanitization removes or transforms sensitive data before it reaches logs, traces, prompt builders, and other intermediate state, while AI Gateway PII filtering provides centralized endpoint enforcement for all callers. B is wrong because instruction-following is not a privacy control. D is wrong because naming conventions do not create enforceable governance boundaries. E is wrong because higher temperature changes generation variability, not privacy guarantees, and may actually make outputs less predictable.

   </details>

4. You must detect U.S. SSNs, phone numbers, and email addresses in a high-volume preprocessing job, but you also need to catch person names in free-text complaint narratives. What detection strategy is best and why?

   <details><summary>Answer</summary>

   A hybrid strategy is best: use regex for rigid identifiers such as SSNs, phone numbers, and email addresses, and use NER or a Presidio-based recognizer stack for contextual entities like person names. This balances speed and precision for fixed formats with contextual understanding for variable language. The main distractor is "regex for everything" because names and many contextual identifiers do not have stable enough patterns to detect reliably with deterministic rules alone.

   </details>

5. A healthcare chatbot team says, "We pseudonymized patient IDs, so our retrieval corpus is no longer sensitive and can be broadly shared." What is the correct response?

   <details><summary>Answer</summary>

   That conclusion is unsafe because pseudonymized data is still sensitive if re-identification remains possible through a mapping key or through linkage with other attributes. The correct response is to keep the corpus governed, restrict access through Unity Catalog, and evaluate whether other fields in the corpus could still identify patients directly or indirectly. The tempting wrong answer is to treat pseudonymization as equivalent to anonymization; that ignores both GDPR logic and practical re-identification risk in real datasets.

   </details>

---

## Further Reading

- [AI Gateway for serving endpoints](https://docs.databricks.com/aws/en/ai-gateway/overview-serving-endpoints) — *verified 2026-07-11* — Overview of Databricks endpoint-layer guardrails including PII detection and safety controls
- [Configure AI Gateway on model serving endpoints](https://docs.databricks.com/aws/en/ai-gateway/configure-ai-gateway-endpoints) — *verified 2026-07-11* — Official configuration guidance for endpoint guardrails, including PII filtering behavior
- [AI governance with Unity AI Gateway](https://docs.databricks.com/aws/en/ai-gateway/) — *verified 2026-07-11* — Current Databricks governance entry point for AI Gateway and related centralized controls
- [Service policies for AI securables](https://docs.databricks.com/aws/en/data-governance/unity-catalog/service-policies/) — *verified 2026-07-11* — Official Unity Catalog service-policy documentation for governing AI-related securables
- [What is Unity Catalog?](https://docs.databricks.com/aws/en/data-governance/unity-catalog/) — *verified 2026-07-11* — Official overview of centralized governance concepts used to separate raw and curated sensitive data
