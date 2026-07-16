# Licensing and Text Mitigation for LLM Applications — Interview Prep

**Section:** 06 Governance | **Role target:** Senior ML Engineer, GenAI Solutions Architect, AI Governance Lead

---

## Core Conceptual Questions

These test whether you understand the fundamentals.

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is the difference between a permissive open-source license and a restrictive open license for foundation models? | • Permissive (Apache 2.0, MIT) allows unrestricted commercial use, modification, redistribution with minimal obligations<br>• Restrictive (Llama Community License) imposes conditions: MAU thresholds, attribution requirements, usage restrictions<br>• License choice constrains deployment architecture and compliance monitoring | Saying "both are open source so they're basically the same" — ignores usage caps and legal constraints that affect production deployments |
| Why is grounding important for copyright risk mitigation in RAG systems? | • Grounding bases responses on retrieved context you control, not memorized training data<br>• Reduces risk of verbatim reproduction of copyrighted material<br>• Combined with citations, provides transparency and attribution to licensed sources | Claiming "the model won't copy training data if we tell it not to" — instruction-following is not a legal guarantee |
| What is the role of AI Gateway safety policies in a production LLM deployment? | • Centralized enforcement at serving endpoint boundary<br>• Applies toxicity detection, PII filtering, rate limiting uniformly to all callers<br>• Complements but does not replace application-layer guardrails | Thinking "AI Gateway handles all safety so we don't need application-layer controls" — ignores data logged/processed before reaching endpoint |
| How does bias detection differ from toxicity detection in LLM outputs? | • Toxicity detects explicit harmful content (hate speech, threats, profanity)<br>• Bias detects unfair treatment or stereotyping of demographic groups (more subtle)<br>• Bias requires demographic parity tests, counterfactual evaluation, embedding analysis | Treating bias as "just another toxicity category" — bias is context-dependent and requires different measurement approaches |

---

## Behavioral Questions

### Q1: Tell me about a time you had to select a foundation model for a commercial application. How did you evaluate licensing constraints?

**Strong answer framework:**
- **Situation:** Describe the application context (commercial use, expected scale, deployment requirements)
- **Task:** Needed to select a model that met both performance and legal requirements
- **Action:** 
  - Documented expected MAU and deployment context (commercial vs research)
  - Reviewed model cards and license terms for candidate models
  - Identified that Llama 2 Community License had 700M MAU cap, ruled it out for billion-user application
  - Selected Apache 2.0 licensed alternative (Mistral) or negotiated enterprise license
  - Documented license decision in model registry metadata for audit
- **Result:** Deployed compliant model, avoided legal risk and potential forced takedown
- **Tradeoff awareness:** Mention that license restrictions sometimes mean choosing a slightly less capable model to ensure legal compliance

**What makes this answer strong:**
- Shows proactive license verification before deployment, not reactive discovery
- Demonstrates understanding that "open source" ≠ "unrestricted use"
- Includes documentation and governance practices

---

### Q2: Describe a situation where you implemented output filtering for toxic or biased content. What challenges did you face?

**Strong answer framework:**
- **Situation:** Production LLM application serving public users, needed to prevent harmful outputs
- **Task:** Implement multi-layer output guardrails for toxicity and bias
- **Action:**
  - Selected toxicity classifier (Detoxify, Perspective API, or Llama Guard)
  - Set threshold based on risk tolerance (lower for high-risk applications)
  - Implemented flag-for-review workflow for borderline cases, not just binary block/allow
  - Added bias testing using demographic parity checks on representative test set
  - Logged all safety decisions to audit table for compliance review
- **Challenge:** False positives where strong opinions were flagged as toxic; solved by adding human review layer
- **Result:** Reduced harmful outputs by 95% while maintaining user experience quality
- **Tradeoff awareness:** Acknowledge that aggressive filtering can block legitimate content; need balance between safety and usability

**What makes this answer strong:**
- Shows understanding that toxicity detection is imperfect and context-dependent
- Demonstrates defence-in-depth approach (automated + human review)
- Includes monitoring and audit practices

---

### Q3: Have you ever had to retrofit copyright mitigation into an existing RAG system? Walk me through your approach.

**Strong answer framework:**
- **Situation:** Existing RAG system relied heavily on model knowledge, legal team raised copyright concerns
- **Task:** Add grounding and attribution without degrading response quality
- **Action:**
  - Audited existing prompts to identify where model was generating from memory vs retrieved context
  - Modified system prompt to explicitly instruct grounding in retrieved documents
  - Added citation formatting to append source references to all responses
  - Implemented similarity check to detect verbatim reproduction of training data
  - Created fallback response for cases where grounding verification failed
- **Challenge:** Initial implementation degraded response quality because model over-relied on exact quotes; refined prompt to encourage synthesis
- **Result:** Reduced copyright risk exposure, improved transparency, passed legal review
- **Tradeoff awareness:** Grounding can make responses more conservative; need to balance risk mitigation with usefulness

**What makes this answer strong:**
- Shows understanding that grounding is not just retrieval but also prompt design
- Demonstrates iterative refinement based on quality metrics
- Includes legal/compliance perspective

---

## Technical Deep-Dive Questions

### Q: You're deploying a Databricks chatbot for a financial services company with 2 billion expected users. Walk me through your model selection and licensing decision process.

**Strong answer:**

1. **Document constraints:**
   - Commercial use required (financial services application)
   - Scale: 2B MAU exceeds most restrictive license thresholds
   - Regulatory requirements: auditability, fair treatment, transparency
   - Need for grounding in governed sources (compliance)

2. **Evaluate license options:**
   - Llama 2/3 Community License: 700M MAU cap → **ruled out** without enterprise agreement
   - Apache 2.0 models (Mistral, older GPT variants): no MAU limit → **candidate**
   - Databricks Foundation Model APIs: check commercial terms → **candidate**
   - Proprietary APIs (OpenAI, Anthropic): allowed under ToS → **candidate** but vendor lock-in risk

3. **Select approach:**
   - Choose Apache 2.0 licensed model (Mistral) or negotiate Llama 3 enterprise license if performance justifies cost
   - Document license decision in Unity Catalog model card
   - Implement usage tracking if license requires it (even if no current cap)

4. **Justify tradeoffs:**
   - Permissive license reduces compliance overhead and legal risk
   - May sacrifice some performance vs restricted models, but legal compliance is non-negotiable
   - Enterprise license negotiation adds cost but may be worth it for best-in-class model

**What interviewers look for:**
- Do you check license terms **before** deployment, not after?
- Do you understand MAU thresholds and commercial use restrictions?
- Do you document decisions for audit and governance?

---

### Q: How would you design an output filtering pipeline that balances safety with user experience quality?

**Strong answer:**

1. **Multi-stage filtering architecture:**
   ```
   LLM Response
     ├─► Schema validation (structural correctness)
     ├─► Toxicity scoring (Detoxify / Llama Guard)
     │     ├─► High confidence toxic (>0.8) → BLOCK
     │     ├─► Borderline (0.5-0.8) → FLAG for review
     │     └─► Safe (<0.5) → PASS
     ├─► Bias detection (demographic parity check)
     │     └─► Significant disparity → FLAG
     ├─► Grounding verification (claims match context?)
     │     └─► Hallucination detected → REJECT or regen
     └─► Validated response → User
   ```

2. **Threshold tuning strategy:**
   - Start with conservative thresholds (0.5 for toxicity)
   - Monitor false positive rate (safe content blocked)
   - Adjust based on use case risk tolerance:
     - High-risk (public-facing, children): lower threshold (0.3-0.5)
     - Internal tools: higher threshold (0.6-0.8)

3. **Human-in-the-loop for borderline cases:**
   - Automated blocking for clear violations
   - Flag borderline cases for SME review
   - Use feedback to refine thresholds and retrain classifiers

4. **Fallback responses:**
   - Neutral, informative message when blocking
   - Don't reveal reason (prevents adversarial probing)
   - Offer alternative phrasing or topic

5. **Monitoring and iteration:**
   - Log all safety decisions to audit table
   - Track false positive/negative rates
   - A/B test threshold changes
   - Retrain classifiers on production feedback

**Tradeoffs to mention:**
- Aggressive filtering improves safety but increases false positives
- Lenient filtering improves UX but increases risk exposure
- Human review adds latency and cost but improves accuracy
- Need continuous monitoring because model behavior and user patterns evolve

---

### Q: A model generates a response that scores 0.52 on toxicity (threshold 0.5) but is actually a legitimate critical opinion. How do you handle this?

**Strong answer:**

This is a **false positive** — the classifier detected strong negative language but not true toxicity.

**Immediate action:**
- Don't automatically block; flag for human review
- Context matters: "That investment strategy is terrible" is blunt financial advice, not abuse
- Allow with warning or request rephrase, don't block outright

**Systemic fix:**
1. **Refine classifier:**
   - Add training examples of strong opinions that are not toxic
   - Fine-tune on domain-specific data (financial language)
   - Consider separate thresholds for different content categories

2. **Improve prompt design:**
   - Instruct model to be professional but direct
   - Provide examples of acceptable critical feedback
   - Reduce likelihood of borderline language

3. **Implement confidence scoring:**
   - Don't treat threshold as hard cutoff
   - Use confidence intervals: high-confidence toxic → block, low-confidence → review
   - Track classifier calibration over time

4. **User feedback loop:**
   - Allow users to report false positives
   - Use reports to retrain classifier
   - Measure user satisfaction with filtering decisions

**What this demonstrates:**
- Understanding that toxicity detection is probabilistic, not deterministic
- Awareness that context and domain matter
- Commitment to iterative improvement based on production feedback

---

## System Design Questions

### Q: Design a governance layer for a multi-tenant LLM application where different customers have different compliance requirements (some need PII filtering, some need toxicity blocking, some need both).

**Approach:**

1. **Clarify requirements:**
   - How many tenants? (affects config management complexity)
   - Real-time vs batch processing? (affects latency budget)
   - Audit requirements? (affects logging scope)
   - Cost constraints? (affects classifier choice)

2. **Propose architecture:**

```
┌─────────────────────────────────────────────┐
│  Request Router                             │
│  ├─► Tenant ID extraction                   │
│  └─► Policy lookup (tenant → config)        │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Tenant-Specific Guardrail Pipeline         │
│  ├─► PII filter (if tenant requires)        │
│  ├─► Toxicity filter (if tenant requires)   │
│  ├─► Bias check (if tenant requires)        │
│  └─► Custom rules (tenant-specific)         │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Foundation Model (shared)                  │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Response Post-Processing                   │
│  ├─► Apply tenant output filters            │
│  ├─► Add tenant-specific citations          │
│  └─► Log to tenant-isolated audit table     │
└─────────────────────────────────────────────┘
```

3. **Key design decisions:**

   **Policy storage:**
   - Unity Catalog table: `tenant_governance_policies`
   - Columns: `tenant_id`, `pii_filter_enabled`, `toxicity_threshold`, `bias_check_enabled`, `custom_rules_json`
   - Cached in application layer for low-latency lookup

   **Filter composition:**
   - Use chain-of-responsibility pattern: each filter is optional and configurable
   - Filters fail-safe: if filter errors, block request (don't bypass)
   - Parallel execution where possible (PII + toxicity can run concurrently)

   **Audit isolation:**
   - Separate Delta tables per tenant: `main.tenant_{id}.audit_log`
   - Unity Catalog access controls ensure tenant isolation
   - Include: timestamp, user_id, request, response, filters_applied, decisions

   **Cost optimization:**
   - Share model endpoint across tenants (multi-tenancy at serving layer)
   - Cache policy lookups (TTL 5 min)
   - Use lightweight classifiers for high-volume tenants

4. **Tradeoffs:**
   - **Shared model:** Cost-efficient but all tenants use same model version
   - **Per-tenant policies:** Flexible but adds config complexity
   - **Centralized audit:** Easier to manage but requires strong access controls
   - **Real-time filtering:** Low latency but higher compute cost vs batch

**What interviewers look for:**
- Do you clarify requirements before proposing a solution?
- Do you consider multi-tenancy isolation and security?
- Do you think about cost, latency, and operational complexity?
- Do you name tradeoffs explicitly?

---

## Troubleshooting Scenarios

### Scenario 1: Production chatbot is blocking 30% of responses as toxic, but users complain that most blocked content is legitimate. What do you investigate?

**Diagnostic steps:**

1. **Analyze blocked samples:**
   - Pull 100 random blocked responses from audit log
   - Manually review: are they actually toxic or false positives?
   - Identify patterns: specific topics, phrases, or user segments

2. **Check classifier calibration:**
   - What is the toxicity score distribution? (are most near threshold or clearly toxic?)
   - Is the classifier trained on domain-relevant data? (financial language may differ from web text)
   - Are there demographic or topic biases in the classifier?

3. **Review threshold setting:**
   - Current threshold: 0.5 (example)
   - What would false positive rate be at 0.6? 0.7?
   - Plot precision-recall curve to find optimal threshold

4. **Investigate prompt design:**
   - Is the system prompt encouraging overly cautious language?
   - Are there examples in the prompt that bias toward bland responses?

**Likely root causes:**
- **Threshold too low:** Classifier is sensitive, catching strong opinions as toxic
- **Domain mismatch:** Classifier trained on social media, not financial/professional language
- **Prompt design:** Model generates blunt language that triggers classifier

**Fixes:**
1. **Immediate:** Raise threshold to 0.6-0.7 to reduce false positives
2. **Short-term:** Add human review layer for borderline cases (0.5-0.7)
3. **Long-term:** Fine-tune classifier on domain-specific labeled data
4. **Prompt refinement:** Add examples of acceptable professional critical feedback

---

### Scenario 2: Legal team reports that your RAG system generated a response that closely matches a copyrighted document not in your retrieval corpus. How do you respond?

**Immediate response:**

1. **Incident containment:**
   - Identify the specific response and user session
   - Check if response was logged and distributed
   - Temporarily increase grounding verification strictness

2. **Root cause analysis:**
   - Was the copyrighted document in the model's training data? (likely yes for public web data)
   - Did the retrieval system fail to find relevant context, causing model to rely on memory?
   - Was the prompt designed to encourage synthesis or verbatim reproduction?

3. **Verify scope:**
   - Search audit logs for similar responses
   - Run similarity check against known copyrighted sources
   - Estimate exposure: how many users saw potentially infringing content?

**Systemic fixes:**

1. **Strengthen grounding enforcement:**
   ```python
   def verify_grounding(response: str, retrieved_docs: list) -> bool:
       """Ensure response is derived from retrieved context, not model memory."""
       # Check that key claims in response appear in retrieved docs
       # Block responses that introduce facts not in context
       return all(claim_in_context(claim, retrieved_docs) for claim in extract_claims(response))
   ```

2. **Add similarity detection:**
   - Implement n-gram overlap check against known copyrighted sources
   - Block responses with >30% verbatim overlap
   - Add to post-processing pipeline

3. **Improve prompt design:**
   - Explicitly instruct: "Synthesize information from the provided context. Do not quote verbatim."
   - Provide examples of acceptable paraphrasing
   - Penalize verbatim reproduction in prompt

4. **Enhance citation:**
   - Make citations mandatory, not optional
   - Include disclaimer: "Response based on [sources]; verify independently"

5. **Legal consultation:**
   - Document incident and remediation steps
   - Review whether fair use applies (depends on jurisdiction and use case)
   - Consider indemnification or insurance for IP risk

**What this demonstrates:**
- Understanding that grounding reduces but doesn't eliminate copyright risk
- Systematic approach to incident response and root cause analysis
- Awareness of legal implications and need for documentation

---

## Code Review Questions

### Q: Review this license compliance check. What's wrong?

```python
def check_license(model_name: str) -> bool:
    """Check if model can be used commercially."""
    if "llama" in model_name.lower():
        return True  # Llama is open source
    return False
```

**Issues:**

1. **Ignores MAU threshold:** Llama Community License allows commercial use only below 700M MAU
2. **No version check:** Llama 2 and Llama 3 have different license terms
3. **Binary decision:** Doesn't capture nuances like attribution requirements or derivative work restrictions
4. **No documentation:** Doesn't log the license decision for audit
5. **Assumes "open source" = "unrestricted":** Conflates availability with permissions

**Corrected version:**

```python
def check_license_compliance(
    model_name: str, 
    expected_mau: int,
    use_case: str  # "commercial", "research", "derivative"
) -> dict:
    """
    Verify model license allows intended use at expected scale.
    Returns compliance status and required actions.
    """
    license_db = {
        "meta-llama/Llama-2-70b-chat-hf": {
            "license": "Llama 2 Community License",
            "commercial_use": True,
            "mau_limit": 700_000_000,
            "derivatives_allowed": True,
            "attribution_required": True,
        },
        "mistralai/Mistral-7B-Instruct-v0.2": {
            "license": "Apache 2.0",
            "commercial_use": True,
            "mau_limit": None,
            "derivatives_allowed": True,
            "attribution_required": False,
        },
    }
    
    info = license_db.get(model_name)
    if not info:
        return {"compliant": False, "reason": "License terms unknown"}
    
    # Check commercial use
    if use_case == "commercial" and not info["commercial_use"]:
        return {"compliant": False, "reason": "Commercial use not allowed"}
    
    # Check MAU limit
    if info["mau_limit"] and expected_mau >= info["mau_limit"]:
        return {
            "compliant": False, 
            "reason": f"MAU {expected_mau} exceeds limit {info['mau_limit']}",
            "action": "Negotiate enterprise license or select alternative model"
        }
    
    # Check derivatives
    if use_case == "derivative" and not info["derivatives_allowed"]:
        return {"compliant": False, "reason": "Derivative works not allowed"}
    
    return {
        "compliant": True,
        "license": info["license"],
        "attribution_required": info["attribution_required"],
        "action": "Document license in model card" if info["attribution_required"] else None
    }
```

---

### Q: This toxicity filter is deployed in production. What would you change?

```python
def filter_toxic(response: str) -> str:
    if "bad" in response or "terrible" in response:
        return "I cannot provide that response."
    return response
```

**Critical issues:**

1. **Keyword matching is too naive:** Misses most toxic content, catches false positives
2. **No scoring or confidence:** Binary decision without nuance
3. **No logging:** Can't audit or improve
4. **Reveals filtering:** Adversaries can probe to find blocked words
5. **No context awareness:** "bad weather" is not toxic

**Production-grade version:**

```python
from detoxify import Detoxify
import logging

toxicity_model = Detoxify("original")
logger = logging.getLogger(__name__)

def filter_toxic(
    response: str, 
    threshold: float = 0.5,
    user_id: str = None
) -> dict:
    """
    Score response for toxicity and return filtered result with audit trail.
    """
    scores = toxicity_model.predict(response)
    max_score = max(scores.values())
    is_toxic = max_score > threshold
    
    # Log decision for audit
    audit_entry = {
        "timestamp": datetime.now(),
        "user_id": user_id,
        "response_preview": response[:100],  # Don't log full toxic content
        "toxicity_score": max_score,
        "category_scores": scores,
        "action": "BLOCK" if is_toxic else "ALLOW",
    }
    logger.info(f"Toxicity check: {audit_entry}")
    
    # Write to audit table (async to avoid latency)
    write_to_audit_table(audit_entry)
    
    if is_toxic:
        return {
            "allowed": False,
            "response": "I apologize, but I cannot provide that response. Please rephrase your question.",
            "reason_internal": f"Toxicity score {max_score} exceeds threshold {threshold}",
        }
    
    return {
        "allowed": True,
        "response": response,
        "toxicity_score": max_score,  # Include for monitoring
    }
```

---

## Trade-off Discussions

### Q: Your team wants to use Llama 3 70B because it outperforms alternatives, but your application will serve 1 billion users. The Llama Community License has a 700M MAU cap. What do you recommend?

**Analysis:**

**Option 1: Negotiate Llama 3 enterprise license**
- ✅ Use best-performing model
- ✅ Legal compliance at scale
- ❌ Cost: enterprise licenses are expensive
- ❌ Time: negotiation may take weeks/months
- ❌ Vendor dependency: terms may change

**Option 2: Use Apache 2.0 alternative (Mistral, DBRX)**
- ✅ No MAU restrictions
- ✅ Lower cost (no license fees)
- ✅ Faster deployment (no negotiation)
- ❌ May sacrifice some performance
- ❌ Less mature ecosystem for some models

**Option 3: Deploy Llama 3 and hope you don't hit 700M MAU**
- ❌ Legal risk: license violation if you exceed threshold
- ❌ Forced takedown if discovered
- ❌ Reputational damage
- ❌ **Never acceptable**

**Recommendation:**
1. **Quantify performance gap:** Benchmark Llama 3 vs alternatives on your specific use case
2. **Calculate cost:** Enterprise license fee vs compute savings from better model
3. **Assess risk:** What's the business impact of forced takedown?
4. **Decision rule:**
   - If performance gap is >10% and justifies license cost → negotiate enterprise license
   - If performance gap is <5% → use Apache 2.0 alternative
   - If uncertain → start with Apache 2.0, migrate to Llama 3 enterprise if needed

**What this demonstrates:**
- You understand license terms are not optional
- You can quantify tradeoffs (performance vs cost vs risk)
- You prioritize legal compliance while advocating for best technical solution

---

### Q: Should toxicity filtering happen at the application layer, the AI Gateway layer, or both?

**Analysis:**

**Application layer only:**
- ✅ Full control over logic and thresholds
- ✅ Can customize per use case
- ❌ Every application must implement correctly
- ❌ No protection for direct endpoint access
- ❌ Inconsistent enforcement across teams

**AI Gateway layer only:**
- ✅ Centralized enforcement
- ✅ Protects all callers uniformly
- ✅ Easier to update policies
- ❌ Cannot protect data logged before endpoint
- ❌ Less flexibility for application-specific rules
- ❌ Adds latency to every request

**Both (defence in depth):**
- ✅ Application layer: sanitize inputs, validate outputs, log safely
- ✅ Gateway layer: backstop for missed cases, enforce org-wide policies
- ✅ Redundancy: if one layer fails, other catches it
- ❌ More complex architecture
- ❌ Higher compute cost

**Recommendation: Both (defence in depth)**

**Rationale:**
- Security principle: never rely on a single control
- Application layer protects data before it reaches endpoint (logs, traces, preprocessing)
- Gateway layer protects endpoint from misconfigured applications
- Cost is justified by risk reduction in production LLM systems

**Implementation:**
```
User Input
  ├─► Application: PII masking, prompt injection detection
  ├─► Application: Log sanitized input
  ├─► AI Gateway: Toxicity check, rate limit
  ├─► Model: Generate response
  ├─► AI Gateway: Output toxicity check
  ├─► Application: Grounding verification, citation
  └─► User
```

---

## Vocabulary That Signals Expertise

Use these terms naturally — don't force them:

- **"Grounding verification"** — when discussing copyright mitigation; shows you understand the mechanism, not just the goal
- **"MAU threshold"** — when discussing Llama licenses; shows you've read the actual terms
- **"Defence in depth"** — when discussing multi-layer guardrails; signals security mindset
- **"False positive rate"** — when discussing toxicity filtering; shows you understand classifier limitations
- **"Demographic parity"** — when discussing bias; shows you know specific fairness metrics
- **"Attribution requirement"** — when discussing licenses; shows you understand compliance obligations beyond just "allowed/not allowed"
- **"Counterfactual evaluation"** — when discussing bias testing; signals advanced fairness knowledge
- **"Inference table audit trail"** — when discussing monitoring; shows Databricks-specific knowledge

---

## Vocabulary That Signals Weakness

Avoid these — they signal outdated or shallow understanding:

- **"It's open source so we can use it however we want"** — ignores restrictive licenses like Llama Community
- **"We'll just tell the model not to be toxic"** — confuses instruction-following with enforcement
- **"Toxicity is subjective so we can't really measure it"** — true but defeatist; production systems must try
- **"We don't need to worry about copyright because we're using RAG"** — grounding reduces but doesn't eliminate risk
- **"Bias is too hard to fix so we'll just document it"** — signals giving up rather than engineering solutions
- **"AI Gateway handles everything"** — ignores need for application-layer controls
- **"We'll add safety features after launch"** — governance must be designed in, not bolted on

---

## STAR Answer Frame

**Situation:** Our financial services client wanted to deploy a Databricks-hosted investment advice chatbot serving 2 billion users globally. The application needed to ground responses in internal policy documents, prevent toxic or biased outputs, and comply with financial regulations requiring auditability and fair treatment.

**Task:** I was responsible for designing the governance layer, including model license compliance, output filtering, copyright mitigation, and audit logging.

**Action:**
1. **License compliance:** Evaluated Llama 3 70B (best performance) but identified 700M MAU cap in Community License. Calculated that enterprise license negotiation would take 3 months. Selected Mistral 7B Instruct (Apache 2.0) as interim solution, documented decision in Unity Catalog model card.

2. **Output filtering pipeline:** Implemented multi-stage guardrails:
   - Input: PII masking (presidio), prompt injection detection
   - AI Gateway: Toxicity detection (Llama Guard, threshold 0.5)
   - Application: Grounding verification (claims must match retrieved context), citation injection
   - Logging: All safety decisions to inference table with restricted access

3. **Copyright mitigation:** Modified RAG prompt to explicitly require grounding in retrieved documents, added mandatory source citations, implemented n-gram similarity check to detect verbatim reproduction.

4. **Bias testing:** Created test set with demographic variations, measured response quality parity across groups, identified and fixed prompt bias toward certain customer segments.

**Result:**
- Deployed compliant system in 6 weeks (vs 3+ months for license negotiation)
- Blocked 2.3% of responses as toxic (false positive rate <0.5% based on human review)
- Zero copyright incidents in first 6 months of production
- Passed regulatory audit with full audit trail of safety decisions
- Client later negotiated Llama 3 enterprise license; we migrated with zero downtime because architecture was model-agnostic

**What made this successful:**
- Proactive license verification prevented legal risk
- Defence-in-depth approach (multiple guardrail layers)
- Quantified tradeoffs (performance vs compliance vs time-to-market)
- Built for auditability from day one

---

## Red Flags Interviewers Watch For

**Specific to licensing and text mitigation:**

1. **Not checking licenses before deployment** — Shows lack of governance maturity; legal risk is unacceptable in production
2. **Treating "open source" as "unrestricted"** — Signals shallow understanding of license terms
3. **Relying only on prompts for safety** — Confuses instruction-following with enforcement; production systems need guardrails
4. **No audit logging of safety decisions** — Violates compliance requirements; can't improve what you don't measure
5. **Binary thinking about toxicity** — "Block everything above threshold" ignores context and false positives
6. **Ignoring bias because it's "too hard"** — Production systems must attempt fairness even if perfect fairness is impossible
7. **Assuming AI Gateway eliminates need for application-layer controls** — Violates defence-in-depth principle
8. **No plan for false positives** — Shows lack of production experience; all classifiers have errors
9. **Treating copyright as "not my problem"** — Legal risk is everyone's problem in GenAI systems
10. **No human review process** — Fully automated filtering without escalation path is brittle and risky

**How to avoid these:**
- Always mention checking licenses **before** deployment
- Discuss tradeoffs explicitly (safety vs UX, performance vs compliance)
- Show awareness that classifiers are imperfect and need monitoring
- Demonstrate defence-in-depth thinking (multiple layers of control)
- Include audit and compliance perspective in technical decisions
