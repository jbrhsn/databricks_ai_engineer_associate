# Licensing and Text Mitigation for LLM Applications on Databricks

**Section:** 06 Governance | **Module:** 01 Guardrails and Compliance | **Est. time:** 2.5 hrs | **Exam mapping:** Governance (8%)

---

## TL;DR

Model licensing and output text mitigation are governance controls that determine which models you can legally deploy, what content they are allowed to generate, and how to prevent harmful or infringing outputs from reaching users. On Databricks, this spans model selection based on license restrictions, output filtering for toxicity and bias, copyright risk mitigation through grounding and attribution, and AI Gateway safety policies that enforce responsible AI practices at the serving boundary. For the exam and for production, the critical insight is that licensing is not just a legal checkbox — it constrains deployment context, commercial use, and derivative work — while output mitigation is not optional polish but a compliance and brand-safety requirement. **The one thing to remember: choose models whose licenses match your deployment context, then layer output guardrails to prevent toxic, biased, or copyright-infringing content from ever reaching production users.**

---

## ELI5 — Explain It Like I'm 5

Imagine you run a library that lends books to readers, but some books come with rules: one says "you can read this at home but not sell copies," another says "you can read this anywhere and even make your own version," and a third says "you can only read this if you are a student, not a business." Before you lend a book, you must check the rules to make sure the borrower is allowed to use it that way. That is model licensing: different models have different rules about who can use them and for what purpose. Now, after someone reads the book and writes a report, you have an editor who checks whether the report copied whole paragraphs from the book without credit, or whether it includes mean or harmful language. If it does, the editor blocks it or asks for a rewrite. That is output text mitigation: making sure what the model generates is safe, fair, and does not violate someone else's rights. The common mistake is thinking that because a model is "open source," you can use it however you want — but many open models still have restrictions on commercial use or require attribution.

---

## Learning Objectives

By the end of this chapter you will be able to:
- [ ] Distinguish between permissive open-source licenses (Apache 2.0, MIT), restrictive open licenses (Llama 2/3 Community License), and commercial/proprietary licenses for foundation models
- [ ] Identify deployment restrictions based on model license terms, including commercial use, derivative works, and attribution requirements
- [ ] Design output filtering pipelines that detect and block toxic, biased, or harmful content before responses reach users
- [ ] Apply copyright and IP risk mitigation strategies such as grounding, citation, and output attribution in RAG systems
- [ ] Configure AI Gateway safety policies (toxicity detection, bias mitigation, factuality checks) on Databricks serving endpoints
- [ ] Explain the role of Llama Guard and similar safety classifiers in production LLM deployments
- [ ] Map responsible AI principles (fairness, transparency, accountability) to concrete engineering controls in Databricks GenAI applications

---

## Visual Overview

### Model License Decision Tree

```
Need to deploy a foundation model
│
├── Is commercial use required?
│     ├── Yes ──► Check license allows commercial deployment
│     │           ├── Permissive (Apache 2.0, MIT) ──► ✓
│     │           ├── Llama 2/3 Community (MAU < 700M) ──► ✓
│     │           ├── Llama 2/3 Community (MAU ≥ 700M) ──► ✗ (need enterprise license)
│     │           └── Research-only license ──► ✗
│     └── No (research/academic)
│           └── Most open licenses allow ──► ✓
│
├── Will you create derivative models (fine-tuning, distillation)?
│     ├── Yes ──► Check license allows derivatives
│     │           ├── Permissive ──► ✓
│     │           ├── Llama 2/3 Community ──► ✓ (with restrictions)
│     │           └── Some proprietary ──► ✗
│     └── No ──► Proceed
│
└── Does license require attribution or disclosure?
      ├── Yes ──► Document model card, license, and usage
      └── No ──► Proceed (but document anyway for governance)
```

### Output Mitigation Pipeline

```
LLM generates response
    │
    ├──► Toxicity detection
    │        ├── Hate speech
    │        ├── Profanity
    │        ├── Threats
    │        └── Sexual content
    │              ├── Block ──► Return fallback
    │              └── Pass ──► Continue
    │
    ├──► Bias detection
    │        ├── Demographic bias
    │        ├── Stereotyping
    │        └── Unfair treatment
    │              ├── Flag / block ──► Return neutral response
    │              └── Pass ──► Continue
    │
    ├──► Factuality / grounding check
    │        ├── Claims match retrieved context?
    │        ├── Hallucination detected?
    │        └── Attribution present?
    │              ├── Fail ──► Reject or request regen
    │              └── Pass ──► Continue
    │
    ├──► Copyright / IP risk check
    │        ├── Verbatim reproduction detected?
    │        ├── Substantial similarity to training data?
    │        └── Proper citation included?
    │              ├── Risk ──► Block or add attribution
    │              └── Safe ──► Continue
    │
    └──► Validated response ──► User
```

### AI Gateway Safety Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│  Application Layer                                  │
│  ├── Input sanitization                             │
│  └── Application-specific guardrails                │
└─────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  AI Gateway (Databricks Serving Endpoint)           │
│  ├── Safety filter (Llama Guard / custom)           │
│  ├── PII detection and masking                      │
│  ├── Rate limiting and cost controls                │
│  └── Audit logging                                  │
└─────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  Foundation Model                                   │
│  (licensed for deployment context)                  │
└─────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  Response Post-Processing                           │
│  ├── Schema validation                              │
│  ├── Toxicity / bias scoring                        │
│  ├── Grounding verification                         │
│  └── Attribution injection                          │
└─────────────────────────────────────────────────────┘
```

### License Types and Deployment Constraints

```
License Type          Commercial Use    Derivatives    Attribution    Restrictions
─────────────────────────────────────────────────────────────────────────────────
Apache 2.0            ✓                 ✓              Optional       Minimal
MIT                   ✓                 ✓              Optional       Minimal
Llama 2/3 Community   ✓ (MAU < 700M)    ✓              Required       MAU threshold
Llama 2/3 Enterprise  ✓                 ✓              Required       Negotiated
Research-only         ✗                 ✗              Required       Non-commercial
Proprietary API       Per agreement     ✗              N/A            Terms of service
```

---

## Key Concepts

### Model Licensing in the Foundation Model Ecosystem

**What it is:** Model licensing defines the legal terms under which a foundation model can be used, modified, distributed, and deployed. Unlike traditional software licenses, LLM licenses often include usage-based restrictions (such as monthly active user thresholds), deployment context constraints (commercial vs research), and requirements around attribution, derivative works, and responsible use.

**How it works under the hood:** When a model is released, the creators attach a license that specifies permitted and prohibited uses. Permissive open-source licenses like Apache 2.0 or MIT impose minimal restrictions, allowing commercial use, modification, and redistribution with few conditions. Restrictive open licenses like the Llama 2 and Llama 3 Community License allow broad use but impose thresholds (such as the 700 million monthly active user limit for commercial deployment) and require a separate enterprise agreement beyond that scale. Proprietary models accessed via API (such as OpenAI GPT-4, Anthropic Claude) are governed by terms of service rather than open licenses, and usage is metered and controlled by the provider. The engineering implication is that license choice constrains not just legal risk but also deployment architecture: a model with a MAU cap may require user tracking and license compliance monitoring, while a fully permissive model does not.

**Where it appears:** On Databricks, model licenses are documented in the Databricks Marketplace model cards, in the Unity Catalog model registry metadata, and in the official model provider documentation. For the exam, you must recognize that "open source" does not automatically mean "unrestricted use" — many popular models have conditional licenses that affect production deployments.

---

### Permissive vs Restrictive Open Licenses

**What it is:** Permissive licenses (Apache 2.0, MIT, BSD) allow nearly unrestricted use, modification, and redistribution, while restrictive open licenses (Llama Community License, some research licenses) impose conditions such as usage scale limits, attribution requirements, or prohibitions on certain use cases.

**How it works under the hood:** Permissive licenses grant broad rights with minimal obligations, typically requiring only that the original license text and copyright notice are preserved in distributions. This makes them ideal for commercial products because there are no usage caps, no approval processes, and no restrictions on derivative works. Restrictive open licenses, by contrast, may allow free use up to a threshold (such as Llama's 700M MAU limit) but require negotiation or payment beyond that, or they may prohibit certain applications (such as military use or content generation for specific industries). The practical difference is that permissive licenses reduce compliance overhead, while restrictive licenses require ongoing monitoring and may trigger legal review as usage scales.

**Where it appears:** Apache 2.0 and MIT are common for models like BERT, GPT-2, and many Hugging Face community models. Llama 2 and Llama 3 use the Meta Community License, which is more restrictive. Databricks Foundation Model APIs may serve models under various licenses, and the model card in the Marketplace or Unity Catalog will specify the terms. Exam questions often test whether you understand that a model being "open" does not mean it is free of deployment restrictions.

---

### Commercial Use and Derivative Work Restrictions

**What it is:** Commercial use restrictions limit whether a model can be deployed in revenue-generating applications, while derivative work restrictions control whether you can fine-tune, distill, or otherwise create modified versions of the model.

**How it works under the hood:** A license that prohibits commercial use (common in research-only licenses) means the model can be used for academic papers, internal experiments, or non-profit projects, but not for customer-facing products or services that generate revenue. A license that prohibits derivative works means you cannot fine-tune the model on your own data, distill it into a smaller model, or use its outputs to train another model. These restrictions are enforced through legal terms, not technical controls, so compliance depends on organizational governance and audit processes. The engineering implication is that if your use case requires fine-tuning or commercial deployment, you must select a model whose license explicitly permits those activities.

**Where it appears:** Llama 2/3 Community License allows commercial use below the MAU threshold and permits fine-tuning with attribution. Some research models (such as early BLOOM variants or academic releases) may prohibit commercial use entirely. Proprietary API models (OpenAI, Anthropic) allow commercial use under their terms of service but prohibit using outputs to train competing models. On Databricks, this information is in the model card and should be verified before production deployment.

---

### Output Toxicity and Bias Detection

**What it is:** Output toxicity detection identifies harmful content such as hate speech, profanity, threats, or sexually explicit material in model-generated text, while bias detection identifies unfair or stereotypical treatment of demographic groups. Both are critical guardrails for production LLM applications.

**How it works under the hood:** Toxicity classifiers are typically fine-tuned language models trained on labeled datasets of toxic vs safe text. They score each generated response on dimensions such as toxicity, severe toxicity, identity attack, insult, profanity, and threat. If the score exceeds a threshold, the response is blocked or flagged for review. Bias detection is more complex because bias can be subtle: it may involve stereotyping (associating certain traits with demographic groups), unfair treatment (different quality of service for different groups), or representation bias (over- or under-representing certain groups). Detection methods include demographic parity checks, counterfactual fairness tests, and embedding-space analysis to identify biased associations. The engineering challenge is that these classifiers are not perfect — they can produce false positives (blocking safe content) and false negatives (missing harmful content) — so production systems often combine automated filtering with human review and feedback loops.

**Where it appears:** On Databricks, toxicity and bias detection can be implemented using open-source libraries (such as Detoxify, Perspective API wrappers), custom fine-tuned classifiers, or as part of AI Gateway safety policies. Llama Guard is a safety classifier specifically designed for LLM outputs and is integrated into some Databricks serving configurations. Exam questions may ask when to block vs flag vs allow with a warning, and the answer depends on risk tolerance and use case.

---

### Copyright and IP Risk Mitigation in LLM Outputs

**What it is:** Copyright risk in LLM systems arises when the model generates text that substantially reproduces copyrighted training data or creates outputs that infringe on intellectual property rights. Mitigation strategies include grounding responses in retrieved context, adding citations, detecting verbatim reproduction, and designing prompts that encourage original synthesis rather than memorization.

**How it works under the hood:** Large language models are trained on vast corpora that include copyrighted material, and they can sometimes reproduce training examples verbatim, especially for well-known texts, code snippets, or distinctive phrases. This creates legal risk because the output may infringe on the original copyright holder's rights. Mitigation approaches include: (1) grounding the model's response in retrieved documents that you have the right to use, so the output is derived from licensed sources rather than memorized training data; (2) adding explicit citations or attributions to retrieved sources, which provides transparency and reduces infringement risk; (3) detecting and blocking outputs that match known copyrighted text using similarity checks or blocklists; (4) designing system prompts that instruct the model to synthesize and paraphrase rather than quote directly. The engineering trade-off is that aggressive filtering may reduce output quality or usefulness, while insufficient filtering increases legal exposure.

**Where it appears:** In Databricks RAG systems, grounding is the primary mitigation: by retrieving context from governed Delta tables or Unity Catalog-managed documents, you ensure the model's response is based on data you control. Citation can be added programmatically by appending source references to the output. For code generation, some organizations use code similarity tools to detect when generated code matches known open-source projects and then add license attribution automatically. Exam questions may contrast grounding (which reduces reliance on memorized training data) with post-hoc filtering (which catches problematic outputs after generation).

---

### AI Gateway Safety Policies and Llama Guard

**What it is:** AI Gateway safety policies are centralized guardrails applied at the Databricks model serving endpoint boundary, and Llama Guard is a safety classifier designed to detect unsafe content in LLM inputs and outputs according to a taxonomy of harm categories.

**How it works under the hood:** When a request is sent to a Databricks serving endpoint with AI Gateway enabled, the gateway intercepts the payload and applies configured policies before forwarding to the model. Safety policies can include toxicity detection, PII filtering, rate limiting, and custom classifiers. Llama Guard is a fine-tuned LLM that classifies text into safety categories such as violence, hate, sexual content, self-harm, and others. It can be applied to both inputs (to block unsafe prompts) and outputs (to block unsafe responses). The advantage of endpoint-level enforcement is that it protects all callers uniformly, including notebooks, SDK clients, and applications that may not implement their own guardrails. The limitation is that endpoint filtering cannot retroactively protect data that was already logged or embedded upstream, so it should be combined with application-layer controls.

**Where it appears:** AI Gateway is configured in the Databricks model serving UI or via REST API when creating or updating an endpoint. Llama Guard can be deployed as a separate endpoint and called before or after the main LLM call, or it can be integrated into a multi-step chain. For the exam, understand that AI Gateway is a centralized enforcement layer, not a replacement for application-layer design, and that Llama Guard is one of several safety classifiers available in the ecosystem.

> ⚠️ Fast-evolving: AI Gateway capabilities, Llama Guard integration, and Unity AI Gateway features are changing rapidly. Verify current official Databricks documentation before relying on exact configuration steps or policy attachment flows.

---

### Responsible AI Principles in Production

**What it is:** Responsible AI is a framework of principles — fairness, transparency, accountability, privacy, safety — that guide the design, deployment, and monitoring of AI systems to ensure they are ethical, trustworthy, and aligned with societal values.

**How it works under the hood:** Responsible AI is not a single feature but a set of engineering practices and governance controls. Fairness means ensuring the system does not produce biased or discriminatory outcomes for different demographic groups, which requires bias testing, counterfactual evaluation, and ongoing monitoring. Transparency means making the system's behavior understandable to users and stakeholders, which includes documenting model cards, providing citations, and explaining decisions. Accountability means establishing clear ownership, audit trails, and mechanisms for redress when the system causes harm. Privacy means protecting sensitive data through PII mitigation, access controls, and data minimization. Safety means preventing harmful outputs through guardrails, testing, and human oversight. The engineering implication is that responsible AI is not a post-deployment checklist but a design philosophy that shapes architecture, testing, and operations from the start.

**Where it appears:** On Databricks, responsible AI practices include Unity Catalog governance for data access, AI Gateway for centralized safety policies, MLflow for model lineage and reproducibility, inference tables for monitoring and audit, and documented model cards in the Marketplace. For the exam, know that responsible AI is not optional or aspirational — it is a compliance and risk-management requirement for production GenAI systems, and it maps to concrete controls such as bias testing, output filtering, and access logging.

---

### Key Parameters / Configuration Knobs

| Parameter | What it controls | Decision rule |
|---|---|---|
| Model license type | Legal permissions for deployment, modification, and commercial use | Choose permissive (Apache 2.0, MIT) for maximum flexibility; accept restrictive (Llama Community) only if usage stays below thresholds or enterprise agreement is in place |
| Toxicity detection threshold | Minimum score required to block or flag a response as toxic | Lower threshold (0.3–0.5) for high-risk applications (public-facing, children, healthcare); higher threshold (0.6–0.8) for internal tools where false positives are costly |
| Bias detection method | How demographic fairness is measured (parity, equalized odds, counterfactual) | Use demographic parity for equal-treatment use cases; use equalized odds when accuracy matters more than equal rates; use counterfactual tests for causal fairness |
| Grounding strategy | Whether responses must cite retrieved sources or can rely on model knowledge | Require grounding and citation for high-stakes domains (legal, medical, financial); allow model knowledge for general Q&A where accuracy is less critical |
| Attribution format | How source citations are presented in outputs | Use inline citations `[1]` for academic-style transparency; use footnotes for user-facing applications; omit only when sources are not user-visible |
| Safety classifier choice | Which model or service detects unsafe content (Llama Guard, Perspective API, custom) | Use Llama Guard for LLM-native safety taxonomy; use Perspective API for web-scale toxicity; build custom classifiers only when domain-specific harm categories are needed |
| Block vs flag vs allow | What happens when unsafe content is detected | Block for high-risk violations (hate speech, threats); flag for review when context matters (political speech, satire); allow with warning for borderline cases |
| Fallback response | What the user sees when a response is blocked | Provide a neutral, informative message that does not reveal the reason for blocking (to avoid adversarial probing); never return the unsafe content with a disclaimer |
| Monitoring and audit scope | Which requests, responses, and decisions are logged | Log all safety decisions (block/flag/allow) with timestamps and reasons; log representative samples of allowed responses for ongoing bias and quality monitoring |
| Human review trigger | When flagged content is escalated to human reviewers | Escalate high-confidence violations for compliance evidence; escalate borderline cases for policy refinement; do not escalate obvious safe content |

### Worked Example: Requirement → Decision

**Given:** A financial services company wants to deploy a Databricks-hosted chatbot that answers customer questions about investment products using internal policy documents and market data. The chatbot must not generate toxic, biased, or misleading content, and it must not reproduce copyrighted financial research verbatim. The company is subject to financial regulations that require auditability and fair treatment of all customers. The application will serve millions of users, so the model license must allow commercial use at scale.

**Step 1 — Identify the goal:** Deploy a compliant, safe, and fair LLM application that grounds responses in governed sources, prevents harmful outputs, and meets regulatory requirements for transparency and accountability.

**Step 2 — Define inputs:** Customer questions (potentially containing PII or adversarial prompts), retrieved policy documents and market data from Unity Catalog, and conversation history.

**Step 3 — Define outputs:** Grounded, cited answers that are factually accurate, free of toxicity and bias, and do not reproduce copyrighted material verbatim.

**Step 4 — Apply constraints:**
- Model license must allow commercial use above 700M MAU (rules out Llama Community without enterprise agreement)
- Responses must cite sources for regulatory transparency
- Toxic, biased, or misleading content must be blocked before reaching users
- PII in user inputs must be masked before logging or model calls
- All safety decisions must be auditable for compliance review

**Step 5 — Select the approach:** Choose a permissively licensed model (such as a Databricks Foundation Model API with Apache 2.0 or commercial terms) or negotiate a Llama 3 enterprise license if that model is preferred. Build a RAG pipeline that retrieves from governed Delta tables and injects citations into the output template. Apply input guardrails (PII masking, prompt injection detection) before the LLM call. Configure AI Gateway safety policies (toxicity detection, bias monitoring) on the serving endpoint. Add output guardrails (schema validation, grounding verification, citation enforcement) after the LLM call. Log all requests, responses, and safety decisions to inference tables with restricted access for audit. This layered approach is better than relying on a single control because it provides defence in depth: license compliance at model selection, input safety at preprocessing, endpoint safety at the gateway, and output safety at post-processing.

---

## Implementation

```python
# Scenario: Select a foundation model for a commercial LLM application on Databricks,
# verifying that the license allows commercial use at scale and documenting the choice.

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# List available foundation models from Databricks Marketplace
models = w.serving_endpoints.list()

def check_license_compliance(model_name: str, expected_mau: int) -> dict:
    """
    Check whether a model's license allows commercial deployment at the expected scale.
    In production, this would query the model card metadata from Unity Catalog or Marketplace.
    """
    # Simplified example: in reality, parse model card or license metadata
    license_info = {
        "meta-llama/Llama-2-70b-chat-hf": {
            "license": "Llama 2 Community License",
            "commercial_use": True,
            "mau_limit": 700_000_000,
            "attribution_required": True,
        },
        "databricks/dbrx-instruct": {
            "license": "Databricks Open Model License",
            "commercial_use": True,
            "mau_limit": None,  # No limit
            "attribution_required": True,
        },
        "mistralai/Mistral-7B-Instruct-v0.2": {
            "license": "Apache 2.0",
            "commercial_use": True,
            "mau_limit": None,
            "attribution_required": False,
        },
    }
    
    info = license_info.get(model_name, {})
    compliant = (
        info.get("commercial_use", False)
        and (info.get("mau_limit") is None or expected_mau < info["mau_limit"])
    )
    
    return {
        "model": model_name,
        "license": info.get("license", "Unknown"),
        "compliant": compliant,
        "reason": (
            "OK" if compliant
            else f"MAU {expected_mau} exceeds limit {info.get('mau_limit')}"
            if info.get("mau_limit")
            else "Commercial use not allowed"
        ),
        "attribution_required": info.get("attribution_required", False),
    }

# Check compliance for a deployment with 1 billion expected MAU
result = check_license_compliance("meta-llama/Llama-2-70b-chat-hf", expected_mau=1_000_000_000)
print(result)
# Output: {'model': 'meta-llama/Llama-2-70b-chat-hf', 'license': 'Llama 2 Community License',
#          'compliant': False, 'reason': 'MAU 1000000000 exceeds limit 700000000', ...}

# Mistral with Apache 2.0 has no MAU limit
result = check_license_compliance("mistralai/Mistral-7B-Instruct-v0.2", expected_mau=1_000_000_000)
print(result)
# Output: {'model': 'mistralai/Mistral-7B-Instruct-v0.2', 'license': 'Apache 2.0',
#          'compliant': True, 'reason': 'OK', 'attribution_required': False}
```

---

```python
# Scenario: Apply output toxicity detection to LLM responses before returning them to users,
# blocking toxic content and logging the decision for audit.

from detoxify import Detoxify

toxicity_model = Detoxify("original")

def check_toxicity(text: str, threshold: float = 0.5) -> dict:
    """
    Score text for toxicity and return whether it should be blocked.
    """
    scores = toxicity_model.predict(text)
    max_score = max(scores.values())
    is_toxic = max_score > threshold
    
    return {
        "text": text,
        "scores": scores,
        "max_score": max_score,
        "is_toxic": is_toxic,
        "action": "BLOCK" if is_toxic else "ALLOW",
    }

# Example: safe response
safe_response = "Your investment portfolio is diversified across equities and bonds."
result = check_toxicity(safe_response)
print(result["action"])  # ALLOW

# Example: toxic response (hypothetical model output)
toxic_response = "You're an idiot for asking that question."
result = check_toxicity(toxic_response)
print(result["action"])  # BLOCK
print(result["scores"])
# {'toxicity': 0.95, 'severe_toxicity': 0.12, 'obscene': 0.08, 'threat': 0.02, ...}

# In production, log the decision to an audit table
if result["is_toxic"]:
    # Log to Delta table for compliance review
    spark.createDataFrame([{
        "timestamp": datetime.now(),
        "user_id": "user_12345",
        "response_text": result["text"],
        "toxicity_score": result["max_score"],
        "action": result["action"],
    }]).write.mode("append").saveAsTable("main.governance.toxicity_audit_log")
```

---

```python
# Scenario: Add source citations to RAG responses to mitigate copyright risk and provide transparency.

def format_response_with_citations(answer: str, sources: list[dict]) -> str:
    """
    Append source citations to the LLM response for transparency and IP risk mitigation.
    """
    citation_text = "\n\nSources:\n"
    for i, source in enumerate(sources, start=1):
        citation_text += f"[{i}] {source['title']} — {source['url']}\n"
    
    return answer + citation_text

# Example usage in a RAG chain
retrieved_docs = [
    {"title": "Investment Policy 2024", "url": "https://internal.example.com/policy/2024", "content": "..."},
    {"title": "Market Analysis Q1", "url": "https://internal.example.com/analysis/q1", "content": "..."},
]

llm_answer = "Based on our investment policy, diversification is recommended for risk management."

final_response = format_response_with_citations(llm_answer, retrieved_docs)
print(final_response)
# Output:
# Based on our investment policy, diversification is recommended for risk management.
#
# Sources:
# [1] Investment Policy 2024 — https://internal.example.com/policy/2024
# [2] Market Analysis Q1 — https://internal.example.com/analysis/q1
```

---

```python
# Anti-pattern: deploying a model without verifying license compliance, then discovering
# usage restrictions only after the application is in production with millions of users.

# Wrong approach:
def deploy_model_wrong():
    # Just pick the most capable model without checking license
    model_name = "meta-llama/Llama-2-70b-chat-hf"
    
    # Deploy to serving endpoint
    w.serving_endpoints.create(
        name="financial-chatbot",
        config={
            "served_models": [{"model_name": model_name, "model_version": "1"}]
        }
    )
    # Later: legal team discovers MAU exceeds 700M limit, application must be taken down

# Correct approach:
def deploy_model_correct():
    # 1. Document expected usage scale
    expected_mau = 1_000_000_000
    
    # 2. Check license compliance before deployment
    compliance = check_license_compliance("meta-llama/Llama-2-70b-chat-hf", expected_mau)
    
    if not compliance["compliant"]:
        print(f"License violation: {compliance['reason']}")
        # 3. Select an alternative model or negotiate enterprise license
        alternative = check_license_compliance("mistralai/Mistral-7B-Instruct-v0.2", expected_mau)
        if alternative["compliant"]:
            model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        else:
            raise ValueError("No compliant model found for expected usage scale")
    else:
        model_name = "meta-llama/Llama-2-70b-chat-hf"
    
    # 4. Document the license decision in model card or deployment metadata
    # 5. Deploy with confidence that license terms are met
    w.serving_endpoints.create(
        name="financial-chatbot",
        config={
            "served_models": [{"model_name": model_name, "model_version": "1"}]
        }
    )

# What breaks in the anti-pattern:
# - No verification of license terms before deployment
# - No consideration of usage scale vs license limits
# - Legal risk discovered only after significant investment in production deployment
# - Potential service disruption and reputational damage from forced takedown
```

---

```python
# Scenario: Configure AI Gateway safety policies on a Databricks serving endpoint to enforce
# toxicity detection and PII filtering at the infrastructure layer.

from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

# Note: AI Gateway configuration syntax is evolving; verify current API in official docs
endpoint_config = EndpointCoreConfigInput(
    name="financial-chatbot-safe",
    served_models=[
        ServedModelInput(
            model_name="main.models.financial_assistant",
            model_version="3",
            scale_to_zero_enabled=True,
        )
    ],
    # AI Gateway guardrails (conceptual example; actual API may differ)
    ai_gateway_config={
        "safety_filter": {
            "enabled": True,
            "toxicity_threshold": 0.5,
            "action": "block",  # or "mask"
        },
        "pii_filter": {
            "enabled": True,
            "entities": ["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"],
            "action": "mask",
        },
        "rate_limit": {
            "requests_per_minute": 1000,
        },
    },
)

# Create or update the endpoint with safety policies
w.serving_endpoints.create_and_wait(config=endpoint_config)

# The gateway now intercepts all requests and responses, applying safety checks
# before forwarding to the model and before returning to the caller.
```

---

## Common Pitfalls & Misconceptions

- **Assuming "open source" means unrestricted use** — Beginners see a model on Hugging Face and assume it can be used for any purpose, but many open models have restrictive licenses that prohibit commercial use or impose usage caps. The correct mental model is that "open" describes availability, not permissions; always read the license before deployment.

- **Relying only on model instructions to prevent harmful outputs** — Teams write system prompts like "never generate toxic content" and assume the model will comply, but instruction-following is not a safety guarantee. The correct mental model is that prompts are suggestions, not enforcement; use output classifiers and guardrails to block unsafe content.

- **Treating toxicity detection as a binary decision** — Beginners set a single threshold and block everything above it, but toxicity is context-dependent and classifiers are imperfect. The correct mental model is that toxicity detection is a risk-scoring tool, not a perfect oracle; combine automated filtering with human review and feedback loops.

- **Ignoring copyright risk in RAG systems** — Teams focus on retrieval quality and forget that the model can still reproduce copyrighted training data even when grounded in retrieved context. The correct mental model is that grounding reduces but does not eliminate copyright risk; add citations, detect verbatim reproduction, and design prompts that encourage synthesis.

- **Deploying without bias testing** — Because bias is harder to measure than toxicity, teams skip it and assume the model is fair. The correct mental model is that all models have bias, and production systems must test for it using demographic parity, counterfactual evaluation, or embedding-space analysis before deployment.

- **Configuring AI Gateway and assuming application-layer controls are unnecessary** — Centralized endpoint filtering sounds comprehensive, so teams skip input sanitization and output validation in application code. The correct mental model is that gateway controls protect the serving boundary, while application-layer controls protect logs, traces, preprocessing, and any path before the endpoint; both are needed for defence in depth.

---

## Key Definitions

| Term | Definition |
|---|---|
| Model license | Legal terms governing how a foundation model can be used, modified, distributed, and deployed, including restrictions on commercial use, derivatives, and attribution |
| Permissive license | An open-source license (such as Apache 2.0 or MIT) that imposes minimal restrictions on use, modification, and redistribution |
| Restrictive open license | An open license that allows broad use but imposes conditions such as usage scale limits, attribution requirements, or prohibitions on certain applications |
| Commercial use | Deployment of a model in revenue-generating applications or services; some licenses prohibit or restrict this |
| Derivative work | A modified version of a model created through fine-tuning, distillation, or other transformations; some licenses prohibit or restrict this |
| MAU (Monthly Active Users) | A usage metric used in some licenses (such as Llama Community License) to define thresholds for commercial deployment |
| Toxicity detection | Automated classification of text to identify harmful content such as hate speech, profanity, threats, or sexually explicit material |
| Bias detection | Identification of unfair or stereotypical treatment of demographic groups in model outputs, measured through parity, equalized odds, or counterfactual tests |
| Grounding | The practice of basing LLM responses on retrieved context rather than relying solely on memorized training data, reducing hallucination and copyright risk |
| Citation / Attribution | Explicit references to source documents in LLM outputs, providing transparency and reducing intellectual property risk |
| Copyright risk | The legal exposure created when an LLM generates text that substantially reproduces copyrighted training data or infringes on intellectual property rights |
| Llama Guard | A safety classifier designed to detect unsafe content in LLM inputs and outputs according to a taxonomy of harm categories |
| AI Gateway safety policies | Centralized guardrails applied at the Databricks serving endpoint boundary, including toxicity detection, PII filtering, and rate limiting |
| Responsible AI | A framework of principles (fairness, transparency, accountability, privacy, safety) that guide ethical and trustworthy AI system design and deployment |

---

## Summary / Quick Recall

- Model licenses constrain deployment context, commercial use, and derivative works; "open source" does not mean unrestricted.
- Permissive licenses (Apache 2.0, MIT) allow broad use; restrictive licenses (Llama Community) impose conditions such as MAU limits.
- Always verify license compliance before production deployment, especially for commercial applications at scale.
- Output toxicity detection blocks harmful content; bias detection ensures fair treatment of demographic groups.
- Grounding responses in retrieved context and adding citations mitigates copyright and IP risk.
- AI Gateway safety policies provide centralized enforcement at the serving endpoint; combine with application-layer guardrails for defence in depth.
- Llama Guard is a safety classifier for LLM-native harm detection; use it for input and output filtering.
- Responsible AI principles (fairness, transparency, accountability) map to concrete controls such as bias testing, audit logging, and output filtering.

---

## Self-Check Questions

1. What is the key difference between a permissive open-source license (such as Apache 2.0) and the Llama 2 Community License?

   <details><summary>Answer</summary>

   Apache 2.0 is a permissive license that allows unrestricted commercial use, modification, and redistribution with minimal obligations (just preserve the license text). The Llama 2 Community License is more restrictive: it allows commercial use but imposes a 700 million monthly active user threshold, beyond which an enterprise agreement is required, and it requires attribution. The tempting wrong answer is that both are "open source" and therefore equivalent; that ignores the usage caps and conditions that make Llama 2 Community License unsuitable for large-scale commercial deployments without negotiation.

   </details>

2. A team is deploying a customer-facing chatbot that will serve 1 billion users. They want to use Llama 2 70B because it performs well in testing. What is the correct licensing decision?

   <details><summary>Answer</summary>

   The Llama 2 Community License allows commercial use only below 700 million MAU, so deploying to 1 billion users violates the license terms. The team must either negotiate a Llama 2 enterprise license with Meta or select a different model with a permissive license (such as Mistral with Apache 2.0 or DBRX with Databricks Open Model License). The main distractor is "just deploy it because it's open source"; that creates legal risk and potential forced takedown.

   </details>

3. **Which TWO** controls best mitigate copyright risk in a Databricks RAG application?
   - A. Ground responses in retrieved context from governed Delta tables
   - B. Increase model temperature to encourage more creative outputs
   - C. Add explicit source citations to all generated responses
   - D. Use a larger model with more parameters
   - E. Disable output logging to prevent evidence of infringement

   <details><summary>Answer</summary>

   **A and C** are correct. Grounding responses in retrieved context reduces reliance on memorized training data, which is the primary source of copyright risk, while adding citations provides transparency and attribution to the sources you have the right to use. B is wrong because higher temperature does not reduce copyright risk and may increase unpredictability. D is wrong because model size does not affect copyright risk. E is wrong because disabling logging hides evidence but does not prevent infringement, and it also eliminates audit trails needed for compliance.

   </details>

4. You are configuring output guardrails for a financial advice chatbot. The toxicity classifier flags a response as toxic with a score of 0.52, just above your threshold of 0.5. The response says "That investment strategy is terrible and will lose you money." What is the best action and why?

   <details><summary>Answer</summary>

   This is a borderline case where the classifier may be detecting strong negative language ("terrible") rather than true toxicity. The best action is to flag for human review rather than automatically blocking, because the content is a legitimate (if blunt) financial opinion, not hate speech or abuse. Blocking it would create a false positive that degrades user experience. The main distractor is "always block anything above the threshold"; that ignores context and the imperfection of toxicity classifiers, which can mistake strong opinions for abuse.

   </details>

5. A team says, "We configured AI Gateway safety filters on our serving endpoint, so we don't need input sanitization or output validation in our application code." What is the correct response?

   <details><summary>Answer</summary>

   That conclusion is unsafe because AI Gateway protects only the serving endpoint boundary; it cannot retroactively protect data that was already logged, embedded, or processed upstream before reaching the endpoint. The correct response is to implement defence in depth: use application-layer input sanitization (PII masking, prompt injection detection) before logging and prompt assembly, use AI Gateway as a centralized backstop at the endpoint, and use application-layer output validation (schema checks, grounding verification) after the model call. The tempting wrong answer is to rely on a single control layer; that violates the principle of defence in depth and leaves gaps in the protection pipeline.

   </details>

---

## Further Reading

- [AI Gateway for serving endpoints](https://docs.databricks.com/aws/en/ai-gateway/overview-serving-endpoints) — *verified 2026-07-11* — Overview of Databricks endpoint-layer guardrails including safety filters and PII detection
- [Configure AI Gateway on model serving endpoints](https://docs.databricks.com/aws/en/ai-gateway/configure-ai-gateway-endpoints) — *verified 2026-07-11* — Official configuration guidance for endpoint guardrails and safety policies
- [Databricks Marketplace model cards](https://docs.databricks.com/aws/en/marketplace/index.html) — *verified 2026-07-11* — Documentation on accessing model metadata including license terms and usage restrictions
- [Unity Catalog model registry](https://docs.databricks.com/aws/en/mlflow/models-in-uc.html) — *verified 2026-07-11* — Official guide to registering and governing models in Unity Catalog with metadata and lineage
- [Responsible AI with Databricks](https://docs.databricks.com/aws/en/machine-learning/responsible-ai/index.html) — *verified 2026-07-11* — Overview of responsible AI practices and tools in the Databricks platform
