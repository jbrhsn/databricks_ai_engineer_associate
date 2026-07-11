# Interview Prep: Unity Catalog Model Registration and Serving

**Section:** 05-assembling-and-deploying-applications | **Module:** 02-vector-search-and-serving | **Type:** Interview Prep | **Roles:** ML Engineer, MLOps Engineer, Data Platform Engineer

---

## Core Conceptual Questions

| Question | Key points to cover | Common weak-answer trap |
|---|---|---|
| What is the purpose of `mlflow.set_registry_uri("databricks-uc")` and when is it required? | Routes MLflow registry API calls to UC instead of the legacy Workspace Model Registry; required when DBR < 13.3 LTS or default catalog is `hive_metastore`; not required on DBR 13.3+ with a UC default catalog; can verify current target with `mlflow.get_registry_uri()`. | Saying "it always needs to be set" — on newer runtimes with a UC default catalog it is the default; omitting the condition shows shallow knowledge. |
| How do model aliases in UC differ from the legacy Workspace Model Registry stages? | Stages (Staging/Production/Archived) are not supported in UC; aliases are arbitrary mutable named pointers to specific version numbers; multiple aliases per model version are allowed; no fixed vocabulary is enforced; aliases decouple promotion from version pinning in code. | Treating aliases as a one-to-one alias-for-stage mapping, or claiming that "Production" is a valid UC stage — neither is accurate. |
| What UC privileges are required to register a model to `prod.ml_team.fraud_detection`? | `USE CATALOG` on `prod`, `USE SCHEMA` on `ml_team`, `CREATE MODEL` (or `CREATE FUNCTION`) on the schema to create the registered model; to add a new version to an existing model, need `CREATE MODEL VERSION` on the registered model (or be its owner) plus `USE SCHEMA` and `USE CATALOG`. | Only listing `CREATE MODEL` and forgetting `USE CATALOG` / `USE SCHEMA` — both hierarchy privileges are required. |
| Why is the serving endpoint's "recorded creator" identity important? | It is the identity validated for `EXECUTE` on the model at endpoint creation and at every config update; it is immutable after creation; if deactivated or de-provisioned, all config updates fail with `PERMISSION_DENIED`; solution is always to use a team-owned service principal, never a personal account. | Saying "the caller's identity is used for UC checks" — the caller only needs workspace-level serving permissions; the creator is the UC-level identity. |
| What is the difference between `mlflow.register_model()` and `MlflowClient.create_registered_model()`? | `register_model()` creates both the registered model shell (if absent) AND a new version, takes a `model_uri` pointing to an existing artifact; `create_registered_model()` creates only the shell with no version; versions must still be added separately. | Saying they do the same thing, or confusing `create_registered_model()` with `log_model(registered_model_name=...)`. |

---

## Applied / Scenario Questions

**Scenario 1:** A batch inference job has been using `mlflow.pyfunc.load_model("models:/prod.ml_team.churn@Champion")` for six months. The team wants to test a new model version (v12) without touching the job code. Walk through the exact steps.

*Strong answer:* (1) Ensure version 12 is already registered under `prod.ml_team.churn` (run the training pipeline which calls `log_model(registered_model_name="prod.ml_team.churn")`). (2) Optionally set a `Challenger` alias on v12: `client.set_registered_model_alias("prod.ml_team.churn", "Challenger", 12)`. (3) When ready to promote, atomically reassign the `Champion` alias: `client.set_registered_model_alias("prod.ml_team.churn", "Champion", 12)`. (4) The next run of the batch job resolves `@Champion` at load time and picks up v12 with zero code changes. Note: if the job is a Databricks Workflow, a new job run is needed to re-execute the load call.

**Scenario 2:** You are setting up a Model Serving endpoint for a fraud model. The business requires p99 < 150 ms and the model takes ~80 ms to execute. How would you configure `workload_size`, `scale_to_zero_enabled`, and concurrency?

*Strong answer:* (1) Set `scale_to_zero_enabled=False` — the bimodal latency of cold starts violates the SLA. (2) Start with `workload_size="Small"` (4 vCPU); if model inference saturates CPU, move to `Medium` or `CPU_MEDIUM` for more memory. (3) For concurrency: `min_provisioned_concurrency` should match expected steady-state QPS × 0.08 s = N; set this to the nearest multiple of 4 above N. `max_provisioned_concurrency` should be the peak QPS × 0.08 s × 1.5 safety factor, again rounded up to a multiple of 4. (4) Enable inference tables for ongoing latency monitoring.

**Scenario 3:** After a routine credential rotation, config updates to all 15 serving endpoints start failing with `PERMISSION_DENIED`. What is the most likely cause and how do you remediate?

*Strong answer:* The recorded creator of the endpoints is a service principal whose client secret was rotated and the SDK/CLI credentials were not updated, OR the service principal lost its `EXECUTE` grant on the registered models during a permission audit. To diagnose: check which identity created the endpoints (`GET /api/2.0/serving-endpoints/{name}` → `creator` field); verify that identity still has `EXECUTE` on the model in UC (`SHOW GRANTS ON MODEL catalog.schema.model_name`). To remediate if the principal was removed from the workspace: delete and recreate all affected endpoints under a new service principal. To prevent recurrence: document that endpoint creator must be a workspace-permanent service principal, and set up alerting on UC permission changes.

---

## System Design / Architecture Questions

**Question:** Design a zero-downtime Challenger evaluation system for a production recommendation model using Databricks Model Serving and Unity Catalog. Cover: registration, endpoint configuration, traffic routing, evaluation, promotion, and rollback.

*Strong answer framework:*

```
1. REGISTRATION
   - CI/CD pipeline trains Challenger → log_model(registered_model_name="prod.rec.ranker")
   - New version N created; alias "Challenger" set on version N
   - Champion stays on version N-1 with alias "Champion"

2. ENDPOINT CONFIG (A/B split)
   - Update existing endpoint with two served entities:
     - ranker-champion: entity_version = version of Champion alias, traffic 90%
     - ranker-challenger: entity_version = N, traffic 10%
   - Both entities in same endpoint; old config stays live until new READY

3. TRAFFIC ROUTING
   - Control plane splits at load balancer; no application code change
   - Inference tables enabled on endpoint → both entities' inputs/outputs logged to Delta

4. EVALUATION (automated)
   - Databricks Workflow reads inference table; computes business metrics per served entity
   - If Challenger CTR > Champion CTR by ≥ 2% over 7-day window → promotion signal

5. PROMOTION
   - client.set_registered_model_alias("prod.rec.ranker", "Champion", N)
   - Update endpoint to 100% ranker-champion, remove ranker-challenger entity
   - Old Champion (N-1) retains alias "rollback" for 30 days

6. ROLLBACK
   - If degradation detected: reassign "Champion" alias back to N-1
   - Update endpoint to 100% traffic on N-1
   - Version N retains its version number; it is not deleted, only de-aliased
```

**Question:** What changes to the above design would be required if the Challenger model is trained in `staging` catalog and production inference must use `prod` catalog?

*Strong answer:* UC serving endpoints cannot directly serve a model from a different catalog context without explicit grants. The correct approach is `copy_model_version()` to copy the Challenger from `staging.rec.ranker` to `prod.rec.ranker` before creating the served entity. This ensures: (1) the endpoint creator's `EXECUTE` grant is on the `prod` model; (2) all lineage, tags, and descriptions transfer; (3) only users with `prod` catalog access can serve the model in production. Do not grant cross-catalog `EXECUTE` to the endpoint creator as a workaround — this violates the environment separation principle.

---

## Vocabulary That Signals Expertise

- **"I set `databricks-uc` as the registry URI before registration to ensure UC routing"** — shows awareness of the runtime dependency.
- **"Aliases replace stages in UC; I use Champion/Challenger over version pinning because it decouples promotion from code changes"** — demonstrates UC mental model.
- **"The recorded creator identity is permanent; we always use a team-owned service principal"** — signals operational maturity.
- **"I call `mlflow.log_input(dataset, 'training')` inside the run to propagate lineage to the model version's Lineage tab in Catalog Explorer"** — shows lineage discipline.
- **"Traffic percentages in `traffic_config.routes` must sum to 100; I always update both routes in a single PUT request"** — shows API-level precision.
- **"I set `min_provisioned_concurrency` to at least 4 for production endpoints and disable scale-to-zero to avoid cold-start SLA violations"** — demonstrates production readiness thinking.
- **"I use `copy_model_version()` to promote from staging to prod catalog rather than granting cross-catalog execution"** — signals security discipline.

---

## Vocabulary That Signals Weakness

- **"I move the model to Production stage"** — stages do not exist in UC; this answer reveals the candidate is using the legacy Workspace Model Registry mental model.
- **"Registration deploys the model"** — fundamental confusion; registration is cataloguing, serving is deployment.
- **"I just use the model name without the catalog and schema"** — this silently targets the legacy registry; shows lack of UC awareness.
- **"Scale-to-zero saves money in production"** — true only if the SLA is elastic; in practice it trades latency for cost with no compensating mechanism, which is rarely an acceptable production trade-off.
- **"I created the endpoint with my personal account"** — signals that the candidate does not understand the creator-identity permanence constraint.
- **"I set `traffic_percentage: 10` for the Challenger and leave the Champion at 100"** — would fail API validation because percentages must sum to 100; shows unfamiliarity with the traffic config schema.

---

## STAR Answer Frame

**Situation:** Our team had 12 production serving endpoints, all created using personal user accounts. Three employees left the company, and on the next quarterly model update cycle, all updates to their 6 endpoints began failing with `PERMISSION_DENIED`.

**Task:** I needed to identify the root cause, remediate the blocked endpoints without prolonged downtime, and prevent recurrence across the remaining 6 endpoints.

**Action:** I first queried the serving endpoints API to identify the `creator` field on all 12 endpoints. For the 6 failing endpoints, I confirmed that the recorded creator accounts had been deactivated in the workspace. I worked with the platform team to: (1) create a new shared service principal `ml-serving-sp`; (2) grant it `EXECUTE` on all affected registered models in UC; (3) delete and recreate each failing endpoint under `ml-serving-sp`, preserving the original config by scripting a `GET` → `DELETE` → `CREATE` loop; (4) update the runbook to require service-principal-owned endpoints going forward; (5) add a monitoring query that alerts when an endpoint's creator account is not in the approved service principals list.

**Result:** All 6 endpoints were restored within 4 hours. The remaining 6 endpoints were migrated to `ml-serving-sp` over the following week. Zero subsequent creator-identity incidents have occurred in the 8 months since.

---

## Red Flags Interviewers Watch For

- **Conflating registration with deployment:** A candidate who says "after registering the model, it is available for inference" does not understand the serving layer architecture.
- **Unaware of the three-part namespace requirement:** A candidate who would pass a single-part name to `register_model()` on a UC-enabled workspace demonstrates a critical gap that would cause silent production misbehavior.
- **No service principal discipline:** A candidate who creates endpoints with personal accounts in any environment demonstrates they have not thought through operational continuity.
- **Cannot describe traffic config validation:** If a candidate cannot explain that `traffic_percentage` values must sum to 100 or cannot describe what happens during an in-flight endpoint config update (old config keeps serving), they have not worked with the API directly.
- **No awareness of cold-start latency:** A candidate who recommends scale-to-zero for all endpoints without mentioning the SLA implication has not operated production ML systems.
- **Stages vocabulary in a UC context:** Mentioning "Staging" or "Production" as model stages (rather than catalog tiers or aliases) in a UC discussion signals the candidate is operating from outdated documentation or experience.
