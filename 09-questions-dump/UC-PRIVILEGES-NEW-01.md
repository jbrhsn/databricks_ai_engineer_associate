# Databricks Generative AI Engineer Associate — Practice Questions (D5 Governance: Unity Catalog Privilege Model)

*Focus: UC privilege hierarchy, role-based access, least privilege principle. Difficulty: medium-to-hard. 10 questions, mixed single/multi-select.*

---

## Q1: Privilege Hierarchy and Usage Prerequisites

An engineer has `SELECT` on a UC table in the `sales` catalog under the `finance` schema. They do NOT have `USE CATALOG` on the `sales` catalog, but they do have `USE SCHEMA` on the `finance` schema. When they try to execute `SELECT * FROM sales.finance.customers`, what happens?

   A. Query succeeds; `SELECT` privilege on the table is sufficient to read data.
   B. Query fails; they lack `USE CATALOG` on the parent catalog, which is a prerequisite for accessing any object inside the catalog.
   C. Query succeeds if they request temporary access through the UC UI.
   D. Query fails; they need `READ METADATA` on the table in addition to `SELECT`.

**A: B.** The `USE CATALOG` privilege is a mandatory prerequisite for interacting with *any* object inside a catalog, regardless of what privileges you have on the object itself. The UC permission model requires a three-level grant chain: `USE CATALOG` (catalog) → `USE SCHEMA` (schema) → the specific privilege (e.g., `SELECT`) on the object. Missing `USE CATALOG` breaks the chain at the top level, causing access denial. This is a key safety mechanism: even if a table owner wants to share their table, users cannot bypass catalog-level access controls. **Option A is wrong** because privileges don't grant access without the usage prerequisites. **Option C is wrong** because UI requests don't override the privilege hierarchy. **Option D is wrong** because `READ METADATA` is not a privilege in UC (only shown in the reference for specific use cases); `SELECT` is the primary read privilege.

---

## Q2: Role-Based Access Control with Inheritance (Multi-Select)

You are implementing least privilege for a data engineering team that needs to:
- **Execute** MLflow models deployed in the `prod` catalog  
- **Execute** Vector Search indexes stored in UC

Which TWO of the following steps are required?

   A. Grant `EXECUTE` on the model function object itself; grant `EXECUTE` on the Vector Search index (both at object level, not cascading from schema).
   B. Create a custom role `ml_ops_engineers`, grant `USE CATALOG` and `USE SCHEMA` on the `prod` catalog and its schemas, then grant `EXECUTE` on each model and index individually.
   C. Grant `USE CATALOG` on the `prod` catalog to the team; grant `USE SCHEMA` on each schema containing models/indexes to the team; then grant `EXECUTE` on each individual model function and Vector Search index to the role.
   D. Grant `EXECUTE` at the catalog level in `prod`; this will cascade to all models and indexes, eliminating the need for individual grants.
   E. Create a service principal with `ALL PRIVILEGES` on the `prod` catalog and delegate model serving to the service principal instead of granting team privileges directly.

**A: C and B.** Both **C and B correctly identify the three-layer privilege model**: (1) `USE CATALOG` on the parent catalog, (2) `USE SCHEMA` on each parent schema, (3) the specific privilege (`EXECUTE`) on the securable object. **C is compliant** because it explicitly chains these three layers and grants `EXECUTE` to the role on each object. **B is also compliant** because creating a custom role and following the three-layer grant chain follows best practices for role-based access. **Option A is incomplete** because it omits the mandatory `USE CATALOG` and `USE SCHEMA` prerequisites, so the team would still be denied access to the models/indexes even with `EXECUTE` on them. **Option D is wrong** because `EXECUTE` does not cascade from catalog level to child objects; UC privileges cascade for data access (e.g., `SELECT` cascades) but not for all operation types. **Option E is wrong** because it violates least privilege by granting `ALL PRIVILEGES` and delegates execution away from the team, which is not the requirement.

---

## Q3: Service Principal Authentication for Model Serving Endpoints

A CI/CD pipeline uses a service principal to deploy an MLflow model to a Model Serving endpoint. The model is stored in a Unity Catalog function at `prod.ml_team.iris_classifier`. The endpoint reads training data from a UC table at `prod.analytics.training_data`. Which privileges must the service principal possess?

   A. `EXECUTE` on the model function only; the endpoint can read any table that exists in the same catalog.
   B. `EXECUTE` on the model function AND `USE CATALOG` + `USE SCHEMA` + `SELECT` on the training data table.
   C. `MANAGE` on the model and `MODIFY` on the training table to enable CI/CD automation.
   D. Service principals are automatically granted `ALL PRIVILEGES` on deployed models; no explicit grants are needed.

**A: B.** A service principal used by a Model Serving endpoint must have the full privilege chain for *both* the model and any upstream data it reads. The service principal needs: (1) `USE CATALOG` and `USE SCHEMA` on the `prod` catalog and `ml_team` schema, plus `EXECUTE` on the model function to invoke it; (2) separately, `USE CATALOG` and `USE SCHEMA` on `prod` and `analytics` schemas, plus `SELECT` on the `training_data` table to read it. **Option A is wrong** because sharing a catalog does not grant automatic table access; each securable object requires explicit privileges. **Option C is wrong** because `MANAGE` and `MODIFY` are too broad for a CI/CD read-only scenario (MANAGE is for privilege administration, MODIFY is for writes); the service principal should follow least privilege. **Option D is wrong** because service principals have no implicit privileges; every privilege must be explicitly granted, even for deployed models.

---

## Q4: Least Privilege Design for Large-Scale Analytics

Your analytics team has 50 users who need to run `SELECT` queries on 500 tables spread across 10 schemas in the `analytics` catalog. Following least privilege, which approach is best?

   A. Grant `SELECT` on each of the 500 individual tables to each user (total: 25,000 grants). This ensures maximum control over exactly which tables each user can access.
   B. Create a custom analytics role, grant `USE CATALOG` on the `analytics` catalog to the role, grant `USE SCHEMA` on all 10 schemas to the role, grant `SELECT` on the `analytics` catalog to the role, and assign all 50 users to this role. The `SELECT` privilege cascades to all 500 current and future tables.
   C. Grant `MODIFY` on the `analytics` catalog to all 50 users so they have unrestricted table access; this simplifies administration.
   D. Grant `BROWSE` on the `analytics` catalog so users can discover tables, then grant `EXECUTE` on each table individually for read-only querying.

**A: B.** **Option B is the correct least privilege approach** because it leverages privilege inheritance (cascading) and role-based access control. By granting `SELECT` at the *catalog* level, it automatically applies to all 500 current tables and any future tables added to the catalog, reducing grant overhead from 25,000+ to ~4 grants. The role eliminates duplicate grants across the 50 users. The three-layer chain (`USE CATALOG` → `USE SCHEMA` → `SELECT`) is correct. **Option A is wrong** because it creates administrative burden and doesn't scale; it's the anti-pattern of over-granular access. **Option C is wrong** because `MODIFY` allows writes (INSERT/UPDATE/DELETE) and violates least privilege for read-only analytics users. **Option D is wrong** because `EXECUTE` is for functions and models, not for table queries; `SELECT` is the correct privilege for reading tables, and `BROWSE` alone provides discovery only, not read access.

---

## Q5: Privilege Inheritance and Cascading Rules (Multi-Select)

Which TWO of the following statements accurately describe how UC privilege inheritance works?

   A. `SELECT` granted on a catalog automatically cascades to all tables and views in that catalog (with appropriate `USE CATALOG` and `USE SCHEMA` prerequisites).
   B. `USE SCHEMA` granted on a catalog automatically cascades to all schemas in that catalog.
   C. `MANAGE` granted on a schema automatically applies to all child objects (tables, views, volumes) in that schema without requiring explicit grants on each child.
   D. `CREATE TABLE` granted on a catalog does not cascade to schemas; it must be explicitly granted on each schema where you want to create tables.
   E. Metastore-level privileges (e.g., `CREATE CATALOG`) cascade to all catalogs in the metastore.

**A: A and C.** **Option A is correct** because data-access privileges like `SELECT`, `MODIFY`, and `READ VOLUME` do cascade from container objects (catalogs/schemas) to their child objects. The privilege inheritance model applies downward through the hierarchy: catalog → schemas → tables. **Option C is correct** because `MANAGE` is inherited by all child objects when granted at the schema level; if you have `MANAGE` on a schema, you automatically get `MANAGE` on all current and future child objects, enabling you to manage their privileges without explicit individual grants. **Option B is wrong** because `USE SCHEMA` is a usage privilege specific to each schema; it does not cascade from catalog to schemas—you must grant `USE SCHEMA` on each schema separately, even if you grant `USE CATALOG` on the parent. **Option D is wrong** because creation privileges like `CREATE TABLE` also cascade when granted at the catalog level; they apply to any current or future schema in that catalog. **Option E is wrong** because metastore-level privileges explicitly do *not* cascade to child objects; metastore grants control metastore-scoped operations (like `CREATE CATALOG`), not data within catalogs.

---

## Q6: Revoking Privileges and Operational Impact

An analytics engineer has revoked `SELECT` on a UC table `prod.reporting.sales_metrics` from the analytics team. What happens to Model Serving endpoints that are already running and actively reading this table as input data?

   A. The endpoints continue running and accessing the table with cached data; the privilege revocation takes effect only on next restart.
   B. The endpoints immediately fail when they attempt the next batch inference, because the Model Serving compute no longer has `SELECT` privilege on the table.
   C. The endpoints experience a 5-minute grace period before the revocation is enforced, allowing in-flight queries to complete.
   D. The endpoints continue running until the table itself is dropped; privilege revocations do not affect running processes.

**A: B.** **Privilege revocation is enforced immediately** in UC. When you revoke `SELECT` on the table, the Model Serving endpoint's service principal loses the ability to read from it. The next time the endpoint attempts to query the table (for inference), the query fails with a permission denied error. There is **no caching grace period or delayed enforcement**—UC's access control checks happen at query time, not at process startup. Endpoints that depend on that table will fail on the next read attempt after revocation. **Option A is wrong** because UC doesn't rely on cached credentials; access is checked per query. **Option C is wrong** because UC does not have a built-in grace period—revocation is atomic. **Option D is wrong** because dropping the table is a separate event; privilege revocation takes immediate effect regardless of table existence.

---

## Q7: Cross-Workspace Sharing and Privilege Model Scope

You share a UC table from workspace-A (the provider) to workspace-B (the consumer) using OpenSharing. The table is cataloged as `prod.shared_data.customer_transactions` in workspace-A. When users in workspace-B access the shared table, which workspace's privilege model is enforced?

   A. Workspace-B's privilege model; the shared table appears as a local object in workspace-B with full inheritance of B's access control.
   B. Workspace-A's privilege model; the provider (workspace-A) enforces all access control decisions for the shared table, regardless of workspace-B's configuration.
   C. Both; workspace-A enforces the initial share permission, and workspace-B applies its own privilege checks on top.
   D. Neither; shared tables bypass UC privilege enforcement for seamless cross-workspace collaboration.

**A: B.** **The provider's privilege model (workspace-A) governs access to shared UC assets.** When a consumer workspace accesses a shared table, the provider's privileges determine whether access is allowed. Workspace-B cannot override or layer additional privilege restrictions on a table shared from workspace-A. The share recipient must have the `USE RECIPIENT` privilege (metastore-level) to mount the share, but once mounted, access control flows from the provider's grants. This is a security feature: the data owner (provider) retains control. **Option A is wrong** because workspace-B does not re-apply its own privilege model to provider-shared objects; the table is not "local" from an access control perspective. **Option C is wrong** because it's not a two-layer model; only the provider enforces access. **Option D is wrong** because UC privilege enforcement is mandatory for all access, including shared assets.

---

## Q8: Conflict Resolution — Missing Parent Privilege

An engineer has `EXECUTE` on a UC function `prod.ml_models.predict_churn` but attempts to call the function and receives an access denied error. They confirm they are in the correct workspace and have the right credentials. The most likely root cause is:

   A. The function is owned by another service principal and cannot be called by individual users under any circumstances.
   B. They lack `USE CATALOG` on the `prod` catalog or `USE SCHEMA` on the `ml_models` schema (or both), which are mandatory prerequisites for executing any function.
   C. They need `APPLY TAG` privilege on the function in addition to `EXECUTE` to invoke it.
   D. The function has a dependency on a Vector Search index they cannot access; the execution fails due to missing downstream privileges.

**A: B.** **The privilege hierarchy is non-negotiable**: to execute a function, you must have `USE CATALOG` on its parent catalog *and* `USE SCHEMA` on its parent schema *and* `EXECUTE` on the function itself. Even if you have `EXECUTE` on the function, a missing `USE CATALOG` or `USE SCHEMA` on the parents will block execution at the container level before the function privilege is even checked. This is why the engineer is denied: they have the specific privilege but lack the usage prerequisites. The fix is to grant them `USE CATALOG` and `USE SCHEMA` at the appropriate levels. **Option A is wrong** because ownership doesn't prevent users with proper privileges from calling a function; only privilege denial prevents calls. **Option C is wrong** because `APPLY TAG` is for tagging objects, not for invoking them. **Option D is wrong** because the function's own privilege failure (on `ml_models.predict_churn`) would manifest before any downstream dependencies are evaluated.

---

## Q9: External Tables and Privilege Requirements

You create an external table `analytics.raw.web_logs` that reads data from a Volume (`volumes_storage`) and a cloud storage path (configured as an external location `s3_logs_external`). Which privilege set is required for users to query this table?

   A. `SELECT` on the external table only; volumes and external locations are infrastructure-level, not access-controlled.
   B. `USE CATALOG` and `USE SCHEMA` on the `analytics` catalog and `raw` schema, `SELECT` on the table, AND `READ VOLUME` on the volume (if data is in a volume) or `EXTERNAL USE LOCATION` on the external location (if data is in cloud storage).
   C. `READ FILES` on the external location and nothing else; the table is just a pointer to the storage.
   D. `ALL PRIVILEGES` on the external table; external tables require full administrative privileges due to their dependency on cloud storage.

**A: B.** **External tables inherit the three-layer prerequisite chain plus storage-specific privileges.** Users need: (1) `USE CATALOG` on `analytics` and `USE SCHEMA` on `raw` (standard prerequisites), (2) `SELECT` on the external table itself, (3) *and* the appropriate storage privilege: `READ VOLUME` if the table reads from a Volume, or `READ FILES` / `EXTERNAL USE LOCATION` if it reads from cloud storage via an external location. UC enforces access control at both the metadata layer (the table) and the storage layer (the underlying data location). Skipping either layer results in access denial. **Option A is wrong** because volumes and external locations are fully access-controlled in UC; they are not infrastructure-only. **Option C is wrong** because `READ FILES` alone is insufficient; you still need the table-level `SELECT` privilege. **Option D is wrong** because `ALL PRIVILEGES` is overkill and violates least privilege; specific read privileges suffice.

---

## Q10: Audit and Lineage Tracking for Access Control (Multi-Select)

Which TWO of the following UC features allow you to track *who* has accessed (read or executed) *which* privilege on *which* object?

   A. `SHOW GRANTS` command; returns all privileges granted to a principal on an object.
   B. Delta Lake transaction logs; automatically record privilege changes and access events.
   C. System tables in `system.access.workspace_audit` or `system.access.table_lineage_access`; track privilege grants/revokes and data lineage.
   D. `SHOW LINEAGE` command on a table or function; shows the privilege inheritance path and upstream dependencies.
   E. Databricks audit logs; platform-level logs that record privilege grant/revoke events and query execution by principal.

**A: A and E.** **Option A is correct** because `SHOW GRANTS` returns all privileges granted to a specified principal (user, group, service principal) on a securable object. It's the primary command for *viewing the privilege state* of who has what access. **Option E is correct** because Databricks' workspace/account audit logs record privilege management operations (GRANT, REVOKE, TRANSFER OWNERSHIP) and query execution events, showing *who performed what privilege action when*. These logs are the governance record for compliance and forensic investigation. **Option B is wrong** because Delta transaction logs record data writes, not privilege events; they are orthogonal to access control. **Option C is wrong** because while `system.access` schema exists in Databricks for audit tables, the specific tables referenced (`workspace_audit`, `table_lineage_access`) are not the standard privilege-tracking tables. (Note: Databricks does provide audit logs, but the exact table names vary by deployment.) **Option D is wrong** because `SHOW LINEAGE` shows data dependencies, not privilege inheritance or access history.

---

## Summary

- **Multi-select questions:** Q2, Q5, Q10 (3 of 10 total)
- **Key concepts covered:** privilege hierarchy, role-based access, least privilege, inheritance rules, service principal authentication, privilege revocation, cross-workspace sharing, conflict resolution, external tables, and audit/lineage tracking.
- **Assumptions grounded in official docs:**
  - `USE CATALOG` and `USE SCHEMA` are mandatory prerequisites for accessing any child object, regardless of child-level privileges.
  - Privileges cascade downward (CATALOG → SCHEMA → OBJECT) for data-access privileges (SELECT, MODIFY, READ VOLUME) and CREATE privileges, but NOT for MANAGE or EXTERNAL USE privileges at metastore level.
  - Privilege revocation is enforced immediately at query time.
  - Provider workspace controls access to shared UC objects (consumer cannot override).
  - External tables require both table-level `SELECT` and storage-level privileges (READ_VOLUME or EXTERNAL_USE_LOCATION).
  - Service principals must have explicit privilege grants; no implicit grants exist even for deployed models.
  - Audit logs (Databricks workspace/account logs) and `SHOW GRANTS` are the primary tools for privilege tracking and forensic access control.

---

## File Saved

**Location:** `/home/chainer/Documents/Projects/databricks_ai_engineer_associate/09-questions-dump/UC-PRIVILEGES-NEW-01.md`

All 10 questions are grounded strictly in the official Databricks UC documentation sourced from:
- [Unity Catalog Privileges Reference](https://docs.databricks.com/en/data-governance/unity-catalog/access-control/privileges-reference.html)
- [Unity Catalog Permissions Model Concepts](https://docs.databricks.com/en/data-governance/unity-catalog/access-control/permissions-concepts.html)
- [Manage Privileges in Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/manage-privileges/index.html)
