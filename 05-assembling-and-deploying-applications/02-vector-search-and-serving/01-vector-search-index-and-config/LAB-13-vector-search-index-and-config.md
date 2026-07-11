# LAB-13: Build, Sync, and Query a Databricks AI Search Index

**Lab:** LAB-13 | **Section:** 05-Assembling & Deploying Applications | **Module:** 02-Vector Search & Serving | **Est. time:** 1.5 hrs

> ⚠️ **Fast-evolving:** Databricks AI Search (formerly Vector Search) is under active development. SDK package names, API parameters, and UI flows may differ from what is documented here. Verify against current official docs before starting.

---

## Objective

Build a working Databricks AI Search index over a synthetic knowledge-base Delta table, trigger a manual sync, execute similarity searches with and without metadata filters, and compare `ann` vs. `hybrid` query types — all using the `AISearchClient` Python SDK.

---

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Serverless compute enabled in the workspace
- `CREATE TABLE` privilege on a Unity Catalog catalog and schema you control
- Ability to create a Databricks AI Search endpoint (workspace admin may need to grant access)
- Access to a cluster running Databricks Runtime 14.0+ or a serverless SQL warehouse for Delta table setup

---

## Setup

Run the following in a Databricks notebook cell to install the AI Search SDK and configure your target catalog and schema:

```python
# Setup: install AI Search SDK and restart Python kernel
%pip install databricks-ai-search
dbutils.library.restartPython()
```

```python
# Setup: define target catalog and schema — edit these to match your environment
CATALOG = "main"          # replace with your catalog name
SCHEMA = "lab13"          # replace with or create your schema
ENDPOINT_NAME = "lab13_search_endpoint"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.kb_articles_index"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.kb_articles"

# Create schema if it does not exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Using: {CATALOG}.{SCHEMA}")
```

---

## Steps

### Step 1 — Create and Populate the Source Delta Table

Create a synthetic knowledge-base table with text articles and metadata, and enable Change Data Feed so the AI Search sync pipeline can read incremental changes.

```python
# Step 1: Create a source Delta table with CDF enabled for AI Search
# CDF is REQUIRED for a Delta Sync Index on a STANDARD endpoint

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {SOURCE_TABLE} (
    id       INT,
    title    STRING,
    content  STRING,
    category STRING,
    sku      STRING
  )
  TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# Insert synthetic knowledge-base articles
spark.sql(f"""
  INSERT INTO {SOURCE_TABLE} VALUES
    (1,  'Return Policy Overview',      'Customers may return any item within 30 days of purchase for a full refund.', 'policy', 'SKU-ALL'),
    (2,  'Shipping to International',   'International orders ship via DHL Express. Standard delivery takes 7-14 business days.', 'shipping', 'SKU-ALL'),
    (3,  'Product XP-7742 Setup Guide', 'To set up model XP-7742, connect the power adapter and press the blue button.', 'setup', 'XP-7742'),
    (4,  'Product XP-7742 Troubleshoot','If model XP-7742 does not power on, check the fuse in compartment A.', 'support', 'XP-7742'),
    (5,  'Warranty Terms',              'All products carry a one-year limited warranty. Register at warranty.example.com.', 'policy', 'SKU-ALL'),
    (6,  'Product ZR-9000 Manual',      'The ZR-9000 features a dual-core processor and 16GB of onboard memory.', 'setup', 'ZR-9000'),
    (7,  'Contact Support',             'Reach our support team at support@example.com or call 1-800-555-0199.', 'support', 'SKU-ALL'),
    (8,  'Bulk Order Discounts',        'Orders over 100 units qualify for a 15% volume discount. Contact sales for details.', 'sales', 'SKU-ALL'),
    (9,  'ZR-9000 Power Issue Fix',     'If the ZR-9000 loses power unexpectedly, update the firmware to version 3.2.', 'support', 'ZR-9000'),
    (10, 'Privacy Policy',              'We collect minimal data and never sell personal information to third parties.', 'policy', 'SKU-ALL')
""")

display(spark.table(SOURCE_TABLE))
```

### Step 2 — Create an AI Search Endpoint

Endpoints are shared infrastructure. Check whether one already exists before creating a new one to avoid errors.

```python
# Step 2: Create the AI Search endpoint (run once; skip if already exists)

from databricks.ai_search.client import AISearchClient

client = AISearchClient()

# Check if endpoint already exists to avoid a duplicate-creation error
try:
    ep = client.get_endpoint(name=ENDPOINT_NAME)
    print(f"Endpoint already exists: {ep['name']} (status: {ep.get('endpoint_status', {}).get('state')})")
except Exception:
    # Endpoint does not exist — create it
    client.create_endpoint(
        name=ENDPOINT_NAME,
        endpoint_type="STANDARD"
    )
    print(f"Endpoint creation initiated: {ENDPOINT_NAME}")
    print("Note: endpoint provisioning may take a few minutes. "
          "Proceed once status is ONLINE.")
```

### Step 3 — Create the Delta Sync Index

Create the index with Databricks-managed embeddings. The SDK will call a Foundation Model API endpoint at sync time to embed the `content` column.

```python
# Step 3: Create a Delta Sync Index with Databricks-managed embeddings
# Uses TRIGGERED pipeline so we control exactly when syncs occur

index = client.create_delta_sync_index(
    endpoint_name=ENDPOINT_NAME,
    source_table_name=SOURCE_TABLE,
    index_name=INDEX_NAME,
    pipeline_type="TRIGGERED",           # we will manually trigger the sync
    primary_key="id",
    embedding_source_column="content",   # Databricks computes embeddings from this column
    embedding_model_endpoint_name="databricks-qwen3-embedding-0-6b",
    columns_to_sync=["id", "title", "content", "category", "sku"]
)

print(f"Index created: {index.name}")
print("The index is not yet populated — proceed to Step 4 to trigger the first sync.")
```

### Step 4 — Trigger the First Sync

With TRIGGERED pipeline, no data appears in the index until `index.sync()` is called. This is the most common operational mistake — the index exists but is empty.

```python
# Step 4: Trigger the first sync to populate the index from the source Delta table
# This step is REQUIRED for TRIGGERED indexes — without it the index is empty.

index = client.get_index(index_name=INDEX_NAME)
index.sync()

print("Sync triggered. This may take several minutes for the first full load.")
print("In production, call index.sync() at the end of your data pipeline.")
```

```python
# Step 4b: Poll index status until it is ONLINE and ready for queries
import time

while True:
    status = client.get_index(index_name=INDEX_NAME)
    state = status.describe().get("status", {}).get("detailed_state", "UNKNOWN")
    print(f"Index state: {state}")
    if state in ("ONLINE", "ONLINE_NO_PENDING_UPDATE"):
        print("Index is ready for queries.")
        break
    if "FAILED" in state.upper():
        print("Index sync failed — check the endpoint logs in the Databricks UI.")
        break
    time.sleep(30)
```

### Step 5 — Run Similarity Searches

Run ANN-only, hybrid, and filtered queries to observe the difference.

```python
# Step 5a: ANN-only query (semantic similarity only)
# Observe whether queries with exact product codes like "XP-7742" are retrieved

index = client.get_index(index_name=INDEX_NAME)

ann_results = index.similarity_search(
    query_text="device not turning on",
    columns=["id", "title", "content", "sku"],
    num_results=5,
    query_type="ann"   # semantic ANN only
)

print("=== ANN Results ===")
for row in ann_results["result"]["data_array"]:
    print(row)
```

```python
# Step 5b: Hybrid query — combines ANN with BM25 keyword matching
# Compare whether XP-7742 or ZR-9000 articles rank differently vs ANN-only

hybrid_results = index.similarity_search(
    query_text="device not turning on",
    columns=["id", "title", "content", "sku"],
    num_results=5,
    query_type="hybrid"   # ANN + BM25 merged via RRF
)

print("=== Hybrid Results ===")
for row in hybrid_results["result"]["data_array"]:
    print(row)
```

```python
# Step 5c: Filtered query — restrict results to a specific SKU
# This is the standard multi-tenant / product-scoped retrieval pattern

filtered_results = index.similarity_search(
    query_text="power problem",
    columns=["id", "title", "content", "sku"],
    filters={"sku": "XP-7742"},   # only return rows where sku = XP-7742
    num_results=5,
    query_type="hybrid"
)

print("=== Filtered Results (sku = XP-7742) ===")
for row in filtered_results["result"]["data_array"]:
    print(row)
```

### Step 6 — Add a New Row and Observe Sync Behavior

Insert a new article and observe that it does NOT appear in search results until `index.sync()` is called. This demonstrates that TRIGGERED mode does not auto-react to table changes.

```python
# Step 6: Insert a new article — it will NOT appear in results until sync is triggered

spark.sql(f"""
  INSERT INTO {SOURCE_TABLE} VALUES
    (11, 'ZR-9000 Battery Replacement',
     'Replace the ZR-9000 battery by removing the back panel with a T6 screwdriver.',
     'support', 'ZR-9000')
""")

# Query BEFORE sync — the new article should NOT appear
pre_sync_results = index.similarity_search(
    query_text="replace battery screwdriver",
    columns=["id", "title"],
    num_results=5
)
print("=== Before sync (article 11 should NOT appear) ===")
for row in pre_sync_results["result"]["data_array"]:
    print(row)

# Now trigger sync
index.sync()
print("Sync triggered. Wait ~1 minute, then re-run the query.")
```

```python
# Step 6b: Re-run the same query after sync completes — article 11 should now appear
post_sync_results = index.similarity_search(
    query_text="replace battery screwdriver",
    columns=["id", "title"],
    num_results=5
)
print("=== After sync (article 11 SHOULD appear) ===")
for row in post_sync_results["result"]["data_array"]:
    print(row)
```

---

## Validation

Run the following validation checks to confirm the lab completed successfully:

```python
# Validation: confirm the index is online and returned expected results

# Check 1: index status
status = client.get_index(index_name=INDEX_NAME).describe()
state = status.get("status", {}).get("detailed_state", "UNKNOWN")
assert state in ("ONLINE", "ONLINE_NO_PENDING_UPDATE"), \
    f"FAIL: index state is {state}, expected ONLINE"
print(f"PASS: index state is {state}")

# Check 2: unfiltered hybrid search returns at least 5 results
results = client.get_index(index_name=INDEX_NAME).similarity_search(
    query_text="return policy",
    columns=["id"],
    num_results=5,
    query_type="hybrid"
)
assert len(results["result"]["data_array"]) == 5, \
    "FAIL: expected 5 results from unfiltered hybrid search"
print("PASS: unfiltered hybrid search returned 5 results")

# Check 3: filtered search only returns rows with the requested SKU
filtered = client.get_index(index_name=INDEX_NAME).similarity_search(
    query_text="power problem",
    columns=["id", "sku"],
    filters={"sku": "XP-7742"},
    num_results=10
)
skus = [row[1] for row in filtered["result"]["data_array"]]
assert all(s == "XP-7742" for s in skus), \
    f"FAIL: filter leak detected — found SKUs: {set(skus)}"
print("PASS: all filtered results have sku = XP-7742")

print("\nAll validation checks passed.")
```

---

## Teardown

Clean up all resources created during the lab to avoid ongoing endpoint costs.

```python
# Teardown: delete the index first, then the endpoint
# WARNING: this permanently deletes all indexed data

# Delete the index
try:
    client.delete_index(index_name=INDEX_NAME)
    print(f"Index deleted: {INDEX_NAME}")
except Exception as e:
    print(f"Index deletion skipped or failed: {e}")

# Delete the endpoint (only if you created it in this lab)
try:
    client.delete_endpoint(name=ENDPOINT_NAME)
    print(f"Endpoint deleted: {ENDPOINT_NAME}")
except Exception as e:
    print(f"Endpoint deletion skipped or failed: {e}")

# Optionally drop the source table and schema
spark.sql(f"DROP TABLE IF EXISTS {SOURCE_TABLE}")
spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")
print("Source table and schema dropped.")
```

---

## Reflection Questions

1. In Step 6, the new article did not appear in search results until `index.sync()` was called. How would you design a Databricks Job workflow that ensures the index is always less than 30 minutes behind the source table, without using CONTINUOUS pipeline mode?

2. The filtered query in Step 5c returned only `sku = XP-7742` results. In a real multi-tenant SaaS application, what would happen if a developer accidentally omitted the `filters` argument from a `similarity_search()` call? How would you prevent this in a production code review?

3. Compare the ANN-only results and the hybrid results from Step 5a vs. Step 5b. If a user searches for the exact product code "XP-7742", which mode produces better recall and why? What does this tell you about which `query_type` to use as the default in production RAG applications?
