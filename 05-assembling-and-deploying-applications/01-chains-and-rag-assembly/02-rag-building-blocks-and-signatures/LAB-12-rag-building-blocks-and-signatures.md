# LAB-12: RAG Building Blocks and Signatures

**Lab:** LAB-12 | **Section:** 05-assembling-and-deploying-applications | **Module:** 01-chains-and-rag-assembly | **Chapter:** 02-rag-building-blocks-and-signatures | **Global lab sequence:** 12 of 22

---

## Objective

Build a complete RAG chain from scratch on Databricks — create a Delta Sync vector index, assemble the retrieval and generation pipeline using LangGraph and LangChain components, define an explicit `ModelSignature` using `ColSpec`, and log the chain with `mlflow.langchain.log_model()` so it is deployable to Model Serving.

---

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Cluster with DBR 15.4 ML or later (includes `mlflow`, `databricks-langchain`, `langchain-core`, `langgraph`)
- Permissions: `CREATE TABLE` on a Unity Catalog schema, `CREATE MODEL` in the model registry, access to a Foundation Model API serving endpoint (e.g. `databricks-meta-llama-3-3-70b-instruct` and `databricks-bge-large-en`)
- A target catalog and schema (replace `lab.rag` with your own throughout the lab)
- Familiarity with: Python, basic LangChain/LangGraph concepts, MLflow run logging

---

## Setup

```bash
# Install required packages (run in a Databricks notebook cell or cluster init script)
%pip install \
  databricks-langchain>=0.3.0 \
  langchain-core>=0.3.0 \
  langgraph>=0.2.0 \
  mlflow>=2.15.0 \
  --quiet

# Restart Python kernel after pip install
dbutils.library.restartPython()
```

```python
# Configure catalog and schema — update these to match your environment
CATALOG     = "lab"
SCHEMA      = "rag"
INDEX_NAME  = f"{CATALOG}.{SCHEMA}.hr_docs_index"
TABLE_NAME  = f"{CATALOG}.{SCHEMA}.hr_docs"
MODEL_NAME  = f"{CATALOG}.{SCHEMA}.hr_rag_chain"

# Foundation model endpoints — verify these are enabled in your workspace
EMBED_ENDPOINT = "databricks-bge-large-en"
LLM_ENDPOINT   = "databricks-meta-llama-3-3-70b-instruct"

# Set MLflow registry to Unity Catalog
import mlflow
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/{CATALOG}/{SCHEMA}/lab12_rag_signatures")
```

---

## Steps

### Step 1 — Create the source Delta table

Create a small corpus of HR policy documents that will back the vector index.

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType

spark = SparkSession.builder.getOrCreate()

# Scenario: Seed an HR policy corpus so we have documents to retrieve
schema = StructType([
    StructField("doc_id",      IntegerType(), False),
    StructField("page_title",  StringType(),  True),
    StructField("page_content",StringType(),  True),
])

hr_docs = [
    (1, "Remote Work Policy",
     "Employees may work remotely up to 3 days per week with manager approval. "
     "A dedicated workspace and reliable internet connection are required."),
    (2, "Parental Leave",
     "Primary caregivers receive 16 weeks of paid parental leave. "
     "Secondary caregivers receive 6 weeks. Leave may begin up to 4 weeks before the due date."),
    (3, "Performance Review Cycle",
     "Performance reviews are conducted semi-annually in June and December. "
     "Employees receive written feedback within 10 business days of the review meeting."),
    (4, "Travel Expense Policy",
     "Business travel must be approved in advance. "
     "Receipts are required for expenses over $25. Reimbursements are processed within 30 days."),
    (5, "Wellness Benefits",
     "Each employee receives a $500 annual wellness stipend for gym memberships, "
     "mental health apps, or fitness equipment. Submit receipts to HR by December 1."),
]

df = spark.createDataFrame(hr_docs, schema=schema)

# Write to Unity Catalog Delta table — required for Delta Sync index
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
df.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(TABLE_NAME)

# Enable Change Data Feed so the Delta Sync index can track updates
spark.sql(f"ALTER TABLE {TABLE_NAME} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"Created table: {TABLE_NAME} with {df.count()} rows")
```

---

### Step 2 — Create a Databricks AI Search (Vector Search) index

```python
# Scenario: Create a Delta Sync index with managed embeddings so Databricks
# handles embedding inference and keeps the index fresh as the Delta table changes.

# Note: Databricks Vector Search was renamed to Databricks AI Search.
# The Python client may still be in the databricks.vector_search namespace.
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

ENDPOINT_NAME = "lab_rag_endpoint"

# Create the vector search endpoint (skip if one already exists in your workspace)
try:
    vsc.create_endpoint(name=ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"Created endpoint: {ENDPOINT_NAME}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"Endpoint {ENDPOINT_NAME} already exists — skipping creation")
    else:
        raise

# Create a Delta Sync index with managed embeddings
# The primary key column and the text column to embed must be specified
try:
    index = vsc.create_delta_sync_index(
        endpoint_name=ENDPOINT_NAME,
        index_name=INDEX_NAME,
        source_table_name=TABLE_NAME,
        pipeline_type="TRIGGERED",           # Use CONTINUOUS for production live updates
        primary_key="doc_id",
        embedding_source_column="page_content",
        embedding_model_endpoint_name=EMBED_ENDPOINT,
    )
    print(f"Index creation started: {INDEX_NAME}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"Index {INDEX_NAME} already exists — fetching existing index")
        index = vsc.get_index(endpoint_name=ENDPOINT_NAME, index_name=INDEX_NAME)
    else:
        raise

# For TRIGGERED pipeline, manually trigger the first sync
import time
index.sync()
time.sleep(30)  # Wait for sync to complete; check status with index.describe()
print(f"Index status: {index.describe()['status']['detailed_state']}")
```

---

### Step 3 — Query the index directly to verify retrieval

```python
# Scenario: Validate that the vector index returns relevant documents before
# wiring it into the full RAG chain.

results = index.similarity_search(
    query_text="How many days can I work from home?",
    columns=["doc_id", "page_title", "page_content"],
    num_results=3,
)

print("Top-3 retrieved documents:")
for i, row in enumerate(results["result"]["data_array"]):
    doc_id, title, content = row[0], row[1], row[2]
    print(f"\n[{i+1}] doc_id={doc_id} | {title}")
    print(f"    {content[:120]}...")

# Expected: doc_id=1 (Remote Work Policy) should rank first
```

---

### Step 4 — Build the LangGraph RAG chain

```python
# Scenario: Assemble a stateful LangGraph RAG chain that retrieves documents,
# assembles a prompt, calls the LLM, and returns both the response and source citations.

from typing import TypedDict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch, DatabricksEmbeddings
from langgraph.graph import StateGraph, END
import json

# --- Define graph state ---
class RAGState(TypedDict):
    query:            str
    docs:             Optional[List[Document]]
    context:          Optional[str]
    response:         Optional[str]
    source_documents: Optional[str]   # JSON string for schema compatibility

# --- Set up retriever ---
embedding = DatabricksEmbeddings(endpoint=EMBED_ENDPOINT)

retriever = DatabricksVectorSearch(
    index_name=INDEX_NAME,
    embedding=embedding,
    text_column="page_content",
    columns=["doc_id", "page_title", "page_content"],
).as_retriever(search_kwargs={"k": 3})

# --- Set up LLM ---
llm = ChatDatabricks(endpoint=LLM_ENDPOINT, max_tokens=256)

# --- Define prompt template ---
prompt_template = ChatPromptTemplate.from_template(
    "You are a helpful HR assistant. Answer ONLY using the context provided.\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n"
    "Answer:"
)

# --- Define graph nodes ---
def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve top-k chunks from Databricks AI Search."""
    docs = retriever.invoke(state["query"])
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    # Build source_documents as a JSON string for schema compatibility
    sources = json.dumps([
        {"doc_id": d.metadata.get("doc_id"), "title": d.metadata.get("page_title"),
         "excerpt": d.page_content[:100]}
        for d in docs
    ])
    return {"docs": docs, "context": context, "source_documents": sources}

def generate_node(state: RAGState) -> RAGState:
    """Call the LLM with the assembled prompt and return the answer."""
    prompt = prompt_template.invoke({"context": state["context"], "query": state["query"]})
    response = llm.invoke(prompt)
    return {"response": response.content}

# --- Build the graph ---
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

rag_app = graph.compile()

# --- Quick smoke test ---
test_result = rag_app.invoke({"query": "What is the parental leave policy?"})
print("Response:", test_result["response"])
print("Sources:", test_result["source_documents"])
```

---

### Step 5 — Wrap the chain in a PyFunc class for clean I/O

```python
# Scenario: Wrap the LangGraph app in an mlflow.pyfunc.PythonModel so the
# predict() method accepts a DataFrame with a "query" column and returns a
# DataFrame with "response" and "source_documents" columns, matching the signature.

import mlflow
import pandas as pd

class RAGChain(mlflow.pyfunc.PythonModel):
    """PyFunc wrapper that exposes the LangGraph RAG app with a clean signature."""

    def load_context(self, context):
        # In production, re-initialise components here using context.model_config
        pass

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in model_input.iterrows():
            output = rag_app.invoke({"query": row["query"]})
            results.append({
                "response":         output.get("response", ""),
                "source_documents": output.get("source_documents", "[]"),
            })
        return pd.DataFrame(results)

# Smoke test the wrapper
test_df = pd.DataFrame([{"query": "How do I claim the wellness stipend?"}])
wrapper = RAGChain()
output_df = wrapper.predict(None, test_df)
print(output_df)
```

---

### Step 6 — Define the `ModelSignature` and log the chain

```python
# Scenario: Formally define the RAG chain's input/output contract using ColSpec
# and log the model so it is deployable to Databricks Model Serving.

from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, DataType, Schema
from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint

# --- Define explicit signature ---
input_schema = Schema([
    ColSpec(name="query", type=DataType.string),
])

output_schema = Schema([
    ColSpec(name="response",         type=DataType.string),
    ColSpec(name="source_documents", type=DataType.string, required=False),
])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Inspect the signature — should show named string columns
print("Signature:\n", signature)

# --- Log the chain ---
input_example = pd.DataFrame([{"query": "What is the parental leave duration?"}])

with mlflow.start_run(run_name="lab12_rag_chain") as run:
    model_info = mlflow.pyfunc.log_model(
        artifact_path="rag_chain",
        python_model=RAGChain(),
        signature=signature,
        input_example=input_example,
        resources=[
            DatabricksVectorSearchIndex(index_name=INDEX_NAME),
            DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
            DatabricksServingEndpoint(endpoint_name=EMBED_ENDPOINT),
        ],
        pip_requirements=[
            "databricks-langchain>=0.3.0",
            "langchain-core>=0.3.0",
            "langgraph>=0.2.0",
        ],
    )
    print(f"Run ID:    {run.info.run_id}")
    print(f"Model URI: {model_info.model_uri}")
```

---

### Step 7 — Register to Unity Catalog and verify the signature

```python
# Scenario: Promote the logged model artifact to the Unity Catalog Model Registry
# so it can be deployed and governed.

uc_model_info = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=MODEL_NAME,
)
print(f"Registered: {MODEL_NAME} version {uc_model_info.version}")

# Verify the signature is present in the registered model
loaded_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{uc_model_info.version}")
registered_sig = mlflow.models.get_model_info(f"models:/{MODEL_NAME}/{uc_model_info.version}").signature

print("\nRegistered model signature:")
print("  Inputs: ", registered_sig.inputs)
print("  Outputs:", registered_sig.outputs)

# Run a test prediction through the registered model
test_result = loaded_model.predict(pd.DataFrame([{"query": "Can I work remotely 5 days a week?"}]))
print("\nTest prediction from registered model:")
print(test_result)
```

---

## Validation

**Expected observable outputs to confirm the lab completed successfully:**

1. **Step 1:** `print(f"Created table: {TABLE_NAME} with {df.count()} rows")` → outputs `Created table: lab.rag.hr_docs with 5 rows`
2. **Step 2:** `index.describe()['status']['detailed_state']` → returns `"ONLINE"` or `"ONLINE_NO_PENDING_UPDATE"`
3. **Step 3:** The top-ranked retrieved document for "How many days can I work from home?" has `doc_id=1` (Remote Work Policy)
4. **Step 5 smoke test:** The PyFunc wrapper returns a DataFrame with columns `response` and `source_documents`; `source_documents` is a non-empty JSON string
5. **Step 6:** `print("Signature:\n", signature)` outputs a signature with:
   ```
   inputs:
     ['query': string (required)]
   outputs:
     ['response': string (required), 'source_documents': string (optional)]
   ```
6. **Step 7:** `registered_sig.inputs` and `registered_sig.outputs` match the signature defined in Step 6; the test prediction returns a DataFrame with a non-empty `response` field

---

## Teardown

```python
# Remove the created resources to avoid ongoing compute charges

# 1. Archive the registered model (prevents accidental serving)
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=uc_model_info.version,
    stage="Archived",
)

# 2. Delete the vector search index
try:
    vsc.delete_index(endpoint_name=ENDPOINT_NAME, index_name=INDEX_NAME)
    print(f"Deleted index: {INDEX_NAME}")
except Exception as e:
    print(f"Index deletion: {e}")

# 3. Delete the vector search endpoint (optional — endpoints have idle costs)
# WARNING: Only delete if no other indexes are using this endpoint
# vsc.delete_endpoint(ENDPOINT_NAME)

# 4. Drop the source Delta table (optional)
# spark.sql(f"DROP TABLE IF EXISTS {TABLE_NAME}")

print("Teardown complete.")
```

---

## Reflection Questions

1. In Step 3, if you change the `query_text` to "What is the expense reimbursement limit?", which document do you expect to rank first and why? What does this tell you about how the embedding model represents semantic similarity across these HR topics?

2. The `RAGChain.predict()` method in Step 5 converts the LangGraph output to a DataFrame by iterating row-by-row. For a production chain serving 100 requests per second, what performance problem does this introduce and how would you redesign the chain to address it?

3. In Step 6, `source_documents` is typed as `DataType.string` (a JSON-serialised list) rather than as a structured array type. What are the tradeoffs of this design choice, and under what circumstances would you use a different output schema to expose the source documents as a typed structure?
