# Extraction and Chunking Strategies

**Section:** Data Preparation | **Module:** Document Processing | **Est. time:** 4.5 hrs
**Exam mapping:** Databricks GenAI Engineer Associate — Data Preparation domain (~20%); chunking strategy selection and trade-offs are directly tested

## Learning Objectives

By the end of this chapter, you will be able to:

- Choose the correct parsing library for PDF, HTML, plain text, and scanned image documents
- Extract and attach document-level, content-based, structural, and contextual metadata to document chunks
- Apply MinHash locality-sensitive hashing in Spark to deduplicate a document corpus
- Implement relevance, toxicity, and PII filtering as pipeline stages
- Implement and compare all four chunking strategies: fixed-size, paragraph-based, format-specific, and semantic
- Calculate optimal chunk size and overlap values given a target context window budget and embedding model token limit
- Select an embedding model appropriate to a given domain and dataset size, using the MTEB leaderboard as a reference

## Core Concepts

### The Unstructured Data Pipeline for RAG

A RAG application's quality ceiling is set by its data pipeline. The Databricks recommended
pipeline (updated 2026-06-30) has five sequential stages:

```
1. Corpus composition & ingestion
        ↓
2. Data preprocessing
   a. Parsing          → extract text from raw files
   b. Enrichment       → metadata, deduplication, filtering
        ↓
3. Chunking            → split text into retrieval units
        ↓
4. Embedding           → vectorize each chunk
        ↓
5. Indexing & storage  → store in Delta table + AI Search index
```

Each stage has direct quality implications. Errors propagate downstream: a poor chunking
decision cannot be compensated for by a better embedding model.

### Stage 1 — Corpus Composition and Ingestion

The first decision is not technical — it is editorial: **what documents should be in your corpus?**

For a customer support bot, appropriate sources include:
- Knowledge base articles and FAQs
- Product manuals and spec sheets
- Troubleshooting guides
- Release notes (if support queries relate to specific versions)

**What to exclude:**
- Duplicate or outdated versions of the same document (handled by deduplication)
- Documents unrelated to the application's scope (handled by filtering)
- Documents with harmful, biased, or privacy-sensitive content (handled by PII filtering)

**Ingestion best practice:** Store raw source files in Unity Catalog Volumes before processing.
This preserves the original, enables reprocessing with new pipeline parameters, and provides
an audit trail. Use Auto Loader or Lakeflow Spark Declarative Pipelines for scalable,
incremental ingestion as new documents arrive.

### Stage 2a — Parsing: Extracting Text from Raw Files

Parsing converts raw file bytes into clean, processable text. The right tool depends on the file format:

#### PDFs

PDFs are the most common unstructured format in enterprise RAG pipelines. Two approaches:

**`PyPDF2` / `pypdf` (simple, fast):**
```python
from pypdf import PdfReader

def parse_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text
```
Works well for text-layer PDFs (born-digital). Fails on scanned PDFs (no text layer).

**`unstructured` (robust, handles mixed content):**
```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(filename="/path/to/doc.pdf")
text = "\n\n".join([str(e) for e in elements])
```
`unstructured` detects document structure (titles, narrative text, tables, list items) and
tags each element by type. More accurate for complex PDFs with tables or mixed layouts.
Databricks explicitly recommends it for production pipelines.

#### HTML Documents

```python
from bs4 import BeautifulSoup

def parse_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "lxml")
    # Remove navigation, scripts, and style noise
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)
```

**Format-aware alternative:** Use LangChain's `HTMLHeaderTextSplitter` to preserve document
structure (H1/H2/H3 headers) as metadata alongside the text chunks — this enables structured
retrieval by section.

#### Scanned Images and Image-Heavy PDFs (OCR)

Optical Character Recognition (OCR) is required when there is no text layer:

| Option | Type | Notes |
|--------|------|-------|
| **Tesseract** | Open-source | Free; good for clean scans; struggles with complex layouts |
| **Amazon Textract** | Cloud API | Strong layout understanding; tables and forms; AWS-specific |
| **Azure AI Vision OCR** | Cloud API | Strong multilingual; document intelligence features |
| **Google Cloud Vision API** | Cloud API | Strong general OCR; handwriting support |

```python
import pytesseract
from PIL import Image

def ocr_image(image_path: str) -> str:
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)
```

**When to use cloud OCR:** When document quality is variable, layouts are complex, or you need
high accuracy on tables and forms. Cloud APIs significantly outperform Tesseract on real-world
enterprise documents at the cost of per-page API fees.

#### Parsing Best Practices

- **Clean as you parse:** Remove headers, footers, page numbers, and boilerplate during parsing,
  not as a separate step. Noise in the text layer propagates to chunks and degrades retrieval.
- **Handle errors explicitly:** Log files that fail to parse; they often reveal upstream data
  quality issues (corrupted files, password-protected PDFs, encoding errors).
- **Evaluate parsing quality:** Manually review a sample of 20–50 parsed documents. Parsing
  errors are invisible in automated metrics but destroy retrieval quality.

### Stage 2b — Enrichment: Metadata, Deduplication, and Filtering

#### Metadata Extraction

Metadata dramatically improves retrieval by enabling hybrid search (vector + keyword filter).
Four metadata categories (from Databricks 2026 documentation):

| Category | Examples |
|----------|---------|
| **Document-level** | File name, source URL, author, creation date, modification date, document version |
| **Content-based** | Extracted keywords, auto-generated summary, named entities, domain tags (PII, HIPAA) |
| **Structural** | Section heading, page number, chapter title, H2/H3 hierarchy |
| **Contextual** | Ingestion date, data sensitivity level, source system, original language |

**LLM-assisted metadata extraction:**
```python
def extract_metadata_with_llm(chunk_text: str, document_title: str) -> dict:
    """Use a small LLM to generate content-based metadata for a chunk."""
    response = client.chat.completions.create(
        model="databricks-meta-llama-3-1-8b-instruct",
        messages=[
            {"role": "system", "content": """Extract metadata from this document chunk.
Return JSON with keys: "keywords" (list of 3-5 key terms), "summary" (1 sentence),
"entities" (list of named entities). Return only JSON."""},
            {"role": "user", "content": f"Title: {document_title}\n\nChunk: {chunk_text}"}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    import json
    return json.loads(response.choices[0].message.content)
```

Store metadata alongside chunk text and embeddings in the same Delta table row. This enables
efficient `WHERE` clause filtering during AI Search queries.

#### Deduplication with MinHash LSH

Duplicate documents create redundant chunks in the index, degrading retrieval quality and
wasting storage. MinHash locality-sensitive hashing is the Databricks-recommended approach
for corpus-scale deduplication (not just exact-match).

**The four-step process (Spark ML):**

```python
from pyspark.ml.feature import HashingTF, MinHashLSH
from pyspark.sql import functions as F
import re

# Step 1: Featurize — tokenize into n-grams and create TF vectors
def tokenize(text):
    tokens = re.sub(r'[^a-z\s]', '', text.lower()).split()
    # 2-grams for better duplicate detection
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]

tokenize_udf = F.udf(tokenize, "array<string>")
df_tokens = df_docs.withColumn("tokens", tokenize_udf(F.col("text")))

# Step 2: Hash features
hashing_tf = HashingTF(inputCol="tokens", outputCol="features", numFeatures=10000)
df_features = hashing_tf.transform(df_tokens)

# Step 3: Apply MinHash
minhash = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
model = minhash.fit(df_features)
df_hashed = model.transform(df_features)

# Step 4: Find and remove near-duplicates (Jaccard similarity threshold)
similar_pairs = model.approxSimilarityJoin(
    df_hashed, df_hashed, threshold=0.8, distCol="jaccard_distance"
)
# Filter to keep only one of each duplicate pair
duplicates = similar_pairs.filter(
    F.col("datasetA.doc_id") < F.col("datasetB.doc_id")
)
duplicate_ids = duplicates.select(F.col("datasetB.doc_id").alias("doc_id")).distinct()
df_deduped = df_docs.join(duplicate_ids, on="doc_id", how="left_anti")
```

**Jaccard similarity threshold:** `0.8` means "documents sharing 80% of their n-gram
vocabulary are considered duplicates." Tune this based on your corpus — technical documentation
with repeated boilerplate may need a higher threshold than general text.

#### Filtering

Remove documents that would harm retrieval quality or introduce compliance risk:

- **Relevance filtering:** Route documents through a classifier (rules-based or small LLM) to
  confirm they are in-scope for the application
- **Toxicity filtering:** Apply a toxicity classifier; set a threshold below which documents are
  rejected from the corpus
- **PII detection:** Use a NER model or regex patterns to flag documents containing names, SSNs,
  credit card numbers, email addresses — then either redact or exclude

---

### Stage 3 — Chunking Strategies

Chunking is **the most important data pipeline decision for retrieval quality.** The Databricks
documentation (2026-06-30) explicitly states: "Choices made on chunking will directly affect the
retrieved data the LLM receives, making it one of the first layers of optimization in a RAG
application."

**The core trade-off:**
- **Smaller chunks:** More precise retrieval; less irrelevant context in the prompt; but may
  lose surrounding context needed to answer the question fully
- **Larger chunks:** More context preserved; but may include irrelevant information that
  distracts the model and consumes context window budget

No single chunk size is universally optimal. The right choice depends on source document
structure, query patterns, and context window budget.

#### Strategy 1 — Fixed-Size Chunking

Split text every N characters or tokens, regardless of content structure.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # characters
    chunk_overlap=200,    # characters of overlap between adjacent chunks
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]  # preference order for split points
)
chunks = splitter.split_text(document_text)
```

**`RecursiveCharacterTextSplitter`** is preferable to `CharacterTextSplitter` because it
tries natural boundaries (paragraphs → sentences → words → characters) before resorting to
arbitrary splits.

**When to use:** As a baseline or for homogeneous plain-text documents with no meaningful
structure. The Databricks docs note this "rarely works for production-grade applications" when
used without any structure awareness.

**Key parameter:** `chunk_overlap` — the number of characters shared between adjacent chunks.
Overlap ensures that a sentence spanning a chunk boundary appears in at least one chunk that
contains both its context and the answer. Typical values: 10–20% of `chunk_size`.

#### Strategy 2 — Paragraph-Based Chunking

Use natural paragraph boundaries (double newlines) to define chunk boundaries.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". "]  # paragraph → sentence → word
)
chunks = splitter.split_text(document_text)
```

**Why it's better than fixed-size:** Paragraphs are written as semantic units — an author
rarely continues a single thought across a paragraph break. Keeping paragraphs intact preserves
semantic coherence.

**When to use:** Most prose documents (articles, documentation, emails, meeting notes). This
is the most common strategy for general-purpose RAG.

**Limitation:** Inconsistent paragraph lengths create variable-size chunks. Some chunks may
be too short (a one-sentence paragraph) or too long (a multi-paragraph section with no breaks).

#### Strategy 3 — Format-Specific Chunking

Leverage document structure (Markdown headers, HTML tags) to define meaningful chunk boundaries.

**Markdown with headers:**
```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)

# Each chunk has metadata automatically attached:
# chunk.metadata = {"h1": "Installation Guide", "h2": "Requirements", "h3": "Python"}
```

**HTML with section boundaries:**
```python
from langchain.text_splitter import HTMLHeaderTextSplitter

headers_to_split_on = [("h1", "header_1"), ("h2", "header_2"), ("h3", "header_3")]
splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text_from_file(html_file_path)
```

**When to use:** Technical documentation in Markdown or HTML; API reference docs; any document
with a well-defined heading hierarchy. The structural metadata (which section each chunk belongs
to) is extremely valuable for retrieval filtering — e.g., "only retrieve from the 'Troubleshooting'
section."

**Limitation:** Requires consistently structured source documents. Documents with inconsistent
or missing headers produce poor results.

#### Strategy 4 — Semantic Chunking

Analyze content meaning to find topic-shift boundaries, then split at natural semantic transitions.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import DatabricksEmbeddings

embedding_model = DatabricksEmbeddings(
    endpoint="databricks-bge-large-en"
)

splitter = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=95           # split where similarity drops to bottom 5%
)
chunks = splitter.split_text(document_text)
```

**How it works:** The splitter embeds consecutive sentence pairs and computes cosine similarity.
Large drops in similarity indicate topic transitions — those are split points.

**When to use:** Long documents with clear topic shifts (e.g., a whitepaper covering multiple
unrelated sections); when other strategies produce semantically incoherent chunks.

**Limitations:** Much slower than other strategies (requires embedding every sentence during
chunking); higher cost; results vary with embedding model quality.

**Summary comparison:**

| Strategy | Best for | Main advantage | Main limitation |
|----------|---------|----------------|-----------------|
| Fixed-size | Plain text baseline | Simple, fast | Ignores meaning |
| Paragraph-based | General prose | Preserves natural units | Variable chunk sizes |
| Format-specific | Structured docs (Markdown/HTML) | Header metadata attached | Requires consistent structure |
| Semantic | Long, varied documents | Respects topic boundaries | Slow, costly |

---

### Stage 4 — Embedding Model Selection

The embedding model converts each chunk into a vector for similarity search. Key selection
criteria from Databricks documentation (2026-06-30):

**Max token limit:** Know your embedding model's maximum input length. `bge-large-en-v1.5` has
a maximum of **512 tokens**. Chunks longer than this are truncated — you lose the tail of the
chunk silently. This is one of the most common production bugs in RAG pipelines.

```python
# Safe chunking that respects the embedding model's token limit
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# For bge-large-en-v1.5: max 512 tokens ≈ ~380 words ≈ ~1,900 characters
# Set chunk_size conservatively below the limit
MAX_CHUNK_TOKENS = 450  # leave headroom
```

**Recommended models (available on Databricks Foundation Model API, 2026):**

| Model | Max tokens | Notes |
|-------|-----------|-------|
| `databricks-bge-large-en` | 512 | Strong English; most used in Databricks RAG tutorials |
| `text-embedding-3-small` (OpenAI) | 8,191 | Large context window; good general-purpose |
| `GTE-Large-v1.5` (Alibaba NLP) | 8,192 | Top MTEB leaderboard performer |

**MTEB leaderboard:** The Massive Text Embedding Benchmark (https://huggingface.co/spaces/mteb/leaderboard)
ranks embedding models by retrieval accuracy across many datasets. Use it to compare models —
but note: the benchmark dataset may not match your domain. Always test on a sample of your
own data before committing to an embedding model.

**Domain-specific fine-tuning:** If your corpus contains proprietary terminology or acronyms
not well-represented in public training data, fine-tuning the embedding model on domain-specific
examples can significantly improve retrieval accuracy.

---

### Putting It All Together: Token Budget Calculation

Before building a pipeline, calculate the token budget that constrains your chunk size:

```
context_window           = 32,768 tokens  (DBRX Instruct)
- system_prompt_tokens   =    500 tokens  (fixed overhead)
- user_query_tokens      =    100 tokens  (estimated max)
- max_response_tokens    =  1,500 tokens  (your SLA)
- safety_margin          =    200 tokens  (buffer)
= available_for_context  = 30,468 tokens

chunks_that_fit = 30,468 ÷ chunk_size_in_tokens

# Example: 400-token chunks → 76 chunks retrievable per call
# In practice, retrieve 5-10 top chunks for quality + latency balance
```

Also constrain by the **embedding model's max tokens per chunk** (e.g., 450 for `bge-large-en-v1.5`).
Your effective chunk size is `min(context_budget_per_chunk, embedding_model_max_tokens)`.

## Deep Dive / Advanced Topics

### Chunk Overlap — Why It Matters More Than It Looks

Without overlap, a sentence that spans the boundary between two chunks appears in neither chunk
completely. A user query that matches the boundary sentence finds neither chunk relevant.

```
Without overlap:
  Chunk 1: "...The maximum throughput is 1,000 requests per second."
  Chunk 2: "This limit applies to all Foundation Model API endpoints..."

Query: "What is the limit on Foundation Model API endpoints?"
→ Neither chunk contains the full sentence; retrieval may miss both
```

```
With overlap (chunk_overlap = 200 chars):
  Chunk 1: "...The maximum throughput is 1,000 requests per second. This limit applies to all Foundation Model API endpoints..."
  Chunk 2: "This limit applies to all Foundation Model API endpoints..."

→ Chunk 1 now contains the full sentence; retrieval succeeds
```

**Rule of thumb:** Set overlap to 10–20% of `chunk_size`. Higher overlap improves recall at the
cost of more chunks stored and slightly more retrieval noise (redundant chunks may both be
retrieved).

### Metadata as a Retrieval Filter

Metadata attached to each chunk enables pre-filtering before vector search, dramatically
reducing the search space and improving precision:

```python
# Without metadata filtering: search all 100,000 chunks
results = index.similarity_search(query=user_query, k=5)

# With metadata filtering: search only the 3,000 chunks from the "API Reference" section
results = index.similarity_search(
    query=user_query,
    k=5,
    filters={"section": "API Reference", "product_version": ">=14.0"}
)
```

This is called **hybrid search**: combining vector similarity with metadata filters. Databricks
AI Search supports this natively. Designing your metadata schema before building the pipeline
is important — retrofitting metadata onto an existing index requires re-indexing.

### Adaptive Chunking (Advanced)

Rather than one fixed chunking strategy for the entire corpus, use routing:

```python
def chunk_document(doc_text: str, doc_format: str) -> list[str]:
    if doc_format == "markdown":
        return format_specific_chunk(doc_text)
    elif doc_format == "html":
        return html_header_chunk(doc_text)
    elif len(doc_text) > 10_000:
        return semantic_chunk(doc_text)  # only use semantic for long docs (worth the cost)
    else:
        return paragraph_chunk(doc_text)  # default
```

This per-document routing is the pattern used in production Databricks RAG pipelines for
mixed-format corpora.

## Worked Examples & Practice

### End-to-End Pipeline: Parsing, Chunking, and Embedding a PDF Corpus

This example processes a collection of PDF product manuals stored in a Unity Catalog Volume
and produces a Delta table of chunks ready for AI Search indexing.

```python
%pip install unstructured pypdf langchain langchain-community tiktoken
dbutils.library.restartPython()

import os
import json
from typing import List, Dict, Any
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# ── 1. Configuration ──────────────────────────────────────────────────────────
CATALOG     = "main"
SCHEMA      = "rag_demo"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_docs"
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.document_chunks"

CHUNK_SIZE    = 400   # tokens (conservative for bge-large-en-v1.5 max=512)
CHUNK_OVERLAP = 60    # tokens (~15% of chunk_size)

client = OpenAI(
    api_key=dbutils.secrets.get(scope="genai-keys", key="databricks-token"),
    base_url=f"https://{spark.conf.get('spark.databricks.workspaceUrl')}/serving-endpoints"
)

# ── 2. Parse PDF ──────────────────────────────────────────────────────────────
def parse_pdf_to_text(file_path: str) -> tuple[str, dict]:
    """Parse a PDF and return (clean_text, document_metadata)."""
    elements = partition_pdf(filename=file_path)
    # Filter to narrative text (skip tables, images for this example)
    text = "\n\n".join([str(e) for e in elements if e.category in ("NarrativeText", "Title")])
    metadata = {
        "source_file": os.path.basename(file_path),
        "source_path": file_path,
        "num_elements": len(elements),
    }
    return text, metadata

# ── 3. Chunk text ─────────────────────────────────────────────────────────────
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")

def token_length(text: str) -> int:
    return len(enc.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=token_length,   # count in tokens, not characters
    separators=["\n\n", "\n", ". ", " "]
)

def chunk_document(text: str, doc_metadata: dict) -> List[Dict[str, Any]]:
    chunks = splitter.split_text(text)
    return [
        {
            "chunk_text": chunk,
            "chunk_index": i,
            "token_count": token_length(chunk),
            **doc_metadata,
        }
        for i, chunk in enumerate(chunks)
    ]

# ── 4. Embed chunks ───────────────────────────────────────────────────────────
def embed_chunks(texts: List[str]) -> List[List[float]]:
    """Batch embed a list of chunk texts using Databricks BGE embedding model."""
    response = client.embeddings.create(
        model="databricks-bge-large-en",
        input=texts
    )
    return [item.embedding for item in response.data]

# ── 5. Process corpus ─────────────────────────────────────────────────────────
all_chunk_records = []
pdf_files = [f for f in dbutils.fs.ls(VOLUME_PATH) if f.name.endswith(".pdf")]

for file_info in pdf_files:
    local_path = f"/dbfs{file_info.path.replace('dbfs:', '')}"
    try:
        text, doc_meta = parse_pdf_to_text(local_path)
        chunks = chunk_document(text, doc_meta)
        
        # Embed in batches of 50 (API rate limit consideration)
        for i in range(0, len(chunks), 50):
            batch = chunks[i:i+50]
            batch_texts = [c["chunk_text"] for c in batch]
            embeddings = embed_chunks(batch_texts)
            for chunk, emb in zip(batch, embeddings):
                chunk["embedding"] = emb
            all_chunk_records.extend(batch)
            
        print(f"Processed {file_info.name}: {len(chunks)} chunks")
    except Exception as e:
        print(f"FAILED: {file_info.name} — {e}")

# ── 6. Write to Delta table ───────────────────────────────────────────────────
chunks_df = spark.createDataFrame(all_chunk_records)
(chunks_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CHUNKS_TABLE))

print(f"Wrote {chunks_df.count()} chunks to {CHUNKS_TABLE}")
```

**Failure mode to observe:** Run this with a PDF that has no text layer (a scanned image PDF).
`partition_pdf` returns elements with no text content. Observe that `text` is empty, and the
pipeline produces zero chunks for that file. **Fix:** Add a check after parsing — if the parsed
text is below a minimum length threshold, route the document to an OCR pipeline.

## Common Pitfalls & Misconceptions

- **Pitfall:** Embedding chunks that exceed the embedding model's max token limit → **Why it happens:** Chunk size is set in characters, not tokens; a 1,500-character chunk may be 400+ tokens for simple text but 600+ tokens for technical jargon → **Fix:** Use a token-counting function (`tiktoken` or model-specific tokenizer) to set `chunk_size` and verify; set chunk size to at most 90% of the embedding model's max

- **Pitfall:** Using fixed-size chunking for structured documents (Markdown docs, HTML pages, API reference) → **Why it happens:** Fixed-size is the simplest to implement → **Fix:** Match chunking strategy to document structure; use `MarkdownHeaderTextSplitter` or `HTMLHeaderTextSplitter` for structured docs — the structural metadata they produce is extremely valuable for filtered retrieval

- **Pitfall:** Skipping overlap entirely to reduce the number of chunks → **Why it happens:** More chunks = higher storage cost and more retrieval noise → **Fix:** 10–20% overlap is almost always worth the cost; boundary-spanning sentences are a major source of retrieval misses without overlap

- **Pitfall:** Building the pipeline without attaching metadata to chunks → **Why it happens:** Getting text into the index feels like the goal; metadata feels optional → **Fix:** Design your metadata schema before writing the first line of pipeline code; retrofitting requires re-indexing the entire corpus

- **Pitfall:** Selecting an embedding model based only on MTEB leaderboard ranking, without testing on your actual data → **Why it happens:** Benchmarks are authoritative-sounding → **Fix:** MTEB benchmarks general retrieval tasks; your domain may have unique vocabulary or query patterns that change the ranking. Always run a retrieval accuracy test on a sample of your own queries

- **Pitfall:** Not deduplicating the corpus before indexing → **Why it happens:** Deduplication feels like a nice-to-have → **Fix:** Duplicate documents create duplicate (or near-duplicate) chunks in the index; the retriever returns several redundant results that consume context window budget without adding information, degrading generation quality

## Key Definitions

| Term | Definition |
|---|---|
| Parsing | The process of extracting usable text from a raw file format (PDF, HTML, image) |
| `unstructured` | A Python library that parses diverse document formats (PDF, DOCX, HTML, images) and preserves structural information (titles, narrative text, tables) |
| Metadata extraction | The process of identifying and attaching contextual attributes (source file, section heading, author, date) to document chunks to enable filtered retrieval |
| MinHash LSH | Locality-Sensitive Hashing using MinHash signatures — a scalable algorithm for identifying near-duplicate documents based on their word-level Jaccard similarity |
| Fixed-size chunking | Splitting text into chunks of a fixed character or token count, ignoring semantic boundaries |
| Paragraph-based chunking | Splitting text at natural paragraph boundaries (double newlines), preserving semantic units |
| Format-specific chunking | Using document structure markers (Markdown headers, HTML tags) as chunk boundaries, automatically attaching structural metadata |
| Semantic chunking | Identifying topic-shift boundaries by computing embedding similarity between consecutive sentence pairs and splitting at large similarity drops |
| Chunk overlap | The number of characters or tokens shared between adjacent chunks to prevent information loss at chunk boundaries |
| MTEB | Massive Text Embedding Benchmark — a public leaderboard ranking embedding models by retrieval and similarity task performance across many datasets |

## Summary / Quick Recall

- The five pipeline stages: ingest → parse → enrich (metadata + dedup + filter) → chunk → embed → index
- Parsing tools: `unstructured`/`pypdf` for PDFs; BeautifulSoup for HTML; Tesseract or cloud OCR for images
- Four metadata categories: document-level, content-based, structural, contextual — store all with the chunk
- MinHash LSH in Spark ML: featurize → hash → similarity join → filter duplicates
- Four chunking strategies: fixed-size (baseline), paragraph (default for prose), format-specific (structured docs), semantic (long varied docs)
- Chunk overlap: 10–20% of chunk size; prevents boundary-spanning sentences from being lost
- Always measure chunk size in **tokens**, not characters; stay below embedding model's max token limit
- Token budget: `context_window - system_prompt - query - max_response - buffer = available_for_chunks`
- MTEB leaderboard is a starting point, not the final word; test on your own data

## Self-Check Questions

1. You have a corpus of Markdown technical documentation files. Which chunking strategy should you use and why?

<details>
<summary>Answer</summary>
Format-specific chunking using LangChain's `MarkdownHeaderTextSplitter`. Markdown documentation has an explicit heading hierarchy (H1/H2/H3) that represents meaningful semantic boundaries. This strategy produces chunks that align with document sections and automatically attaches structural metadata (which heading each chunk belongs to). This metadata can later be used for filtered retrieval — e.g., "only search the 'Troubleshooting' section." Fixed-size or paragraph chunking would ignore this structure and produce semantically incoherent chunks.
</details>

2. Your embedding model (`bge-large-en-v1.5`) has a 512-token maximum. You set `chunk_size=2000` characters. What is the likely outcome and how do you fix it?

<details>
<summary>Answer</summary>
At ~4 characters per token, a 2,000-character chunk is approximately 500 tokens — close to the limit. For technical content with shorter words (code, acronyms), the token count could easily exceed 512. Chunks exceeding the limit are silently truncated by the embedding model — the tail of the chunk (which may contain the key answer to a query) is dropped without any error. Fix: set `chunk_size` using a token-counting function (`tiktoken`) rather than character count. Set `chunk_size=450` tokens (90% of the 512-token limit as a safety margin).
</details>

3. After building your RAG pipeline, you notice that several queries return highly redundant chunks — the same information appears in 3–4 of the top-5 retrieved results. What is the most likely data pipeline cause?

<details>
<summary>Answer</summary>
Near-duplicate documents in the corpus that were not deduplicated before chunking. If a document exists in three slightly different versions (e.g., from different source systems or with minor edits), all three produce nearly identical chunks. The retriever correctly identifies all of them as relevant to the query, but they consume context window budget without adding new information. Fix: add a deduplication step using MinHash LSH before chunking, with a Jaccard similarity threshold around 0.8–0.9.
</details>

4. What is the purpose of chunk overlap and what is the risk of setting it too high?

<details>
<summary>Answer</summary>
Chunk overlap shares N characters/tokens between adjacent chunks to prevent information loss at chunk boundaries. A sentence spanning a boundary appears in at least one chunk that contains it fully, making it retrievable. If overlap is too high (e.g., 50% of chunk size), adjacent chunks become nearly identical — retrieval returns both, wasting context window space with redundant content. The sweet spot is 10–20% of chunk size.
</details>

5. You want to filter out PII-containing documents from your RAG corpus before indexing. Name two approaches and describe the trade-off between them.

<details>
<summary>Answer</summary>
Two approaches: (1) **Rule-based regex patterns** — match known PII formats (SSNs, credit card numbers, email addresses, phone numbers) using regex. Fast, cheap, zero false positives for well-defined patterns, but misses context-dependent PII (names, addresses without postal codes, indirect identifiers). (2) **Named Entity Recognition (NER) model** — a fine-tuned NER model (e.g., spaCy or Hugging Face NER) identifies person names, locations, organizations, and other entity types. Captures more PII types including names, but has false positives (e.g., identifying a product name as a person name) and is slower/more expensive. For high-compliance environments (HIPAA, GDPR), use both in combination: regex for known-format PII, NER for entity-type PII.
</details>

## Further Reading

- [Databricks — Build an unstructured data pipeline for RAG](https://docs.databricks.com/en/agents/tutorials/ai-cookbook/quality-data-pipeline-rag.html) — official step-by-step guide (updated 2026-06-30); the primary source for this chapter
- [LangChain — Text splitters documentation](https://python.langchain.com/docs/how_to/#text-splitters) — reference for all four chunking strategies with code examples
- [Unstructured.io documentation](https://docs.unstructured.io/) — official docs for the `unstructured` PDF/document parsing library
- [MTEB Leaderboard — Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — current ranking of embedding models by retrieval benchmark performance
- [ChunkViz](https://chunkviz.up.railway.app/) — interactive tool for visualizing how `RecursiveCharacterTextSplitter` parameters affect chunk boundaries
