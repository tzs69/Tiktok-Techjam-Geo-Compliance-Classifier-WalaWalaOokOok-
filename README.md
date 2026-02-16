# Geo Compliance Classifier (LangGraph + Qdrant Hybrid)

This repository has a split architecture:

- `offline/`: Qdrant-backed ingestion, parent-child chunking, and hybrid retrieval
- `online/`: LangGraph workflow for batch geo-compliance classification

## Offline Architecture

The offline layer is Qdrant-only and uses `data_sources/` as the corpus root.

### Ingestion and indexing

1. Parse `.txt` and extensionless legal files from `data_sources/`.
2. Extract title/body (`/()/()/` split when available, fallback to first-line title).
3. Infer domain from folder path or content keywords.
4. Parent-child chunking:
- Parent: markdown-aware splitter first (`MarkdownHeaderTextSplitter`)
- Fallback parent splitter: recursive character splitter
- Child: recursive character splitter over parent chunks
5. Index child chunks into a single Qdrant collection with hybrid vectors:
- Dense vector: `dense` (FastEmbed)
- Sparse vector: `sparse` (FastEmbed BM25)
6. Store metadata payload including `parent_id` and `parent_text` snippet.

### Retrieval

Retriever runs hybrid query (dense + sparse fusion) with optional domain filtering and returns:
- `chunk_id`, `domain`, `law_name`, `source_path`, `text`, `score`
- `parent_id`, `parent_snippet`

## Online Workflow Nodes

1. `query_enhancer`
2. `retrieve`
3. `classifier`
4. `hitl_router`
5. `finalize`

## Input and Output

Input CSV must contain:
- `feature_name`
- `feature_description`

Output CSV includes:
- `needs_geo_compliance`
- `reasoning`
- `citations` (JSON list)
- `confidence`
- `needs_hitl`
- `hitl_reason`
- `audit_trail` (JSON list)

## Setup

Start Qdrant locally:

```bash
docker compose up -d qdrant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Rebuild Qdrant index from `data_sources/` and run batch:

```bash
python3 main.py \
  --input input_data/sample_features.csv \
  --output outputs/classification_output.csv \
  --rebuild-index
```

Use existing Qdrant collection:

```bash
python3 main.py \
  --input input_data/sample_features.csv \
  --output outputs/classification_output.csv
```

Optional overrides:

- `--data-root data_sources`
- `--collection-name geo_compliance_hybrid_v1`
- `--qdrant-url http://localhost:6333`
- `--qdrant-api-key <key>`

## Environment Variables

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION`
- `QDRANT_TIMEOUT`
- `QDRANT_PREFER_GRPC`
- `QDRANT_DISTANCE`
- `QDRANT_RECREATE_COLLECTION`
- `QDRANT_HYBRID_RECALL_MULTIPLIER`
- `QDRANT_HYBRID_RECALL_MIN`
- `QDRANT_LATE_INTERACTION_MODEL`
- `QDRANT_DENSE_VECTOR_NAME`
- `QDRANT_SPARSE_VECTOR_NAME`
- `QDRANT_LATE_VECTOR_NAME`
- `OPENAI_API_KEY` (for online LLM nodes)
