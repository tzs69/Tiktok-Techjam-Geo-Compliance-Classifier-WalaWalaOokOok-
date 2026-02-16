# Repository Map

## Root
- `main.py`: CLI entrypoint for Qdrant indexing + batch classification
- `requirements.txt`: runtime dependencies
- `docker-compose.yml`: local Qdrant service
- `README.md`: setup and usage
- `data_sources/`: legal/compliance source corpus
- `input_data/`: input CSV datasets
- `outputs/`: generated output CSV files
- `offline/`: Qdrant ingestion and retrieval
- `online/`: LangGraph inference pipeline

## offline/
- `offline/qdrant_config.py`
  - Qdrant settings model from env/CLI overrides
- `offline/domain_inference.py`
  - Domain inference rules for folder-based and standalone files
- `offline/chunking.py`
  - Source parsing + markdown-aware parent-child chunking
- `offline/index_builder.py`
  - Builds and upserts hybrid vectors into Qdrant collection
  - Ensures collection/indexes and reports ingest stats
- `offline/retriever.py`
  - Hybrid search (dense+sparse fusion)
  - Supports domain filtering and fallback search
  - Returns child hit + parent snippet metadata

## online/
- `online/schemas.py`
  - `FeatureRecord`
  - `WorkflowOutput`
- `online/pipeline.py`
  - Reads input CSV rows
  - Invokes compiled LangGraph app per row
  - Writes output CSV

### online/graph/
- `online/graph/state.py`
  - Typed workflow state contract
- `online/graph/workflow.py`
  - LangGraph topology and retriever wiring

### online/graph/nodes/
- `query_enhance_route.py`
  - Query clarification + domain/law/region hints
  - Optional LLM fallback
- `retrieve.py`
  - Retrieves evidence from offline Qdrant retriever (hybrid retrieval + late-interaction reranking)
- `classify.py`
  - LLM-backed classification with deterministic confidence fallback
- `hitl_router.py`
  - Deterministic-first HITL escalation router
- `finalize.py`
  - Final output formatting
