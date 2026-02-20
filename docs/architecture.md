# Architecture Documentation

## System Overview

The Support Intelligence System is a **modular monolith** — logically separated components communicating via function calls within a single deployable unit. Each module has a clean interface and could be split into a separate service later without architectural changes.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Gateway (:8000)                      │
│                                                                  │
│  POST /api/v1/tickets/process    Full pipeline                  │
│  POST /api/v1/tickets/classify   Classification only            │
│  POST /api/v1/tickets/search     Retrieval only                 │
│  POST /api/v1/feedback           Agent corrections              │
│  GET  /api/v1/anomalies          Current anomalies              │
│  GET  /api/v1/health             System health                  │
└──────┬──────────────────┬───────────────────┬───────────────────┘
       │                  │                   │
┌──────▼──────┐   ┌──────▼───────┐   ┌───────▼────────┐
│  Classifier  │   │   Hybrid     │   │   Anomaly      │
│  Registry    │   │   Retriever  │   │   Detector     │
│              │   │              │   │                │
│ ┌──────────┐ │   │ ┌──────────┐ │   │ • Volume       │
│ │ CatBoost │ │──▶│ │ Vector   │ │   │   z-scores     │
│ │ (primary)│ │   │ │ Search   │ │   │ • Sentiment    │
│ ├──────────┤ │   │ ├──────────┤ │   │   EWMA drift   │
│ │DistilBERT│ │   │ │ BM25     │ │   │ • New issue    │
│ │(fallback)│ │   │ │ Keywords │ │   │   detection    │
│ └──────────┘ │   │ ├──────────┤ │   └────────────────┘
│              │   │ │ Graph-RAG│ │
│  Ensemble    │   │ ├──────────┤ │
│  routing     │   │ │ RRF + Re-│ │
│              │   │ │ rank     │ │
└──────────────┘   └──────┬──────┘
                          │
       ┌──────────────────┼──────────────────┐
       ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌───────────────┐
│ PostgreSQL   │  │   Qdrant     │  │   MLflow       │
│ (:5432)      │  │   (:6333)    │  │   (:5000)      │
│              │  │              │  │                │
│ tickets      │  │ resolution   │  │ experiments    │
│ features     │  │ embeddings   │  │ metrics        │
│ graph tables │  │ + metadata   │  │ artifacts      │
│ feedback     │  │ payloads     │  │ model versions │
└──────────────┘  └──────────────┘  └────────────────┘
```

## Component Interaction Flow

### Inference (Real-time)

```
New Ticket → API Gateway
    │
    ├── Step 1: Classification
    │     CatBoost predicts category (1-5ms)
    │     ├─ Confidence ≥ 0.6 → Use CatBoost result
    │     └─ Confidence < 0.6 → Also run DistilBERT → Ensemble
    │
    ├── Step 2: Retrieval (filtered by predicted category)
    │     ├─ Vector Search (Qdrant, cosine similarity)
    │     ├─ BM25 Keyword Match (exact error codes)
    │     ├─ Graph Traversal (product→issues→resolutions)
    │     └─ RRF Fusion → Quality Re-ranking → Top-5
    │
    └── Response: category + confidence + resolutions + graph context
```

### Ingestion (Batch)

```
JSON → Validate → Temporal Split (70/15/15)
    → Feature Engineering → PostgreSQL
    → Embed resolutions → Qdrant
    → Build BM25 index
    → Populate graph tables
```

### Feedback Loop

```
Agent correction → agent_feedback table
    → Correction rate monitoring
    → Confusion pair analysis → targeted retraining data
```

## Technology Justifications

| Component | Choice | Why Over Alternatives |
|-----------|--------|----------------------|
| API | FastAPI | Async, auto OpenAPI docs, Pydantic validation. Flask lacks async; Django too heavy. |
| Database | PostgreSQL 16 | JSONB for semi-structured data, materialized views. SQLite can't do concurrent writes. |
| Vector Store | Qdrant | Metadata filtering during search (critical for category-filtered retrieval). FAISS has no filtering; Pinecone requires vendor account. |
| Classical ML | CatBoost | Native categorical handling, ordered boosting for temporal data. XGBoost needs manual encoding. |
| Deep Learning | DistilBERT | 2× faster than BERT-base, <1% F1 drop, CPU-feasible. |
| Embeddings | all-MiniLM-L6-v2 | 384-dim, fast, Apache 2.0. OpenAI adds API dependency. |
| Search | BM25Okapi | Exact error code matching that embeddings miss. |
| Tracking | MLflow | Self-hosted, no account needed for reviewer. W&B requires signup. |
| Fusion | RRF | Parameter-free, robust across heterogeneous score distributions. |

## Graceful Degradation

| Component Failure | System Behavior |
|-------------------|-----------------|
| CatBoost unavailable | DistilBERT becomes primary |
| Both classifiers down | Returns "Unknown" category, retrieval still works |
| Qdrant down | BM25-only retrieval |
| BM25 not indexed | Vector-only retrieval |
| Graph query fails | Skip graph context, proceed with vector+BM25 |
| PostgreSQL down | API returns 503 |

Every retrieval component is wrapped in `try/except`. The system proceeds with whatever results are available.

## Production Considerations (Documented, Not Implemented)

- **Kubernetes**: Helm chart for horizontal API scaling
- **CI/CD**: GitHub Actions → train → compare metrics → blue-green deploy
- **Monitoring**: Prometheus + Grafana for latency, throughput, model drift
- **A/B Testing**: Traffic splitting between model versions
- **Feature Store**: Feast for sub-ms real-time feature serving
- **Stream Processing**: Kafka + Flink for real-time anomaly detection at scale
- **Model Serving**: Triton for GPU-accelerated transformer inference
