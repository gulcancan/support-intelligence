# Architecture Documentation

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Gateway (:8000)                        │
│                                                                     │
│  Processing      Analytics          Monitoring        Feedback      │
│  /tickets/*      /analytics/*       /anomalies        /feedback     │
│                                     /monitoring/*                   │
└──────┬──────────────┬──────────────────┬──────────────┬─────────────┘
       │              │                  │              │
┌──────▼──────┐ ┌─────▼──────┐  ┌───────▼───────┐ ┌───▼──────────┐
│ Classifier   │ │ Business   │  │  Anomaly      │ │  Feedback    │
│ Registry     │ │ Analytics  │  │  Detector     │ │  Collector   │
│              │ │            │  │               │ │              │
│ CatBoost ◄───┤ │ Resol.Time │  │ Volume Spikes │ │ Corrections  │
│ DistilBERT   │ │ Satisfact. │  │ Sentiment     │ │ Correction   │
│ Ensemble     │ │ Agent Perf │  │ New Issues    │ │ Rate Monitor │
└──────┬───────┘ │ Quality    │  └───────────────┘ └──────────────┘
       │         │ Scorer     │
       │         └────────────┘
┌──────▼──────────────────────────────┐  ┌─────────────────────────┐
│       Hybrid Retriever              │  │  Drift Monitor          │
│                                     │  │  (Streaming)            │
│  Vector Search ──► RRF ──► Quality  │  │                         │
│  BM25 Search  ──► Fusion   Re-rank  │  │  PSI Feature Drift     │
│  Graph-RAG   ──►                    │  │  Prediction Drift       │
└──────┬────────────┬─────────────────┘  │  Confidence Drift       │
       │            │                    │  Label Drift             │
┌──────▼──────┐ ┌───▼──────┐            │  Shadow Deploy           │
│ PostgreSQL   │ │ Qdrant   │            │  Auto-Retrain Pipeline   │
│ (:5432)      │ │ (:6333)  │            └─────────────────────────┘
│              │ │          │
│ tickets      │ │ embeddings│  ┌─────────────────────────┐
│ features     │ │ + payload │  │ MLflow (:5000)           │
│ graph tables │ │ filtering │  │ Experiments, Metrics     │
│ feedback     │ └──────────┘  │ Model Artifacts          │
│ pred_log     │               └─────────────────────────┘
└──────────────┘
```

## Data Flow

### Ingestion (Batch — 300K tickets)

```
tickets.json (300K) → Validate (DQ checks) → Temporal Split (70/15/15)
    → Feature Engineering (text, customer, product, temporal features)
    → PostgreSQL: tickets + ticket_features + graph tables
    → Qdrant: resolution embeddings (all-MiniLM-L6-v2, 384-dim)
    → BM25: tokenized index for keyword matching
```

### Inference (Real-time)

```
New Ticket → Classify (CatBoost 1-5ms, DistilBERT fallback)
    → Retrieve (Vector + BM25 + Graph → RRF fusion → Quality re-rank)
    → Log prediction (for drift monitoring)
    → Shadow predict (if shadow model deployed)
    → Return: category + confidence + top-5 scored resolutions
```

### Feedback Loop

```
Agent Correction → feedback table → Correction rate ↑?
    → YES: retrain on historical + corrected labels
    → Validate new model (must beat current by ≥0.5% F1)
    → Shadow deploy 24h → Promote if outperforms
```

## Technology Choices

| Component | Choice | Why |
|-----------|--------|-----|
| API | FastAPI | Async, auto-OpenAPI, Pydantic validation |
| DB | PostgreSQL 16 | Reliable, JSONB, materialized views, graph-as-SQL |
| Vector Store | Qdrant | Metadata filtering during search, Docker-native |
| Classical ML | CatBoost | Native categoricals, ordered boosting, 1-5ms |
| Deep Learning | DistilBERT (PyTorch) | 2× faster than BERT, <1% F1 drop, HuggingFace ecosystem |
| Embeddings | all-MiniLM-L6-v2 | 384-dim, fast, Apache 2.0 |
| Search | BM25Okapi | Exact error code matching |
| Fusion | RRF (k=60) | Parameter-free, robust across heterogeneous scores |
| Tracking | MLflow | Self-hosted, no external account needed |
| Drift | PSI + EWMA | Industry-standard thresholds, interpretable |
| Container | Docker Compose | Single-command deployment with health checks |

### Note on PyTorch vs TensorFlow

The task spec mentions TensorFlow/Keras for the deep learning approach. We chose PyTorch + HuggingFace Transformers because DistilBERT's ecosystem is more mature in PyTorch, HuggingFace provides superior model management, and PyTorch's dynamic graph enables easier debugging. The architectural principles (transfer learning, layer freezing, structured feature concatenation) are framework-agnostic and directly transferable.

## Graceful Degradation

| Failure | Behavior |
|---------|----------|
| CatBoost down | DistilBERT becomes primary |
| Both classifiers | Returns "Unknown", retrieval still works |
| Qdrant down | BM25-only retrieval |
| BM25 not built | Vector-only retrieval |
| Graph query fails | Skip graph context |
| PostgreSQL down | API returns 503 |

## Event Schema

All system interactions are tracked via `SystemEvent` (see `analytics/business.py`):

```
event_id, event_type, timestamp, ticket_id, agent_id,
model_name, model_version, predicted_category, confidence,
latency_ms, n_results_returned, quality_score,
corrected_category, resolution_accepted, status, error_message
```

This enables auditing, analytics on feature usage, latency monitoring, and connecting predictions to outcomes.

## Production Deployment (Documented)

- **Kubernetes**: Helm chart, separate deployments for API / model serving / jobs
- **CI/CD**: Push → test → train → validate → blue-green deploy
- **Monitoring**: Prometheus + Grafana for latency, throughput, drift
- **A/B Testing**: Traffic split between model versions via feature flags
- **Feature Store**: Feast for sub-ms real-time feature serving
- **Stream Processing**: Kafka + Flink for real-time anomaly detection at scale
