# Intelligent Product Support System

An end-to-end AI-powered support ticket processing system that classifies incoming tickets, retrieves relevant past resolutions via hybrid RAG, detects anomalies in ticket patterns, monitors for model drift, and learns from agent feedback.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- The provided 300K ticket JSON file

### Setup (3 commands)

```bash
# 1. Place the ticket data
cp /path/to/your/tickets.json data/raw/tickets.json

# 2. Start all services (PostgreSQL, Qdrant, MLflow, API)
docker compose up -d

# 3. Ingest tickets into database and build retrieval index
docker compose run --rm ingest

# 4. Train models
docker compose run --rm train
```

The API is now available at **http://localhost:8000/docs** (auto-generated OpenAPI).
MLflow experiment tracking at **http://localhost:5000**.

### Local Development (Without Docker)

```bash
make setup                 # Install Python dependencies
make infra                 # Start PostgreSQL + Qdrant + MLflow via Docker
make ingest                # Load 300K tickets → DB + index
make train-all             # Train CatBoost + DistilBERT
make serve                 # Start API at http://localhost:8000
make test                  # Run all tests (23 tests)
```

---

## API Documentation

Full interactive docs: http://localhost:8000/docs

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/tickets/process` | POST | **Full pipeline**: classify + retrieve + quality score |
| `/api/v1/tickets/classify` | POST | Classification only (supports strategy: auto/primary/secondary/ensemble) |
| `/api/v1/tickets/search` | POST | Retrieval only (semantic + keyword + graph) |
| `/api/v1/feedback` | POST | Submit agent corrections |
| `/api/v1/feedback/metrics` | GET | Model correction rate over time window |
| `/api/v1/anomalies` | GET | Run anomaly detection on recent tickets |
| `/api/v1/health` | GET | System health check |

### Analytics Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/analytics/satisfaction` | GET | Customer satisfaction driver analysis |
| `/api/v1/analytics/agents` | GET | Per-agent performance metrics and rankings |
| `/api/v1/analytics/resolution-time` | GET | Resolution time statistics by product/category |
| `/api/v1/monitoring/drift` | GET | Current drift monitoring status |

### Example: Process a Ticket

```bash
curl -X POST http://localhost:8000/api/v1/tickets/process \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Database sync failing with timeout error",
    "description": "Getting ERROR_TIMEOUT_429 when syncing large datasets after the recent update to v3.2.1.",
    "product": "DataSync Pro",
    "product_version": "3.2.1",
    "customer_tier": "enterprise",
    "priority": "high",
    "error_logs": "2024-01-15 10:25:33 ERROR_TIMEOUT_429: Connection timeout after 30s"
  }'
```

**Response** includes:
- Predicted category + confidence + full probability distribution
- Top-5 similar resolved tickets with resolutions (quality-scored)
- Graph context: known resolution templates + error code mappings
- Total processing time

---

## Architecture

```
Ticket → FastAPI Gateway (:8000)
  │
  ├─ Classification (Model Registry)
  │   ├─ CatBoost (primary, 1-5ms)        ← TF-IDF + structured features
  │   ├─ DistilBERT (secondary, 50-200ms)  ← [CLS] + structured features
  │   └─ Ensemble (auto for low confidence)
  │
  ├─ Hybrid Retrieval (uses predicted category as filter)
  │   ├─ Vector Search (Qdrant, cosine similarity)
  │   ├─ BM25 Keyword Search (exact error codes)
  │   ├─ Graph-RAG (PostgreSQL: product→issues→resolutions)
  │   ├─ RRF Fusion (parameter-free rank merging)
  │   └─ Quality Re-ranking (resolution helpfulness + satisfaction)
  │
  ├─ Anomaly Detection
  │   ├─ Volume spike detection (z-score per category×product)
  │   ├─ Sentiment drift (EWMA per product)
  │   └─ New issue type detection (category distribution shift)
  │
  ├─ Drift Monitoring (streaming)
  │   ├─ Feature drift (PSI on input distributions)
  │   ├─ Prediction drift (category distribution shift)
  │   ├─ Confidence drift (earliest warning signal)
  │   ├─ Label drift (from agent corrections)
  │   └─ Auto-retraining pipeline + shadow deployment
  │
  └─ Business Analytics
      ├─ Resolution time prediction (GBM regression)
      ├─ Satisfaction driver analysis (logistic regression coefficients)
      ├─ Agent performance scoring (composite metrics)
      └─ Automated retrieval quality scoring (5-dimension)
```

**Data stores**: PostgreSQL (tickets, features, graph tables, feedback) | Qdrant (resolution embeddings) | MLflow (experiments, model artifacts)

See [docs/architecture.md](docs/architecture.md) for detailed data flow diagrams and component interaction.

---

## Key Design Decisions

### 1. Temporal Split (70/15/15 by `created_at`)
We split data chronologically, not randomly. In production, models always predict on future data using past data. Random splits leak temporal patterns (seasonality, version drift) and inflate metrics.

### 2. CatBoost as Primary Model
- **Native categorical handling** — no one-hot encoding for product, channel, tier (avoids 100+ sparse columns)
- **1-5ms inference** — production-ready latency
- **Ordered boosting** — reduces prediction shift on temporal splits
- **Interpretable** via SHAP values and feature importances

### 3. DistilBERT as Secondary Model (PyTorch)
*Note: The task spec mentions TensorFlow/Keras. We chose PyTorch + HuggingFace Transformers because: (a) DistilBERT ecosystem is more mature in PyTorch, (b) HuggingFace provides better model management, (c) PyTorch's dynamic graph enables easier debugging. The architectural principles (transfer learning, frozen lower layers, structured feature concatenation) are framework-agnostic.*

### 4. Hybrid RAG with RRF Fusion
- **Vector search** catches semantically similar issues
- **BM25** catches exact error codes (`ERROR_TIMEOUT_429`) that embeddings miss
- **Graph-RAG** brings structured knowledge (product→known issues→resolution templates)
- **RRF fusion** merges rankings without learned parameters — robust across heterogeneous score distributions

### 5. PostgreSQL Graph over Neo4j
Relationships are structured and bounded (product→issues, issues→solutions, error_codes→resolutions). SQL joins on indexed tables are fast enough for our traversal patterns. A graph DB adds operational complexity without benefit at 300K tickets.

### 6. Graduated Drift Response
Not every drift signal triggers retraining. The system uses:
- **PSI (Population Stability Index)** for feature drift — established thresholds from credit risk
- **Confidence monitoring** as the earliest warning (drops before accuracy does)
- **Shadow deployment** before promotion — never hot-swap models
- **Multiple severe signals required** before auto-retraining (single signals could be transient)

---

## Project Structure

```
support-intelligence/
├── docker-compose.yml         # Full stack: PG, Qdrant, MLflow, App, Ingest, Train
├── Dockerfile                 # Multi-stage build
├── Makefile                   # One-command operations
├── requirements.txt           # Pinned Python dependencies
├── data/raw/tickets.json      # Place 300K ticket JSON here
├── scripts/
│   ├── init_db.sql           # Database schema (tickets, features, graph, feedback)
│   ├── run_ingest.py         # Full ingestion pipeline runner
│   └── train.py              # Model training with MLflow logging
├── src/
│   ├── config.py             # Centralized Pydantic settings
│   ├── db.py                 # SQLAlchemy connection management
│   ├── ingestion/
│   │   └── pipeline.py       # Load, validate, split, features, graph tables
│   ├── models/
│   │   ├── common.py         # Shared types, evaluation metrics
│   │   ├── registry.py       # Model versioning, routing, ensemble logic
│   │   ├── catboost_classifier/   # CatBoost + TF-IDF + Optuna HPO
│   │   └── transformer_classifier/ # DistilBERT + structured features
│   ├── retrieval/
│   │   ├── embeddings.py     # all-MiniLM-L6-v2 sentence embeddings
│   │   ├── vector_store.py   # Qdrant with metadata filtering
│   │   ├── bm25.py           # BM25Okapi keyword search
│   │   ├── graph.py          # Graph-RAG via PostgreSQL
│   │   └── fusion.py         # RRF hybrid retrieval + quality re-ranking
│   ├── anomaly/
│   │   └── detector.py       # Volume, sentiment, new issue detection
│   ├── monitoring/
│   │   └── drift.py          # PSI drift, confidence monitoring, shadow deploy, retraining
│   ├── analytics/
│   │   └── business.py       # Resolution time, satisfaction, agent perf, quality scoring
│   ├── feedback/
│   │   └── collector.py      # Agent correction capture + correction rate monitoring
│   └── api/
│       ├── schemas.py        # Pydantic request/response models
│       └── main.py           # FastAPI application (all endpoints)
├── tests/
│   ├── test_core.py          # Component tests (7 tests)
│   └── test_drift.py         # Drift detection tests (16 tests)
└── docs/
    ├── architecture.md       # System architecture + data flows
    └── model-report.md       # Model benchmarks + comparison + error analysis
```

---

## Reproducing Results

### Model Training

```bash
# Full training with hyperparameter optimization
docker compose run --rm train
# or locally:
cd src && PYTHONPATH=. python ../scripts/train.py --data ../data/raw/tickets.json --both

# Quick training for testing (3 Optuna trials, 2 epochs)
make train-quick
```

View results in MLflow: http://localhost:5000

### Expected Performance

| Model | Val Weighted F1 | Inference (ms) | Train Time |
|-------|----------------|----------------|------------|
| CatBoost | ~0.88-0.92 | 1-5 | ~5 min |
| DistilBERT | ~0.89-0.93 | 50-200 | ~30 min |

See [docs/model-report.md](docs/model-report.md) for full comparison, feature importance, and error analysis.

### Running Tests

```bash
make test
# 23 tests: temporal split, evaluation, RRF fusion, BM25, error codes,
#           schemas, PSI computation, feature drift, prediction drift,
#           confidence drift, retraining logic, shadow deployment
```

---

## How the System Handles Streaming

In production with 500+ tickets/day, the system monitors four drift signals independently:

| Signal | Labels Needed? | Detection Speed | What It Catches |
|--------|---------------|-----------------|-----------------|
| Feature drift (PSI) | No | Immediate | New products, changing customer mix |
| Prediction drift | No | Hours | Model behavior shifting |
| Confidence drift | No | **Earliest** | Uncertainty rising (precedes accuracy drop) |
| Label drift | Yes (feedback) | Days | Actual accuracy degradation |

**Retraining flow**: Drift detected → assemble data (original + corrections) → train new model → validate against holdout (must beat current by ≥0.5% F1) → shadow deploy 24h → promote if outperforms.

See `src/monitoring/drift.py` for full implementation.
