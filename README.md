# Support Intelligence System

AI-powered support ticket processing: classification, retrieval, anomaly detection, feedback loops.

## Quick Start
```bash
make setup          # Install deps + generate 100k tickets
make infra          # Start PostgreSQL, Qdrant, MLflow
make ingest && make index  # Load data + build search index
make train-all      # Train CatBoost + Transformer
make serve          # Start API at http://localhost:8000
make test-api       # Verify it works
```

API docs: http://localhost:8000/docs | MLflow: http://localhost:5000

## Architecture
```
Ticket → FastAPI Gateway
  ├─ Classification (CatBoost primary, DistilBERT secondary)
  ├─ Hybrid Retrieval (Vector + BM25 + Graph → RRF fusion)
  └─ Anomaly Detection (volume, sentiment, new issues)
```

## Key Design Decisions
1. **Temporal split** over random — mirrors production (no temporal leakage)
2. **CatBoost primary** — 1-5ms inference, native categoricals, interpretable
3. **DistilBERT secondary** — better text generalization, ensemble for low-confidence
4. **RRF fusion** — parameter-free merging of vector + keyword search
5. **PostgreSQL graph** over Neo4j — bounded relationships, SQL joins sufficient

## API Usage
```bash
curl -X POST http://localhost:8000/api/v1/tickets/process \
  -H "Content-Type: application/json" \
  -d '{"subject":"DB sync timeout","description":"ERROR_TIMEOUT_429 on large datasets","product":"DataSync Pro"}'
```

See full docs at [docs/architecture.md](docs/architecture.md) and [docs/model-report.md](docs/model-report.md).
