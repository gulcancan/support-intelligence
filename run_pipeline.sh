#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════
# Full Pipeline: Ingest → Train → Serve
#
# Run after deploy.sh and uploading your tickets.json
# Usage: cd ~/support-intelligence && bash run_pipeline.sh
# ═══════════════════════════════════════════════════════════

cd "$(dirname "$0")"

# Verify data exists
if [ ! -f "data/raw/tickets.json" ]; then
    echo "ERROR: data/raw/tickets.json not found!"
    echo "Upload your ticket data first:"
    echo "  scp tickets.json user@server:~/support-intelligence/data/raw/tickets.json"
    exit 1
fi

TICKET_COUNT=$(python3 -c "import json; print(len(json.load(open('data/raw/tickets.json'))))" 2>/dev/null || echo "?")
echo "Found data/raw/tickets.json ($TICKET_COUNT tickets)"
echo ""

# Ensure services are running
echo "═══ Step 0: Ensuring infrastructure is up ═══"
docker compose up -d postgres qdrant mlflow
echo "Waiting for health checks..."
sleep 10

# Step 1: Ingest
echo ""
echo "═══ Step 1: Ingesting tickets into PostgreSQL + Qdrant ═══"
echo "  This loads all tickets, computes features, builds graph tables,"
echo "  and creates the vector + BM25 retrieval index."
echo "  Expected time: ~5-15 min for 300K tickets"
echo ""
docker compose run --rm ingest
echo "✓ Ingestion complete"

# Step 2: Train
echo ""
echo "═══ Step 2: Training models (CatBoost + DistilBERT) ═══"
echo "  CatBoost: ~5 min (with 20 Optuna trials)"
echo "  DistilBERT: ~30-60 min on CPU (5 epochs)"
echo "  Results will be logged to MLflow at http://$(hostname -I | awk '{print $1}'):5000"
echo ""

# Use quick training for faster validation, or full training for best results
# Uncomment the one you want:

# Quick training (~10 min total) — good for testing the pipeline works:
docker compose run --rm train python -m scripts.train --data /app/data/raw/tickets.json --both --optuna-trials 5 --epochs 2 --model-dir /app/models

# Full training (~45 min total) — best results:
# docker compose run --rm train

echo "✓ Training complete"

# Step 3: Start the API
echo ""
echo "═══ Step 3: Starting the API server ═══"
docker compose up -d app
sleep 5

# Step 4: Verify
echo ""
echo "═══ Step 4: Verifying ═══"
echo ""

# Health check
echo "Health check:"
curl -s http://localhost:8000/api/v1/health | python3 -m json.tool 2>/dev/null || echo "  (API still starting, try again in a few seconds)"

echo ""

# Test classification
echo "Test classification:"
curl -s -X POST http://localhost:8000/api/v1/tickets/process \
    -H "Content-Type: application/json" \
    -d '{
        "subject": "Database sync failing with timeout error",
        "description": "Getting ERROR_TIMEOUT_429 when syncing large datasets. Started after the recent update.",
        "product": "DataSync Pro",
        "product_version": "3.2.1",
        "customer_tier": "enterprise",
        "priority": "high",
        "error_logs": "2024-01-15 10:25:33 ERROR_TIMEOUT_429: Connection timeout after 30s"
    }' | python3 -m json.tool 2>/dev/null || echo "  (API still loading models, try again in ~30s)"

IP=$(hostname -I | awk '{print $1}')
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Pipeline complete!                                  ║"
echo "║                                                      ║"
echo "║  API Docs:   http://$IP:8000/docs          ║"
echo "║  MLflow:     http://$IP:5000               ║"
echo "║                                                      ║"
echo "║  Try it:                                             ║"
echo "║  curl -X POST http://$IP:8000/api/v1/tickets/process ║"
echo "║    -H 'Content-Type: application/json'               ║"
echo "║    -d '{\"subject\":\"test\",\"description\":\"help\"}'  ║"
echo "║                                                      ║"
echo "║  Analytics:                                          ║"
echo "║  curl http://$IP:8000/api/v1/analytics/agents        ║"
echo "║  curl http://$IP:8000/api/v1/analytics/satisfaction  ║"
echo "║  curl http://$IP:8000/api/v1/anomalies               ║"
echo "║                                                      ║"
echo "║  Logs:  docker compose logs -f app                   ║"
echo "║  Stop:  docker compose down                          ║"
echo "╚══════════════════════════════════════════════════════╝"
