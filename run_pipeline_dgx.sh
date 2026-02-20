#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════
# DGX Spark Pipeline: Ingest → Train (GPU) → Serve
#
# Prerequisites:
#   - NVIDIA Container Toolkit installed (comes pre-installed on DGX)
#   - data/raw/tickets.json present
#
# Usage: bash run_pipeline_dgx.sh
# ═══════════════════════════════════════════════════════════

cd "$(dirname "$0")"
COMPOSE="docker compose -f docker-compose.dgx.yml"

# Verify GPU access
echo "═══ Checking GPU ═══"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. GPU may not be accessible inside containers."
fi

# Verify data
if [ ! -f "data/raw/tickets.json" ]; then
    echo "ERROR: data/raw/tickets.json not found"
    exit 1
fi
TICKET_COUNT=$(python3 -c "import json; print(len(json.load(open('data/raw/tickets.json'))))" 2>/dev/null || echo "?")
echo "Data: $TICKET_COUNT tickets"
echo ""

# Step 0: Build and start infra
echo "═══ Step 0: Building images and starting infrastructure ═══"
$COMPOSE build
$COMPOSE up -d postgres qdrant mlflow
echo "Waiting for services..."
sleep 15

# Step 1: Ingest
echo ""
echo "═══ Step 1: Ingesting tickets ═══"
$COMPOSE run --rm ingest
echo "✓ Ingestion complete"

# Step 2: Train on GPU
echo ""
echo "═══ Step 2: Training models (GPU-accelerated) ═══"
echo "  CatBoost: ~5 min (CPU, Optuna HPO)"
echo "  DistilBERT: ~3-5 min on Blackwell GPU (batch_size=128, 5 epochs)"
echo ""
$COMPOSE run --rm train
echo "✓ Training complete"

# Step 3: Start API
echo ""
echo "═══ Step 3: Starting API ═══"
$COMPOSE up -d app
sleep 5

# Step 4: Verify
echo ""
echo "═══ Verification ═══"
curl -s http://localhost:8000/api/v1/health | python3 -m json.tool 2>/dev/null || echo "(API still starting...)"

echo ""
echo "Test inference:"
curl -s -X POST http://localhost:8000/api/v1/tickets/process \
    -H "Content-Type: application/json" \
    -d '{"subject":"DB sync timeout","description":"ERROR_TIMEOUT_429 on large datasets","product":"DataSync Pro"}' \
    | python3 -m json.tool 2>/dev/null || echo "(Models still loading...)"

IP=$(hostname -I | awk '{print $1}')
echo ""
echo "═══════════════════════════════════════════════"
echo "  API:    http://$IP:8000/docs"
echo "  MLflow: http://$IP:5000"
echo "  Logs:   $COMPOSE logs -f app"
echo "  Stop:   $COMPOSE down"
echo "═══════════════════════════════════════════════"
