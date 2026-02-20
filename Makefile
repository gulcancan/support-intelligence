.PHONY: help setup infra ingest train-all serve test clean

PYTHON := python3
DATA := data/raw/tickets.json

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Setup ──

setup: ## Install Python dependencies
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages 2>/dev/null || $(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -r requirements.txt --break-system-packages 2>/dev/null || $(PYTHON) -m pip install -r requirements.txt
	@echo "✅ Dependencies installed. Place your ticket data at $(DATA)"

# ── Docker workflow (recommended) ──

up: ## Start full stack (PostgreSQL, Qdrant, MLflow, API)
	docker compose up -d
	@echo "✅ Services starting. API: http://localhost:8000 | MLflow: http://localhost:5000"

down: ## Stop all services
	docker compose down

ingest-docker: ## Ingest 300K tickets (run once after 'up')
	docker compose run --rm ingest

train-docker: ## Train both models inside Docker
	docker compose run --rm train

# ── Local workflow ──

infra: ## Start infrastructure only (PostgreSQL, Qdrant, MLflow)
	docker compose up -d postgres qdrant mlflow
	@echo "Waiting for services..." && sleep 10
	@echo "✅ PostgreSQL:5432 | Qdrant:6333 | MLflow:5000"

ingest: ## Ingest tickets into database + build retrieval index
	cd src && PYTHONPATH=. $(PYTHON) ../scripts/run_ingest.py

train-catboost: ## Train CatBoost classifier
	cd src && PYTHONPATH=. $(PYTHON) ../scripts/train.py --data ../$(DATA) --catboost --model-dir ../models

train-transformer: ## Train DistilBERT classifier
	cd src && PYTHONPATH=. $(PYTHON) ../scripts/train.py --data ../$(DATA) --transformer --model-dir ../models

train-all: ## Train both models and compare
	cd src && PYTHONPATH=. $(PYTHON) ../scripts/train.py --data ../$(DATA) --both --model-dir ../models

train-quick: ## Quick training (3 Optuna trials, 2 epochs)
	cd src && PYTHONPATH=. $(PYTHON) ../scripts/train.py --data ../$(DATA) --both --optuna-trials 3 --epochs 2 --model-dir ../models

serve: ## Start FastAPI server locally
	cd src && PYTHONPATH=. uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# ── Testing ──

test: ## Run all tests
	cd src && PYTHONPATH=. $(PYTHON) -m pytest ../tests/ -v

test-api: ## Test API endpoints with curl
	@echo "=== Health ===" && curl -s http://localhost:8000/api/v1/health | python3 -m json.tool
	@echo "\n=== Classify ===" && curl -s -X POST http://localhost:8000/api/v1/tickets/process \
		-H "Content-Type: application/json" \
		-d '{"subject":"Database sync failing","description":"ERROR_TIMEOUT_429 on large datasets","product":"DataSync Pro","priority":"high"}' \
		| python3 -m json.tool

# ── Cleanup ──

clean: ## Remove trained models and caches
	rm -rf models/ __pycache__ src/__pycache__ src/**/__pycache__

clean-all: clean down ## Full cleanup including Docker volumes
	docker compose down -v
