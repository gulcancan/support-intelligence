.PHONY: help setup generate-data train-all serve test

PYTHON := python3
DATA := data/raw/tickets_100k.json

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Install deps + generate data
	$(PYTHON) -m pip install -r requirements.txt --break-system-packages 2>/dev/null || $(PYTHON) -m pip install -r requirements.txt
	$(MAKE) generate-data

generate-data: ## Generate 100k synthetic tickets
	cd scripts && $(PYTHON) generate_data.py

infra: ## Start PostgreSQL, Qdrant, MLflow
	docker compose up -d postgres qdrant mlflow && sleep 10 && echo "✅ Infra ready"

infra-down: ## Stop infrastructure
	docker compose down

ingest: ## Load tickets into database
	cd src && PYTHONPATH=. $(PYTHON) -c "from ingestion.pipeline import ingest; import json; print(json.dumps(ingest('../$(DATA)'),indent=2,default=str))"

index: ## Build retrieval index
	cd src && PYTHONPATH=. $(PYTHON) -c "import json; from retrieval.fusion import HybridRetriever; r=HybridRetriever(); r.index_tickets(json.load(open('../$(DATA)')))"

train-catboost: ## Train CatBoost
	cd src && PYTHONPATH=. $(PYTHON) ../scripts/train.py --data ../$(DATA) --catboost --model-dir ../models

train-transformer: ## Train DistilBERT
	cd src && PYTHONPATH=. $(PYTHON) ../scripts/train.py --data ../$(DATA) --transformer --model-dir ../models

train-all: ## Train both + compare
	cd src && PYTHONPATH=. $(PYTHON) ../scripts/train.py --data ../$(DATA) --both --model-dir ../models

train-quick: ## Quick train (3 trials, 2 epochs)
	cd src && PYTHONPATH=. $(PYTHON) ../scripts/train.py --data ../$(DATA) --both --optuna-trials 3 --epochs 2 --model-dir ../models

serve: ## Start FastAPI server
	cd src && PYTHONPATH=. uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run tests
	cd src && PYTHONPATH=. $(PYTHON) -m pytest ../tests/ -v

test-api: ## Test API with curl
	@curl -s http://localhost:8000/api/v1/health | python3 -m json.tool

clean: ## Remove generated files
	rm -rf models/ data/raw/tickets_100k.json __pycache__
