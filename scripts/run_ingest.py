"""
Run full ingestion pipeline: load 300K tickets → PostgreSQL → Qdrant → BM25.

Usage:
    python -m scripts.run_ingest
    # or via docker compose:
    docker compose run --rm ingest
"""
import json
import logging
import sys
import os
from pathlib import Path

# Ensure /app/src is on the path — NVIDIA base image entrypoint can override PYTHONPATH
_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    from config import get_settings
    settings = get_settings()

    # Find the ticket data file
    data_dir = Path(settings.data_dir)
    candidates = [
        data_dir / "raw" / "tickets.json",
        data_dir / "raw" / "tickets_300k.json",
        data_dir / "raw" / "tickets_100k.json",
    ]
    data_path = None
    for p in candidates:
        if p.exists():
            data_path = p
            break

    if data_path is None:
        logger.error(f"No ticket data found. Searched: {[str(p) for p in candidates]}")
        logger.error("Place your ticket JSON file at data/raw/tickets.json")
        sys.exit(1)

    logger.info(f"Found ticket data: {data_path}")

    # Step 1: Ingest into PostgreSQL
    logger.info("=" * 60)
    logger.info("Step 1: Ingesting tickets into PostgreSQL")
    logger.info("=" * 60)
    from ingestion.pipeline import ingest
    result = ingest(str(data_path))
    logger.info(f"Ingestion result: {json.dumps(result, indent=2, default=str)}")

    # Step 2: Build retrieval index
    logger.info("=" * 60)
    logger.info("Step 2: Building retrieval index (embeddings + BM25)")
    logger.info("=" * 60)
    with open(data_path) as f:
        tickets = json.load(f)

    # Only index resolved tickets
    resolved = [t for t in tickets if t.get("resolution")]
    logger.info(f"Indexing {len(resolved):,} resolved tickets out of {len(tickets):,} total")

    from retrieval.fusion import HybridRetriever
    retriever = HybridRetriever()
    retriever.index_tickets(resolved)

    # Step 3: Populate graph tables
    logger.info("=" * 60)
    logger.info("Step 3: Populating graph tables")
    logger.info("=" * 60)
    from ingestion.pipeline import populate_graph_tables
    populate_graph_tables(tickets)

    logger.info("=" * 60)
    logger.info("Ingestion complete!")
    logger.info(f"  Tickets loaded: {len(tickets):,}")
    logger.info(f"  Tickets indexed for retrieval: {len(resolved):,}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
