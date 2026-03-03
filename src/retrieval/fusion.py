"""
Hybrid retrieval: Vector + BM25 + Graph, fused via RRF.

Now uses multi-task classification outputs to improve retrieval:
  - category    → filters vector/BM25 search to same category
  - subcategory → boosts results matching same subcategory
  - priority    → boosts results from similar priority tickets (high→critical)
  - sentiment   → boosts results that resolved similar sentiment tickets successfully

The flow is:
  1. Vector search (Qdrant) — filtered by category + product
  2. BM25 keyword search — filtered by category, boosted by subcategory match
  3. Graph-RAG — uses category + subcategory for resolution templates
  4. RRF fusion — parameter-free rank merging
  5. Classification-aware re-ranking — boost by subcategory, priority, resolution quality
"""
import re, logging
from typing import Optional
from collections import defaultdict
from retrieval.embeddings import EmbeddingService
from retrieval.vector_store import VectorStore
from retrieval.bm25 import BM25Search
from retrieval.graph import GraphRetriever

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(result_lists, k=60, id_key="ticket_id"):
    """Merge ranked lists via RRF. Parameter-free, robust across score distributions."""
    scores = defaultdict(float)
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            did = doc.get(id_key)
            if did:
                scores[did] += 1.0 / (k + rank + 1)
                docs.setdefault(did, doc)
    return [
        {**docs[d], "rrf_score": s}
        for d, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if d in docs
    ]


def extract_error_codes(text):
    return list(set(re.findall(r"ERROR_\w+", text or "")))


# Priority adjacency — tickets with "nearby" priority are still relevant
PRIORITY_ADJACENCY = {
    "critical": {"critical": 1.0, "high": 0.7, "medium": 0.3, "low": 0.1},
    "high":     {"critical": 0.7, "high": 1.0, "medium": 0.5, "low": 0.2},
    "medium":   {"critical": 0.3, "high": 0.5, "medium": 1.0, "low": 0.5},
    "low":      {"critical": 0.1, "high": 0.2, "medium": 0.5, "low": 1.0},
}


class HybridRetriever:
    def __init__(self, embedding_service=None, vector_store=None,
                 bm25_search=None, graph_retriever=None):
        self.embedder = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()
        self.bm25 = bm25_search or BM25Search()
        self.graph = graph_retriever

    def retrieve(self, ticket, predicted_category=None, predicted_subcategory=None,
                 predicted_priority=None, predicted_sentiment=None, top_k=5, use_graph=True):
        """
        Retrieve relevant past resolutions using classification predictions.

        Args:
            ticket: dict with ticket fields
            predicted_category: e.g. "Technical Issue"
            predicted_subcategory: e.g. "Configuration"
            predicted_priority: e.g. "high"
            predicted_sentiment: e.g. "frustrated"
            top_k: number of results to return
            use_graph: whether to query the knowledge graph
        """
        query = f"{ticket.get('subject', '')} {ticket.get('description', '')}"
        product = ticket.get("product")
        ec = extract_error_codes(f"{ticket.get('description', '')} {ticket.get('error_logs', '')}")

        rlists = []

        # ── Vector search (filtered by category + product) ──
        try:
            qv = self.embedder.embed_ticket(ticket)
            rlists.append(self.vector_store.search(
                qv, top_k * 2,
                category_filter=predicted_category,
                product_filter=product,
            ))
        except Exception as e:
            logger.error(f"Vector search failed: {e}")

        # ── BM25 keyword search (filtered by category) ──
        try:
            rlists.append(self.bm25.search(
                query, top_k * 2,
                category_filter=predicted_category,
            ))
        except Exception as e:
            logger.error(f"BM25 failed: {e}")

        # ── Graph-RAG (uses category + subcategory) ──
        gc = {}
        if use_graph and self.graph and product and predicted_category:
            try:
                gc = self.graph.get_related_context(
                    product, predicted_category,
                    error_codes=ec,
                    subcategory=predicted_subcategory,
                )
            except Exception as e:
                logger.error(f"Graph failed: {e}")

        # ── RRF fusion ──
        fused = reciprocal_rank_fusion(rlists) if rlists else []

        # ── Classification-aware re-ranking ──
        for r in fused:
            score = r.get("rrf_score", 0)

            # Base quality signals (unchanged)
            if r.get("resolution_helpful"):
                score *= 1.5
            if (r.get("satisfaction_score") or 0) >= 4:
                score *= 1.3

            # NEW: Subcategory match boost
            # If the retrieved ticket has the same subcategory, it's much more relevant
            if predicted_subcategory and r.get("subcategory"):
                if r["subcategory"] == predicted_subcategory:
                    score *= 1.8  # strong boost — same fine-grained issue type
                elif r.get("category") == predicted_category:
                    score *= 1.1  # mild boost — same category at least

            # NEW: Priority proximity boost
            # Resolutions from similar-priority tickets are more applicable
            # (a fix for a critical issue may not apply to a low-priority one)
            if predicted_priority and r.get("priority"):
                adjacency = PRIORITY_ADJACENCY.get(predicted_priority, {})
                pri_boost = adjacency.get(r["priority"], 0.5)
                score *= (0.8 + 0.4 * pri_boost)  # range: 0.84 to 1.2

            # NEW: Sentiment-aware boost
            # If the customer is frustrated/angry, prioritize resolutions that
            # previously resolved similar sentiment tickets with high satisfaction
            if predicted_sentiment in ("frustrated", "angry"):
                if (r.get("satisfaction_score") or 0) >= 4:
                    score *= 1.2  # prioritize proven crowd-pleasers
                if r.get("resolution_helpful"):
                    score *= 1.1

            r["final_score"] = score

        fused.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        return {
            "results": fused[:top_k],
            "graph_context": gc,
            "metadata": {
                "predicted_category": predicted_category,
                "predicted_subcategory": predicted_subcategory,
                "predicted_priority": predicted_priority,
                "predicted_sentiment": predicted_sentiment,
                "product": product,
                "error_codes": ec,
            },
        }

    def index_tickets(self, tickets, batch_size=500):
        """Build search indices from resolved tickets."""
        with_res = [t for t in tickets if t.get("resolution")]
        logger.info(f"Indexing {len(with_res)} resolved tickets...")

        self.bm25.build_index(with_res)

        texts = [" | ".join([t.get("subject", ""), t.get("resolution", "")]) for t in with_res]
        embs = self.embedder.embed_batch(texts, batch_size=64)
        ids = [t["ticket_id"] for t in with_res]

        # Include subcategory and priority in payloads for re-ranking
        payloads = [{
            "category": t.get("category"),
            "subcategory": t.get("subcategory"),
            "priority": t.get("priority"),
            "customer_sentiment": t.get("customer_sentiment"),
            "product": t.get("product"),
            "resolution": t.get("resolution"),
            "resolution_code": t.get("resolution_code"),
            "resolution_helpful": t.get("resolution_helpful"),
            "satisfaction_score": t.get("satisfaction_score"),
            "subject": t.get("subject"),
        } for t in with_res]

        self.vector_store.create_collection(vector_dim=embs.shape[1])
        self.vector_store.upsert_batch(ids, embs, payloads, batch_size)
        logger.info("Indexing complete")
