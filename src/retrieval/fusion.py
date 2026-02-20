"""Hybrid retrieval: Vector + BM25 + Graph, fused via RRF."""
import re, logging
from typing import Optional
from collections import defaultdict
from retrieval.embeddings import EmbeddingService
from retrieval.vector_store import VectorStore
from retrieval.bm25 import BM25Search
from retrieval.graph import GraphRetriever
logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(result_lists, k=60, id_key="ticket_id"):
    scores = defaultdict(float); docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            did = doc.get(id_key)
            if did: scores[did] += 1.0/(k+rank+1); docs.setdefault(did, doc)
    return [{**docs[d],"rrf_score":s} for d,s in sorted(scores.items(), key=lambda x:x[1], reverse=True) if d in docs]

def extract_error_codes(text): return list(set(re.findall(r"ERROR_\w+", text or "")))

class HybridRetriever:
    def __init__(self, embedding_service=None, vector_store=None, bm25_search=None, graph_retriever=None):
        self.embedder = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()
        self.bm25 = bm25_search or BM25Search()
        self.graph = graph_retriever

    def retrieve(self, ticket, predicted_category=None, top_k=5, use_graph=True):
        query = f"{ticket.get('subject','')} {ticket.get('description','')}"
        product = ticket.get("product"); ec = extract_error_codes(f"{ticket.get('description','')} {ticket.get('error_logs','')}")
        rlists = []
        try:
            qv = self.embedder.embed_ticket(ticket)
            rlists.append(self.vector_store.search(qv, top_k*2, category_filter=predicted_category, product_filter=product))
        except Exception as e: logger.error(f"Vector search failed: {e}")
        try: rlists.append(self.bm25.search(query, top_k*2, category_filter=predicted_category))
        except Exception as e: logger.error(f"BM25 failed: {e}")
        gc = {}
        if use_graph and self.graph and product and predicted_category:
            try: gc = self.graph.get_related_context(product, predicted_category, ec)
            except Exception as e: logger.error(f"Graph failed: {e}")
        fused = reciprocal_rank_fusion(rlists) if rlists else []
        for r in fused:
            s = r.get("rrf_score",0)
            if r.get("resolution_helpful"): s *= 1.5
            if (r.get("satisfaction_score") or 0) >= 4: s *= 1.3
            r["final_score"] = s
        fused.sort(key=lambda x:x.get("final_score",0), reverse=True)
        return {"results":fused[:top_k],"graph_context":gc,"metadata":{"predicted_category":predicted_category,"product":product,"error_codes":ec}}

    def index_tickets(self, tickets, batch_size=500):
        with_res = [t for t in tickets if t.get("resolution")]
        logger.info(f"Indexing {len(with_res)} resolved tickets...")
        self.bm25.build_index(with_res)
        texts = [" | ".join([t.get("subject",""),t.get("resolution","")]) for t in with_res]
        embs = self.embedder.embed_batch(texts, batch_size=64)
        ids = [t["ticket_id"] for t in with_res]
        payloads = [{"category":t.get("category"),"product":t.get("product"),"resolution":t.get("resolution"),"resolution_code":t.get("resolution_code"),"resolution_helpful":t.get("resolution_helpful"),"satisfaction_score":t.get("satisfaction_score"),"subject":t.get("subject")} for t in with_res]
        self.vector_store.create_collection(vector_dim=embs.shape[1])
        self.vector_store.upsert_batch(ids, embs, payloads, batch_size)
        logger.info("Indexing complete")
