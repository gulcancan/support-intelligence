"""Qdrant vector store wrapper with metadata filtering."""
import logging, numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from config import get_settings
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name=None):
        s = get_settings(); self.collection_name = collection_name or s.qdrant_collection
        self.client = QdrantClient(host=s.qdrant_host, port=s.qdrant_port, timeout=30)

    def create_collection(self, vector_dim=384):
        self.client.recreate_collection(collection_name=self.collection_name, vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE))

    def upsert_batch(self, ids, vectors, payloads, batch_size=500):
        points = [PointStruct(id=i, vector=vectors[i].tolist(), payload={**payloads[i],"ticket_id":ids[i]}) for i in range(len(ids))]
        for s in range(0,len(points),batch_size): self.client.upsert(collection_name=self.collection_name, points=points[s:s+batch_size])
        logger.info(f"Upserted {len(points)} vectors")

    def search(self, query_vector, top_k=10, category_filter=None, product_filter=None):
        conds = []
        if category_filter: conds.append(FieldCondition(key="category",match=MatchValue(value=category_filter)))
        if product_filter: conds.append(FieldCondition(key="product",match=MatchValue(value=product_filter)))
        filt = Filter(must=conds) if conds else None
        results = self.client.search(collection_name=self.collection_name, query_vector=query_vector.tolist(), limit=top_k, query_filter=filt, with_payload=True)
        return [{"ticket_id":h.payload.get("ticket_id"),"score":h.score,"category":h.payload.get("category"),"product":h.payload.get("product"),"resolution":h.payload.get("resolution"),"resolution_code":h.payload.get("resolution_code"),"resolution_helpful":h.payload.get("resolution_helpful"),"satisfaction_score":h.payload.get("satisfaction_score"),"subject":h.payload.get("subject")} for h in results]

    def count(self): return self.client.get_collection(self.collection_name).points_count
