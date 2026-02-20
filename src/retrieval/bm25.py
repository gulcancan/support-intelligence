"""BM25 keyword search for exact error code matching."""
import re, logging, numpy as np
from typing import Optional
from rank_bm25 import BM25Okapi
logger = logging.getLogger(__name__)

class BM25Search:
    def __init__(self): self._index = None; self._documents = []; self._tokenized = []

    @staticmethod
    def _tokenize(text): return re.findall(r"error_\w+|\w+", text.lower())

    def build_index(self, documents):
        self._documents = documents
        self._tokenized = [self._tokenize(" ".join([d.get("subject",""),d.get("description",""),d.get("resolution",""),d.get("error_logs","") or ""]+([t for t in d.get("tags",[]) if isinstance(t,str)]))) for d in documents]
        self._index = BM25Okapi(self._tokenized)
        logger.info(f"BM25 index: {len(documents)} docs")

    def search(self, query, top_k=10, category_filter=None):
        if not self._index: raise ValueError("Index not built")
        scores = self._index.get_scores(self._tokenize(query))
        if category_filter:
            idxs = [i for i,d in enumerate(self._documents) if d.get("category")==category_filter]
            if idxs:
                pairs = sorted([(i,scores[i]) for i in idxs], key=lambda x:x[1], reverse=True)[:top_k]
                top_idxs = [i for i,_ in pairs]
            else: top_idxs = np.argsort(scores)[::-1][:top_k].tolist()
        else: top_idxs = np.argsort(scores)[::-1][:top_k].tolist()
        return [{"ticket_id":self._documents[i].get("ticket_id"),"score":float(scores[i]),"category":self._documents[i].get("category"),"product":self._documents[i].get("product"),"resolution":self._documents[i].get("resolution"),"resolution_code":self._documents[i].get("resolution_code"),"resolution_helpful":self._documents[i].get("resolution_helpful"),"satisfaction_score":self._documents[i].get("satisfaction_score"),"subject":self._documents[i].get("subject")} for i in top_idxs if scores[i]>0]
