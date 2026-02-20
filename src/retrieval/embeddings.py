"""Sentence-transformer embeddings (all-MiniLM-L6-v2)."""
import logging, numpy as np
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name; self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self): return self.model.get_sentence_embedding_dimension()
    def embed(self, text): return self.model.encode(text, normalize_embeddings=True)
    def embed_batch(self, texts, batch_size=64, show_progress=True): return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=show_progress)

    def embed_ticket(self, ticket):
        parts = [ticket.get("subject",""), ticket.get("description","")]
        if ticket.get("product"): parts.append(f"Product: {ticket['product']}")
        if ticket.get("error_logs"): parts.append(f"Error: {ticket['error_logs'].split(chr(10))[0]}")
        return self.embed(" | ".join(parts))
