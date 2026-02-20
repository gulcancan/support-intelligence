"""Lightweight Graph-RAG using PostgreSQL relational tables."""
import logging
from typing import Optional
from sqlalchemy import text
from db import get_engine
logger = logging.getLogger(__name__)

class GraphRetriever:
    def __init__(self, engine=None): self.engine = engine or get_engine()

    def get_product_resolutions(self, product, category, top_k=5):
        q = text("SELECT iss.resolution_code, iss.resolution_template, iss.success_rate, iss.usage_count, pi.avg_resolution_hrs FROM issue_solutions iss JOIN product_issues pi ON pi.issue_type=iss.issue_type AND pi.product=:product WHERE iss.issue_type=:category ORDER BY iss.success_rate DESC LIMIT :top_k")
        with self.engine.connect() as c:
            rows = c.execute(q, {"product":product,"category":category,"top_k":top_k}).fetchall()
        return [{"resolution_code":r[0],"resolution_template":r[1],"success_rate":r[2],"usage_count":r[3],"avg_resolution_hrs":r[4],"source":"graph"} for r in rows]

    def get_error_code_resolutions(self, error_codes, product=None, top_k=5):
        if not error_codes: return []
        q = text("SELECT error_code, product, issue_type, resolution_code, resolution_text, occurrences FROM error_code_mapping WHERE error_code = ANY(:codes) ORDER BY occurrences DESC LIMIT :top_k")
        params = {"codes":error_codes,"top_k":top_k}
        with self.engine.connect() as c: rows = c.execute(q, params).fetchall()
        return [{"error_code":r[0],"product":r[1],"issue_type":r[2],"resolution_code":r[3],"resolution_text":r[4],"occurrences":r[5]} for r in rows]

    def get_related_context(self, product, category, error_codes=None):
        return {"resolution_templates":self.get_product_resolutions(product,category),"error_code_resolutions":self.get_error_code_resolutions(error_codes or [],product),"product_stats":{}}
