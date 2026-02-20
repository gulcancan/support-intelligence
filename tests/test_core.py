"""Tests for core components."""
import pytest, pandas as pd, numpy as np

class TestTemporalSplit:
    def test_ordering(self):
        from ingestion.pipeline import temporal_split
        df = pd.DataFrame({"ticket_id":[f"T{i}" for i in range(100)],"created_at":pd.date_range("2024-01-01",periods=100,freq="D"),"category":["A"]*50+["B"]*50})
        df = temporal_split(df)
        tr = df[df["data_split"]=="train"]; val = df[df["data_split"]=="val"]; te = df[df["data_split"]=="test"]
        if len(tr)>0 and len(val)>0: assert tr["created_at"].max() <= val["created_at"].min()
        if len(val)>0 and len(te)>0: assert val["created_at"].max() <= te["created_at"].min()

class TestEvaluation:
    def test_evaluate_model(self):
        from models.common import evaluate_model
        m = evaluate_model([0,1,2,0,1,2],[0,1,2,0,2,1],["A","B","C"])
        assert 0<=m.weighted_f1<=1 and len(m.per_class_f1)==3

class TestRRF:
    def test_fusion(self):
        from retrieval.fusion import reciprocal_rank_fusion
        l1 = [{"ticket_id":"A","score":0.9},{"ticket_id":"B","score":0.7}]
        l2 = [{"ticket_id":"B","score":0.95},{"ticket_id":"A","score":0.6}]
        fused = reciprocal_rank_fusion([l1,l2])
        assert len(fused)==2 and all("rrf_score" in r for r in fused)

class TestBM25:
    def test_search(self):
        from retrieval.bm25 import BM25Search
        docs = [
            {"ticket_id":"1","subject":"database timeout error","description":"ERROR_TIMEOUT_429 on large sync","resolution":"increased timeout","tags":["database","timeout"],"category":"Tech"},
            {"ticket_id":"2","subject":"billing question","description":"need invoice for Q4","resolution":"sent invoice","tags":["billing"],"category":"Billing"},
            {"ticket_id":"3","subject":"auth failure","description":"users cannot login","resolution":"reset tokens","tags":["auth"],"category":"Tech"},
        ]
        bm25 = BM25Search(); bm25.build_index(docs)
        results = bm25.search("database timeout ERROR_TIMEOUT_429")
        assert len(results) > 0
        assert results[0]["ticket_id"] == "1"

class TestErrorCodes:
    def test_extraction(self):
        from retrieval.fusion import extract_error_codes
        assert "ERROR_TIMEOUT_429" in extract_error_codes("Got ERROR_TIMEOUT_429")
        assert extract_error_codes("")==[]

class TestSchemas:
    def test_ticket_input(self):
        from api.schemas import TicketInput
        t = TicketInput(subject="Test", description="desc"); assert t.subject=="Test"
    def test_requires_subject(self):
        from api.schemas import TicketInput
        with pytest.raises(Exception): TicketInput(description="no subject")
