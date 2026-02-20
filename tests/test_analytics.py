"""Tests for business analytics and quality scoring."""
import pytest
import numpy as np
import pandas as pd


class TestSatisfactionAnalyzer:
    def test_analyze(self):
        from analytics.business import SatisfactionAnalyzer
        df = pd.DataFrame({
            "satisfaction_score": np.random.choice([1,2,3,4,5], 500, p=[0.05,0.1,0.2,0.35,0.3]),
            "resolution_time_hours": np.random.lognormal(2, 1, 500),
            "resolution_attempts": np.random.choice([1,2,3], 500),
            "transferred_count": np.random.choice([0,1,2], 500, p=[0.7,0.2,0.1]),
            "response_count": np.random.choice([1,2,3,4,5], 500),
            "escalated": np.random.choice([True,False], 500, p=[0.1,0.9]),
            "resolution_helpful": np.random.choice([True,False], 500, p=[0.8,0.2]),
            "agent_experience_months": np.random.randint(3, 120, 500),
            "customer_tier": np.random.choice(["free","starter","professional","enterprise"], 500),
            "priority": np.random.choice(["low","medium","high","critical"], 500),
        })
        result = SatisfactionAnalyzer().analyze(df)
        assert "drivers" in result
        assert len(result["drivers"]) > 0
        assert 0 <= result["satisfaction_rate"] <= 1
        assert 0 <= result["model_auc"] <= 1


class TestAgentPerformance:
    def test_analyze(self):
        from analytics.business import AgentPerformanceAnalyzer
        df = pd.DataFrame({
            "ticket_id": [f"T{i}" for i in range(200)],
            "agent_id": [f"AGENT-{i%10}" for i in range(200)],
            "resolution_time_hours": np.random.lognormal(2, 1, 200),
            "satisfaction_score": np.random.choice([1,2,3,4,5], 200),
            "escalated": np.random.choice([True,False], 200, p=[0.1,0.9]),
            "resolution_helpful": np.random.choice([True,False], 200, p=[0.8,0.2]),
            "response_count": np.random.randint(1, 8, 200),
            "agent_experience_months": [12 + i%10 * 5 for i in range(200)],
            "agent_specialization": ["general"] * 200,
            "category": np.random.choice(["Tech","Billing","Feature"], 200),
        })
        result = AgentPerformanceAnalyzer().analyze(df)
        assert result["n_agents"] == 10
        assert len(result["agents"]) == 10
        assert len(result["top_performers"]) == 5


class TestQualityScorer:
    def test_score_result(self):
        from analytics.business import RetrievalQualityScorer
        scorer = RetrievalQualityScorer()
        ticket = {"category": "Technical Issue", "product": "DataSync Pro"}
        result = {
            "category": "Technical Issue", "product": "DataSync Pro",
            "resolution_helpful": True, "satisfaction_score": 4,
            "resolution": "Increased batch size in config.yaml from 100MB to 500MB. Customer confirmed resolved.",
        }
        score = scorer.score_result(ticket, result)
        assert 0 <= score["composite_score"] <= 1
        assert "relevance" in score["dimensions"]
        # Matching category + product + helpful + detailed resolution = high score
        assert score["composite_score"] > 0.5

    def test_low_quality_result(self):
        from analytics.business import RetrievalQualityScorer
        scorer = RetrievalQualityScorer()
        ticket = {"category": "Technical Issue", "product": "DataSync Pro"}
        result = {
            "category": "Account & Billing", "product": "CloudStore",
            "resolution_helpful": False, "satisfaction_score": 1,
            "resolution": "Closed",
        }
        score = scorer.score_result(ticket, result)
        assert score["composite_score"] < 0.4  # Bad match

    def test_batch_scoring(self):
        from analytics.business import RetrievalQualityScorer
        scorer = RetrievalQualityScorer()
        ticket = {"category": "Tech", "product": "A"}
        results = [
            {"category": "Tech", "product": "A", "resolution": "Fixed config", "resolution_helpful": True, "satisfaction_score": 5},
            {"category": "Billing", "product": "B", "resolution": "ok", "resolution_helpful": False, "satisfaction_score": 2},
        ]
        scored = scorer.score_retrieval_batch(ticket, results)
        assert scored[0]["quality_score"] > scored[1]["quality_score"]


class TestEventSchema:
    def test_system_event_creation(self):
        from analytics.business import SystemEvent
        event = SystemEvent(
            event_id="EVT-001", event_type="classification",
            timestamp="2024-01-15T10:30:00Z", ticket_id="TK-001",
            predicted_category="Tech", confidence=0.92,
            model_name="catboost", latency_ms=3.5,
        )
        assert event.event_type == "classification"
        assert event.status == "success"
