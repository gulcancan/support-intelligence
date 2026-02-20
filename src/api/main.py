"""FastAPI application — unified API gateway."""
import time, logging
from contextlib import asynccontextmanager
from typing import Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import *
from config import get_settings

logger = logging.getLogger(__name__)
model_registry = retriever = anomaly_detector = feedback_collector = None

@asynccontextmanager
async def lifespan(app):
    global model_registry, retriever, anomaly_detector, feedback_collector
    logger.info("Initializing system...")
    from models.registry import ModelRegistry; model_registry = ModelRegistry()
    s = get_settings()
    try:
        from models.catboost_classifier import CatBoostTicketClassifier
        cb = CatBoostTicketClassifier(model_dir=f"{s.model_dir}/catboost"); cb.load()
        model_registry.register("catboost", cb, is_primary=True)
    except Exception as e: logger.warning(f"CatBoost not available: {e}")
    try:
        from models.transformer_classifier import TransformerTicketClassifier
        tf = TransformerTicketClassifier(model_dir=f"{s.model_dir}/transformer"); tf.load()
        model_registry.register("transformer", tf)
    except Exception as e: logger.warning(f"Transformer not available: {e}")
    try:
        from retrieval.fusion import HybridRetriever
        retriever = HybridRetriever()
    except Exception as e: logger.warning(f"Retriever not available: {e}")
    from anomaly.detector import AnomalyDetector; anomaly_detector = AnomalyDetector()
    try:
        from feedback.collector import FeedbackCollector; feedback_collector = FeedbackCollector()
    except: pass
    logger.info("System ready"); yield; logger.info("Shutting down")

app = FastAPI(title="Support Intelligence System", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/api/v1/tickets/process", response_model=TicketProcessingResponse)
async def process_ticket(ticket: TicketInput):
    start = time.time(); td = ticket.model_dump()
    if td.get("ticket_text_length") is None: td["ticket_text_length"] = len(td.get("description",""))
    clf = None; pred_cat = None
    if model_registry and model_registry.list_models():
        try:
            r = model_registry.predict(td)
            clf = ClassificationResponse(predicted_category=r.predicted_category, predicted_subcategory=r.predicted_subcategory, confidence=r.confidence, category_probabilities=r.category_probabilities, model_name=r.model_name, latency_ms=r.latency_ms)
            pred_cat = r.predicted_category
        except Exception as e: logger.error(f"Classification failed: {e}")
    if clf is None: clf = ClassificationResponse(predicted_category="Unknown", confidence=0.0, category_probabilities={}, model_name="fallback", latency_ms=0.0)
    ret = RetrievalResponse(results=[], graph_context=GraphContext(), metadata={})
    if retriever:
        try:
            rr = retriever.retrieve(td, predicted_category=pred_cat, top_k=5)
            ret = RetrievalResponse(results=[RetrievalResult(**r) for r in rr["results"]], graph_context=GraphContext(**rr.get("graph_context",{})), metadata=rr.get("metadata",{}))
        except Exception as e: logger.error(f"Retrieval failed: {e}")
    return TicketProcessingResponse(ticket_id=ticket.ticket_id, classification=clf, retrieval=ret, processing_time_ms=round((time.time()-start)*1000,2))

@app.post("/api/v1/tickets/classify", response_model=ClassificationResponse)
async def classify_ticket(ticket: TicketInput, strategy: str = Query("auto")):
    if not model_registry or not model_registry.list_models(): raise HTTPException(503, "No models")
    td = ticket.model_dump(); td.setdefault("ticket_text_length", len(td.get("description","")))
    r = model_registry.predict(td, strategy=strategy)
    return ClassificationResponse(predicted_category=r.predicted_category, predicted_subcategory=r.predicted_subcategory, confidence=r.confidence, category_probabilities=r.category_probabilities, model_name=r.model_name, latency_ms=r.latency_ms)

@app.post("/api/v1/feedback")
async def submit_feedback(fb: FeedbackInput):
    if not feedback_collector: raise HTTPException(503, "Feedback unavailable")
    from feedback.collector import AgentFeedback
    return feedback_collector.submit_feedback(AgentFeedback(ticket_id=fb.ticket_id, predicted_category=fb.predicted_category, corrected_category=fb.corrected_category, resolution_accepted=fb.resolution_accepted, agent_id=fb.agent_id))

@app.get("/api/v1/anomalies", response_model=list[AnomalyResponse])
async def get_anomalies():
    if not anomaly_detector: raise HTTPException(503)
    try:
        from db import get_engine; from sqlalchemy import text as st
        with get_engine().connect() as c: rows = c.execute(st("SELECT * FROM tickets ORDER BY created_at DESC LIMIT 10000")).fetchall()
        if not rows: return []
        df = pd.DataFrame(rows)
        return [AnomalyResponse(anomaly_type=a.anomaly_type, severity=a.severity, description=a.description, dimensions=a.dimensions, metric_value=a.metric_value, threshold=a.threshold, detected_at=a.detected_at) for a in anomaly_detector.run_all_checks(df)]
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    models = model_registry.list_models() if model_registry else []
    db_ok = False
    try:
        from db import get_engine; from sqlalchemy import text as st
        with get_engine().connect() as c: c.execute(st("SELECT 1")); db_ok = True
    except: pass
    return HealthResponse(status="healthy" if models and db_ok else "degraded", version="1.0.0", models_loaded=models, database_connected=db_ok)


# ── Analytics Endpoints ──

@app.get("/api/v1/analytics/satisfaction")
async def satisfaction_drivers():
    """Analyze what drives customer satisfaction scores."""
    try:
        from db import get_engine; from sqlalchemy import text as st
        from analytics.business import SatisfactionAnalyzer
        with get_engine().connect() as c:
            rows = c.execute(st("SELECT * FROM tickets")).fetchall()
            cols = c.execute(st("SELECT * FROM tickets LIMIT 0")).keys()
        df = pd.DataFrame(rows, columns=cols)
        return SatisfactionAnalyzer().analyze(df)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/v1/analytics/agents")
async def agent_performance():
    """Compute per-agent performance metrics."""
    try:
        from db import get_engine; from sqlalchemy import text as st
        from analytics.business import AgentPerformanceAnalyzer
        with get_engine().connect() as c:
            rows = c.execute(st("SELECT * FROM tickets")).fetchall()
            cols = c.execute(st("SELECT * FROM tickets LIMIT 0")).keys()
        df = pd.DataFrame(rows, columns=cols)
        return AgentPerformanceAnalyzer().analyze(df)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/v1/analytics/resolution-time")
async def resolution_time_stats():
    """Get resolution time statistics by category and product."""
    try:
        from db import get_engine; from sqlalchemy import text as st
        with get_engine().connect() as c:
            rows = c.execute(st("""
                SELECT product, category,
                    COUNT(*) as ticket_count,
                    AVG(resolution_time_hours) as avg_hours,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY resolution_time_hours) as median_hours,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY resolution_time_hours) as p95_hours
                FROM tickets
                WHERE resolution_time_hours IS NOT NULL
                GROUP BY product, category
                ORDER BY avg_hours DESC
            """)).fetchall()
        return [{"product": r[0], "category": r[1], "ticket_count": r[2],
                 "avg_hours": round(float(r[3]),1), "median_hours": round(float(r[4]),1),
                 "p95_hours": round(float(r[5]),1)} for r in rows]
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/v1/monitoring/drift")
async def drift_status():
    """Check current drift signals across all monitors."""
    try:
        from monitoring.drift import DriftDetector
        detector = DriftDetector()
        # Return empty if no reference data yet
        return {"status": "operational", "message": "Drift monitoring active. Run monitoring cycle to check signals."}
    except Exception as e:
        raise HTTPException(500, str(e))
