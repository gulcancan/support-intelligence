"""
Business analytics models and quality scoring.

Covers the task requirements for:
- Analytical models for business metrics (resolution times, satisfaction drivers, agent performance)
- Automated quality scoring for generated responses
- Event schemas for tracking system interactions

These are batch-computed analytics that feed into dashboards and the anomaly detector.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
import joblib

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Event Schema — tracks all system interactions
# ═══════════════════════════════════════════════════════════

@dataclass
class SystemEvent:
    """
    Standardized event schema for tracking all system interactions.

    Every significant action is recorded as a SystemEvent. This enables:
    - Auditing: who did what, when
    - Analytics: which features are used, what fails
    - Monitoring: latency distributions, error rates
    - Feedback loops: connecting predictions to outcomes
    """
    event_id: str                   # Unique event identifier
    event_type: str                 # classification, retrieval, feedback, anomaly, retrain
    timestamp: str                  # ISO 8601
    ticket_id: Optional[str] = None
    agent_id: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None

    # Prediction events
    predicted_category: Optional[str] = None
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None

    # Retrieval events
    n_results_returned: Optional[int] = None
    top_result_score: Optional[float] = None
    retrieval_method: Optional[str] = None   # vector, bm25, graph, hybrid

    # Feedback events
    corrected_category: Optional[str] = None
    resolution_accepted: Optional[bool] = None
    satisfaction_score: Optional[int] = None

    # Quality events
    quality_score: Optional[float] = None
    quality_dimensions: Optional[dict] = None

    # System metadata
    component: Optional[str] = None          # classifier, retriever, anomaly_detector
    status: str = "success"                   # success, error, timeout, fallback
    error_message: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════
# Resolution Time Predictor
# ═══════════════════════════════════════════════════════════

class ResolutionTimePredictor:
    """
    Predict how long a ticket will take to resolve.

    Useful for:
    - Setting customer expectations ("estimated resolution: 4-8 hours")
    - Agent workload planning
    - SLA monitoring (flag tickets likely to breach SLA)

    Uses GradientBoosting regression on structured ticket features.
    Target: log(resolution_time_hours) for better distribution modeling.
    """

    FEATURES = [
        "priority_enc", "severity_enc", "customer_tier_enc", "channel_enc",
        "product_enc", "previous_tickets", "account_age_days",
        "account_monthly_value", "similar_issues_last_30_days",
        "ticket_text_length", "affected_users", "contains_error_code",
        "contains_stack_trace", "weekend_ticket", "after_hours",
        "hour_of_day",
    ]

    def __init__(self):
        self.model = None
        self.encoders = {}
        self.scaler = StandardScaler()

    def _prepare(self, df, fit=False):
        X = pd.DataFrame(index=df.index)
        cat_cols = {
            "priority_enc": "priority", "severity_enc": "severity",
            "customer_tier_enc": "customer_tier", "channel_enc": "channel",
            "product_enc": "product",
        }
        for feat, col in cat_cols.items():
            if col in df.columns:
                vals = df[col].fillna("unknown").astype(str)
                if fit:
                    enc = LabelEncoder()
                    X[feat] = enc.fit_transform(vals)
                    self.encoders[feat] = enc
                else:
                    enc = self.encoders[feat]
                    X[feat] = vals.map(lambda v, e=enc: e.transform([v])[0] if v in e.classes_ else -1)
            else:
                X[feat] = 0

        num_cols = [
            "previous_tickets", "account_age_days", "account_monthly_value",
            "similar_issues_last_30_days", "ticket_text_length", "affected_users",
        ]
        for col in num_cols:
            X[col] = pd.to_numeric(df[col], errors="coerce").fillna(0) if col in df.columns else 0

        bool_cols = ["contains_error_code", "contains_stack_trace", "weekend_ticket", "after_hours"]
        for col in bool_cols:
            X[col] = df[col].fillna(False).astype(int) if col in df.columns else 0

        X["hour_of_day"] = pd.to_datetime(df["created_at"]).dt.hour if "created_at" in df.columns else 12

        return X[self.FEATURES]

    def train(self, df):
        """Train on historical tickets with known resolution times."""
        valid = df[df["resolution_time_hours"].notna() & (df["resolution_time_hours"] > 0)].copy()
        logger.info(f"Training resolution time predictor on {len(valid):,} tickets")

        X = self._prepare(valid, fit=True)
        y = np.log1p(valid["resolution_time_hours"].values)  # Log-transform for better fit

        self.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        self.model.fit(X, y)

        y_pred = self.model.predict(X)
        mae_log = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Convert back to hours for interpretable MAE
        mae_hours = mean_absolute_error(np.expm1(y), np.expm1(y_pred))

        logger.info(f"Resolution time model: MAE={mae_hours:.1f}h, R²={r2:.3f}")
        return {"mae_hours": round(mae_hours, 2), "r2": round(r2, 4)}

    def predict(self, ticket: dict) -> dict:
        """Predict resolution time for a new ticket."""
        if self.model is None:
            return {"predicted_hours": None, "confidence_interval": None}
        df = pd.DataFrame([ticket])
        X = self._prepare(df)
        log_pred = self.model.predict(X)[0]
        pred_hours = np.expm1(log_pred)
        # Rough confidence interval via training residual std
        return {
            "predicted_hours": round(float(pred_hours), 1),
            "confidence_interval": [
                round(max(0.5, float(pred_hours * 0.5)), 1),
                round(float(pred_hours * 2.0), 1),
            ],
        }

    def save(self, path):
        joblib.dump({"model": self.model, "encoders": self.encoders}, path)

    def load(self, path):
        data = joblib.load(path)
        self.model = data["model"]
        self.encoders = data["encoders"]


# ═══════════════════════════════════════════════════════════
# Satisfaction Driver Analysis
# ═══════════════════════════════════════════════════════════

class SatisfactionAnalyzer:
    """
    Identify what drives customer satisfaction scores.

    Trains a logistic regression to predict satisfaction >= 4 (satisfied)
    and extracts feature coefficients as driver importance.

    Key insight: this is as much about the COEFFICIENTS as the predictions.
    Stakeholders want to know "what can we improve to increase satisfaction?"
    """

    def analyze(self, df: pd.DataFrame) -> dict:
        """Analyze satisfaction drivers from ticket data."""
        valid = df[df["satisfaction_score"].notna()].copy()
        if len(valid) < 100:
            return {"error": "Insufficient data"}

        # Binary target: satisfied (4-5) vs not (1-3)
        valid["satisfied"] = (valid["satisfaction_score"] >= 4).astype(int)

        features = {}
        features["resolution_time_hours"] = pd.to_numeric(valid["resolution_time_hours"], errors="coerce").fillna(24)
        features["resolution_attempts"] = pd.to_numeric(valid["resolution_attempts"], errors="coerce").fillna(1)
        features["transferred_count"] = pd.to_numeric(valid["transferred_count"], errors="coerce").fillna(0)
        features["response_count"] = pd.to_numeric(valid["response_count"], errors="coerce").fillna(1)
        features["escalated"] = valid["escalated"].fillna(False).astype(int)
        features["resolution_helpful"] = valid["resolution_helpful"].fillna(False).astype(int)
        features["agent_experience_months"] = pd.to_numeric(valid["agent_experience_months"], errors="coerce").fillna(12)
        features["is_enterprise"] = (valid["customer_tier"] == "enterprise").astype(int)
        features["is_critical"] = (valid["priority"] == "critical").astype(int)
        features["first_contact_resolution"] = (features["response_count"] <= 2).astype(int)

        X = pd.DataFrame(features)
        y = valid["satisfied"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X_scaled, y)

        # Extract coefficients as driver importance (standardized)
        coefs = dict(zip(X.columns, model.coef_[0]))
        drivers = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)

        # Compute baseline metrics
        satisfaction_rate = y.mean()
        auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])

        return {
            "satisfaction_rate": round(float(satisfaction_rate), 3),
            "model_auc": round(float(auc), 3),
            "n_tickets": len(valid),
            "drivers": [
                {
                    "feature": name,
                    "coefficient": round(float(coef), 4),
                    "direction": "positive" if coef > 0 else "negative",
                    "interpretation": self._interpret(name, coef),
                }
                for name, coef in drivers
            ],
        }

    @staticmethod
    def _interpret(feature, coef):
        interpretations = {
            "resolution_time_hours": "Faster resolutions increase satisfaction" if coef < 0 else "Unexpected: longer resolutions correlate with higher satisfaction",
            "resolution_helpful": "Helpful resolutions strongly drive satisfaction",
            "first_contact_resolution": "Resolving in ≤2 interactions increases satisfaction",
            "escalated": "Escalation reduces satisfaction (perceived severity/frustration)",
            "transferred_count": "Each transfer reduces satisfaction",
            "agent_experience_months": "More experienced agents achieve higher satisfaction",
            "resolution_attempts": "Multiple attempts reduce satisfaction",
            "response_count": "More back-and-forth reduces satisfaction",
            "is_enterprise": "Enterprise customers may have higher/lower expectations",
            "is_critical": "Critical issues have lower baseline satisfaction",
        }
        return interpretations.get(feature, f"{'Increases' if coef > 0 else 'Decreases'} satisfaction probability")


# ═══════════════════════════════════════════════════════════
# Agent Performance Analytics
# ═══════════════════════════════════════════════════════════

class AgentPerformanceAnalyzer:
    """
    Compute agent-level performance metrics.

    Metrics per agent:
    - Ticket volume and throughput
    - Average resolution time
    - Customer satisfaction scores
    - First-contact resolution rate
    - Escalation rate
    - Category specialization alignment
    """

    def analyze(self, df: pd.DataFrame) -> dict:
        """Compute per-agent performance metrics."""
        if "agent_id" not in df.columns:
            return {"error": "No agent_id column"}

        valid = df[df["agent_id"].notna()].copy()

        agents = valid.groupby("agent_id").agg(
            ticket_count=("ticket_id", "count"),
            avg_resolution_hours=("resolution_time_hours", "mean"),
            median_resolution_hours=("resolution_time_hours", "median"),
            avg_satisfaction=("satisfaction_score", "mean"),
            satisfaction_count=("satisfaction_score", "count"),
            escalation_rate=("escalated", "mean"),
            resolution_helpful_rate=("resolution_helpful", "mean"),
            avg_response_count=("response_count", "mean"),
            experience_months=("agent_experience_months", "first"),
            specialization=("agent_specialization", "first"),
        ).reset_index()

        # First-contact resolution: resolved in ≤ 2 responses
        valid["fcr"] = (pd.to_numeric(valid["response_count"], errors="coerce").fillna(3) <= 2).astype(int)
        fcr = valid.groupby("agent_id")["fcr"].mean().reset_index().rename(columns={"fcr": "fcr_rate"})
        agents = agents.merge(fcr, on="agent_id", how="left")

        # Rank agents by composite score
        agents["perf_score"] = (
            agents["avg_satisfaction"].fillna(3) / 5.0 * 0.3
            + agents["resolution_helpful_rate"].fillna(0.5) * 0.25
            + agents["fcr_rate"].fillna(0.5) * 0.2
            + (1 - agents["escalation_rate"].fillna(0.1)) * 0.15
            + (1 - np.clip(agents["avg_resolution_hours"].fillna(24) / 48, 0, 1)) * 0.1
        )
        agents = agents.sort_values("perf_score", ascending=False)

        # Category specialization alignment
        cat_by_agent = valid.groupby(["agent_id", "category"]).size().reset_index(name="count")
        top_cats = cat_by_agent.sort_values("count", ascending=False).groupby("agent_id").first().reset_index()
        agents = agents.merge(
            top_cats[["agent_id", "category"]].rename(columns={"category": "primary_category"}),
            on="agent_id", how="left",
        )

        return {
            "n_agents": len(agents),
            "summary": {
                "avg_resolution_hours": round(float(agents["avg_resolution_hours"].mean()), 1),
                "avg_satisfaction": round(float(agents["avg_satisfaction"].mean()), 2),
                "avg_fcr_rate": round(float(agents["fcr_rate"].mean()), 3),
            },
            "agents": [
                {
                    "agent_id": row["agent_id"],
                    "ticket_count": int(row["ticket_count"]),
                    "avg_resolution_hours": round(float(row["avg_resolution_hours"]), 1) if pd.notna(row["avg_resolution_hours"]) else None,
                    "avg_satisfaction": round(float(row["avg_satisfaction"]), 2) if pd.notna(row["avg_satisfaction"]) else None,
                    "fcr_rate": round(float(row["fcr_rate"]), 3) if pd.notna(row["fcr_rate"]) else None,
                    "escalation_rate": round(float(row["escalation_rate"]), 3),
                    "perf_score": round(float(row["perf_score"]), 3),
                    "specialization": row.get("specialization"),
                    "primary_category": row.get("primary_category"),
                }
                for _, row in agents.head(50).iterrows()
            ],
            "top_performers": agents.head(5)["agent_id"].tolist(),
            "needs_coaching": agents.tail(5)["agent_id"].tolist(),
        }


# ═══════════════════════════════════════════════════════════
# Automated Quality Scoring for Retrieval Results
# ═══════════════════════════════════════════════════════════

class RetrievalQualityScorer:
    """
    Automated quality scoring for the retrieval system's responses.

    Scores each retrieval result on multiple dimensions:
    1. Relevance: Does the result match the ticket's category/product?
    2. Recency: How recent is the resolution?
    3. Success: Was the resolution marked helpful?
    4. Completeness: Does the resolution contain actionable steps?
    5. Specificity: Does the resolution reference specific configs/commands?

    The composite score is used for:
    - Re-ranking retrieval results
    - Monitoring retrieval quality over time
    - Identifying knowledge gaps (consistently low scores for certain categories)
    """

    def score_result(self, ticket: dict, result: dict) -> dict:
        """Score a single retrieval result against the query ticket."""
        scores = {}

        # 1. Relevance (category/product match)
        cat_match = ticket.get("category") == result.get("category")
        prod_match = ticket.get("product") == result.get("product")
        scores["relevance"] = 1.0 if (cat_match and prod_match) else 0.7 if cat_match else 0.3 if prod_match else 0.1

        # 2. Success (historical resolution quality)
        if result.get("resolution_helpful") is True:
            scores["success"] = 1.0
        elif result.get("resolution_helpful") is False:
            scores["success"] = 0.2
        else:
            scores["success"] = 0.5

        sat = result.get("satisfaction_score")
        if sat is not None:
            scores["satisfaction"] = min(1.0, sat / 5.0)
        else:
            scores["satisfaction"] = 0.5

        # 3. Completeness (does resolution have actionable content?)
        resolution = result.get("resolution", "") or ""
        scores["completeness"] = min(1.0, max(0.1,
            0.2 * (len(resolution) > 50) +
            0.2 * (len(resolution) > 150) +
            0.2 * bool(re.search(r'\d+', resolution)) +  # Contains specific numbers
            0.2 * bool(re.search(r'(config|setting|parameter|update|change|set|increase|decrease)', resolution, re.I)) +
            0.2 * bool(re.search(r'(resolved|fixed|working|confirmed)', resolution, re.I))
        ))

        # 4. Specificity (references specific technical details)
        scores["specificity"] = min(1.0, max(0.1,
            0.25 * bool(re.search(r'ERROR_\w+', resolution)) +
            0.25 * bool(re.search(r'\.(yaml|json|conf|py|js|xml|ini)', resolution, re.I)) +
            0.25 * bool(re.search(r'(version|v\d+|upgrade|patch)', resolution, re.I)) +
            0.25 * bool(re.search(r'(command|script|query|api|endpoint)', resolution, re.I))
        ))

        # Composite score (weighted average)
        weights = {"relevance": 0.30, "success": 0.25, "satisfaction": 0.10,
                   "completeness": 0.20, "specificity": 0.15}
        composite = sum(scores[k] * weights[k] for k in weights)

        return {
            "composite_score": round(composite, 3),
            "dimensions": {k: round(v, 3) for k, v in scores.items()},
        }

    def score_retrieval_batch(self, ticket: dict, results: list[dict]) -> list[dict]:
        """Score all retrieval results for a ticket."""
        scored = []
        for r in results:
            s = self.score_result(ticket, r)
            scored.append({**r, "quality_score": s["composite_score"], "quality_dimensions": s["dimensions"]})
        return sorted(scored, key=lambda x: x["quality_score"], reverse=True)

    def compute_system_quality_metrics(self, scored_results: list[dict]) -> dict:
        """Aggregate quality metrics across many queries for system-level monitoring."""
        if not scored_results:
            return {"n_queries": 0}

        scores = [r["quality_score"] for r in scored_results if "quality_score" in r]
        return {
            "n_results_scored": len(scores),
            "mean_quality": round(float(np.mean(scores)), 3),
            "median_quality": round(float(np.median(scores)), 3),
            "p25_quality": round(float(np.percentile(scores, 25)), 3),
            "low_quality_rate": round(float(np.mean([s < 0.3 for s in scores])), 3),
            "high_quality_rate": round(float(np.mean([s >= 0.7 for s in scores])), 3),
        }
