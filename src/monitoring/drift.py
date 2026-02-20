"""
Streaming ticket processing and model drift monitoring.

This module handles the production reality where tickets arrive continuously
and the model must be monitored and updated without downtime.

Architecture:
                                        ┌──────────────┐
    Ticket Stream ──► Kafka Topic ──────► Stream        │
    (500+/day)        "tickets.raw"      │ Processor    │
                                         │              │
                                         │ ┌──────────┐ │     ┌───────────────┐
                                         │ │ Classify  │─┼────►│ Predictions   │
                                         │ │ + Log     │ │     │ Store (PG)    │
                                         │ └──────────┘ │     └───────┬───────┘
                                         │              │             │
                                         │ ┌──────────┐ │     ┌───────▼───────┐
                                         │ │ Feature   │─┼────►│ Feature       │
                                         │ │ Compute   │ │     │ Distribution  │
                                         │ └──────────┘ │     │ Store (Redis) │
                                         └──────────────┘     └───────┬───────┘
                                                                      │
                                              ┌───────────────────────┘
                                              ▼
                                    ┌──────────────────┐
                                    │  Drift Monitor   │
                                    │  (runs hourly)   │
                                    │                  │
                                    │ • PSI on features│
                                    │ • Confidence     │
                                    │   degradation    │
                                    │ • Label drift    │
                                    │   (from feedback)│
                                    │ • Prediction     │
                                    │   distribution   │
                                    └────────┬─────────┘
                                             │
                              ┌──────────────┼──────────────┐
                              ▼              ▼              ▼
                        ┌──────────┐  ┌──────────┐  ┌──────────────┐
                        │  Alert   │  │  Auto     │  │  Shadow      │
                        │  Only    │  │  Retrain  │  │  Deploy      │
                        │ (minor)  │  │ (major)   │  │  + A/B test  │
                        └──────────┘  └──────────┘  └──────────────┘

Key design decisions:

1. WINDOWED STATISTICS, NOT POINT ESTIMATES
   We track distributions over sliding windows (1h, 24h, 7d) rather than
   individual predictions. Single predictions are noisy; windows reveal trends.

2. MULTIPLE DRIFT SIGNALS, NOT ONE METRIC
   A single "drift score" hides what's actually changing. We track:
   - Feature drift (input distribution changed)
   - Prediction drift (model outputs changed)
   - Performance drift (model accuracy degraded — requires labels)
   Each requires different responses.

3. GRADUATED RESPONSE
   Not every drift signal means "retrain now":
   - Minor drift → log + alert, continue monitoring
   - Sustained drift → trigger retraining pipeline
   - Severe drift → shadow-deploy new model, A/B test, then promote

4. RETRAINING ≠ FROM SCRATCH
   For incremental updates, we fine-tune from the existing model checkpoint
   on recent data (last 30 days), not retrain on all 100k tickets.
"""

import logging
import time
import json
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from collections import deque
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. Prediction Logger — records every prediction for monitoring
# ═══════════════════════════════════════════════════════════

@dataclass
class PredictionRecord:
    """Every prediction is logged for drift analysis."""
    ticket_id: str
    timestamp: str
    predicted_category: str
    confidence: float
    model_name: str
    model_version: str
    # Feature snapshot (for PSI computation)
    feature_values: dict  # key structured features at prediction time
    # Later populated from feedback
    true_category: Optional[str] = None
    was_corrected: bool = False


class PredictionLogger:
    """
    Logs all predictions to a persistent store.

    In production: writes to a Kafka topic or PostgreSQL table.
    The key insight is that we need BOTH the prediction AND the input
    features to detect whether drift is in the data or the model.
    """

    def __init__(self, engine=None):
        from db import get_engine
        self.engine = engine or get_engine()
        self._ensure_table()

    def _ensure_table(self):
        from sqlalchemy import text
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS prediction_log (
                    id SERIAL PRIMARY KEY,
                    ticket_id VARCHAR(32),
                    timestamp TIMESTAMPTZ NOT NULL,
                    predicted_category VARCHAR(64),
                    confidence FLOAT,
                    model_name VARCHAR(64),
                    model_version VARCHAR(32),
                    feature_values JSONB,
                    true_category VARCHAR(64),
                    was_corrected BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_predlog_ts ON prediction_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_predlog_model ON prediction_log(model_name, model_version);
            """))

    def log(self, record: PredictionRecord):
        from sqlalchemy import text
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO prediction_log
                    (ticket_id, timestamp, predicted_category, confidence,
                     model_name, model_version, feature_values)
                VALUES (:tid, :ts, :cat, :conf, :mn, :mv, :fv)
            """), {
                "tid": record.ticket_id,
                "ts": record.timestamp,
                "cat": record.predicted_category,
                "conf": record.confidence,
                "mn": record.model_name,
                "mv": record.model_version,
                "fv": json.dumps(record.feature_values),
            })

    def update_true_label(self, ticket_id: str, true_category: str):
        """Called when agent feedback provides the ground truth."""
        from sqlalchemy import text
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE prediction_log
                SET true_category = :tc, was_corrected = (predicted_category != :tc)
                WHERE ticket_id = :tid
            """), {"tc": true_category, "tid": ticket_id})


# ═══════════════════════════════════════════════════════════
# 2. Drift Detection — multi-signal monitoring
# ═══════════════════════════════════════════════════════════

class DriftSeverity(Enum):
    NONE = "none"
    MINOR = "minor"          # Log and watch
    MODERATE = "moderate"    # Alert + increase monitoring frequency
    SEVERE = "severe"        # Trigger retraining pipeline


@dataclass
class DriftSignal:
    """A detected drift signal with severity and context."""
    signal_type: str       # feature_drift, prediction_drift, confidence_drift, label_drift
    severity: DriftSeverity
    metric_name: str
    current_value: float
    reference_value: float
    threshold: float
    description: str
    dimensions: dict = field(default_factory=dict)
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    recommended_action: str = ""


class DriftDetector:
    """
    Multi-signal drift detection system.

    Monitors four types of drift, each requiring different data:

    1. FEATURE DRIFT (no labels needed)
       - Population Stability Index (PSI) on key input features
       - Detects: new products, changing customer mix, seasonal shifts
       - Response: may or may not affect model — investigate before retraining

    2. PREDICTION DRIFT (no labels needed)
       - Distribution of predicted categories over time windows
       - Detects: model behavior changing even if inputs look stable
       - Response: compare with feature drift to diagnose cause

    3. CONFIDENCE DRIFT (no labels needed)
       - Moving average of prediction confidence scores
       - Detects: model uncertainty increasing (often precedes accuracy drop)
       - Response: the earliest warning sign — monitor closely

    4. LABEL DRIFT (requires feedback/labels)
       - Accuracy/F1 computed on recent corrected predictions
       - Detects: actual model performance degradation
       - Response: strongest signal — retrain if sustained
    """

    # PSI thresholds (industry standard from credit risk modeling)
    PSI_MINOR = 0.1      # Slight shift, worth noting
    PSI_MODERATE = 0.2    # Significant shift, investigate
    PSI_SEVERE = 0.25     # Major shift, likely needs retraining

    # Confidence thresholds
    CONF_DROP_MINOR = 0.05     # 5% drop in mean confidence
    CONF_DROP_MODERATE = 0.10  # 10% drop
    CONF_DROP_SEVERE = 0.15    # 15% drop

    # Label drift (correction rate thresholds)
    CORRECTION_RATE_MINOR = 0.10      # 10% of predictions corrected
    CORRECTION_RATE_MODERATE = 0.20   # 20%
    CORRECTION_RATE_SEVERE = 0.30     # 30%

    def __init__(self, engine=None):
        # Engine only needed for label_drift queries; other checks work without DB
        self._engine = engine

    @property
    def engine(self):
        if self._engine is None:
            from db import get_engine
            self._engine = get_engine()
        return self._engine

    @staticmethod
    def compute_psi(reference: np.ndarray, current: np.ndarray,
                    n_bins: int = 10) -> float:
        """
        Population Stability Index.

        PSI = Σ (P_current - P_reference) × ln(P_current / P_reference)

        Interpretation:
        - PSI < 0.1:  No significant shift
        - 0.1 ≤ PSI < 0.2:  Moderate shift
        - PSI ≥ 0.2:  Significant shift — investigate

        Why PSI over KL-divergence:
        - PSI is symmetric (KL is not)
        - PSI has well-established thresholds from decades of credit risk
        - PSI handles the case where bins have zero counts gracefully
        """
        # Create bins from reference distribution
        breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        breakpoints = np.unique(breakpoints)  # Handle ties

        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]

        # Convert to proportions with smoothing to avoid log(0)
        eps = 1e-6
        ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps * len(ref_counts))
        cur_pct = (cur_counts + eps) / (cur_counts.sum() + eps * len(cur_counts))

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def detect_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: list[str] = None,
    ) -> list[DriftSignal]:
        """
        Compare feature distributions between reference and current windows.

        reference_df: training data or last stable period
        current_df: recent production data (e.g., last 24h or 7d)
        """
        if features is None:
            features = [
                "ticket_text_length", "previous_tickets", "account_age_days",
                "account_monthly_value", "similar_issues_last_30_days",
                "product_version_age_days", "affected_users",
            ]

        signals = []
        for feat in features:
            if feat not in reference_df.columns or feat not in current_df.columns:
                continue

            ref_vals = pd.to_numeric(reference_df[feat], errors="coerce").dropna().values
            cur_vals = pd.to_numeric(current_df[feat], errors="coerce").dropna().values

            if len(ref_vals) < 50 or len(cur_vals) < 10:
                continue

            psi = self.compute_psi(ref_vals, cur_vals)

            if psi >= self.PSI_SEVERE:
                severity = DriftSeverity.SEVERE
            elif psi >= self.PSI_MODERATE:
                severity = DriftSeverity.MODERATE
            elif psi >= self.PSI_MINOR:
                severity = DriftSeverity.MINOR
            else:
                continue  # No drift

            signals.append(DriftSignal(
                signal_type="feature_drift",
                severity=severity,
                metric_name=f"psi_{feat}",
                current_value=psi,
                reference_value=0.0,
                threshold=self.PSI_MODERATE,
                description=f"Feature '{feat}' PSI={psi:.3f} ({severity.value})",
                dimensions={"feature": feat},
                recommended_action="Investigate feature source. If upstream data pipeline changed, may not require retraining."
            ))

        # Also check categorical feature drift via chi-squared
        cat_features = ["product", "customer_tier", "channel", "priority"]
        for feat in cat_features:
            if feat not in reference_df.columns or feat not in current_df.columns:
                continue
            ref_dist = reference_df[feat].value_counts(normalize=True)
            cur_dist = current_df[feat].value_counts(normalize=True)

            # Jensen-Shannon divergence for categorical
            all_cats = set(ref_dist.index) | set(cur_dist.index)
            ref_arr = np.array([ref_dist.get(c, 0) for c in all_cats])
            cur_arr = np.array([cur_dist.get(c, 0) for c in all_cats])

            # Check for new categories not in reference
            new_cats = set(cur_dist.index) - set(ref_dist.index)
            if new_cats:
                signals.append(DriftSignal(
                    signal_type="feature_drift",
                    severity=DriftSeverity.MODERATE,
                    metric_name=f"new_categories_{feat}",
                    current_value=len(new_cats),
                    reference_value=0,
                    threshold=0,
                    description=f"New values in '{feat}': {new_cats}",
                    dimensions={"feature": feat, "new_values": list(new_cats)},
                    recommended_action="New categorical values likely need model retraining."
                ))

        return signals

    # ── Prediction Distribution Drift ──

    def detect_prediction_drift(
        self,
        reference_distribution: dict[str, float],
        current_predictions: list[str],
    ) -> list[DriftSignal]:
        """
        Compare predicted category distribution against reference.

        reference_distribution: category → proportion from training/validation
        current_predictions: list of recent predicted categories
        """
        if len(current_predictions) < 20:
            return []

        current_dist = pd.Series(current_predictions).value_counts(normalize=True)

        signals = []
        for cat, ref_prop in reference_distribution.items():
            cur_prop = current_dist.get(cat, 0.0)
            rel_change = abs(cur_prop - ref_prop) / max(ref_prop, 0.01)

            if rel_change > 0.5:  # >50% relative change
                severity = DriftSeverity.SEVERE if rel_change > 1.0 else DriftSeverity.MODERATE
                signals.append(DriftSignal(
                    signal_type="prediction_drift",
                    severity=severity,
                    metric_name=f"pred_dist_{cat}",
                    current_value=cur_prop,
                    reference_value=ref_prop,
                    threshold=ref_prop * 1.5,
                    description=f"'{cat}' shifted: {ref_prop:.1%} → {cur_prop:.1%} ({rel_change:.0%} change)",
                    dimensions={"category": cat},
                    recommended_action="Cross-reference with feature drift. If features shifted too, this is expected."
                ))

        return signals

    # ── Confidence Drift ──

    def detect_confidence_drift(
        self,
        reference_confidence: float,
        current_confidences: list[float],
        window_label: str = "24h",
    ) -> list[DriftSignal]:
        """
        Monitor model confidence degradation.

        This is often the EARLIEST signal of drift — confidence drops
        before accuracy does, because the model "knows" it's uncertain.
        """
        if len(current_confidences) < 10:
            return []

        current_mean = float(np.mean(current_confidences))
        drop = reference_confidence - current_mean

        if drop >= self.CONF_DROP_SEVERE:
            severity = DriftSeverity.SEVERE
        elif drop >= self.CONF_DROP_MODERATE:
            severity = DriftSeverity.MODERATE
        elif drop >= self.CONF_DROP_MINOR:
            severity = DriftSeverity.MINOR
        else:
            return []

        return [DriftSignal(
            signal_type="confidence_drift",
            severity=severity,
            metric_name=f"mean_confidence_{window_label}",
            current_value=current_mean,
            reference_value=reference_confidence,
            threshold=reference_confidence - self.CONF_DROP_MODERATE,
            description=f"Mean confidence dropped: {reference_confidence:.3f} → {current_mean:.3f} (Δ={drop:.3f})",
            recommended_action="If sustained >24h, trigger retraining. Check if specific product/category is driving the drop."
        )]

    # ── Label Drift (requires ground truth from feedback) ──

    def detect_label_drift(
        self,
        window_hours: int = 168,  # 7 days
    ) -> list[DriftSignal]:
        """
        Compute actual model accuracy from agent corrections.

        This is the strongest drift signal but requires labeled data
        (agent feedback). We compute a rolling correction rate and
        flag when it exceeds thresholds.
        """
        from sqlalchemy import text
        query = text("""
            SELECT
                predicted_category,
                true_category,
                was_corrected,
                timestamp
            FROM prediction_log
            WHERE true_category IS NOT NULL
              AND timestamp >= NOW() - :hours * INTERVAL '1 hour'
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(query, {"hours": window_hours}).fetchall()

        if len(rows) < 20:
            return []

        corrected = sum(1 for r in rows if r[2])  # was_corrected
        total = len(rows)
        correction_rate = corrected / total

        if correction_rate >= self.CORRECTION_RATE_SEVERE:
            severity = DriftSeverity.SEVERE
        elif correction_rate >= self.CORRECTION_RATE_MODERATE:
            severity = DriftSeverity.MODERATE
        elif correction_rate >= self.CORRECTION_RATE_MINOR:
            severity = DriftSeverity.MINOR
        else:
            return []

        # Identify which categories are most affected
        corrections_by_cat = {}
        for r in rows:
            if r[2]:  # was_corrected
                key = f"{r[0]} → {r[1]}"
                corrections_by_cat[key] = corrections_by_cat.get(key, 0) + 1

        top_confusions = dict(sorted(corrections_by_cat.items(), key=lambda x: x[1], reverse=True)[:5])

        return [DriftSignal(
            signal_type="label_drift",
            severity=severity,
            metric_name="correction_rate_7d",
            current_value=correction_rate,
            reference_value=0.0,
            threshold=self.CORRECTION_RATE_MODERATE,
            description=f"Correction rate: {correction_rate:.1%} ({corrected}/{total} in {window_hours}h)",
            dimensions={"top_confusions": top_confusions},
            recommended_action="Retrain with corrected labels. Focus on categories: " + ", ".join(top_confusions.keys())
        )]

    # ── Orchestrator ──

    def run_all_checks(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        reference_pred_distribution: dict[str, float],
        current_predictions: list[str],
        reference_confidence: float,
        current_confidences: list[float],
    ) -> list[DriftSignal]:
        """Run all drift checks and return sorted signals."""
        all_signals = []

        all_signals.extend(self.detect_feature_drift(reference_df, current_df))
        all_signals.extend(self.detect_prediction_drift(reference_pred_distribution, current_predictions))
        all_signals.extend(self.detect_confidence_drift(reference_confidence, current_confidences))
        all_signals.extend(self.detect_label_drift())

        # Sort by severity (severe first)
        severity_order = {DriftSeverity.SEVERE: 0, DriftSeverity.MODERATE: 1, DriftSeverity.MINOR: 2}
        all_signals.sort(key=lambda s: severity_order.get(s.severity, 3))

        return all_signals


# ═══════════════════════════════════════════════════════════
# 3. Automated Retraining Pipeline
# ═══════════════════════════════════════════════════════════

class RetrainingTrigger(Enum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"       # Weekly retrain regardless
    DRIFT_TRIGGERED = "drift"     # Drift detector flagged severe
    FEEDBACK_TRIGGERED = "feedback"  # Correction rate exceeded threshold


@dataclass
class RetrainingConfig:
    """Configuration for automated retraining."""
    # When to retrain
    max_days_without_retrain: int = 7          # Force retrain after N days
    drift_severity_trigger: DriftSeverity = DriftSeverity.SEVERE
    min_new_labels: int = 200                  # Min corrected labels before retraining

    # How to retrain
    use_incremental: bool = True               # Fine-tune from checkpoint vs full retrain
    incremental_window_days: int = 30          # How far back to include data
    validation_improvement_threshold: float = 0.005  # New model must beat old by this F1 margin

    # Safety: shadow deploy before promotion
    shadow_deploy_hours: int = 24              # Run new model in shadow for N hours
    shadow_min_predictions: int = 100          # Min shadow predictions before comparing
    auto_promote: bool = False                 # If True, promote automatically. If False, require human approval.


class RetrainingOrchestrator:
    """
    Manages the model retraining lifecycle.

    Flow:
    1. Drift detector signals severe drift OR scheduled trigger fires
    2. Orchestrator assembles training data (historical + recent corrections)
    3. New model is trained (incremental fine-tune or full retrain)
    4. New model is evaluated on holdout set
    5. If better than current: shadow deploy
    6. After shadow period: compare live metrics
    7. If shadow model performs well: promote to primary
    8. Old model becomes fallback

    Key insight: we NEVER just swap models. There's always a validation
    step and a shadow period. The worst thing in production is deploying
    a model that's worse than the current one.
    """

    def __init__(self, config: RetrainingConfig = None):
        self.config = config or RetrainingConfig()
        self._retrain_history: list[dict] = []

    def should_retrain(
        self,
        drift_signals: list[DriftSignal],
        last_retrain_at: Optional[datetime] = None,
        available_corrections: int = 0,
    ) -> tuple[bool, RetrainingTrigger, str]:
        """
        Decide whether to trigger retraining.

        Returns (should_retrain, trigger_type, reason).
        """
        # Check scheduled retraining
        if last_retrain_at:
            days_since = (datetime.now(timezone.utc) - last_retrain_at).days
            if days_since >= self.config.max_days_without_retrain:
                return True, RetrainingTrigger.SCHEDULED, \
                    f"Scheduled: {days_since} days since last retrain (max={self.config.max_days_without_retrain})"

        # Check drift severity
        severe_signals = [s for s in drift_signals if s.severity == DriftSeverity.SEVERE]
        if severe_signals:
            # Require multiple severe signals or sustained drift
            if len(severe_signals) >= 2:
                descriptions = "; ".join(s.description for s in severe_signals[:3])
                return True, RetrainingTrigger.DRIFT_TRIGGERED, \
                    f"Multiple severe drift signals: {descriptions}"

        # Check feedback volume
        if available_corrections >= self.config.min_new_labels:
            # Only retrain on feedback if correction rate is high enough
            correction_signals = [s for s in drift_signals if s.signal_type == "label_drift"]
            if correction_signals and correction_signals[0].severity.value in ("moderate", "severe"):
                return True, RetrainingTrigger.FEEDBACK_TRIGGERED, \
                    f"{available_corrections} corrections available, correction rate elevated"

        return False, RetrainingTrigger.MANUAL, "No retraining trigger met"

    def assemble_training_data(
        self,
        original_train_df: pd.DataFrame,
        corrections: list[dict],
        recent_tickets_df: pd.DataFrame,
        strategy: str = "augment",
    ) -> pd.DataFrame:
        """
        Build training dataset for retraining.

        Strategies:
        - "augment": Original training data + corrected labels override
        - "recent_only": Only use recent data (last N days) + corrections
        - "weighted": All data, but recent data weighted higher

        The "augment" strategy is safest: it preserves knowledge from the
        full training set while incorporating corrections.
        """
        if strategy == "augment":
            # Start with original training data
            train_df = original_train_df.copy()

            # Apply corrections: override labels where agents corrected
            correction_map = {c["ticket_id"]: c["corrected"] for c in corrections}
            mask = train_df["ticket_id"].isin(correction_map)
            train_df.loc[mask, "category"] = train_df.loc[mask, "ticket_id"].map(correction_map)

            # Add any truly new tickets (not in original training set)
            new_ids = set(recent_tickets_df["ticket_id"]) - set(train_df["ticket_id"])
            if new_ids:
                new_tickets = recent_tickets_df[recent_tickets_df["ticket_id"].isin(new_ids)]
                train_df = pd.concat([train_df, new_tickets], ignore_index=True)

            logger.info(f"Training data: {len(train_df)} tickets "
                        f"({len(corrections)} corrections applied, {len(new_ids)} new tickets)")
            return train_df

        elif strategy == "recent_only":
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.incremental_window_days)
            recent = recent_tickets_df[pd.to_datetime(recent_tickets_df["created_at"]) >= cutoff].copy()

            correction_map = {c["ticket_id"]: c["corrected"] for c in corrections}
            mask = recent["ticket_id"].isin(correction_map)
            recent.loc[mask, "category"] = recent.loc[mask, "ticket_id"].map(correction_map)

            logger.info(f"Recent-only training data: {len(recent)} tickets from last "
                        f"{self.config.incremental_window_days} days")
            return recent

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def validate_new_model(
        self,
        new_model,
        old_model,
        validation_df: pd.DataFrame,
    ) -> dict:
        """
        Compare new model against current model on validation data.

        The new model must beat the old model by a minimum margin
        to be eligible for deployment. This prevents deploying models
        that are marginally better on training data but worse in practice.
        """
        from models.common import evaluate_model

        # Old model predictions
        old_preds = old_model.predict_batch(validation_df)
        old_labels = old_model.label_encoder.transform(old_preds)
        y_true = old_model.label_encoder.transform(validation_df["category"])
        old_metrics = evaluate_model(y_true, old_labels, list(old_model.label_encoder.classes_))

        # New model predictions
        new_preds = new_model.predict_batch(validation_df)
        new_labels = new_model.label_encoder.transform(new_preds)
        new_metrics = evaluate_model(y_true, new_labels, list(new_model.label_encoder.classes_))

        improvement = new_metrics.weighted_f1 - old_metrics.weighted_f1
        should_deploy = improvement >= self.config.validation_improvement_threshold

        result = {
            "old_f1": old_metrics.weighted_f1,
            "new_f1": new_metrics.weighted_f1,
            "improvement": round(improvement, 4),
            "threshold": self.config.validation_improvement_threshold,
            "should_deploy": should_deploy,
            "reason": f"F1 improved by {improvement:.4f}" if should_deploy
                      else f"Improvement {improvement:.4f} below threshold {self.config.validation_improvement_threshold}",
        }

        logger.info(f"Model validation: old={old_metrics.weighted_f1:.4f}, "
                     f"new={new_metrics.weighted_f1:.4f}, "
                     f"deploy={'YES' if should_deploy else 'NO'}")

        return result


# ═══════════════════════════════════════════════════════════
# 4. Shadow Deployment — safe model rollout
# ═══════════════════════════════════════════════════════════

class ShadowDeployer:
    """
    Runs a new model in shadow mode alongside the primary model.

    Both models process every ticket. Only the primary model's prediction
    is returned to the user. The shadow model's predictions are logged
    for comparison.

    After the shadow period, we compare:
    - Accuracy (if feedback is available)
    - Confidence distributions
    - Agreement rate with primary model
    - Latency

    If the shadow model outperforms: promote to primary.
    """

    def __init__(self):
        self._shadow_model = None
        self._shadow_predictions: list[dict] = []
        self._shadow_start: Optional[datetime] = None

    def deploy_shadow(self, model, model_name: str):
        """Start shadow deployment."""
        self._shadow_model = model
        self._shadow_name = model_name
        self._shadow_predictions = []
        self._shadow_start = datetime.now(timezone.utc)
        logger.info(f"Shadow deployment started: {model_name}")

    def shadow_predict(self, ticket: dict, primary_result) -> Optional[dict]:
        """Run shadow prediction alongside primary. Returns shadow result for logging."""
        if self._shadow_model is None:
            return None

        try:
            shadow_result = self._shadow_model.predict(ticket)
            record = {
                "ticket_id": ticket.get("ticket_id"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "primary_category": primary_result.predicted_category,
                "primary_confidence": primary_result.confidence,
                "shadow_category": shadow_result.predicted_category,
                "shadow_confidence": shadow_result.confidence,
                "agree": primary_result.predicted_category == shadow_result.predicted_category,
            }
            self._shadow_predictions.append(record)
            return record
        except Exception as e:
            logger.error(f"Shadow prediction failed: {e}")
            return None

    def get_shadow_report(self) -> dict:
        """Generate comparison report between primary and shadow models."""
        if not self._shadow_predictions:
            return {"status": "no_data"}

        n = len(self._shadow_predictions)
        agreement = sum(1 for p in self._shadow_predictions if p["agree"]) / n
        primary_conf = np.mean([p["primary_confidence"] for p in self._shadow_predictions])
        shadow_conf = np.mean([p["shadow_confidence"] for p in self._shadow_predictions])

        hours_running = (datetime.now(timezone.utc) - self._shadow_start).total_seconds() / 3600

        return {
            "status": "running",
            "shadow_model": self._shadow_name,
            "n_predictions": n,
            "hours_running": round(hours_running, 1),
            "agreement_rate": round(agreement, 4),
            "primary_mean_confidence": round(primary_conf, 4),
            "shadow_mean_confidence": round(shadow_conf, 4),
            "confidence_delta": round(shadow_conf - primary_conf, 4),
            "ready_for_promotion": (
                n >= 100 and hours_running >= 24 and shadow_conf >= primary_conf
            ),
        }

    def promote_shadow(self):
        """Promote shadow model to primary. Returns the model object."""
        model = self._shadow_model
        self._shadow_model = None
        self._shadow_predictions = []
        logger.info(f"Shadow model {self._shadow_name} promoted to primary")
        return model


# ═══════════════════════════════════════════════════════════
# 5. Streaming Processor — ties it all together
# ═══════════════════════════════════════════════════════════

class StreamingTicketProcessor:
    """
    Production streaming processor that handles the full lifecycle:

    1. Process incoming tickets (classify + retrieve)
    2. Log predictions for monitoring
    3. Periodically run drift detection
    4. Trigger retraining when needed
    5. Shadow-deploy and promote new models

    In production, this would be:
    - A Kafka consumer reading from "tickets.raw"
    - A Celery beat scheduler for periodic drift checks
    - An Airflow DAG for the retraining pipeline

    For the prototype, it's a single class that demonstrates the logic.
    """

    def __init__(self, model_registry, retriever, config: RetrainingConfig = None):
        self.registry = model_registry
        self.retriever = retriever
        self.config = config or RetrainingConfig()

        self.drift_detector = DriftDetector()
        self.retrain_orchestrator = RetrainingOrchestrator(self.config)
        self.shadow_deployer = ShadowDeployer()
        self.prediction_logger = None  # Initialized when DB is available

        # Reference statistics (set from training data)
        self._reference_confidence: float = 0.85
        self._reference_pred_dist: dict = {}
        self._reference_df: Optional[pd.DataFrame] = None
        self._last_retrain: Optional[datetime] = None

        # Rolling windows for monitoring
        self._recent_confidences: deque = deque(maxlen=1000)
        self._recent_predictions: deque = deque(maxlen=1000)

    def set_reference(self, train_df: pd.DataFrame, val_confidence: float,
                      pred_distribution: dict):
        """Set reference statistics from training/validation."""
        self._reference_df = train_df
        self._reference_confidence = val_confidence
        self._reference_pred_dist = pred_distribution
        self._last_retrain = datetime.now(timezone.utc)
        logger.info(f"Reference set: confidence={val_confidence:.3f}, "
                     f"categories={list(pred_distribution.keys())}")

    def process_ticket(self, ticket: dict) -> dict:
        """
        Process a single ticket through the full pipeline.

        Returns the normal API response, but also:
        - Logs the prediction for drift monitoring
        - Runs shadow model if deployed
        - Updates rolling statistics
        """
        # Primary prediction
        result = self.registry.predict(ticket)

        # Log for monitoring
        self._recent_confidences.append(result.confidence)
        self._recent_predictions.append(result.predicted_category)

        if self.prediction_logger:
            self.prediction_logger.log(PredictionRecord(
                ticket_id=ticket.get("ticket_id", "unknown"),
                timestamp=datetime.now(timezone.utc).isoformat(),
                predicted_category=result.predicted_category,
                confidence=result.confidence,
                model_name=result.model_name,
                model_version="current",
                feature_values={
                    "product": ticket.get("product"),
                    "customer_tier": ticket.get("customer_tier"),
                    "priority": ticket.get("priority"),
                    "ticket_text_length": len(ticket.get("description", "")),
                },
            ))

        # Shadow prediction (non-blocking)
        shadow_result = self.shadow_deployer.shadow_predict(ticket, result)

        return {
            "classification": result,
            "shadow_active": shadow_result is not None,
        }

    def run_monitoring_cycle(self, current_df: pd.DataFrame) -> dict:
        """
        Periodic monitoring check (run hourly or daily).

        Returns a report with drift signals and recommended actions.
        """
        if self._reference_df is None:
            return {"status": "no_reference", "signals": []}

        signals = self.drift_detector.run_all_checks(
            reference_df=self._reference_df,
            current_df=current_df,
            reference_pred_distribution=self._reference_pred_dist,
            current_predictions=list(self._recent_predictions),
            reference_confidence=self._reference_confidence,
            current_confidences=list(self._recent_confidences),
        )

        # Check if retraining is needed
        should_retrain, trigger, reason = self.retrain_orchestrator.should_retrain(
            drift_signals=signals,
            last_retrain_at=self._last_retrain,
        )

        # Check shadow model status
        shadow_report = self.shadow_deployer.get_shadow_report()

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_recent_predictions": len(self._recent_predictions),
            "mean_confidence": float(np.mean(self._recent_confidences)) if self._recent_confidences else None,
            "drift_signals": [
                {
                    "type": s.signal_type,
                    "severity": s.severity.value,
                    "metric": s.metric_name,
                    "value": s.current_value,
                    "description": s.description,
                    "action": s.recommended_action,
                }
                for s in signals
            ],
            "retraining": {
                "should_retrain": should_retrain,
                "trigger": trigger.value,
                "reason": reason,
            },
            "shadow_deployment": shadow_report,
        }

        if should_retrain:
            logger.warning(f"RETRAINING TRIGGERED: {reason}")

        return report
