"""Tests for drift detection and retraining logic."""
import pytest
import numpy as np
import pandas as pd
from monitoring.drift import (
    DriftDetector, DriftSeverity, DriftSignal,
    RetrainingOrchestrator, RetrainingConfig, RetrainingTrigger,
    ShadowDeployer,
)


class TestPSI:
    """Population Stability Index computation."""

    def test_identical_distributions(self):
        """PSI should be ~0 for identical distributions."""
        np.random.seed(42)
        data = np.random.normal(100, 15, 1000)
        psi = DriftDetector.compute_psi(data, data)
        assert psi < 0.01  # Essentially zero

    def test_shifted_distribution(self):
        """PSI should be high for shifted distributions."""
        np.random.seed(42)
        reference = np.random.normal(100, 15, 1000)
        shifted = np.random.normal(120, 15, 1000)  # Mean shifted by 20
        psi = DriftDetector.compute_psi(reference, shifted)
        assert psi > 0.1  # Should detect significant shift

    def test_different_variance(self):
        """PSI detects variance changes too."""
        np.random.seed(42)
        reference = np.random.normal(100, 10, 1000)
        wider = np.random.normal(100, 30, 1000)  # Same mean, 3x variance
        psi = DriftDetector.compute_psi(reference, wider)
        assert psi > 0.05  # Detectable

    def test_psi_symmetry(self):
        """PSI is approximately symmetric (unlike KL divergence)."""
        np.random.seed(42)
        a = np.random.normal(100, 15, 1000)
        b = np.random.normal(110, 15, 1000)
        psi_ab = DriftDetector.compute_psi(a, b)
        psi_ba = DriftDetector.compute_psi(b, a)
        assert abs(psi_ab - psi_ba) < 0.05  # Should be close


class TestFeatureDrift:
    """Feature distribution drift detection."""

    def test_no_drift(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "ticket_text_length": np.random.normal(200, 50, 1000),
            "previous_tickets": np.random.poisson(3, 1000),
        })
        detector = DriftDetector()
        signals = detector.detect_feature_drift(df, df)
        assert len(signals) == 0  # Same data → no drift

    def test_detects_shift(self):
        np.random.seed(42)
        reference = pd.DataFrame({
            "ticket_text_length": np.random.normal(200, 50, 1000),
            "account_age_days": np.random.normal(365, 100, 1000),
        })
        # Current data has very different text lengths
        current = pd.DataFrame({
            "ticket_text_length": np.random.normal(500, 50, 200),  # Much longer texts
            "account_age_days": np.random.normal(365, 100, 200),   # Same
        })
        detector = DriftDetector()
        signals = detector.detect_feature_drift(reference, current)
        # Should detect drift in ticket_text_length but not account_age_days
        text_signals = [s for s in signals if "ticket_text_length" in s.metric_name]
        assert len(text_signals) > 0
        assert text_signals[0].severity in (DriftSeverity.MODERATE, DriftSeverity.SEVERE)

    def test_new_categorical_values(self):
        reference = pd.DataFrame({"product": ["A", "B", "C"] * 100})
        current = pd.DataFrame({"product": ["A", "B", "D"] * 30})  # D is new
        detector = DriftDetector()
        signals = detector.detect_feature_drift(reference, current, features=[])
        new_cat_signals = [s for s in signals if "new_categories" in s.metric_name]
        assert len(new_cat_signals) > 0
        assert "D" in str(new_cat_signals[0].dimensions)


class TestPredictionDrift:
    def test_stable_predictions(self):
        ref_dist = {"Tech": 0.35, "Billing": 0.15, "Feature": 0.12}
        # Match reference proportions exactly with 100 samples
        current = ["Tech"] * 35 + ["Billing"] * 15 + ["Feature"] * 12 + ["Other"] * 38
        detector = DriftDetector()
        signals = detector.detect_prediction_drift(ref_dist, current)
        # Should not detect drift for categories that match reference
        tech_signals = [s for s in signals if "Tech" in s.metric_name]
        assert len(tech_signals) == 0

    def test_shifted_predictions(self):
        ref_dist = {"Tech": 0.35, "Billing": 0.15, "Feature": 0.12}
        # Suddenly 80% of predictions are "Tech"
        current = ["Tech"] * 80 + ["Billing"] * 10 + ["Feature"] * 10
        detector = DriftDetector()
        signals = detector.detect_prediction_drift(ref_dist, current)
        assert len(signals) > 0


class TestConfidenceDrift:
    def test_stable_confidence(self):
        detector = DriftDetector()
        signals = detector.detect_confidence_drift(0.85, [0.84, 0.86, 0.83, 0.87] * 25)
        assert len(signals) == 0

    def test_dropping_confidence(self):
        detector = DriftDetector()
        # Confidence dropped from 0.85 to 0.65
        signals = detector.detect_confidence_drift(0.85, [0.65, 0.63, 0.68, 0.62] * 25)
        assert len(signals) > 0
        assert signals[0].severity == DriftSeverity.SEVERE


class TestRetrainingOrchestrator:
    def test_no_retrain_when_stable(self):
        orch = RetrainingOrchestrator(RetrainingConfig(max_days_without_retrain=7))
        from datetime import datetime, timezone
        should, trigger, reason = orch.should_retrain(
            drift_signals=[],
            last_retrain_at=datetime.now(timezone.utc),
            available_corrections=0,
        )
        assert should is False

    def test_scheduled_retrain(self):
        orch = RetrainingOrchestrator(RetrainingConfig(max_days_without_retrain=7))
        from datetime import datetime, timedelta, timezone
        should, trigger, reason = orch.should_retrain(
            drift_signals=[],
            last_retrain_at=datetime.now(timezone.utc) - timedelta(days=10),
            available_corrections=0,
        )
        assert should is True
        assert trigger == RetrainingTrigger.SCHEDULED

    def test_drift_triggered_retrain(self):
        orch = RetrainingOrchestrator()
        severe_signals = [
            DriftSignal("feature_drift", DriftSeverity.SEVERE, "psi_text_len", 0.3, 0, 0.2, "big shift"),
            DriftSignal("confidence_drift", DriftSeverity.SEVERE, "conf_24h", 0.65, 0.85, 0.75, "conf drop"),
        ]
        should, trigger, reason = orch.should_retrain(drift_signals=severe_signals)
        assert should is True
        assert trigger == RetrainingTrigger.DRIFT_TRIGGERED

    def test_single_severe_not_enough(self):
        """A single severe signal shouldn't immediately trigger retraining — could be transient."""
        orch = RetrainingOrchestrator()
        signals = [
            DriftSignal("feature_drift", DriftSeverity.SEVERE, "psi_x", 0.3, 0, 0.2, "shift"),
        ]
        should, trigger, _ = orch.should_retrain(drift_signals=signals)
        assert should is False  # Need multiple signals


class TestShadowDeployer:
    def test_shadow_lifecycle(self):
        from dataclasses import dataclass

        @dataclass
        class MockResult:
            predicted_category: str = "Tech"
            confidence: float = 0.9

        class MockModel:
            def predict(self, ticket):
                return MockResult(predicted_category="Tech", confidence=0.88)

        deployer = ShadowDeployer()
        assert deployer.get_shadow_report()["status"] == "no_data"

        deployer.deploy_shadow(MockModel(), "catboost-v2")

        primary = MockResult()
        for i in range(50):
            deployer.shadow_predict({"ticket_id": f"T{i}"}, primary)

        report = deployer.get_shadow_report()
        assert report["status"] == "running"
        assert report["n_predictions"] == 50
        assert report["agreement_rate"] == 1.0  # Both predict "Tech"
