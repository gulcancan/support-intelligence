"""
Multi-task CatBoost classifier with sentence transformer embeddings.

Text features: all-MiniLM-L6-v2 → 384-dim dense embedding
  (replaces TF-IDF — captures semantics, synonyms, word order)

Feature vector per ticket:
  384 (text embedding) + 9 (categorical, label-encoded) + 9 (numerical) + 4 (boolean) = 406 features

Trains 4 independent CatBoost classifiers:
  1. Category      (7 classes)
  2. Subcategory   (27 classes)
  3. Priority      (4 classes)
  4. Sentiment     (6 classes)

Why sentence embeddings over TF-IDF:
  - TF-IDF is bag-of-words: no word order, no synonyms, no semantics
  - "Configuration error" and "Setup problem" share zero TF-IDF features
  - For 27 subcategories with subtle distinctions, TF-IDF F1 ≈ 17-20%
  - Dense embeddings capture semantic similarity → expected F1 ≈ 60-80%
  - We already load the embedding model for RAG retrieval → zero extra cost

Why still CatBoost (not just the transformer classifier):
  - 3-5ms inference vs 50-200ms for transformer classifier
  - Native categorical feature handling (product, tier, channel)
  - SHAP interpretability over structured features
  - Optuna hyperparameter tuning
  - Can run on CPU-only deployment
"""
import time, logging, json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from models.common import ClassificationResult, ModelMetrics, evaluate_model

logger = logging.getLogger(__name__)

# ── Task definitions ────────────────────────────────────────────────────────
TASK_DEFS = {
    "category":    {"column": "category"},
    "subcategory": {"column": "subcategory"},
    "priority":    {"column": "priority"},
    "sentiment":   {"column": "customer_sentiment"},
}

# ── Embedding cache ─────────────────────────────────────────────────────────
_embedding_model = None


def _get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Lazy-load sentence transformer. Shared across all tasks."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name} (dim={_embedding_model.get_sentence_embedding_dimension()})")
    return _embedding_model


def _embed_tickets(df, model=None, batch_size=256):
    """
    Compute sentence embeddings for all tickets in a DataFrame.

    Combines subject + description into a single text.
    Returns: np.ndarray of shape (n_tickets, embedding_dim)
    """
    model = model or _get_embedding_model()
    texts = (df["subject"].fillna("") + " [SEP] " + df["description"].fillna("")).tolist()
    embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 5000)
    return embeddings


class _SingleTaskCatBoost:
    """One CatBoost classifier for a single task."""

    CAT_FEATURES = [
        "product", "product_module", "customer_tier", "priority",
        "severity", "channel", "environment", "region", "business_impact",
    ]
    NUM_FEATURES = [
        "previous_tickets", "account_age_days", "account_monthly_value",
        "similar_issues_last_30_days", "product_version_age_days",
        "ticket_text_length", "affected_users", "attachments_count",
        "response_count",
    ]
    BOOL_FEATURES = [
        "contains_error_code", "contains_stack_trace",
        "weekend_ticket", "after_hours",
    ]

    def __init__(self, task_name, task_def):
        self.task_name = task_name
        self.column = task_def["column"]
        self.label_encoder: Optional[LabelEncoder] = None
        self.model: Optional[CatBoostClassifier] = None
        self._cat_encoders: Dict[str, LabelEncoder] = {}

    def _build_structured_features(self, df, fit=False):
        """Build the structured (non-text) features. Returns np.ndarray."""
        struct = pd.DataFrame(index=df.index)

        for col in self.CAT_FEATURES:
            if col in df.columns:
                struct[col] = df[col].fillna("unknown").astype(str)
                if fit:
                    enc = LabelEncoder()
                    struct[col] = enc.fit_transform(struct[col])
                    self._cat_encoders[col] = enc
                else:
                    enc = self._cat_encoders[col]
                    struct[col] = struct[col].map(
                        lambda x, e=enc: e.transform([x])[0] if x in e.classes_ else -1
                    )

        for col in self.NUM_FEATURES:
            if col in df.columns:
                struct[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        for col in self.BOOL_FEATURES:
            if col in df.columns:
                struct[col] = df[col].fillna(False).astype(int)

        return struct.values.astype(np.float64)

    def _combine_features(self, embeddings, df, fit=False):
        """Concatenate text embeddings + structured features."""
        struct = self._build_structured_features(df, fit=fit)
        return np.hstack([embeddings, struct])

    def train(self, train_df, val_df, train_emb, val_emb, n_trials=20, use_optuna=True):
        """Train on pre-computed embeddings + structured features."""
        logger.info(f"  [{self.task_name}] Training CatBoost (col={self.column})...")

        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(train_df[self.column].fillna("UNKNOWN"))
        y_val = self.label_encoder.transform(val_df[self.column].fillna("UNKNOWN"))
        n_classes = len(self.label_encoder.classes_)

        class_counts = np.bincount(y_train, minlength=n_classes).astype(float)
        class_counts[class_counts == 0] = 1
        sample_weights = (len(y_train) / (n_classes * class_counts))[y_train]

        X_train = self._combine_features(train_emb, train_df, fit=True)
        X_val = self._combine_features(val_emb, val_df)

        logger.info(f"  [{self.task_name}] Features: {X_train.shape[1]} (384 emb + {X_train.shape[1]-384} struct), Classes: {n_classes}")

        if use_optuna and n_trials > 0:
            params = self._optuna_tune(X_train, y_train, X_val, y_val, sample_weights, n_trials)
        else:
            params = {"iterations": 1000, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0}

        self.model = CatBoostClassifier(
            **params,
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Weighted",
            early_stopping_rounds=50,
            verbose=100,
            random_seed=42,
        )
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val), sample_weight=sample_weights)

        y_pred = self.model.predict(X_val).flatten().astype(int)
        metrics = evaluate_model(y_val, y_pred, list(self.label_encoder.classes_))
        logger.info(f"  [{self.task_name}] Val F1: {metrics.weighted_f1}")
        return metrics

    def _optuna_tune(self, X_tr, y_tr, X_val, y_val, sw, n_trials):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def obj(trial):
            p = {
                "iterations": trial.suggest_int("iterations", 300, 1500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
            }
            m = CatBoostClassifier(
                **p, loss_function="MultiClass",
                eval_metric="TotalF1:average=Weighted",
                early_stopping_rounds=30, verbose=0, random_seed=42,
            )
            m.fit(X_tr, y_tr, eval_set=(X_val, y_val), sample_weight=sw)
            return f1_score(y_val, m.predict(X_val).flatten().astype(int), average="weighted")

        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=n_trials)
        logger.info(f"  [{self.task_name}] Optuna best F1: {study.best_value:.4f}")
        return study.best_params

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_label(self, X):
        idx = self.model.predict(X).flatten().astype(int)
        return self.label_encoder.inverse_transform(idx)

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(d / "model.cbm"))
        joblib.dump(self.label_encoder, d / "label_encoder.joblib")
        joblib.dump(self._cat_encoders, d / "cat_encoders.joblib")

    def load(self, directory):
        d = Path(directory)
        self.model = CatBoostClassifier()
        self.model.load_model(str(d / "model.cbm"))
        self.label_encoder = joblib.load(d / "label_encoder.joblib")
        self._cat_encoders = joblib.load(d / "cat_encoders.joblib")


# ── Multi-task wrapper ──────────────────────────────────────────────────────

class CatBoostTicketClassifier:
    """
    Multi-task CatBoost with sentence transformer embeddings.

    Embeddings are computed once and shared across all 4 task classifiers.
    This makes training efficient: the embedding step (~2-5 min for 100K tickets)
    happens once, then each CatBoost trains in ~1-2 min on the dense features.

    Usage:
        clf = CatBoostTicketClassifier()
        metrics = clf.train(train_df, val_df, n_trials=20)
        result = clf.predict(ticket_dict)
    """

    def __init__(self, model_dir="./models/catboost", tasks=None,
                 embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tasks = tasks or TASK_DEFS
        self.embedding_model_name = embedding_model_name
        self.classifiers: Dict[str, _SingleTaskCatBoost] = {
            name: _SingleTaskCatBoost(name, tdef)
            for name, tdef in self.tasks.items()
        }
        logger.info(f"CatBoost multi-task: {list(self.tasks.keys())} with {embedding_model_name}")

    @property
    def label_encoders(self) -> Dict[str, LabelEncoder]:
        return {name: clf.label_encoder for name, clf in self.classifiers.items() if clf.label_encoder}

    def train(self, train_df, val_df, n_trials=20, use_optuna=True, embedding_batch_size=256):
        """Train all task classifiers with shared embeddings."""
        logger.info(f"Training CatBoost multi-task ({len(self.classifiers)} tasks)...")
        t0 = time.time()

        # Compute embeddings once for all tasks
        logger.info(f"  Computing embeddings for {len(train_df)+len(val_df):,} tickets...")
        emb_model = _get_embedding_model(self.embedding_model_name)
        train_emb = _embed_tickets(train_df, emb_model, batch_size=embedding_batch_size)
        val_emb = _embed_tickets(val_df, emb_model, batch_size=embedding_batch_size)
        emb_time = time.time() - t0
        logger.info(f"  Embeddings: {train_emb.shape[1]}-dim, computed in {emb_time:.0f}s")

        # Train each task
        all_metrics = {}
        for task_name, clf in self.classifiers.items():
            col = clf.column
            if col not in train_df.columns:
                logger.warning(f"  [{task_name}] Column '{col}' missing, skipping")
                continue
            metrics = clf.train(train_df, val_df, train_emb, val_emb,
                                n_trials=n_trials, use_optuna=use_optuna)
            all_metrics[task_name] = metrics

        tt = time.time() - t0
        logger.info(f"CatBoost multi-task training complete in {tt:.0f}s (embedding: {emb_time:.0f}s)")
        for task, m in all_metrics.items():
            n_cls = len(self.classifiers[task].label_encoder.classes_)
            logger.info(f"  {task:15s}  val_f1={m.weighted_f1:.4f}  ({n_cls} classes)")

        cat_metrics = all_metrics.get("category", list(all_metrics.values())[0])
        cat_metrics.task_f1_scores = {t: m.weighted_f1 for t, m in all_metrics.items()}
        return cat_metrics

    def _embed_single(self, ticket_dict):
        """Embed a single ticket for inference."""
        model = _get_embedding_model(self.embedding_model_name)
        text = f"{ticket_dict.get('subject', '')} [SEP] {ticket_dict.get('description', '')}"
        return model.encode([text], normalize_embeddings=True)

    def predict(self, ticket) -> ClassificationResult:
        """Predict all tasks for a single ticket."""
        start = time.time()
        df = pd.DataFrame([ticket])
        emb = self._embed_single(ticket)

        results = {}
        probabilities = {}

        for task_name, clf in self.classifiers.items():
            if clf.model is None:
                continue
            X = clf._combine_features(emb, df)
            proba = clf.predict_proba(X)[0]
            idx = int(np.argmax(proba))
            le = clf.label_encoder
            results[task_name] = le.inverse_transform([idx])[0]
            probabilities[task_name] = {
                le.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba)
            }

        latency = round((time.time() - start) * 1000, 2)

        return ClassificationResult(
            predicted_category=results.get("category", "UNKNOWN"),
            predicted_subcategory=results.get("subcategory"),
            predicted_priority=results.get("priority"),
            predicted_sentiment=results.get("sentiment"),
            category_probabilities=probabilities.get("category", {}),
            subcategory_probabilities=probabilities.get("subcategory", {}),
            priority_probabilities=probabilities.get("priority", {}),
            sentiment_probabilities=probabilities.get("sentiment", {}),
            confidence=max(probabilities.get("category", {}).values(), default=0.0),
            model_name="catboost-multitask",
            latency_ms=latency,
        )

    def predict_batch(self, df, embedding_batch_size=256) -> pd.DataFrame:
        """Predict all tasks for a DataFrame."""
        emb = _embed_tickets(df, _get_embedding_model(self.embedding_model_name),
                             batch_size=embedding_batch_size)
        result = pd.DataFrame(index=df.index)
        for task_name, clf in self.classifiers.items():
            if clf.model is None:
                continue
            X = clf._combine_features(emb, df)
            result[f"pred_{task_name}"] = clf.predict_label(X)
        return result

    def feature_importance(self, task="category", top_n=30):
        """Get feature importance with meaningful names."""
        clf = self.classifiers.get(task)
        if not clf or not clf.model:
            return []
        imp = clf.model.feature_importances_

        # Build feature names: first 384 are embedding dims, rest are structured
        n_emb = 384  # sentence transformer dimension
        struct_names = (
            [c for c in clf.CAT_FEATURES] +
            [c for c in clf.NUM_FEATURES] +
            [c for c in clf.BOOL_FEATURES]
        )
        names = [f"emb_{i}" for i in range(n_emb)] + struct_names

        results = []
        for i, v in sorted(enumerate(imp), key=lambda x: x[1], reverse=True)[:top_n]:
            fname = names[i] if i < len(names) else f"f_{i}"
            results.append({"feature": fname, "importance": round(float(v), 4)})
        return results

    def save(self, path=None):
        d = Path(path) if path else self.model_dir
        d.mkdir(parents=True, exist_ok=True)

        for task_name, clf in self.classifiers.items():
            if clf.model is not None:
                clf.save(d / task_name)

        config = {
            "embedding_model": self.embedding_model_name,
            "tasks": {
                name: {
                    "column": self.tasks[name]["column"],
                    "n_classes": len(self.classifiers[name].label_encoder.classes_)
                    if self.classifiers[name].label_encoder else 0,
                }
                for name in self.tasks
                if name in self.classifiers and self.classifiers[name].model is not None
            },
        }
        with open(d / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved CatBoost multi-task to {d}")

    def load(self, path=None):
        d = Path(path) if path else self.model_dir

        with open(d / "config.json") as f:
            config = json.load(f)

        self.embedding_model_name = config.get("embedding_model", self.embedding_model_name)

        for task_name, task_info in config["tasks"].items():
            task_dir = d / task_name
            if task_dir.exists():
                if task_name not in self.classifiers:
                    self.classifiers[task_name] = _SingleTaskCatBoost(task_name, {
                        "column": task_info["column"],
                    })
                self.classifiers[task_name].load(task_dir)
                logger.info(f"  Loaded [{task_name}] from {task_dir}")

        logger.info(f"Loaded CatBoost multi-task ({len(self.classifiers)} tasks, emb={self.embedding_model_name})")
