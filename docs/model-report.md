# Model Documentation & Comparison Report

## 1. CatBoost Ticket Classifier

### Architecture
```
subject + description → TF-IDF (5000 features, bigrams, sublinear TF)
    → Chi-squared selection (top 3000)
    → Concatenate with 22 structured features
    → CatBoost gradient-boosted ensemble
    → Softmax over 7 categories
```

### Feature Groups

**Text features (TF-IDF):**
- Bigram TF-IDF on `subject + description` (5000 initial → 3000 selected)
- `sublinear_tf=True` for better term frequency weighting
- `min_df=5, max_df=0.95` to filter noise and ubiquitous terms
- Chi-squared feature selection keeps the most discriminative terms

**Categorical features (9):**
product, product_module, customer_tier, priority, severity, channel, environment, region, business_impact

CatBoost handles these natively — no one-hot encoding needed. This is a practical advantage: with 5 products × 25 modules × 4 tiers × 4 priorities, one-hot encoding would create 100+ sparse columns.

**Numerical features (9):**
previous_tickets, account_age_days, account_monthly_value, similar_issues_last_30_days, product_version_age_days, ticket_text_length, affected_users, attachments_count, response_count

**Boolean features (4):**
contains_error_code, contains_stack_trace, weekend_ticket, after_hours

### Hyperparameter Tuning

Optuna Bayesian optimization (20 trials, maximizing weighted F1):

| Parameter | Search Range | Rationale |
|-----------|-------------|-----------|
| iterations | [300, 1500] | Trade training time vs convergence |
| learning_rate | [0.01, 0.3] log | Lower = smoother, higher = faster |
| depth | [4, 10] | Deeper = more feature interactions |
| l2_leaf_reg | [1e-3, 10] log | Regularization strength |
| random_strength | [0.5, 5.0] | Randomization for robustness |

### Class Imbalance Handling

Inverse frequency class weights via `sample_weight`:
```
weight_i = n_total / (n_classes × n_class_i)
```

Why not SMOTE: Synthetic samples on TF-IDF feature vectors don't correspond to real language patterns. The generated points in TF-IDF space are not meaningful text representations. Class weighting is more principled for NLP-derived features.

---

## 2. DistilBERT Ticket Classifier

### Architecture
```
subject + " [SEP] " + description
    → DistilBERT tokenizer (max 128 tokens)
    → DistilBERT encoder (6 layers, 768-dim)
    → [CLS] token embedding (768-dim)
    → Concatenate with 13 standardized numerical features
    → Dropout(0.3) → Linear(781, 256) → ReLU
    → Dropout(0.3) → Linear(256, 7)
    → Softmax
```

### Key Design Choices

**Why [CLS] + structured features (not text-only):**
A pure text model ignores valuable metadata. A ticket saying "this is broken" means very different things depending on whether `customer_tier=enterprise` and `priority=critical` vs `customer_tier=free` and `priority=low`. The concatenation architecture captures both.

**Why freeze bottom 4/6 layers:**
- Reduces trainable parameters from 66M to ~15M
- Prevents catastrophic forgetting of pre-trained representations
- Bottom layers capture general syntax; top layers adapt to task
- 3× faster training with negligible quality loss

**Training configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Weight decay regularization |
| Learning rate | 2e-5 | Standard for BERT fine-tuning |
| Warmup | 10% of steps | Prevents early gradient explosions |
| Schedule | Linear decay | Gradual learning rate reduction |
| Batch size | 32 | Balance memory usage and gradient quality |
| Max epochs | 5 | Early stopping usually triggers at 3-4 |
| Patience | 2 epochs | Stop if no F1 improvement |
| Max length | 128 tokens | Covers 95%+ of ticket text |
| Gradient clipping | 1.0 | Stabilizes training |

### Why DistilBERT over BERT/RoBERTa

| Model | Params | Inference (CPU) | F1 Delta |
|-------|--------|-----------------|----------|
| BERT-base | 110M | ~300ms | baseline |
| DistilBERT | 66M | ~150ms | -0.5% |
| RoBERTa | 125M | ~350ms | +0.3% |

For ticket classification (short text, 7 classes), the marginal quality gain of RoBERTa doesn't justify 2× slower inference. DistilBERT is the Pareto-optimal choice.

---

## 3. Model Comparison

### Performance Metrics

| Metric | CatBoost | DistilBERT | Notes |
|--------|----------|------------|-------|
| Val Weighted F1 | ~0.88-0.92 | ~0.89-0.93 | Transformer slightly better |
| Val Macro F1 | ~0.84-0.88 | ~0.86-0.90 | Larger gap on minority classes |
| Test Weighted F1 | ~0.87-0.91 | ~0.88-0.92 | Temporal test set |
| Inference latency | 1-5 ms | 50-200 ms | CatBoost 10-40× faster |
| Training time | ~5 min | ~30 min | With Optuna / 5 epochs |

*Exact numbers depend on Optuna trial results and random seed.*

### When to Use Which

| Scenario | Best Model | Why |
|----------|-----------|-----|
| Production default | CatBoost | Fast, interpretable, easy to deploy |
| High-throughput (>100 tickets/sec) | CatBoost | 1-5ms vs 50-200ms |
| Stakeholder explanations | CatBoost | SHAP values, feature importances |
| Rich text, sparse metadata | DistilBERT | Better text generalization |
| New/unseen product names | DistilBERT | Handles OOV via subword tokenization |
| Low-confidence predictions | Ensemble | Average probabilities for robustness |

### Ensemble Strategy

The `ModelRegistry` implements automatic routing:

1. CatBoost predicts first (fast, always available)
2. If `confidence < 0.6` → also run DistilBERT
3. Average probabilities: `P_ensemble(c) = 0.5 × P_catboost(c) + 0.5 × P_transformer(c)`
4. Select argmax category

This gives us the speed of CatBoost for easy cases (majority) and the quality of the ensemble for ambiguous cases (minority).

---

## 4. Error Analysis

### Expected Confusion Patterns

| Confused Pair | Root Cause | Mitigation |
|--------------|-----------|------------|
| "Technical Issue" ↔ "Bug Report" | Both describe broken functionality. Distinction is intent (config error vs code defect) | Hierarchical classification: first technical/non-technical |
| "Feature Request" ↔ "How-To / Guidance" | "How do I do X?" could be either | Add subcategory prediction; disambiguate via resolution_code patterns |
| "Outage / Downtime" ↔ "Technical Issue" | Outages are technical issues at scale | Priority/severity features help distinguish |

### Minority Class Performance

Classes with <5% representation ("Outage / Downtime", "Compliance / Security") will have:
- Lower recall (fewer training examples)
- Higher variance in per-class F1 across folds

Mitigations applied:
- Class-weighted loss function
- Stratified consideration in temporal split
- Monitoring per-class F1 in MLflow (not just aggregate)

---

## 5. Feature Importance Analysis

### Top CatBoost Features (Expected)

| Rank | Feature Type | Feature | Rationale |
|------|-------------|---------|-----------|
| 1-5 | TF-IDF | Error-related terms | Discriminate technical vs non-technical |
| 6-10 | Categorical | product, product_module | Strong category priors per product |
| 11-15 | TF-IDF | Action verbs ("configure", "upgrade", "request") | Distinguish guidance/feature/technical |
| 16-20 | Numerical | severity, previous_tickets, account_value | Priority signals |
| 20-30 | Boolean | contains_error_code, contains_stack_trace | Binary technical indicators |

Use `model.feature_importance()` or SHAP values post-training for exact rankings.

---

## 6. Experiment Tracking

All experiments logged to MLflow (`http://localhost:5000`):

### Logged Artifacts

| Artifact | Description |
|----------|-------------|
| Parameters | All hyperparameters (learning rate, depth, iterations, etc.) |
| Metrics | Weighted F1, Macro F1, Accuracy (val and test) |
| Per-class F1 | Individual F1 for each of 7 categories |
| Inference latency | Average ms over 50 predictions |
| Training time | Wall-clock seconds |
| Feature importance | Top-30 features (CatBoost only) |
| Model files | Serialized model artifacts |

### Reproducibility Checklist

- [x] Data generation seed: `random.seed(42), np.random.seed(42)`
- [x] CatBoost: `random_seed=42`
- [x] Temporal split (deterministic by `created_at` sorting)
- [x] Dependencies pinned in `requirements.txt`
- [x] Docker Compose for consistent environment
- [x] MLflow logs all parameters + data split sizes
