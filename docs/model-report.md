# Model Documentation & Findings Report

## 1. Multi-Task Classification

The system predicts four targets simultaneously for each incoming ticket:

| Task | Classes | Column | Purpose |
|------|---------|--------|---------|
| Category | 5–7 | `category` | Route to correct team |
| Subcategory | 25–27 | `subcategory` | Identify specific issue type |
| Priority | 4 | `priority` | Determine response urgency |
| Sentiment | 6 | `customer_sentiment` | Adapt tone and resolution strategy |

---

## 2. CatBoost Classifier (Primary)

### Architecture

```
subject + description → all-MiniLM-L6-v2 sentence embedding (384-dim)
                              ↓
    + 9 categorical features (native handling, no one-hot)
    + 9 numerical features
    + 4 boolean features
    = 406-dim feature vector
                              ↓
    4 × independent CatBoost classifiers (one per task)
    Each with independent Optuna HPO (20 trials)
```

### Feature groups

**Text features — sentence embeddings (384 dims):**
All-MiniLM-L6-v2 encodes `subject + description` into a single 384-dimensional dense vector. This replaced TF-IDF (bag-of-words, 5000 sparse features) after experiments showed both representations hit the same F1 ceiling on this data. Sentence embeddings were chosen because they capture semantics, synonyms, and word order — properties that matter when real data has subtle text differences between subcategories. The same embedding model is already loaded for RAG retrieval, so there is zero additional memory cost.

**Categorical features (9):**
product, product_module, customer_tier, priority, severity, channel, environment, region, business_impact. CatBoost handles these natively — no one-hot encoding needed.

**Numerical features (9):**
previous_tickets, account_age_days, account_monthly_value, similar_issues_last_30_days, product_version_age_days, ticket_text_length, affected_users, attachments_count, response_count.

**Boolean features (4):**
contains_error_code, contains_stack_trace, weekend_ticket, after_hours.

### Hyperparameter tuning

Optuna Bayesian optimization (20 trials per task, maximizing weighted F1):

| Parameter | Search range | Rationale |
|-----------|-------------|-----------|
| iterations | [300, 1500] | Trade training time vs convergence |
| learning_rate | [0.01, 0.3] log | Lower = smoother, higher = faster |
| depth | [4, 10] | Deeper = more feature interactions |
| l2_leaf_reg | [1e-3, 10] log | Regularization strength |
| random_strength | [0.5, 5.0] | Randomization for robustness |

Each task gets independent tuning because optimal hyperparameters differ: category (easy, few iterations) vs subcategory (hard, deeper trees, more regularization).

### Why separate classifiers per task (not multi-output)

CatBoost is a tree ensemble — it has no shared representations between tasks. Unlike a neural network where a shared encoder computes text features once, CatBoost trees split on individual features independently. Per-task classifiers allow independent Optuna tuning and independent retraining (if sentiment degrades, retrain only the sentiment model without touching category).

---

## 3. ModernBERT + LoRA Classifier (Secondary)

### Architecture

```
subject + " [SEP] " + description
    → ModernBERT tokenizer (max 512 tokens)
    → ModernBERT encoder (FROZEN, 149M params)
      + LoRA adapters on Q/K/V attention (rank=16, alpha=32)
    → [CLS] token embedding (768-dim)
    → concat with 13 standardized numerical features
    → shared trunk: Dropout → Linear(781→512) → GELU → Dropout
                              ↓
              ┌──────────┬────┼──────────┬───────────┐
              ↓          ↓    ↓          ↓           ↓
         category   subcategory  priority  sentiment
          head        head       head       head
         (MLP)       (MLP)      (MLP)      (MLP)
```

### Why LoRA over full fine-tuning

With ~70K training tickets (70% of 100K), full fine-tuning updates 149M parameters — massively over-parameterized for the data regime. LoRA adds only ~300K trainable parameters (0.2% of the encoder), which better matches the amount of available training signal.

| | Full fine-tuning | LoRA (r=16) |
|---|---|---|
| Encoder params trainable | ~75M (top 50% unfrozen) | ~300K (A/B matrices on Q/K/V) |
| Trunk + heads | ~500K | ~500K |
| Total trainable | ~75.5M | ~800K |
| Risk of catastrophic forgetting | Moderate | None (base weights frozen) |
| VRAM for optimizer states | High (Adam stores 2× trainable) | Low |
| Learning rate | 2e-5 (standard BERT) | 2e-4 (10× higher, LoRA needs it) |
| Training speed | Slower (backprop through encoder) | Faster |
| Can swap/retrain per-task adapters | No (shared encoder changes) | Yes (freeze trunk, swap adapter) |

### LoRA implementation details

`LoRALinear` wraps each attention Q/K/V projection:

```
output = W_frozen @ x  +  (B @ A) @ x × (alpha / rank)
```

- `A` initialized with Kaiming uniform, `B` initialized with zeros → LoRA starts as identity (no perturbation to pretrained weights at initialization)
- `alpha/rank = 32/16 = 2` — standard scaling factor
- Applied to all attention Q/K/V projections across 12 transformer layers
- Auto-detects projection names across architectures (ModernBERT's `Wqkv`, DistilBERT's `q_lin`/`k_lin`/`v_lin`, etc.)

### Multi-task loss

Per-task focal loss with class weighting:

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)    where γ=2
```

Focal loss down-weights easy examples exponentially: at 95% confidence, gradient is reduced 400× compared to standard cross-entropy. Combined with per-task loss weights (subcategory: 1.5×, sentiment: 0.8×, category/priority: 1.0×), this focuses training on hard, informative examples.

### Training configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW (only trainable params) | Weight decay regularization |
| Learning rate | 2e-4 | Higher than full fine-tuning — LoRA adapters need larger LR |
| Warmup | 10% of steps | Prevents early gradient explosions |
| Schedule | Linear decay | Gradual learning rate reduction |
| Batch size | 32 | Constrained by VRAM (increase on larger GPUs) |
| Max epochs | 5 | Early stopping usually triggers at 3–4 |
| Patience | 2 epochs | Stop if mean F1 across tasks doesn't improve |
| Max length | 512 tokens | Covers 99%+ of ticket text |
| Gradient clipping | 1.0 | Stabilizes training |
| Focal gamma | 2.0 | Standard focusing parameter |

---

## 4. Multi-Task Results on Synthetic Data

### Measured performance

| Task | CatBoost F1 | LoRA F1 | Random baseline | Assessment |
|------|-------------|---------|-----------------|------------|
| Category (5 cls) | **1.0000** | ~1.0000 | ~0.20 | Trivially solvable |
| Subcategory (25 cls) | **0.1988** | ~0.20 | ~0.04 | Above random, but weak signal |
| Priority (4 cls) | **0.4843** | ~0.48 | ~0.25 | Moderate signal from metadata |
| Sentiment (6 cls) | **0.1624** | ~0.16 | ~0.17 | Near random — no signal |

Training time: ~27 min (CatBoost) | Inference latency: ~13.5 ms (CatBoost, all 4 tasks)

### Root cause: synthetic data generation artifacts

Both model families hit the same F1 ceiling. The bottleneck is in the synthetic data, not the models. Here is the concrete evidence from the actual ticket data:

**Subcategory labels are random with respect to text.** Within the same category, every subcategory uses identical subject and description templates. For example, all 5 subcategories under "Account Management" produce the same subject:

```
[Account Management]
  Access Control (111 tickets): "License upgrade needed for Analytics Dashboard"
  Billing (118 tickets):        "License upgrade needed for DataSync Pro"
  License (117 tickets):        "License upgrade needed for DataSync Pro"
  Subscription (120 tickets):   "License upgrade needed for CloudBackup Enterprise"
  Upgrade (109 tickets):        "License upgrade needed for DataSync Pro"
```

The only variation is the product name, which does not predict subcategory. The same pattern holds for every category — "Data Issue" subcategories (Corruption, Data Loss, Import/Export, Sync Error, Validation) all produce "Data inconsistency in {product}."

**Sentiment labels are decorrelated from text tone.** The same template appears under opposite sentiments:

```
[angry]:      "We would like to request a feature for StreamProcessor that allows bulk operations..."
[frustrated]: "We would like to request a feature for CloudBackup Enterprise that allows bulk operations..."
[satisfied]:  "The Analytics Dashboard has been running extremely slowly for the past 2 days..."
[grateful]:   "We've noticed data inconsistencies in Analytics Dashboard..."
```

A "satisfied" customer complains about slow performance. An "angry" customer politely requests a feature. No model can learn sentiment from text that doesn't express it.

**Priority partially correlates with `affected_users` but nothing else.** From a sample of 3000 tickets:

```
  critical: avg_affected_users=483, business_impact={high:198, medium:218, low:182, critical:160}
  high:     avg_affected_users=494, business_impact={medium:202, high:185, critical:176, low:188}
  medium:   avg_affected_users=26,  business_impact={medium:185, high:193, critical:200, low:179}
  low:      avg_affected_users=26,  business_impact={critical:179, high:192, medium:175, low:188}
```

Critical/high tickets have ~480 affected users vs ~26 for medium/low — a real signal that CatBoost can learn (explaining F1=0.48). But `business_impact` is uniformly distributed across all priorities, meaning the generator randomizes it. The text doesn't correlate with priority at all.

**Category F1=1.0 explained.** All 27 subcategories map to exactly one category, and the generator uses category-specific templates with distinctive vocabulary ("License upgrade" → Account Management, "Data inconsistency" → Data Issue, "running extremely slowly" → Technical Issue). Even 2 CatBoost trees suffice — the overfitting detector stops at iteration 2.

### What this means for the architecture

The F1 results validate the architecture rather than exposing model weakness:
- Models correctly learn learnable patterns (category from vocabulary, priority from `affected_users`)
- Models correctly fail on unlearnable patterns (random subcategory/sentiment labels)
- Replacing TF-IDF with sentence embeddings produced zero improvement — confirming the bottleneck is data, not representation
- Both CatBoost and LoRA-ModernBERT hit the same ceiling — confirming this is not a model capacity issue

### Expected performance on real data

With real customer tickets where text genuinely reflects the issue type, urgency, and tone:

| Task | Synthetic F1 | Expected real F1 | Why improvement expected |
|------|-------------|-------------------|--------------------------|
| Category | 1.00 | 0.85–0.93 | Harder (ambiguous text) but still learnable |
| Subcategory | 0.20 | 0.60–0.80 | Real descriptions distinguish "Configuration" from "Crash/Bug" |
| Priority | 0.48 | 0.55–0.70 | Text urgency cues + metadata combine |
| Sentiment | 0.16 | 0.55–0.75 | Real frustration, gratitude visible in language |

---

## 5. Model Comparison

### When to use which

| Scenario | Best model | Why |
|----------|-----------|-----|
| Production default | CatBoost | ~13ms for all 4 tasks, CPU-only, interpretable |
| High throughput (>100 tickets/sec) | CatBoost | 13ms vs 50–150ms |
| Stakeholder explanations | CatBoost | SHAP values, feature importances |
| Rich text, sparse metadata | ModernBERT + LoRA | Better text generalization |
| New/unseen product names | ModernBERT + LoRA | Subword tokenization handles OOV |
| Long ticket descriptions | ModernBERT + LoRA | 8192 token context |
| Low-confidence predictions | Ensemble | Average per-task probability distributions |

### Ensemble strategy

The `ModelRegistry` implements automatic per-task routing:

1. CatBoost predicts first (fast, always available)
2. If `confidence < 0.6` → also run ModernBERT + LoRA
3. Average per-task probabilities: `P_ensemble(c) = 0.5 × P_catboost(c) + 0.5 × P_lora(c)`
4. Select argmax per task

This gives CatBoost speed for easy cases (majority) and ensemble quality for ambiguous cases.

---

## 6. Dual-Representation Retrieval

### How classification predictions improve retrieval

All four multi-task predictions feed into hybrid retrieval re-ranking:

| Prediction | Re-ranking effect | Boost factor |
|---|---|---|
| Category | Pre-filters vector search and BM25 to same category | Binary filter |
| Subcategory | Boosts results matching same subcategory | 1.8× |
| Priority | Adjacency scoring — nearby priority tickets rank higher | 0.84–1.2× |
| Sentiment | When frustrated/angry, boosts high-satisfaction resolutions | 1.2× |

### Dual text representation

| Search method | Text indexed | Text queried | Why |
|---|---|---|---|
| BM25 keyword | Original (raw text + error logs) | Original query | Exact tokens: `ERROR_TIMEOUT_429`, version `3.2.1` |
| Vector semantic | Cleaned (noise stripped) | Cleaned query | Better embeddings after removing HTML, greetings, stack traces |

Text cleaning operations (lightweight regex, no LLM, at index time for resolutions):
- Strip HTML tags, email headers, quoted replies
- Remove greetings ("Hi team,") and closings ("Best regards,")
- Collapse stack traces to first + last frame
- Normalize repeated punctuation and whitespace

---

## 7. Feature Importance (CatBoost)

Post-training, `model.feature_importance()` returns per-task rankings:

**Expected for category:** embedding dimensions dominate (they encode the distinctive vocabulary per category). Structured features contribute minimally since the text alone achieves F1=1.0.

**Expected for priority:** structured features rank high — `affected_users`, `customer_tier`, `business_impact` are the primary signals. Embedding dimensions contribute less since priority is metadata-driven in this data.

**Expected for subcategory/sentiment on real data:** embedding dimensions should dominate as text becomes the primary discriminator. On synthetic data, no features are informative (confirming the random-label finding).

---

## 8. Improving Synthetic Data

If re-generating synthetic data, injecting these patterns would immediately lift F1:

**Subcategory-specific descriptions:**
Instead of "Data inconsistency in {product}" for all Data Issue subcategories, use templates like:
- Sync Error: "Data sync between {product} instances is failing with conflict errors"
- Corruption: "Records in {product} are showing garbled/corrupted values after migration"
- Import/Export: "CSV export from {product} is producing malformed files"

**Sentiment-specific language:**
Instead of random sentiment assignment, match text tone:
- frustrated: "I've been trying to resolve this for 3 days and nothing works..."
- angry: "This is completely unacceptable. We're paying enterprise rates for..."
- satisfied: "Thank you for the quick turnaround on this issue..."
- neutral: "We'd like to report an issue with the following configuration..."

**Priority-correlated text urgency:**
- critical: "URGENT: Production is down, affecting all users..."
- low: "Minor cosmetic issue in the dashboard, no rush..."

### Alternative: LLM label distillation

Rather than re-generating synthetic data, we can correct the existing labels using a strong open-source foundation model (Qwen2.5-72B, Llama 3.1-70B) running locally. This is implemented in `scripts/correct_labels.py`.

The approach prompts the LLM with the full ticket context — not just subject and description, but also error logs, resolution text, customer feedback, escalation reason, affected users, customer tier, environment, and tags. The model infers what the subcategory, priority, and sentiment *should* be based on the actual content, then we replace the random labels with the LLM's predictions.

**Why this works well:**
- Sentiment is near-trivial for a foundation model (F1 ~0.85–0.95 expected) — text tone is exactly what LLMs understand
- Subcategory benefits from resolution text and error logs that the original generator ignored when assigning labels
- Priority can be inferred from affected users + urgency language + escalation status
- Category labels are NOT corrected — they already achieve F1=1.0

**Why open-source and local:**
- Zero API cost for 100K tickets (vs ~$100–200 for Claude/GPT-4 API)
- No rate limits — vLLM with continuous batching achieves 3–8 tickets/sec
- Full data privacy — tickets never leave the machine
- Reproducible — same model weights, temperature=0.1 for deterministic outputs

**Estimated time:** 4–9 hours for 100K tickets on a DGX Spark with Qwen2.5-72B-AWQ via vLLM. Checkpointing every 50 tickets makes it safe to interrupt and resume.

See `README.md → Label Correction via LLM Distillation` for setup instructions.

---

## 9. Experiment Tracking

All experiments logged to MLflow (`http://localhost:5000`):

| Artifact | Description |
|----------|-------------|
| Parameters | All hyperparameters per task (learning rate, depth, LoRA rank, etc.) |
| Metrics | Per-task weighted F1 (category, subcategory, priority, sentiment) |
| Inference latency | Average ms over 50 predictions |
| Training time | Wall-clock seconds |
| Feature importance | Per-task feature rankings (CatBoost) |
| Model files | Serialized model artifacts |

### Reproducibility checklist

- [x] CatBoost: `random_seed=42`
- [x] Temporal split (deterministic by `created_at` sorting)
- [x] Dependencies pinned in `requirements.txt`
- [x] Docker Compose for consistent environment
- [x] MLflow logs all parameters + data split sizes
- [x] LoRA: deterministic initialization (Kaiming A, zeros B)
