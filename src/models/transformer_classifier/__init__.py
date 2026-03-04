"""
Multi-task transformer classifier with LoRA adapters.

Architecture:
  text → ModernBERT (FROZEN) + LoRA adapters → [CLS] (768-dim)
                                                    ↓
  structured (13-dim) → concat → shared trunk (512-dim, GELU, Dropout)
                                      ↓
                  ┌──────────┬────────┼──────────┬───────────┐
                  ↓          ↓        ↓          ↓           ↓
             category   subcategory  priority  sentiment
              head        head       head       head

Why LoRA over full fine-tuning:
  - ~100K training tickets is too few for 149M params (over-parameterized)
  - LoRA adds ~300K trainable params (r=16) — better matched to data regime
  - No risk of catastrophic forgetting (base weights are frozen)
  - Faster training: only backprop through LoRA + heads
  - Lower VRAM: no optimizer states for 149M frozen params
  - Can swap/retrain individual task adapters independently

LoRA config:
  - rank=16, alpha=32 (alpha/rank = 2 is standard scaling)
  - Applied to query, key, value projections in attention
  - ~0.2% of base model params are trainable via LoRA

Loss: Multi-task focal loss (γ=2) with per-task weighting.
Default encoder: ModernBERT-base (answerdotai/ModernBERT-base)
"""
import time, logging, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from models.common import ClassificationResult, ModelMetrics, evaluate_model

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model registry ──────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "modernbert":       {"name": "answerdotai/ModernBERT-base",     "max_length": 512},
    "modernbert-large": {"name": "answerdotai/ModernBERT-large",    "max_length": 512},
    "distilbert":       {"name": "distilbert-base-uncased",         "max_length": 128},
    "deberta-small":    {"name": "microsoft/deberta-v3-small",      "max_length": 256},
}
DEFAULT_MODEL = "modernbert"

# ── Task definitions ────────────────────────────────────────────────────────
TASK_DEFS = {
    "category":    {"column": "category",           "weight": 1.0},
    "subcategory": {"column": "subcategory",        "weight": 1.5},
    "priority":    {"column": "priority",           "weight": 1.0},
    "sentiment":   {"column": "customer_sentiment", "weight": 0.8},
}


# ── Focal Loss ──────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017): FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Down-weights well-classified examples, focuses on hard ones.
    γ=0 → standard CE.  γ=2 → easy examples (p_t>0.9) get 100× less gradient.
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


# ── Dataset ─────────────────────────────────────────────────────────────────
class MultiTaskTicketDataset(Dataset):
    def __init__(self, texts, structured, labels_dict, tokenizer, max_length=512):
        self.texts = texts
        self.structured = structured
        self.labels = labels_dict
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "structured": torch.tensor(self.structured[idx], dtype=torch.float32),
        }
        for task_name, labels in self.labels.items():
            item[f"label_{task_name}"] = torch.tensor(labels[idx], dtype=torch.long)
        return item


# ── LoRA layer ──────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """
    LoRA adapter wrapping an existing nn.Linear.

    Adds low-rank decomposition: output = W_frozen @ x + (B @ A) @ x * (alpha/rank)
    Only A and B are trainable. W_frozen stays frozen.
    """

    def __init__(self, original_linear: nn.Linear, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        # Low-rank decomposition
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Kaiming init for A, zero init for B → LoRA starts as identity
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.original(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    @property
    def n_trainable(self):
        return self.lora_A.numel() + self.lora_B.numel()


def apply_lora_to_model(encoder, rank=16, alpha=32, target_modules=None):
    """
    Replace attention Q/K/V linear layers with LoRA-wrapped versions.

    Walks the encoder module tree, finds linear layers whose name matches
    target_modules patterns, wraps them with LoRALinear.
    """
    if target_modules is None:
        # Common attention projection names across architectures
        target_modules = {"query", "key", "value", "q_proj", "k_proj", "v_proj",
                          "Wqkv", "q_lin", "k_lin", "v_lin"}

    lora_params = []
    replaced = 0

    for name, module in encoder.named_modules():
        for attr_name in list(vars(module).keys()):
            if attr_name.startswith("_"):
                continue
            try:
                child = getattr(module, attr_name)
            except Exception:
                continue
            if isinstance(child, nn.Linear) and attr_name in target_modules:
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, attr_name, lora_layer)
                lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
                replaced += 1

    # Also check named_children for nested modules
    if replaced == 0:
        # Fallback: search all children for Linear layers with matching names
        for name, module in encoder.named_modules():
            last_part = name.split(".")[-1] if "." in name else name
            if isinstance(module, nn.Linear) and last_part in target_modules:
                # Find parent and attribute name
                parts = name.split(".")
                parent = encoder
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                attr = parts[-1]
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, attr, lora_layer)
                lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
                replaced += 1

    logger.info(f"LoRA: replaced {replaced} layers (rank={rank}, alpha={alpha})")
    return lora_params


# ── Model ───────────────────────────────────────────────────────────────────
class MultiTaskTicketModel(nn.Module):
    """
    Frozen encoder + LoRA adapters + shared trunk + per-task heads.

    Trainable parameters:
      - LoRA A/B matrices in attention Q/K/V (~300K params for r=16)
      - Shared trunk (~400K params)
      - Task heads (~100K params total)
    Total: ~800K trainable vs 149M total (0.5%)
    """

    def __init__(self, model_name, n_structured, task_n_classes,
                 dropout=0.3, lora_rank=16, lora_alpha=32):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Freeze ALL encoder parameters first
        for param in self.encoder.parameters():
            param.requires_grad = False

        n_frozen = sum(p.numel() for p in self.encoder.parameters())
        logger.info(f"Encoder: {model_name}, hidden={hidden_size}, frozen={n_frozen:,} params")

        # Apply LoRA — this unfreezes only the LoRA A/B matrices
        self.lora_params = apply_lora_to_model(
            self.encoder, rank=lora_rank, alpha=lora_alpha,
        )
        n_lora = sum(p.numel() for p in self.lora_params)
        logger.info(f"LoRA trainable: {n_lora:,} params ({n_lora/n_frozen*100:.2f}% of encoder)")

        # Shared trunk: [CLS] + structured → 512-dim
        trunk_dim = 512
        self.shared_trunk = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + n_structured, trunk_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )

        # Per-task heads
        self.task_heads = nn.ModuleDict()
        for task_name, n_classes in task_n_classes.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(trunk_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(128, n_classes),
            )

        # Log total trainable
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_all = sum(p.numel() for p in self.parameters())
        logger.info(f"Total trainable: {total_trainable:,} / {total_all:,} ({total_trainable/total_all*100:.1f}%)")
        logger.info(f"Task heads: {', '.join(f'{k}({v} cls)' for k, v in task_n_classes.items())}")

    def forward(self, input_ids, attention_mask, structured):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        shared = self.shared_trunk(torch.cat([cls, structured], dim=1))

        logits = {}
        for task_name, head in self.task_heads.items():
            logits[task_name] = head(shared)
        return logits


# ── Classifier ──────────────────────────────────────────────────────────────
class TransformerTicketClassifier:
    """
    Multi-task ticket classifier with LoRA.

    Usage:
        clf = TransformerTicketClassifier()
        metrics = clf.train(train_df, val_df)
        result = clf.predict(ticket_dict)
    """

    STRUCT_COLS = [
        "previous_tickets", "account_age_days", "account_monthly_value",
        "similar_issues_last_30_days", "product_version_age_days",
        "ticket_text_length", "affected_users", "attachments_count",
        "response_count", "contains_error_code", "contains_stack_trace",
        "weekend_ticket", "after_hours",
    ]

    def __init__(self, model_dir="./models/transformer", model_key=None, model_name=None,
                 tasks=None, focal_gamma=2.0, lora_rank=16, lora_alpha=32):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if model_name:
            self.model_name = model_name
            self.max_length = 512
        else:
            key = model_key or DEFAULT_MODEL
            if key not in MODEL_REGISTRY:
                logger.warning(f"Unknown model key '{key}', falling back to {DEFAULT_MODEL}")
                key = DEFAULT_MODEL
            entry = MODEL_REGISTRY[key]
            self.model_name = entry["name"]
            self.max_length = entry["max_length"]

        self.tasks = tasks or TASK_DEFS
        self.focal_gamma = focal_gamma
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        self.tokenizer = None
        self.model = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = None
        self.structured_cols = self.STRUCT_COLS

        logger.info(f"LoRA multi-task: {self.model_name}, r={lora_rank}, α={lora_alpha}, "
                     f"tasks={list(self.tasks.keys())}, focal_γ={focal_gamma}")

    def _prepare_structured(self, df, fit=False):
        from sklearn.preprocessing import StandardScaler
        s = pd.DataFrame(
            {c: pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df.columns else 0
             for c in self.structured_cols},
            index=df.index,
        )
        v = s.values.astype(np.float32)
        if fit:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(v)
        return self.scaler.transform(v)

    def _get_texts(self, df):
        return (df["subject"].fillna("") + " [SEP] " + df["description"].fillna("")).tolist()

    def _encode_labels(self, df, fit=False) -> Dict[str, np.ndarray]:
        labels = {}
        for task_name, task_def in self.tasks.items():
            col = task_def["column"]
            if col not in df.columns:
                logger.warning(f"Task '{task_name}': column '{col}' not in data, skipping")
                continue
            if fit:
                le = LabelEncoder()
                labels[task_name] = le.fit_transform(df[col].fillna("UNKNOWN"))
                self.label_encoders[task_name] = le
                logger.info(f"  {task_name}: {len(le.classes_)} classes — {list(le.classes_)}")
            else:
                le = self.label_encoders[task_name]
                vals = df[col].fillna("UNKNOWN").values
                known = set(le.classes_)
                mapped = np.array([le.transform([v])[0] if v in known else 0 for v in vals])
                labels[task_name] = mapped
        return labels

    def _build_focal_losses(self, y_train_dict) -> Dict[str, FocalLoss]:
        losses = {}
        for task_name, y in y_train_dict.items():
            n_classes = len(self.label_encoders[task_name].classes_)
            counts = np.bincount(y, minlength=n_classes).astype(float)
            counts[counts == 0] = 1
            alpha = torch.tensor(len(y) / (n_classes * counts), dtype=torch.float32).to(DEVICE)
            losses[task_name] = FocalLoss(gamma=self.focal_gamma, alpha=alpha)
        return losses

    def train(self, train_df, val_df, epochs=5, batch_size=32, lr=2e-4):
        """
        Train with LoRA.

        Note: lr=2e-4 (10x higher than full fine-tuning) because LoRA adapters
        need larger learning rates — the frozen encoder provides stable gradients.
        """
        logger.info(f"Training LoRA multi-task model on {DEVICE}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        y_train = self._encode_labels(train_df, fit=True)
        y_val = self._encode_labels(val_df)

        active_tasks = [t for t in y_train if t in y_val]
        task_n_classes = {t: len(self.label_encoders[t].classes_) for t in active_tasks}

        tr_ds = MultiTaskTicketDataset(
            self._get_texts(train_df), self._prepare_structured(train_df, fit=True),
            {t: y_train[t] for t in active_tasks}, self.tokenizer, self.max_length,
        )
        val_ds = MultiTaskTicketDataset(
            self._get_texts(val_df), self._prepare_structured(val_df),
            {t: y_val[t] for t in active_tasks}, self.tokenizer, self.max_length,
        )
        tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size * 2)

        # Build model with LoRA
        self.model = MultiTaskTicketModel(
            self.model_name, len(self.structured_cols), task_n_classes,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
        ).to(DEVICE)

        # Focal losses per task
        focal_losses = self._build_focal_losses({t: y_train[t] for t in active_tasks})

        # Optimizer — only trainable params (LoRA + trunk + heads)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        total_steps = len(tr_dl) * epochs
        sched = get_linear_schedule_with_warmup(opt, int(total_steps * 0.1), total_steps)

        best_f1, best_state, patience, no_imp = 0.0, None, 2, 0

        for ep in range(epochs):
            self.model.train()
            total_loss = 0

            for bi, batch in enumerate(tr_dl):
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                st = batch["structured"].to(DEVICE)

                opt.zero_grad()
                logits = self.model(ids, mask, st)

                loss = torch.tensor(0.0, device=DEVICE)
                for task_name in active_tasks:
                    task_label = batch[f"label_{task_name}"].to(DEVICE)
                    task_loss = focal_losses[task_name](logits[task_name], task_label)
                    task_weight = self.tasks[task_name]["weight"]
                    loss = loss + task_weight * task_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                sched.step()
                total_loss += loss.item()

                if (bi + 1) % 100 == 0:
                    logger.info(f"  Ep{ep+1} batch {bi+1}/{len(tr_dl)} loss={loss.item():.4f}")

            val_metrics = self._validate_all(val_dl, {t: y_val[t] for t in active_tasks})
            avg_loss = total_loss / len(tr_dl)
            task_summary = " | ".join(f"{t}={val_metrics[t]:.4f}" for t in active_tasks)
            logger.info(f"Epoch {ep+1} loss={avg_loss:.4f} | {task_summary}")

            mean_f1 = np.mean([val_metrics[t] for t in active_tasks])
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    logger.info(f"Early stop ep{ep+1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(DEVICE)

        final_metrics = self._validate_all(val_dl, {t: y_val[t] for t in active_tasks})
        logger.info(f"Best mean val F1: {best_f1:.4f}")
        for task_name, f1 in final_metrics.items():
            logger.info(f"  {task_name}: F1={f1:.4f} ({len(self.label_encoders[task_name].classes_)} cls)")

        y_val_cat = y_val["category"]
        preds = self._predict_batch_internal(val_dl)
        cat_metrics = evaluate_model(y_val_cat, preds["category"], list(self.label_encoders["category"].classes_))
        cat_metrics.task_f1_scores = final_metrics
        return cat_metrics

    def _validate_all(self, dl, y_val_dict) -> Dict[str, float]:
        preds = self._predict_batch_internal(dl)
        metrics = {}
        for task_name, y_true in y_val_dict.items():
            if task_name in preds:
                metrics[task_name] = f1_score(y_true, preds[task_name], average="weighted")
        return metrics

    @torch.no_grad()
    def _predict_batch_internal(self, loader) -> Dict[str, np.ndarray]:
        self.model.eval()
        all_preds = {t: [] for t in self.model.task_heads.keys()}
        for batch in loader:
            logits = self.model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
                batch["structured"].to(DEVICE),
            )
            for task_name, task_logits in logits.items():
                all_preds[task_name].extend(torch.argmax(task_logits, dim=1).cpu().numpy())
        return {t: np.array(p) for t, p in all_preds.items()}

    def predict(self, ticket) -> ClassificationResult:
        start = time.time()
        self.model.eval()
        df = pd.DataFrame([ticket])
        text = self._get_texts(df)[0]
        st = self._prepare_structured(df)
        enc = self.tokenizer(
            text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(
                enc["input_ids"].to(DEVICE),
                enc["attention_mask"].to(DEVICE),
                torch.tensor(st, dtype=torch.float32).to(DEVICE),
            )

        results = {}
        probabilities = {}
        for task_name, task_logits in logits.items():
            proba = torch.softmax(task_logits, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(proba))
            le = self.label_encoders[task_name]
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
            model_name=f"lora-multitask-{self.model_name.split('/')[-1]}",
            latency_ms=latency,
        )

    def predict_batch(self, df) -> pd.DataFrame:
        texts = self._get_texts(df)
        structured = self._prepare_structured(df)
        dummy_labels = {t: np.zeros(len(df), dtype=int) for t in self.label_encoders}
        ds = MultiTaskTicketDataset(texts, structured, dummy_labels, self.tokenizer, self.max_length)
        dl = DataLoader(ds, batch_size=64)

        preds = self._predict_batch_internal(dl)
        result = pd.DataFrame(index=df.index)
        for task_name, indices in preds.items():
            result[f"pred_{task_name}"] = self.label_encoders[task_name].inverse_transform(indices)
        return result

    def save(self, path=None):
        d = Path(path) if path else self.model_dir
        d.mkdir(parents=True, exist_ok=True)

        # Save full state dict (includes frozen weights + LoRA)
        # For production, could save only LoRA + heads to reduce size
        torch.save(self.model.state_dict(), d / "model.pt")
        self.tokenizer.save_pretrained(str(d / "tokenizer"))

        import joblib
        for task_name, le in self.label_encoders.items():
            joblib.dump(le, d / f"label_encoder_{task_name}.joblib")
        joblib.dump(self.scaler, d / "scaler.joblib")

        config = {
            "model_name": self.model_name,
            "n_structured": len(self.structured_cols),
            "max_length": self.max_length,
            "structured_cols": self.structured_cols,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "tasks": {
                task_name: {
                    "n_classes": len(self.label_encoders[task_name].classes_),
                    "classes": list(self.label_encoders[task_name].classes_),
                    "weight": self.tasks[task_name]["weight"],
                }
                for task_name in self.label_encoders
            },
            "focal_gamma": self.focal_gamma,
        }
        with open(d / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved LoRA multi-task model to {d}")

    def load(self, path=None):
        d = Path(path) if path else self.model_dir
        import joblib

        with open(d / "config.json") as f:
            cfg = json.load(f)

        self.model_name = cfg["model_name"]
        self.max_length = cfg["max_length"]
        self.structured_cols = cfg["structured_cols"]
        self.focal_gamma = cfg.get("focal_gamma", 2.0)
        self.lora_rank = cfg.get("lora_rank", 16)
        self.lora_alpha = cfg.get("lora_alpha", 32)

        self.tokenizer = AutoTokenizer.from_pretrained(str(d / "tokenizer"))
        self.scaler = joblib.load(d / "scaler.joblib")

        task_n_classes = {}
        for task_name, task_info in cfg["tasks"].items():
            self.label_encoders[task_name] = joblib.load(d / f"label_encoder_{task_name}.joblib")
            task_n_classes[task_name] = task_info["n_classes"]

        self.model = MultiTaskTicketModel(
            cfg["model_name"], cfg["n_structured"], task_n_classes,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
        )
        self.model.load_state_dict(torch.load(d / "model.pt", map_location=DEVICE, weights_only=True))
        self.model.to(DEVICE)
        self.model.eval()
        logger.info(f"Loaded LoRA model from {d} ({self.model_name}, r={self.lora_rank}, tasks={list(task_n_classes.keys())})")


# ── Backward-compatible aliases ─────────────────────────────────────────────
TicketDataset = MultiTaskTicketDataset
TicketClassifierModel = MultiTaskTicketModel
