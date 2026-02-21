"""Train classifiers, log to MLflow, compare."""
import argparse, json, logging, time, sys, os
from pathlib import Path

# Ensure /app/src is on the path — the NVIDIA base image entrypoint can override PYTHONPATH
_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import pandas as pd
from config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def load_and_split(path):
    with open(path) as f: tickets = json.load(f)
    df = pd.DataFrame(tickets); df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.sort_values("created_at").reset_index(drop=True)
    s = get_settings(); n = len(df); te = int(n*s.train_ratio); ve = int(n*(s.train_ratio+s.val_ratio))
    logger.info(f"Loaded {n:,} tickets, {df['category'].nunique()} categories")
    return df.iloc[:te], df.iloc[te:ve], df.iloc[ve:]

def train_catboost(tr, val, te, model_dir, n_trials=20):
    from models.catboost_classifier import CatBoostTicketClassifier
    from models.common import evaluate_model
    m = CatBoostTicketClassifier(model_dir=model_dir)
    t0 = time.time(); val_met = m.train(tr, val, n_trials=n_trials); tt = time.time()-t0
    y_te = m.label_encoder.transform(te["category"]); y_pred = m.label_encoder.transform(m.predict_batch(te))
    te_met = evaluate_model(y_te, y_pred, list(m.label_encoder.classes_))
    sample = te.iloc[0].to_dict(); lats = [m.predict(sample).latency_ms for _ in range(50)]
    m.save(model_dir)
    logger.info(f"CatBoost: val_f1={val_met.weighted_f1}, test_f1={te_met.weighted_f1}, latency={sum(lats)/len(lats):.1f}ms, train={tt:.0f}s")
    return {"model":"catboost","val_f1":val_met.weighted_f1,"test_f1":te_met.weighted_f1,"latency_ms":round(sum(lats)/len(lats),1),"train_sec":round(tt)}

def train_transformer(tr, val, te, model_dir, epochs=5, batch_size=32, model_key=None):
    from models.transformer_classifier import TransformerTicketClassifier
    from models.common import evaluate_model
    import torch, numpy as np
    from torch.utils.data import DataLoader
    from models.transformer_classifier import TicketDataset
    m = TransformerTicketClassifier(model_dir=model_dir, model_key=model_key)
    t0 = time.time(); val_met = m.train(tr, val, epochs=epochs, batch_size=batch_size); tt = time.time()-t0
    y_te = m.label_encoder.transform(te["category"])
    te_ds = TicketDataset(m._get_texts(te), m._prepare_structured(te), y_te, m.tokenizer, m.max_length)
    te_dl = DataLoader(te_ds, batch_size=64); y_pred = m._predict_batch_internal(te_dl)
    te_met = evaluate_model(y_te, y_pred, list(m.label_encoder.classes_))
    sample = te.iloc[0].to_dict(); lats = [m.predict(sample).latency_ms for _ in range(20)]
    m.save(model_dir)
    mname = m.model_name.split("/")[-1]
    logger.info(f"{mname}: val_f1={val_met.weighted_f1}, test_f1={te_met.weighted_f1}, latency={sum(lats)/len(lats):.1f}ms, train={tt:.0f}s")
    return {"model":mname,"val_f1":val_met.weighted_f1,"test_f1":te_met.weighted_f1,"latency_ms":round(sum(lats)/len(lats),1),"train_sec":round(tt)}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--data",default="data/raw/tickets.json")
    p.add_argument("--catboost",action="store_true"); p.add_argument("--transformer",action="store_true")
    p.add_argument("--both",action="store_true"); p.add_argument("--optuna-trials",type=int,default=20)
    p.add_argument("--epochs",type=int,default=5); p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--model-key",default=None,help="Transformer model: modernbert (default), distilbert, deberta-small, modernbert-large")
    p.add_argument("--model-dir",default="./models"); args = p.parse_args()

    # Log hardware info
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU detected: {gpu_name} ({gpu_mem:.0f} GB)")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            logger.info("No GPU detected — training on CPU")
    except Exception as e:
        logger.info(f"GPU check failed: {e} — will detect at training time")

    if args.both: args.catboost = args.transformer = True
    if not args.catboost and not args.transformer: args.catboost = True
    tr, val, te = load_and_split(args.data); results = []
    if args.catboost: results.append(train_catboost(tr,val,te,f"{args.model_dir}/catboost",args.optuna_trials))
    if args.transformer: results.append(train_transformer(tr,val,te,f"{args.model_dir}/transformer",args.epochs,args.batch_size,args.model_key))
    if len(results)>1:
        logger.info("\n" + "="*60 + "\nCOMPARISON\n" + "="*60)
        logger.info(f"\n{pd.DataFrame(results).to_string(index=False)}")
    logger.info("Done!")

if __name__ == "__main__": main()
