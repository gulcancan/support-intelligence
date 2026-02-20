"""Data ingestion: load JSON, validate, temporal split, compute features, store."""
import json, logging, re
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np, pandas as pd
from sqlalchemy import text
from config import get_settings
from db import get_engine

logger = logging.getLogger(__name__)

class DataQualityReport:
    def __init__(self): self.checks, self.passed = [], True
    def check(self, name, condition, details=""):
        self.checks.append({"name":name,"passed":condition,"details":details})
        if not condition: self.passed = False; logger.warning(f"DQ FAIL: {name} — {details}")
    def summary(self): return {"total":len(self.checks),"passed":sum(1 for c in self.checks if c["passed"]),"failed":sum(1 for c in self.checks if not c["passed"])}

def validate_tickets(df):
    r = DataQualityReport()
    r.check("row_count", len(df) > 0, f"{len(df)} rows")
    r.check("no_duplicate_ids", df["ticket_id"].nunique() == len(df))
    r.check("required_fields", all(c in df.columns for c in ["ticket_id","created_at","category","description"]))
    r.check("category_not_null", df["category"].notna().all(), f"Null: {df['category'].isna().sum()}")
    return r

def temporal_split(df, train_ratio=0.70, val_ratio=0.15):
    """Temporal split by created_at — mirrors production where we predict on future data."""
    df = df.sort_values("created_at").reset_index(drop=True)
    n = len(df); t_end = int(n*train_ratio); v_end = int(n*(train_ratio+val_ratio))
    df["data_split"] = "test"
    df.loc[:t_end-1,"data_split"] = "train"
    df.loc[t_end:v_end-1,"data_split"] = "val"
    for s in ["train","val","test"]:
        sub = df[df["data_split"]==s]
        logger.info(f"  {s}: {len(sub):,} ({sub['created_at'].min()} -> {sub['created_at'].max()})")
    return df

def compute_features(df):
    f = pd.DataFrame({"ticket_id": df["ticket_id"]})
    f["subject_length"] = df["subject"].fillna("").str.len()
    f["description_length"] = df["description"].fillna("").str.len()
    f["combined_text_length"] = f["subject_length"] + f["description_length"]
    f["word_count"] = df["description"].fillna("").str.split().str.len()
    f["has_error_code"] = df["contains_error_code"].fillna(False)
    f["has_stack_trace"] = df["contains_stack_trace"].fillna(False)
    f["error_code_count"] = df["description"].fillna("").apply(lambda t: len(re.findall(r"ERROR_\w+",str(t))))
    cust = df.groupby("customer_id").agg(cnt=("ticket_id","count"),avg_sat=("satisfaction_score","mean"),esc_rate=("escalated","mean")).reset_index()
    m = df.merge(cust, on="customer_id", how="left")
    f["customer_ticket_count_30d"] = m["cnt"].fillna(0).astype(int)
    f["customer_avg_satisfaction"] = m["avg_sat"].fillna(3.5)
    f["customer_escalation_rate"] = m["esc_rate"].fillna(0.0)
    prod = df.groupby("product").agg(avg_hrs=("resolution_time_hours","mean"),vol=("ticket_id","count")).reset_index()
    m2 = df.merge(prod, on="product", how="left")
    f["product_avg_resolution_hrs"] = m2["avg_hrs"].fillna(24.0)
    f["product_ticket_volume_7d"] = m2["vol"].fillna(0).astype(int)
    created = pd.to_datetime(df["created_at"])
    f["hour_of_day"] = created.dt.hour
    f["day_of_week"] = created.dt.dayofweek
    f["is_weekend"] = created.dt.dayofweek >= 5
    f["is_after_hours"] = (created.dt.hour < 8) | (created.dt.hour > 18)
    f["days_since_product_update"] = df["product_version_age_days"].fillna(90)
    return f

def ingest(json_path, db_url=None):
    settings = get_settings()
    engine = get_engine() if not db_url else __import__("sqlalchemy").create_engine(db_url)
    logger.info(f"Loading {json_path}")
    with open(json_path) as f: raw = json.load(f)
    df = pd.DataFrame(raw); df["created_at"] = pd.to_datetime(df["created_at"])
    logger.info(f"Loaded {len(df):,} tickets")
    dq = validate_tickets(df)
    df = temporal_split(df, settings.train_ratio, settings.val_ratio)
    features = compute_features(df)
    json_cols = ["agent_actions","tags","related_tickets","kb_articles_viewed","kb_articles_helpful","auto_suggested_solutions"]
    for col in json_cols:
        if col in df.columns: df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x,(list,dict)) else x)
    for i in range(0,len(df),5000):
        df.iloc[i:i+5000].to_sql("tickets",engine,if_exists="append",index=False,method="multi")
        logger.info(f"  Wrote {min(i+5000,len(df)):,}")
    features.to_sql("ticket_features",engine,if_exists="append",index=False,method="multi")
    logger.info("Ingestion complete")
    return {"total":len(df),"splits":df["data_split"].value_counts().to_dict(),"categories":df["category"].value_counts().to_dict(),"dq":dq.summary()}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(ingest("data/raw/tickets_100k.json"), indent=2, default=str))
