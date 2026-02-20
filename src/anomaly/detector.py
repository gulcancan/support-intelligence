"""Anomaly detection: volume spikes, sentiment drift, new issue types."""
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional
import numpy as np, pandas as pd
logger = logging.getLogger(__name__)

@dataclass
class Anomaly:
    anomaly_type: str; severity: str; description: str; dimensions: dict
    metric_value: float; threshold: float
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class AnomalyDetector:
    def __init__(self, zscore_threshold=2.0, window_days=30):
        self.zscore_threshold = zscore_threshold; self.window_days = window_days

    def detect_volume_anomalies(self, df, group_cols=["product","category"]):
        anomalies = []; df = df.copy(); df["created_at"] = pd.to_datetime(df["created_at"])
        cutoff = df["created_at"].max() - timedelta(days=7)
        recent, baseline = df[df["created_at"]>=cutoff], df[df["created_at"]<cutoff]
        if baseline.empty or recent.empty: return []
        baseline["date"] = baseline["created_at"].dt.date
        bstats = baseline.groupby(group_cols+["date"]).size().reset_index(name="count").groupby(group_cols)["count"].agg(["mean","std"]).reset_index()
        bstats.columns = group_cols+["mean_daily","std_daily"]; bstats["std_daily"] = bstats["std_daily"].fillna(1)
        ndays = max(1,(recent["created_at"].max()-recent["created_at"].min()).days)
        rvol = recent.groupby(group_cols).size().reset_index(name="rc"); rvol["rdaily"] = rvol["rc"]/ndays
        m = rvol.merge(bstats, on=group_cols, how="left"); m["z"] = (m["rdaily"]-m["mean_daily"])/m["std_daily"]
        for _,r in m[m["z"]>self.zscore_threshold].iterrows():
            anomalies.append(Anomaly("volume_spike","critical" if r["z"]>3 else "warning",f"Volume: {r['rdaily']:.1f}/day vs baseline {r['mean_daily']:.1f}/day (z={r['z']:.2f})",{c:r[c] for c in group_cols},r["rdaily"],r["mean_daily"]+self.zscore_threshold*r["std_daily"]))
        return anomalies

    def detect_sentiment_drift(self, df, group_col="product"):
        anomalies = []; df = df.copy()
        smap = {"angry":1,"frustrated":2,"anxious":2.5,"confused":3,"neutral":3.5,"satisfied":5}
        df["sscore"] = df["customer_sentiment"].map(smap).fillna(3)
        df["created_at"] = pd.to_datetime(df["created_at"]); df = df.sort_values("created_at")
        for g in df[group_col].unique():
            gdf = df[df[group_col]==g]
            if len(gdf)<50: continue
            ewma = gdf["sscore"].ewm(span=30).mean().iloc[-1]
            mu, std = gdf["sscore"].mean(), max(gdf["sscore"].std(), 0.1)
            z = (ewma-mu)/std
            if z < -self.zscore_threshold:
                anomalies.append(Anomaly("sentiment_drift","warning" if z>-3 else "critical",f"Sentiment drift {g}: EWMA={ewma:.2f} vs mean={mu:.2f} (z={z:.2f})",{group_col:g},ewma,mu-self.zscore_threshold*std))
        return anomalies

    def detect_new_issue_types(self, df, model_confidences=None):
        anomalies = []; df = df.copy()
        recent = df.tail(int(len(df)*0.1)); hist = df.head(int(len(df)*0.9))
        for prod in df["product"].unique():
            hd = hist[hist["product"]==prod]["category"].value_counts(normalize=True)
            rd = recent[recent["product"]==prod]["category"].value_counts(normalize=True)
            for cat in rd.index:
                if (cat not in hd.index or hd[cat]<0.01) and rd[cat]>0.05:
                    anomalies.append(Anomaly("new_issue_type","info",f"'{cat}' emerging for {prod}: {rd[cat]:.1%}",{"product":prod,"category":cat},rd[cat],0.05))
        return anomalies

    def run_all_checks(self, df, model_confidences=None):
        all_a = self.detect_volume_anomalies(df) + self.detect_new_issue_types(df, model_confidences) + self.detect_sentiment_drift(df)
        sev = {"critical":0,"warning":1,"info":2}; all_a.sort(key=lambda a:sev.get(a.severity,3))
        logger.info(f"Detected {len(all_a)} anomalies"); return all_a
