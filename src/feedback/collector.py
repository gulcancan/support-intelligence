"""Agent feedback capture for model improvement."""
import logging
from dataclasses import dataclass, asdict
from typing import Optional
from sqlalchemy import text
logger = logging.getLogger(__name__)

@dataclass
class AgentFeedback:
    ticket_id: str; predicted_category: str
    corrected_category: Optional[str] = None; predicted_subcategory: Optional[str] = None
    corrected_subcategory: Optional[str] = None; suggested_resolution: Optional[str] = None
    resolution_accepted: Optional[bool] = None; agent_id: Optional[str] = None

class FeedbackCollector:
    def __init__(self, engine=None):
        from db import get_engine; self.engine = engine or get_engine()

    def submit_feedback(self, fb):
        q = text("INSERT INTO agent_feedback (ticket_id,predicted_category,corrected_category,predicted_subcategory,corrected_subcategory,suggested_resolution,resolution_accepted,agent_id) VALUES (:ticket_id,:predicted_category,:corrected_category,:predicted_subcategory,:corrected_subcategory,:suggested_resolution,:resolution_accepted,:agent_id) RETURNING id")
        with self.engine.begin() as c: fid = c.execute(q, asdict(fb)).fetchone()[0]
        return {"feedback_id":fid,"category_corrected":fb.corrected_category is not None,"resolution_accepted":fb.resolution_accepted}

    def get_correction_rate(self, window_days=7):
        q = text("SELECT COUNT(*),SUM(CASE WHEN corrected_category IS NOT NULL THEN 1 ELSE 0 END),SUM(CASE WHEN resolution_accepted=true THEN 1 ELSE 0 END),SUM(CASE WHEN resolution_accepted=false THEN 1 ELSE 0 END) FROM agent_feedback WHERE feedback_at >= NOW()-:w*INTERVAL '1 day'")
        with self.engine.connect() as c: r = c.execute(q,{"w":window_days}).fetchone()
        t,cor,acc,rej = r[0] or 0, r[1] or 0, r[2] or 0, r[3] or 0
        return {"window_days":window_days,"total":t,"correction_rate":cor/t if t else 0,"acceptance_rate":acc/(acc+rej) if acc+rej else 0}
