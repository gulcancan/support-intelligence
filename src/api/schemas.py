"""Pydantic request/response schemas."""
from pydantic import BaseModel, Field
from typing import Optional

class TicketInput(BaseModel):
    ticket_id: Optional[str] = None
    subject: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=1, max_length=10000)
    product: Optional[str] = None; product_version: Optional[str] = None; product_module: Optional[str] = None
    customer_id: Optional[str] = None; customer_tier: Optional[str] = None
    priority: Optional[str] = None; severity: Optional[str] = None; channel: Optional[str] = None
    error_logs: Optional[str] = None; stack_trace: Optional[str] = None
    environment: Optional[str] = None; region: Optional[str] = None
    previous_tickets: Optional[int] = 0; account_age_days: Optional[int] = 0
    account_monthly_value: Optional[float] = 0; similar_issues_last_30_days: Optional[int] = 0
    product_version_age_days: Optional[int] = 0; ticket_text_length: Optional[int] = None
    affected_users: Optional[int] = 1; attachments_count: Optional[int] = 0; response_count: Optional[int] = 1
    contains_error_code: Optional[bool] = False; contains_stack_trace: Optional[bool] = False
    weekend_ticket: Optional[bool] = False; after_hours: Optional[bool] = False
    business_impact: Optional[str] = "medium"

class ClassificationResponse(BaseModel):
    predicted_category: str; predicted_subcategory: Optional[str] = None
    confidence: float; category_probabilities: dict; model_name: str; latency_ms: float

class RetrievalResult(BaseModel):
    ticket_id: Optional[str] = None; resolution: Optional[str] = None; resolution_code: Optional[str] = None
    score: Optional[float] = None; final_score: Optional[float] = None; category: Optional[str] = None
    product: Optional[str] = None; subject: Optional[str] = None

class GraphContext(BaseModel):
    resolution_templates: list = []; error_code_resolutions: list = []; product_stats: dict = {}

class RetrievalResponse(BaseModel):
    results: list[RetrievalResult]; graph_context: GraphContext; metadata: dict

class TicketProcessingResponse(BaseModel):
    ticket_id: Optional[str] = None; classification: ClassificationResponse
    retrieval: RetrievalResponse; processing_time_ms: float

class FeedbackInput(BaseModel):
    ticket_id: str; predicted_category: str; corrected_category: Optional[str] = None
    resolution_accepted: Optional[bool] = None; agent_id: Optional[str] = None

class AnomalyResponse(BaseModel):
    anomaly_type: str; severity: str; description: str; dimensions: dict
    metric_value: float; threshold: float; detected_at: str

class HealthResponse(BaseModel):
    status: str; version: str; models_loaded: list; vector_store_count: Optional[int] = None; database_connected: bool
