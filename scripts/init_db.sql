CREATE TABLE IF NOT EXISTS tickets (
    ticket_id VARCHAR(32) PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ,
    customer_id VARCHAR(32),
    customer_tier VARCHAR(16),
    organization_id VARCHAR(32),
    product VARCHAR(64),
    product_version VARCHAR(16),
    product_module VARCHAR(64),
    category VARCHAR(64),
    subcategory VARCHAR(64),
    priority VARCHAR(16),
    severity VARCHAR(8),
    channel VARCHAR(16),
    subject TEXT,
    description TEXT,
    error_logs TEXT,
    stack_trace TEXT,
    customer_sentiment VARCHAR(16),
    previous_tickets INTEGER,
    resolution TEXT,
    resolution_code VARCHAR(64),
    resolved_at TIMESTAMPTZ,
    resolution_time_hours FLOAT,
    resolution_attempts INTEGER,
    agent_id VARCHAR(32),
    agent_experience_months INTEGER,
    agent_specialization VARCHAR(32),
    agent_actions JSONB,
    escalated BOOLEAN,
    escalation_reason TEXT,
    transferred_count INTEGER,
    satisfaction_score INTEGER,
    feedback_text TEXT,
    resolution_helpful BOOLEAN,
    tags JSONB,
    related_tickets JSONB,
    kb_articles_viewed JSONB,
    kb_articles_helpful JSONB,
    environment VARCHAR(16),
    account_age_days INTEGER,
    account_monthly_value FLOAT,
    similar_issues_last_30_days INTEGER,
    product_version_age_days INTEGER,
    known_issue BOOLEAN,
    bug_report_filed BOOLEAN,
    resolution_template_used VARCHAR(64),
    auto_suggested_solutions JSONB,
    auto_suggestion_accepted BOOLEAN,
    ticket_text_length INTEGER,
    response_count INTEGER,
    attachments_count INTEGER,
    contains_error_code BOOLEAN,
    contains_stack_trace BOOLEAN,
    business_impact VARCHAR(16),
    affected_users INTEGER,
    weekend_ticket BOOLEAN,
    after_hours BOOLEAN,
    language VARCHAR(8),
    region VARCHAR(8),
    data_split VARCHAR(8),
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tickets_created ON tickets(created_at);
CREATE INDEX IF NOT EXISTS idx_tickets_product ON tickets(product);
CREATE INDEX IF NOT EXISTS idx_tickets_category ON tickets(category);
CREATE INDEX IF NOT EXISTS idx_tickets_split ON tickets(data_split);

CREATE TABLE IF NOT EXISTS ticket_features (
    ticket_id VARCHAR(32) PRIMARY KEY REFERENCES tickets(ticket_id),
    subject_length INTEGER,
    description_length INTEGER,
    combined_text_length INTEGER,
    word_count INTEGER,
    has_error_code BOOLEAN,
    has_stack_trace BOOLEAN,
    error_code_count INTEGER,
    customer_ticket_count_30d INTEGER,
    customer_avg_satisfaction FLOAT,
    customer_escalation_rate FLOAT,
    product_avg_resolution_hrs FLOAT,
    product_ticket_volume_7d INTEGER,
    product_category_dist JSONB,
    hour_of_day INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    is_after_hours BOOLEAN,
    days_since_product_update INTEGER,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS product_issues (
    id SERIAL PRIMARY KEY,
    product VARCHAR(64) NOT NULL,
    issue_type VARCHAR(64) NOT NULL,
    frequency INTEGER DEFAULT 1,
    avg_resolution_hrs FLOAT,
    UNIQUE(product, issue_type)
);

CREATE TABLE IF NOT EXISTS issue_solutions (
    id SERIAL PRIMARY KEY,
    issue_type VARCHAR(64) NOT NULL,
    resolution_code VARCHAR(64) NOT NULL,
    resolution_template TEXT,
    success_rate FLOAT,
    usage_count INTEGER DEFAULT 1,
    UNIQUE(issue_type, resolution_code)
);

CREATE TABLE IF NOT EXISTS error_code_mapping (
    id SERIAL PRIMARY KEY,
    error_code VARCHAR(128) NOT NULL,
    product VARCHAR(64),
    issue_type VARCHAR(64),
    resolution_code VARCHAR(64),
    resolution_text TEXT,
    occurrences INTEGER DEFAULT 1,
    UNIQUE(error_code, product, issue_type)
);

CREATE TABLE IF NOT EXISTS metric_snapshots (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(128) NOT NULL,
    metric_value FLOAT NOT NULL,
    dimensions JSONB,
    snapshot_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON metric_snapshots(metric_name, snapshot_at);

CREATE TABLE IF NOT EXISTS agent_feedback (
    id SERIAL PRIMARY KEY,
    ticket_id VARCHAR(32) REFERENCES tickets(ticket_id),
    predicted_category VARCHAR(64),
    corrected_category VARCHAR(64),
    predicted_subcategory VARCHAR(64),
    corrected_subcategory VARCHAR(64),
    suggested_resolution TEXT,
    resolution_accepted BOOLEAN,
    agent_id VARCHAR(32),
    feedback_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(64) NOT NULL,
    version VARCHAR(32) NOT NULL,
    mlflow_run_id VARCHAR(64),
    metrics JSONB,
    is_active BOOLEAN DEFAULT FALSE,
    trained_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(model_name, version)
);
