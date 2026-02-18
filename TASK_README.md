# Full Stack AI Engineer Technical Assessment: Intelligent Product Support System

## The Challenge

Your company's product support team is drowning in tickets. With 100,000+ historical tickets and 500+ new ones daily, they need an intelligent system that can learn from past resolutions to automatically handle routine issues, surface emerging problems, and help agents resolve complex cases faster.

## Your Mission

Build an end-to-end intelligent support system that:

1. **Understands** incoming support tickets (categorization, priority, sentiment)
2. **Retrieves** relevant solutions from past resolutions and documentation
3. **Detects** emerging issues and anomalies in real-time
4. **Learns** from agent actions and customer feedback


## Important Notes

### Freedom of Choice
- Use any programming languages, frameworks, or cloud services you prefer
- Choose appropriate MLOps tools and practices that align with industry standards
- You can simulate external services if needed
- Focus on demonstrating your approach rather than perfect implementation of every feature

### What We're Looking For
- **System thinking**: How components work together
- **Practical choices**: Real-world trade-offs and constraints
- **Production mindset**: Building for reliability, reproducibility, and scale
- **Engineering excellence**: Clean, maintainable code with proper dependency management
- **Clear communication**: Well-documented decisions and code

## Technical Requirements

### The System You'll Build

Create a unified AI system that processes support tickets through multiple intelligent components working together. Your implementation should include:

**Data Pipeline & Feature Engineering**
- Ingest ticket data from provided JSON files
- Build a feature store that serves both real-time and batch features
- Create analytical models for business metrics (resolution times, satisfaction drivers, agent performance)
- Implement data quality checks and monitoring
- Design event schemas for tracking all system interactions

**Intelligent Processing Engine (Components should build on each other)**

1. **Multi-Model Categorization** (Foundation Layer)
   - Implement both traditional ML (XGBoost/CatBoost) and deep learning (TensorFlow/Keras) approaches for ticket classification
   - Compare their performance/latency trade-offs
   - Target >85% weighted F1 score
   - Output categories feed into the retrieval system

2. **RAG + Graph-RAG Solution Finder** (Uses categorization output)
   - Build a hybrid retrieval system that uses the predicted categories to improve search
   - Combines semantic search with keyword matching and metadata filtering
   - Maps relationships: Products ↔ Issues ↔ Solutions ↔ Historical Tickets
   - Links entities (error codes, product names, technical terms)
   - Re-ranks results based on resolution success rates and category relevance

3. **Anomaly Detection** (Analyzes patterns from both above components)
   - Create an analyzer that monitors categorization patterns and retrieval failures to identify:
     - Unusual ticket volume patterns per category
     - New issue types not matching existing categories
     - Sentiment shifts in specific product areas
     - Emerging problems based on retrieval system query patterns

**System Enhancements**
- Implement proper model versioning and experiment tracking using industry-standard MLOps practices
- Ensure reproducible model training and deployment through appropriate tooling
- Design for containerized deployment to ensure consistency across environments
- Performance monitoring and drift detection  
- Fallback mechanisms for system failures
- Capture agent corrections and customer feedback
- Update models based on resolution success/failure
- Automated quality scoring for generated responses

## Provided Resources

### Dataset
You will receive a single JSON file containing 100,000 historical support tickets. You should:
- Split this into training, validation, and test sets (suggested: 70/15/15 split)
- Use appropriate techniques to handle class imbalance
- Document your data splitting strategy and rationale
- Consider how your system would handle streaming data in production

### Dataset Structure
```json
{
  "ticket_id": "TK-2024-001234",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-16T14:20:00Z",
  "customer_id": "CUST-5678",
  "customer_tier": "enterprise",
  "organization_id": "ORG-123",
  "product": "DataSync Pro",
  "product_version": "3.2.1",
  "product_module": "sync_engine",
  "category": "Technical Issue",  // Target variable to predict
  "subcategory": "Configuration",  // Target variable to predict  
  "priority": "high",
  "severity": "P2",
  "channel": "email",
  "subject": "Database sync failing with timeout error",
  "description": "Getting ERROR_TIMEOUT_429 when syncing large datasets. This started happening after the recent update. The sync works for small datasets but fails consistently for anything over 1GB.",
  "error_logs": "2024-01-15 10:25:33 ERROR_TIMEOUT_429: Connection timeout after 30s\n2024-01-15 10:25:34 RETRY_FAILED: Max retries exceeded",
  "stack_trace": "at SyncEngine.execute(sync_engine.py:156)\nat DataProcessor.run(processor.py:89)",
  "customer_sentiment": "frustrated",  // Derived from text analysis
  "previous_tickets": 3,  // Number of previous tickets from this customer
  "resolution": "Increased batch size limits in config.yaml from 100MB to 500MB. Optimized query performance by adding index on timestamp field. Customer confirmed issue resolved.",
  "resolution_code": "CONFIG_CHANGE",
  "resolved_at": "2024-01-16T14:20:00Z",
  "resolution_time_hours": 27.83,
  "resolution_attempts": 2,
  "agent_id": "AGENT-101",
  "agent_experience_months": 18,
  "agent_specialization": "database",
  "agent_actions": ["viewed_logs", "checked_config", "applied_fix", "verified_resolution"],
  "escalated": false,
  "escalation_reason": null,
  "transferred_count": 1,
  "satisfaction_score": 4,  // 1-5 scale
  "feedback_text": "Issue resolved but took too long",
  "resolution_helpful": true,  // Did the suggested resolution work?
  "tags": ["database", "sync", "timeout", "configuration"],
  "related_tickets": ["TK-2024-001230", "TK-2024-001145"],  // Similar issues
  "kb_articles_viewed": ["KB-429", "KB-887"],
  "kb_articles_helpful": ["KB-887"],  // Which articles actually helped
  "environment": "production",
  "account_age_days": 456,
  "account_monthly_value": 5000,
  "similar_issues_last_30_days": 45,  
  "product_version_age_days": 15,
  "known_issue": false,
  "bug_report_filed": false,
  "resolution_template_used": "TEMPLATE-DB-TIMEOUT",
  "auto_suggested_solutions": ["KB-887", "KB-429", "KB-1001"],  // What system previously suggested
  "auto_suggestion_accepted": false,  // Whether agent used the suggestion
  "ticket_text_length": 287,
  "response_count": 3,  // Number of back-and-forth messages
  "attachments_count": 2,
  "contains_error_code": true,
  "contains_stack_trace": true,
  "business_impact": "medium",  // low/medium/high/critical
  "affected_users": 15,
  "weekend_ticket": false,
  "after_hours": false,
  "language": "en",
  "region": "NA"
}
```

## Deliverables

### 1. Project Code & Working System
- Complete codebase with all components integrated
- API endpoints for ticket processing and solution retrieval
- Data pipeline for JSON file ingestion and processing
- Environment setup that ensures reproducibility across different machines
- Clear dependencies management and isolation strategy
- Fully containerized solution with orchestration for easy deployment

### 2. Architecture Documentation
- System architecture diagram showing all components and data flows
- Technology choices with justifications
- Description of how components interact and build on each other
- Deployment strategy for consistent execution across environments

### 3. Model Documentation
- Performance benchmarks for all models
- Comparison of XGBoost/CatBoost vs TensorFlow approaches
- Feature importance analysis
- Error analysis and failure cases
- Experiment tracking and model lineage documentation

### 4. README
- Clear setup and installation instructions with minimal manual configuration
- API documentation with sample requests/responses
- Key design decisions and trade-offs
- Instructions for running the system with the provided JSON file
- Guide for reproducing training and evaluation results


