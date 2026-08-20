from datetime import datetime
from pydantic import BaseModel, Field

class Evidence(BaseModel):
    agent_name: str
    finding: str                # human-readable summary
    confidence_score: float     # 0.0–1.0
    supporting_data: dict       # raw metrics/evidence backing the finding
    recommended_action: str | None = None
    predicted_impact: str | None = None      # e.g. "accuracy may drop 3-5%"
    estimated_cost: str | None = None        # e.g. "low compute, ~2 min"
    timestamp: datetime = Field(default_factory=datetime.now)

class PipelineState(BaseModel):
    stage: str                  # e.g., "data_ingestion", "preprocessing", "training", "evaluation"
    status: str                 # "success", "failed"
    metrics: dict = Field(default_factory=dict)       # e.g., {"loss": 0.25, "accuracy": 0.85, "execution_time_sec": 45.2, "memory_mb": 512}
    data_schema: dict | None = None  # e.g., {"columns": {"age": "int64", "income": "float64"}}
    data_summary: dict | None = None # e.g., {"row_count": 1000, "missing_percentages": {"age": 0.05}}
    logs: list[str] = Field(default_factory=list)
    error_message: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class RecoveryAction(BaseModel):
    """Records the details and outcome of a simulated recovery strategy execution."""
    action_id: str
    strategy_name: str
    description: str
    executed: bool
    success: bool | None = None
    execution_log: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class VerificationResult(BaseModel):
    """Records a single verification check performed after a recovery action."""
    check_name: str
    passed: bool
    details: str
    metric_before: float | None = None
    metric_after: float | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class KnowledgeEntry(BaseModel):
    """A historical record of a failure, the recovery strategy used, and its outcome."""
    failure_signature: str          # hashable/comparable summary of the failure
    root_cause_type: str
    recovery_strategy_used: str
    outcome: str                    # "success" | "failure" | "partial"
    notes: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
