import pytest
from datetime import datetime
from agents.schemas import PipelineState
from agents.failure_detection import FailureDetectionAgent

@pytest.fixture
def agent():
    ref_schema = {
        "age": "int64",
        "income": "float64",
        "target": "int64"
    }
    return FailureDetectionAgent(
        reference_schema=ref_schema,
        min_accuracy=0.80,
        min_f1=0.75,
        max_execution_time_sec=100.0,
        max_memory_mb=1024.0
    )

def test_healthy_pipeline(agent):
    state = PipelineState(
        stage="evaluation",
        status="success",
        metrics={"accuracy": 0.85, "f1": 0.78, "memory_mb": 500.0, "execution_time_sec": 30.0},
        timestamp=datetime.now()
    )
    evidences = agent.monitor(state)
    assert len(evidences) == 0

def test_ingestion_failure_explicit(agent):
    state = PipelineState(
        stage="data_ingestion",
        status="failed",
        error_message="FileNotFoundError: dataset.csv not found",
        timestamp=datetime.now()
    )
    evidences = agent.monitor(state)
    assert len(evidences) == 1
    evidence = evidences[0]
    assert evidence.supporting_data["failure_type"] == "ingestion_failure"
    assert "FileNotFoundError" in evidence.finding
    assert evidence.confidence_score == 1.0

def test_ingestion_failure_empty(agent):
    state = PipelineState(
        stage="data_ingestion",
        status="success",
        data_summary={"row_count": 0},
        timestamp=datetime.now()
    )
    evidences = agent.monitor(state)
    assert len(evidences) == 1
    assert evidences[0].supporting_data["failure_type"] == "ingestion_failure"
    assert "dataset is empty" in evidences[0].finding

def test_schema_drift_missing_column(agent):
    state = PipelineState(
        stage="preprocessing",
        status="success",
        data_schema={"columns": {"age": "int64", "target": "int64"}}, # income missing
        timestamp=datetime.now()
    )
    evidences = agent.monitor(state)
    assert len(evidences) == 1
    assert evidences[0].supporting_data["failure_type"] == "schema_drift"
    assert "income" in evidences[0].supporting_data["missing_columns"]
    assert evidences[0].supporting_data["severity"] == "critical"

def test_schema_drift_type_mismatch(agent):
    state = PipelineState(
        stage="preprocessing",
        status="success",
        data_schema={"columns": {"age": "int64", "income": "object", "target": "int64"}}, # income is object instead of float64
        timestamp=datetime.now()
    )
    evidences = agent.monitor(state)
    assert len(evidences) == 1
    assert evidences[0].supporting_data["failure_type"] == "schema_drift"
    assert "income" in evidences[0].supporting_data["type_mismatches"]
    assert evidences[0].supporting_data["severity"] == "high"

def test_training_failure_nan_loss(agent):
    state = PipelineState(
        stage="training",
        status="success",
        metrics={"loss": "NaN", "epochs": 10},
        timestamp=datetime.now()
    )
    evidences = agent.monitor(state)
    assert len(evidences) == 1
    assert evidences[0].supporting_data["failure_type"] == "training_divergence"
    assert "NaN" in evidences[0].finding

def test_training_failure_non_convergence(agent):
    state = PipelineState(
        stage="training",
        status="success",
        metrics={"loss": 2.5, "train_accuracy": 0.12, "epochs": 6},
        timestamp=datetime.now()
    )
    evidences = agent.monitor(state)
    assert len(evidences) == 1
    assert evidences[0].supporting_data["failure_type"] == "training_divergence"
    assert "Non-convergence" in evidences[0].finding

def test_resource_failure_oom_log(agent):
    state = PipelineState(
        stage="training",
        status="failed",
        error_message="Process terminated with Exit code 137 (OOM)",
        timestamp=datetime.now()
    )
    evidences = agent.monitor(state)
    assert len(evidences) == 2
    types = [e.supporting_data["failure_type"] for e in evidences]
    assert "training_divergence" in types
    assert "resource_exhaustion" in types
    resource_ev = next(e for e in evidences if e.supporting_data["failure_type"] == "resource_exhaustion")
    assert resource_ev.supporting_data["resource"] == "memory"

def test_resource_failure_oom_threshold(agent):
    state = PipelineState(
        stage="preprocessing",
        status="success",
        metrics={"memory_mb": 1500.0}, # max is 1024
        timestamp=datetime.now()
    )
    evidences = agent.monitor(state)
    assert len(evidences) == 1
    assert evidences[0].supporting_data["failure_type"] == "resource_exhaustion"
    assert evidences[0].supporting_data["resource"] == "memory"

def test_resource_failure_timeout(agent):
    state = PipelineState(
        stage="training",
        status="success",
        metrics={"execution_time_sec": 120.0}, # max is 100
        timestamp=datetime.now()
    )
    evidences = agent.monitor(state)
    assert len(evidences) == 1
    assert evidences[0].supporting_data["failure_type"] == "resource_exhaustion"
    assert evidences[0].supporting_data["resource"] == "time"

def test_evaluation_collapse(agent):
    state = PipelineState(
        stage="evaluation",
        status="success",
        metrics={"accuracy": 0.72, "f1": 0.65}, # both below threshold
        timestamp=datetime.now()
    )
    evidences = agent.monitor(state)
    assert len(evidences) == 1
    assert evidences[0].supporting_data["failure_type"] == "evaluation_collapse"
    assert "Accuracy 0.72 < Threshold 0.80" in evidences[0].finding
    assert "F1 Score 0.65 < Threshold 0.75" in evidences[0].finding
