import pytest
from datetime import datetime
from agents.schemas import Evidence, PipelineState
from agents.root_cause_analysis import RootCauseAnalysisAgent

@pytest.fixture
def agent():
    return RootCauseAnalysisAgent()

def test_cascading_schema_drift_failure(agent):
    # Failure Detection flagged training crash AND schema drift
    evidence_list = [
        Evidence(
            agent_name="FailureDetectionAgent",
            finding="Training crashed: TypeError",
            confidence_score=1.0,
            supporting_data={"failure_type": "training_divergence", "error_message": "TypeError: 'float' object..."}
        ),
        Evidence(
            agent_name="FailureDetectionAgent",
            finding="Schema Drift: column 'income' type mismatch",
            confidence_score=0.9,
            supporting_data={"failure_type": "schema_drift", "type_mismatches": {"income": {"expected": "float64", "actual": "object"}}}
        )
    ]
    
    history = [
        PipelineState(stage="data_ingestion", status="success", timestamp=datetime.now()),
        PipelineState(stage="preprocessing", status="success", timestamp=datetime.now())
    ]
    
    diagnoses = agent.diagnose(evidence_list, history)
    assert len(diagnoses) == 1
    diag = diagnoses[0]
    assert diag.supporting_data["root_cause_type"] == "data_schema_cascade"
    assert diag.confidence_score == 0.90
    assert "schema drift" in diag.finding.lower()
    assert "income" in diag.supporting_data["drifted_columns"]

def test_resource_exhaustion_diagnose(agent):
    # Failure Detection flagged training crash and OOM
    evidence_list = [
        Evidence(
            agent_name="FailureDetectionAgent",
            finding="Training crashed: Process killed",
            confidence_score=1.0,
            supporting_data={"failure_type": "training_divergence", "error_message": "Process killed"}
        ),
        Evidence(
            agent_name="FailureDetectionAgent",
            finding="Resource exhaustion (OOM): Limit 1024MB exceeded",
            confidence_score=1.0,
            supporting_data={"failure_type": "resource_exhaustion", "resource": "memory", "limit_mb": 1024.0}
        )
    ]
    
    diagnoses = agent.diagnose(evidence_list)
    # This might match both data schema cascade or resource constraint if any other flags are there.
    # But here we only have training_divergence and resource_exhaustion.
    assert len(diagnoses) == 1
    diag = diagnoses[0]
    assert diag.supporting_data["root_cause_type"] == "resource_constraint"
    assert diag.supporting_data["resource_type"] == "memory"
    assert diag.confidence_score == 0.95

def test_algorithmic_instability(agent):
    # Only training divergence with NaN loss, no schema drift
    evidence_list = [
        Evidence(
            agent_name="FailureDetectionAgent",
            finding="Training divergence: loss is NaN",
            confidence_score=1.0,
            supporting_data={"failure_type": "training_divergence", "metric": "loss", "value": "NaN"}
        )
    ]
    
    diagnoses = agent.diagnose(evidence_list)
    assert len(diagnoses) == 1
    diag = diagnoses[0]
    assert diag.supporting_data["root_cause_type"] == "algorithmic_instability"
    assert diag.confidence_score == 0.85

def test_model_overfitting_underfitting(agent):
    # Evaluation collapse only
    evidence_list = [
        Evidence(
            agent_name="FailureDetectionAgent",
            finding="Evaluation collapse: Accuracy 0.65 below threshold",
            confidence_score=0.9,
            supporting_data={"failure_type": "evaluation_collapse", "accuracy": 0.65}
        )
    ]
    
    diagnoses = agent.diagnose(evidence_list)
    assert len(diagnoses) == 1
    diag = diagnoses[0]
    assert diag.supporting_data["root_cause_type"] == "model_underfitting_overfitting"
    assert diag.confidence_score == 0.75

def test_ambiguous_failure(agent):
    # A generic training crash with no specific indicators
    evidence_list = [
        Evidence(
            agent_name="FailureDetectionAgent",
            finding="Stage failed: exit code 1",
            confidence_score=1.0,
            supporting_data={"failure_type": "some_unknown_failure"}
        )
    ]

    diagnoses = agent.diagnose(evidence_list)
    # Should generate multiple hypotheses
    assert len(diagnoses) == 2
    assert diagnoses[0].supporting_data["root_cause_type"] == "uncaught_exception"
    assert diagnoses[0].confidence_score == 0.60
    assert diagnoses[1].supporting_data["root_cause_type"] == "environment_issue"
    assert diagnoses[1].confidence_score == 0.40


def test_ingestion_failure_no_downstream_symptom():
    """Pipeline dies at ingestion (empty dataset). No training/eval evidence exists.
    Agent should attribute directly to ingestion, NOT fall back to generic hypothesis."""
    evidence = [
        Evidence(
            agent_name="FailureDetectionAgent",
            finding="Ingestion succeeded but dataset is empty (0 rows).",
            confidence_score=1.0,
            supporting_data={
                "failure_type": "ingestion_failure",
                "row_count": 0,
                "stage": "data_ingestion",
            },
        )
    ]
    result = RootCauseAnalysisAgent().diagnose(evidence, pipeline_history=[])
    assert len(result) == 1
    top = result[0]
    assert top.supporting_data["root_cause_type"] == "ingestion_failure_direct"
    assert top.confidence_score == pytest.approx(0.95)
    assert top.supporting_data["root_cause_type"] != "uncaught_exception"


def test_schema_drift_no_downstream_symptom():
    """Schema drift during preprocessing, before training starts.
    Agent should attribute directly to schema drift, NOT fall back generically."""
    evidence = [
        Evidence(
            agent_name="FailureDetectionAgent",
            finding="Critical Schema Drift: Missing expected columns: ['income']",
            confidence_score=1.0,
            supporting_data={
                "failure_type": "schema_drift",
                "severity": "critical",
                "missing_columns": ["income"],
                "current_columns": ["age", "gender"],
            },
        )
    ]
    result = RootCauseAnalysisAgent().diagnose(evidence, pipeline_history=[])
    assert len(result) == 1
    top = result[0]
    assert top.supporting_data["root_cause_type"] == "schema_drift_direct"
    assert top.confidence_score == pytest.approx(0.85)
    assert "income" in top.supporting_data["drifted_columns"]


def test_ingestion_prefers_cascade_when_downstream_symptom_present():
    """Regression guard: when a downstream training_divergence IS present alongside
    ingestion/schema evidence, the cascade rule (data_schema_cascade) must still win
    over the new standalone rules. Standalone rules only fire when `not hypotheses`."""
    evidence = [
        Evidence(
            agent_name="FailureDetectionAgent",
            finding="Critical Schema Drift: Missing expected columns: ['income']",
            confidence_score=1.0,
            supporting_data={"failure_type": "schema_drift", "missing_columns": ["income"]},
        ),
        Evidence(
            agent_name="FailureDetectionAgent",
            finding="Training crashed: KeyError 'income'",
            confidence_score=1.0,
            supporting_data={"failure_type": "training_divergence"},
        ),
    ]
    result = RootCauseAnalysisAgent().diagnose(evidence, pipeline_history=[])
    top = result[0]
    assert top.supporting_data["root_cause_type"] == "data_schema_cascade"
