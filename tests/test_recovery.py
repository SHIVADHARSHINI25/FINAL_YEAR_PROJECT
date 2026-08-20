"""Unit tests for RecoveryAgent."""
import pytest
from agents.recovery import RecoveryAgent
from agents.schemas import Evidence, RecoveryAction


@pytest.fixture
def agent():
    return RecoveryAgent()


def _ev(finding: str) -> Evidence:
    return Evidence(
        agent_name="test",
        finding=finding,
        confidence_score=1.0,
        supporting_data={},
    )


# ------------------------------------------------------------------
# Strategy selection tests
# ------------------------------------------------------------------

def test_gpu_oom_selects_reduce_batch_size(agent):
    ev = [_ev("CRITICAL GPU Memory: 98.0% utilisation")]
    action, evidence = agent.plan(ev)
    assert action.strategy_name == "reduce_batch_size"
    assert action.executed is True


def test_ram_oom_selects_reduce_batch_size(agent):
    ev = [_ev("CRITICAL RAM: 97.0% utilisation")]
    action, _ = agent.plan(ev)
    assert action.strategy_name == "reduce_batch_size"


def test_exploding_gradient_selects_clipping(agent):
    ev = [_ev("Exploding gradients: max gradient norm = 500.00")]
    action, _ = agent.plan(ev)
    assert action.strategy_name == "add_gradient_clipping"


def test_vanishing_gradient_selects_activation_switch(agent):
    ev = [_ev("Vanishing gradients: min gradient norm = 1.00e-08")]
    action, _ = agent.plan(ev)
    assert action.strategy_name == "switch_activation_function"


def test_training_divergence_selects_reduce_lr(agent):
    ev = [_ev("Training divergence: Loss contains NaN/Inf values")]
    action, _ = agent.plan(ev)
    assert action.strategy_name == "reduce_learning_rate"


def test_overfitting_selects_regularisation(agent):
    ev = [_ev("Overfitting detected: train_accuracy=0.99, val_accuracy=0.75")]
    action, _ = agent.plan(ev)
    assert action.strategy_name == "apply_regularisation"


def test_underfitting_selects_capacity_increase(agent):
    ev = [_ev("Underfitting detected: train_accuracy=0.52, val_accuracy=0.50")]
    action, _ = agent.plan(ev)
    assert action.strategy_name == "increase_model_capacity"


def test_plateau_selects_lr_schedule(agent):
    ev = [_ev("Loss plateau detected: std=0.0001 over last 5 epochs")]
    action, _ = agent.plan(ev)
    assert action.strategy_name == "apply_lr_schedule"


def test_schema_drift_selects_preprocessing(agent):
    ev = [_ev("Critical Schema Drift: Missing expected columns: ['income']")]
    action, _ = agent.plan(ev)
    assert action.strategy_name == "retrigger_preprocessing"


def test_ingestion_failure_selects_reingestion(agent):
    ev = [_ev("Ingestion succeeded but dataset is empty (0 rows)")]
    action, _ = agent.plan(ev)
    assert action.strategy_name == "reingestion"


def test_fallback_for_unknown_failure(agent):
    ev = [_ev("Some completely unknown issue occurred")]
    action, evidence = agent.plan(ev)
    assert action.strategy_name == "retrigger_last_stage"
    assert evidence.confidence_score == 0.55


# ------------------------------------------------------------------
# RecoveryAction structure
# ------------------------------------------------------------------

def test_action_has_unique_id(agent):
    ev = [_ev("Training divergence: loss NaN")]
    a1, _ = agent.plan(ev)
    a2, _ = agent.plan(ev)
    assert a1.action_id != a2.action_id


def test_action_success_is_true(agent):
    ev = [_ev("Training divergence: loss NaN")]
    action, _ = agent.plan(ev)
    assert action.success is True


def test_evidence_contains_strategy_name(agent):
    ev = [_ev("Exploding gradients")]
    _, evidence = agent.plan(ev)
    assert evidence.supporting_data["strategy_name"] == "add_gradient_clipping"


def test_evidence_confidence_is_positive(agent):
    ev = [_ev("Training divergence")]
    _, evidence = agent.plan(ev)
    assert 0 < evidence.confidence_score <= 1.0


# ------------------------------------------------------------------
# Priority ordering
# ------------------------------------------------------------------

def test_gpu_oom_takes_priority_over_overfitting(agent):
    """GPU OOM (rule 1) should beat Overfitting (rule 6) in priority."""
    ev = [
        _ev("CRITICAL GPU Memory: 98.0% utilisation"),
        _ev("Overfitting detected: gap=0.30"),
    ]
    action, _ = agent.plan(ev)
    assert action.strategy_name == "reduce_batch_size"


def test_empty_evidence_list_falls_back(agent):
    action, _ = agent.plan([])
    assert action.strategy_name == "retrigger_last_stage"
