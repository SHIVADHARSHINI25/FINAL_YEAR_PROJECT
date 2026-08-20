"""Unit tests for ModelAnalysisAgent."""
import math
import pytest
from agents.model_analysis import ModelAnalysisAgent


@pytest.fixture
def agent():
    return ModelAnalysisAgent()


# ------------------------------------------------------------------
# Loss curve tests
# ------------------------------------------------------------------

def test_loss_nan_detected(agent):
    history = [{"loss": float("nan")}]
    ev = agent.analyze(history)
    assert any("divergence" in e.finding.lower() for e in ev)


def test_loss_monotonically_increasing_detected(agent):
    history = [{"loss": v} for v in [0.5, 0.6, 0.7, 0.8]]
    ev = agent.analyze(history)
    checks = [e.supporting_data["check_type"] for e in ev]
    assert "loss_divergence" in checks


def test_loss_plateau_detected(agent):
    # Flat loss for 6 epochs — well above std threshold
    history = [{"loss": 0.45 + i * 0.0001} for i in range(6)]
    ev = agent.analyze(history)
    checks = [e.supporting_data["check_type"] for e in ev]
    assert "loss_plateau" in checks


def test_loss_plateau_not_triggered_on_converging_loss(agent):
    history = [{"loss": 1.0 - i * 0.1} for i in range(6)]
    ev = agent.analyze(history)
    checks = [e.supporting_data["check_type"] for e in ev]
    assert "loss_plateau" not in checks


def test_loss_oscillation_detected(agent):
    history = [{"loss": v} for v in [1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0]]
    ev = agent.analyze(history)
    checks = [e.supporting_data["check_type"] for e in ev]
    assert "loss_oscillation" in checks


def test_healthy_loss_no_issues(agent):
    # Smoothly decreasing loss — should produce zero findings
    history = [{"loss": 1.0 - i * 0.1} for i in range(5)]
    ev = agent.analyze(history)
    assert ev == []


# ------------------------------------------------------------------
# Overfitting / underfitting tests
# ------------------------------------------------------------------

def test_overfitting_detected(agent):
    history = [{"train_accuracy": 0.99, "val_accuracy": 0.75}]
    ev = agent.analyze(history)
    checks = [e.supporting_data["check_type"] for e in ev]
    assert "overfitting" in checks


def test_underfitting_detected(agent):
    history = [{"train_accuracy": 0.52, "val_accuracy": 0.50}]
    ev = agent.analyze(history)
    checks = [e.supporting_data["check_type"] for e in ev]
    assert "underfitting" in checks


def test_healthy_accuracy_no_flag(agent):
    history = [{"train_accuracy": 0.88, "val_accuracy": 0.85}]
    ev = agent.analyze(history)
    assert ev == []


# ------------------------------------------------------------------
# Learning rate spike tests
# ------------------------------------------------------------------

def test_lr_spike_detected(agent):
    history = [
        {"learning_rate": 0.001, "loss": 0.5},
        {"learning_rate": 0.1,   "loss": 1.5},   # both LR and loss spiked
    ]
    ev = agent.analyze(history)
    checks = [e.supporting_data["check_type"] for e in ev]
    assert "lr_spike" in checks


def test_lr_increase_without_loss_spike_not_flagged(agent):
    history = [
        {"learning_rate": 0.001, "loss": 0.5},
        {"learning_rate": 0.002, "loss": 0.48},  # LR up, loss down — healthy warm-up
    ]
    ev = agent.analyze(history)
    checks = [e.supporting_data["check_type"] for e in ev]
    assert "lr_spike" not in checks


# ------------------------------------------------------------------
# Gradient tests
# ------------------------------------------------------------------

def test_exploding_gradients_nan(agent):
    history = [{"grad_norm": float("nan")}]
    ev = agent.analyze(history)
    checks = [e.supporting_data["check_type"] for e in ev]
    assert "exploding_gradients" in checks


def test_exploding_gradients_large_value(agent):
    history = [{"grad_norm": 500.0}]
    ev = agent.analyze(history)
    checks = [e.supporting_data["check_type"] for e in ev]
    assert "exploding_gradients" in checks


def test_vanishing_gradients_detected(agent):
    history = [{"grad_norm": 1e-8}]
    ev = agent.analyze(history)
    checks = [e.supporting_data["check_type"] for e in ev]
    assert "vanishing_gradients" in checks


def test_normal_gradient_no_flag(agent):
    history = [{"grad_norm": 0.5}]
    ev = agent.analyze(history)
    assert ev == []


# ------------------------------------------------------------------
# Hyperparameter tests
# ------------------------------------------------------------------

def test_high_lr_flagged(agent):
    ev = agent.analyze([], hyperparameters={"learning_rate": 5.0})
    checks = [e.supporting_data["check_type"] for e in ev]
    assert "hyperparameter_sanity" in checks


def test_zero_lr_flagged(agent):
    ev = agent.analyze([], hyperparameters={"learning_rate": 0})
    assert any("non-positive" in e.finding for e in ev)


def test_batch_larger_than_dataset_flagged(agent):
    ev = agent.analyze([], hyperparameters={"batch_size": 512, "dataset_size": 100})
    checks = [e.supporting_data["check_type"] for e in ev]
    assert "hyperparameter_sanity" in checks


def test_negative_epochs_flagged(agent):
    ev = agent.analyze([], hyperparameters={"epochs": -5})
    assert any("epochs" in e.finding for e in ev)


def test_reasonable_hyperparams_no_flag(agent):
    ev = agent.analyze([], hyperparameters={
        "learning_rate": 0.001, "batch_size": 32, "dataset_size": 1000, "epochs": 20
    })
    assert ev == []


# ------------------------------------------------------------------
# Empty / minimal input edge cases
# ------------------------------------------------------------------

def test_empty_history_no_crash(agent):
    ev = agent.analyze([])
    assert ev == []


def test_single_epoch_history_no_crash(agent):
    ev = agent.analyze([{"loss": 0.5}])
    assert ev == []


def test_no_metrics_no_crash(agent):
    ev = agent.analyze([{}])
    assert ev == []
