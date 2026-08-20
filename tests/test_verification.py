"""Unit tests for VerificationAgent."""
import math
import pytest
from agents.verification import VerificationAgent


@pytest.fixture
def agent():
    return VerificationAgent()


# ------------------------------------------------------------------
# Loss improvement tests
# ------------------------------------------------------------------

def test_loss_improvement_passes(agent):
    results = agent.verify(
        pre_metrics={"val_loss": 0.8},
        post_metrics={"val_loss": 0.5},
    )
    loss_r = next(r for r in results if r.check_name == "loss_improvement")
    assert loss_r.passed is True
    assert loss_r.metric_before == pytest.approx(0.8)
    assert loss_r.metric_after == pytest.approx(0.5)


def test_loss_improvement_fails_when_loss_unchanged(agent):
    results = agent.verify(
        pre_metrics={"val_loss": 0.8},
        post_metrics={"val_loss": 0.8},
    )
    loss_r = next(r for r in results if r.check_name == "loss_improvement")
    assert loss_r.passed is False


def test_loss_improvement_fails_when_loss_worsens(agent):
    results = agent.verify(
        pre_metrics={"loss": 0.5},
        post_metrics={"loss": 0.9},
    )
    loss_r = next(r for r in results if r.check_name == "loss_improvement")
    assert loss_r.passed is False


def test_loss_check_skipped_gracefully_when_no_loss_key(agent):
    results = agent.verify(pre_metrics={}, post_metrics={})
    loss_r = next(r for r in results if r.check_name == "loss_improvement")
    # No data → passes by default (skip)
    assert loss_r.passed is True


# ------------------------------------------------------------------
# Accuracy recovery tests
# ------------------------------------------------------------------

def test_accuracy_recovery_passes_when_improved(agent):
    results = agent.verify(
        pre_metrics={"val_accuracy": 0.75},
        post_metrics={"val_accuracy": 0.85},
    )
    acc_r = next(r for r in results if r.check_name == "accuracy_recovery")
    assert acc_r.passed is True


def test_accuracy_recovery_passes_within_allowed_drop(agent):
    # Drop of 0.01 ≤ allowed 0.02
    results = agent.verify(
        pre_metrics={"val_accuracy": 0.85},
        post_metrics={"val_accuracy": 0.84},
    )
    acc_r = next(r for r in results if r.check_name == "accuracy_recovery")
    assert acc_r.passed is True


def test_accuracy_recovery_fails_when_large_drop(agent):
    results = agent.verify(
        pre_metrics={"val_accuracy": 0.90},
        post_metrics={"val_accuracy": 0.70},
    )
    acc_r = next(r for r in results if r.check_name == "accuracy_recovery")
    assert acc_r.passed is False


# ------------------------------------------------------------------
# NaN detection tests
# ------------------------------------------------------------------

def test_no_nan_passes_clean_metrics(agent):
    results = agent.verify(
        pre_metrics={"loss": 0.5},
        post_metrics={"loss": 0.3, "accuracy": 0.88},
    )
    nan_r = next(r for r in results if r.check_name == "no_new_nan")
    assert nan_r.passed is True


def test_nan_in_post_metrics_fails(agent):
    results = agent.verify(
        pre_metrics={"loss": 0.5},
        post_metrics={"loss": float("nan")},
    )
    nan_r = next(r for r in results if r.check_name == "no_new_nan")
    assert nan_r.passed is False


def test_inf_in_post_metrics_fails(agent):
    results = agent.verify(
        pre_metrics={"loss": 0.5},
        post_metrics={"loss": float("inf")},
    )
    nan_r = next(r for r in results if r.check_name == "no_new_nan")
    assert nan_r.passed is False


# ------------------------------------------------------------------
# Resource check tests
# ------------------------------------------------------------------

def test_resource_ok_when_below_critical(agent):
    results = agent.verify(
        pre_metrics={},
        post_metrics={"memory_percent": 70.0, "cpu_percent": 60.0},
    )
    resource_results = [r for r in results if r.check_name.startswith("resource_ok")]
    assert all(r.passed for r in resource_results)


def test_resource_fails_when_still_critical(agent):
    results = agent.verify(
        pre_metrics={},
        post_metrics={"memory_percent": 97.0},
    )
    mem_r = next(r for r in results if "memory" in r.check_name)
    assert mem_r.passed is False


# ------------------------------------------------------------------
# Schema validation tests
# ------------------------------------------------------------------

def test_schema_valid_when_matching(agent):
    ref = {"age": "int64", "income": "float64"}
    results = agent.verify(
        pre_metrics={},
        post_metrics={"data_schema": {"age": "int64", "income": "float64"}},
        reference_schema=ref,
    )
    schema_r = next(r for r in results if r.check_name == "schema_valid")
    assert schema_r.passed is True


def test_schema_fails_on_missing_column(agent):
    ref = {"age": "int64", "income": "float64"}
    results = agent.verify(
        pre_metrics={},
        post_metrics={"data_schema": {"age": "int64"}},  # income missing
        reference_schema=ref,
    )
    schema_r = next(r for r in results if r.check_name == "schema_valid")
    assert schema_r.passed is False
    assert "income" in schema_r.details


def test_schema_fails_on_type_mismatch(agent):
    ref = {"age": "int64", "income": "float64"}
    results = agent.verify(
        pre_metrics={},
        post_metrics={"data_schema": {"age": "object", "income": "float64"}},
        reference_schema=ref,
    )
    schema_r = next(r for r in results if r.check_name == "schema_valid")
    assert schema_r.passed is False


def test_schema_fails_when_no_data_schema_key(agent):
    results = agent.verify(
        pre_metrics={},
        post_metrics={},
        reference_schema={"age": "int64"},
    )
    schema_r = next(r for r in results if r.check_name == "schema_valid")
    assert schema_r.passed is False


# ------------------------------------------------------------------
# All-clear scenario
# ------------------------------------------------------------------

def test_all_checks_pass_healthy_recovery(agent):
    """A clean recovery where everything improved."""
    ref = {"age": "int64", "income": "float64"}
    results = agent.verify(
        pre_metrics={"val_loss": 0.9, "val_accuracy": 0.72, "memory_percent": 60.0},
        post_metrics={
            "val_loss": 0.4,
            "val_accuracy": 0.88,
            "memory_percent": 55.0,
            "cpu_percent": 40.0,
            "data_schema": {"age": "int64", "income": "float64"},
        },
        reference_schema=ref,
    )
    assert all(r.passed for r in results), [r for r in results if not r.passed]
