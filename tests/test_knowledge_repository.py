"""Unit tests for KnowledgeRepositoryAgent."""
import pytest
from agents.knowledge_repository import KnowledgeRepositoryAgent
from agents.schemas import KnowledgeEntry


@pytest.fixture
def agent():
    return KnowledgeRepositoryAgent()


def _entry(
    signature: str,
    root_cause: str,
    strategy: str,
    outcome: str,
    notes: str | None = None,
) -> KnowledgeEntry:
    return KnowledgeEntry(
        failure_signature=signature,
        root_cause_type=root_cause,
        recovery_strategy_used=strategy,
        outcome=outcome,
        notes=notes,
    )


# ------------------------------------------------------------------
# record() tests
# ------------------------------------------------------------------

def test_record_returns_evidence(agent):
    entry = _entry("nan_loss", "training_divergence", "reduce_learning_rate", "success")
    ev = agent.record(entry)
    assert ev.agent_name == "KnowledgeRepositoryAgent"
    assert "nan_loss" in ev.finding


def test_record_increments_length(agent):
    assert len(agent) == 0
    agent.record(_entry("nan_loss", "training_divergence", "reduce_learning_rate", "success"))
    assert len(agent) == 1
    agent.record(_entry("oom", "memory_pressure", "reduce_batch_size", "success"))
    assert len(agent) == 2


# ------------------------------------------------------------------
# lookup() tests
# ------------------------------------------------------------------

def test_lookup_by_signature(agent):
    agent.record(_entry("nan_loss", "training_divergence", "reduce_learning_rate", "success"))
    agent.record(_entry("oom",      "memory_pressure",    "reduce_batch_size",    "success"))
    results = agent.lookup(failure_signature="nan_loss")
    assert len(results) == 1
    assert results[0].failure_signature == "nan_loss"


def test_lookup_by_root_cause(agent):
    agent.record(_entry("nan_loss_1", "training_divergence", "reduce_learning_rate", "success"))
    agent.record(_entry("nan_loss_2", "training_divergence", "reduce_learning_rate", "failure"))
    agent.record(_entry("oom",        "memory_pressure",    "reduce_batch_size",    "success"))
    results = agent.lookup(root_cause_type="training_divergence")
    assert len(results) == 2


def test_lookup_no_results(agent):
    results = agent.lookup(failure_signature="nonexistent")
    assert results == []


def test_lookup_with_both_filters_acts_as_AND(agent):
    agent.record(_entry("nan_loss", "training_divergence", "reduce_learning_rate", "success"))
    # Same root cause, different signature
    agent.record(_entry("gradient_nan", "training_divergence", "add_gradient_clipping", "success"))
    results = agent.lookup(root_cause_type="training_divergence", failure_signature="nan_loss")
    assert len(results) == 1
    assert results[0].failure_signature == "nan_loss"


# ------------------------------------------------------------------
# advise() tests
# ------------------------------------------------------------------

def test_advise_recommends_best_strategy(agent):
    # 3 successes with reduce_lr, 1 failure
    for _ in range(3):
        agent.record(_entry("nan_loss", "training_divergence", "reduce_learning_rate", "success"))
    agent.record(_entry("nan_loss", "training_divergence", "reduce_learning_rate", "failure"))
    ev = agent.advise("nan_loss")
    assert ev.supporting_data["recommended_strategy"] == "reduce_learning_rate"
    assert 0 < ev.confidence_score <= 1.0


def test_advise_prefers_better_strategy_across_multiple(agent):
    # reduce_lr: 2 successes (score=2), add_clip: 1 success + 1 failure (score=0.5)
    agent.record(_entry("nan_loss", "training_divergence", "reduce_learning_rate", "success"))
    agent.record(_entry("nan_loss", "training_divergence", "reduce_learning_rate", "success"))
    agent.record(_entry("nan_loss", "training_divergence", "add_gradient_clipping", "success"))
    agent.record(_entry("nan_loss", "training_divergence", "add_gradient_clipping", "failure"))
    ev = agent.advise("nan_loss")
    assert ev.supporting_data["recommended_strategy"] == "reduce_learning_rate"


def test_advise_returns_zero_confidence_for_unknown(agent):
    ev = agent.advise("completely_unknown_failure")
    assert ev.confidence_score == 0.0
    assert "No historical data" in ev.finding


def test_advise_counts_cases(agent):
    agent.record(_entry("nan_loss", "training_divergence", "reduce_learning_rate", "success"))
    agent.record(_entry("nan_loss", "training_divergence", "reduce_learning_rate", "partial"))
    ev = agent.advise("nan_loss")
    assert ev.supporting_data["cases_examined"] == 2


# ------------------------------------------------------------------
# summary() tests
# ------------------------------------------------------------------

def test_summary_empty_repository(agent):
    ev = agent.summary()
    assert "empty" in ev.finding.lower()
    assert ev.supporting_data["total_entries"] == 0


def test_summary_reflects_correct_counts(agent):
    agent.record(_entry("nan_loss", "training_divergence", "reduce_learning_rate", "success"))
    agent.record(_entry("oom",      "memory_pressure",    "reduce_batch_size",    "failure"))
    agent.record(_entry("nan_loss", "training_divergence", "reduce_learning_rate", "success"))
    ev = agent.summary()
    assert ev.supporting_data["total_entries"] == 3
    assert ev.supporting_data["outcome_distribution"]["success"] == 2
    assert ev.supporting_data["outcome_distribution"]["failure"] == 1
    assert ev.supporting_data["success_rate"] == pytest.approx(2 / 3)


def test_summary_identifies_top_cause_and_strategy(agent):
    for _ in range(3):
        agent.record(_entry("nan_loss", "training_divergence", "reduce_learning_rate", "success"))
    agent.record(_entry("oom", "memory_pressure", "reduce_batch_size", "success"))
    ev = agent.summary()
    top_cause = ev.supporting_data["top_root_causes"][0][0]
    top_strategy = ev.supporting_data["top_strategies"][0][0]
    assert top_cause == "training_divergence"
    assert top_strategy == "reduce_learning_rate"


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_partial_outcome_scoring(agent):
    """Partial outcome should score 0.25, between success (1.0) and failure (-0.5)."""
    for _ in range(4):
        agent.record(_entry("test", "some_cause", "strategy_a", "partial"))
    ev = agent.advise("test")
    # confidence = score(0.25) / count(4) normalised → 0.25
    assert ev.confidence_score == pytest.approx(0.25)


def test_failure_only_history_gives_low_confidence(agent):
    for _ in range(3):
        agent.record(_entry("nan_loss", "training_divergence", "reduce_learning_rate", "failure"))
    ev = agent.advise("nan_loss")
    # score = -0.5 * 3 = -1.5 → clamped to 0.0
    assert ev.confidence_score == pytest.approx(0.0)
