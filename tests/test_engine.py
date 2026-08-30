import pytest
from agents.schemas import Evidence, KnowledgeEntry
from agents.knowledge_repository import KnowledgeRepositoryAgent
from decision_engine.engine import DecisionEngine


def test_decision_engine_ranking():
    kra = KnowledgeRepositoryAgent()
    
    # We manipulate _entries directly to avoid hitting the DB in unit test
    # (or if it hits DB, it just saves a test record which is fine).
    kra.record(KnowledgeEntry(
        failure_signature="test_fail",
        root_cause_type="test_cause",
        recovery_strategy_used="good_strategy",
        outcome="success",
        notes=""
    ))
    kra.record(KnowledgeEntry(
        failure_signature="test_fail",
        root_cause_type="test_cause",
        recovery_strategy_used="bad_strategy",
        outcome="failure",
        notes=""
    ))
    
    engine = DecisionEngine()
    
    hypotheses = [
        Evidence(
            agent_name="RCA",
            finding="Test failure cause",
            confidence_score=0.9,
            supporting_data={}
        )
    ]
    
    options = [
        Evidence(
            agent_name="Recovery",
            finding="Try bad strategy",
            confidence_score=0.8,
            supporting_data={"strategy_name": "bad_strategy"},
            estimated_cost="high compute",
            predicted_impact="high risk"
        ),
        Evidence(
            agent_name="Recovery",
            finding="Try good strategy",
            confidence_score=0.9,
            supporting_data={"strategy_name": "good_strategy"},
            estimated_cost="low compute",
            predicted_impact="improve accuracy"
        )
    ]
    
    ranked = engine.rank_strategies(hypotheses, options, kra)
    
    assert len(ranked) == 2
    assert ranked[0].strategy_name == "good_strategy"
    assert ranked[1].strategy_name == "bad_strategy"
    assert ranked[0].total_score > ranked[1].total_score
