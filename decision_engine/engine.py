from typing import List
from pydantic import BaseModel
from agents.schemas import Evidence
from agents.knowledge_repository import KnowledgeRepositoryAgent


class RankedStrategy(BaseModel):
    strategy_name: str
    description: str
    total_score: float
    confidence_score: float
    historical_success_rate: float
    expected_accuracy_preservation: float
    recovery_time_cost: float
    risk_level: float


class DecisionEngine:
    def __init__(self):
        # Weights for the scoring formula
        self.weights = {
            "confidence": 0.3,
            "historical_success": 0.3,
            "accuracy_preservation": 0.2,
            "time_cost": 0.1,
            "risk": 0.1
        }

    def rank_strategies(
        self,
        root_cause_hypotheses: List[Evidence],
        recovery_options: List[Evidence],
        knowledge_repository: KnowledgeRepositoryAgent
    ) -> List[RankedStrategy]:
        """
        Score each candidate recovery strategy against multiple criteria using a weighted sum.
        """
        ranked_strategies = []

        for option in recovery_options:
            strategy_name = option.supporting_data.get("strategy_name", "unknown") if option.supporting_data else "unknown"
            description = option.recommended_action or option.finding

            # 1. Confidence score from upstream agent
            confidence_score = option.confidence_score

            # 2. Historical success rate from Knowledge Repository
            all_entries = knowledge_repository.lookup() # Returns all if no args provided
            strategy_entries = [e for e in all_entries if e.recovery_strategy_used == strategy_name]
            successes = sum(1 for e in strategy_entries if e.outcome == "success")
            partials = sum(1 for e in strategy_entries if e.outcome == "partial")
            total = len(strategy_entries)

            if total > 0:
                historical_success_rate = (successes + 0.5 * partials) / total
            else:
                historical_success_rate = 0.5  # Neutral default if no history

            # 3. Expected accuracy preservation (heuristic)
            predicted_impact = str(option.predicted_impact).lower()
            expected_accuracy_preservation = 0.8
            if "drop" in predicted_impact or "decrease" in predicted_impact:
                expected_accuracy_preservation = 0.4
            elif "improve" in predicted_impact or "increase" in predicted_impact:
                expected_accuracy_preservation = 0.95

            # 4. Recovery time / cost (heuristic, higher score = better/less cost)
            estimated_cost = str(option.estimated_cost).lower()
            recovery_time_cost = 0.8
            if "high" in estimated_cost:
                recovery_time_cost = 0.2
            elif "moderate" in estimated_cost:
                recovery_time_cost = 0.5

            # 5. Risk level (heuristic, higher score = less risk)
            risk_level = 0.7
            if "high risk" in predicted_impact or "divergence" in predicted_impact:
                risk_level = 0.3

            total_score = (
                (confidence_score * self.weights["confidence"]) +
                (historical_success_rate * self.weights["historical_success"]) +
                (expected_accuracy_preservation * self.weights["accuracy_preservation"]) +
                (recovery_time_cost * self.weights["time_cost"]) +
                (risk_level * self.weights["risk"])
            )

            ranked = RankedStrategy(
                strategy_name=strategy_name,
                description=description,
                total_score=total_score,
                confidence_score=confidence_score,
                historical_success_rate=historical_success_rate,
                expected_accuracy_preservation=expected_accuracy_preservation,
                recovery_time_cost=recovery_time_cost,
                risk_level=risk_level
            )
            ranked_strategies.append(ranked)

        # Sort highest score first
        ranked_strategies.sort(key=lambda x: x.total_score, reverse=True)
        return ranked_strategies
