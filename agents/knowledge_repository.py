"""
Knowledge Repository Agent
===========================
Role: An in-memory store of past failure/recovery records (KnowledgeEntry).
Provides:
  - record()   -- add a new entry after a recovery cycle completes
  - lookup()   -- search entries by root cause type and/or failure signature
  - advise()   -- given a failure signature, return the best historical strategy
  - summary()  -- aggregated stats for reporting / the decision engine

The repository is intentionally in-memory so the prototype stays dependency-free.
Persistence (JSON / SQLite) can be added later without changing the public API.
"""

from collections import Counter, defaultdict
from agents.base import BaseAgent
from agents.schemas import Evidence, KnowledgeEntry


class KnowledgeRepositoryAgent(BaseAgent):
    """Manages historical knowledge about failures and successful recovery strategies."""

    def __init__(self, name: str = "KnowledgeRepositoryAgent"):
        super().__init__(name)
        self._entries: list[KnowledgeEntry] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, entry: KnowledgeEntry) -> Evidence:
        """Store a new KnowledgeEntry and return a confirming Evidence."""
        self._entries.append(entry)
        return Evidence(
            agent_name=self.name,
            finding=f"Recorded knowledge entry: failure='{entry.failure_signature}', strategy='{entry.recovery_strategy_used}', outcome='{entry.outcome}'.",
            confidence_score=1.0,
            supporting_data={
                "failure_signature": entry.failure_signature,
                "root_cause_type": entry.root_cause_type,
                "recovery_strategy_used": entry.recovery_strategy_used,
                "outcome": entry.outcome,
                "total_entries": len(self._entries),
            },
        )

    def lookup(
        self,
        root_cause_type: str | None = None,
        failure_signature: str | None = None,
    ) -> list[KnowledgeEntry]:
        """Return entries matching root_cause_type and/or failure_signature (OR logic)."""
        results = []
        for entry in self._entries:
            match_rc = root_cause_type is None or entry.root_cause_type == root_cause_type
            match_sig = failure_signature is None or entry.failure_signature == failure_signature
            if match_rc and match_sig:
                results.append(entry)
        return results

    def advise(self, failure_signature: str) -> Evidence:
        """
        Given a failure signature, find the historically most successful strategy
        for that failure type. Returns an Evidence with recommendation.
        """
        candidates = self.lookup(failure_signature=failure_signature)
        if not candidates:
            # Try broader lookup via root_cause_type embedded in signature
            candidates = self.lookup(root_cause_type=failure_signature)

        if not candidates:
            return Evidence(
                agent_name=self.name,
                finding=f"No historical data for failure signature '{failure_signature}'. No advice available.",
                confidence_score=0.0,
                supporting_data={"failure_signature": failure_signature, "matched_entries": 0},
            )

        # Score strategies: successes weighted +1, failures weighted -0.5, partials +0.25
        strategy_scores: dict[str, float] = defaultdict(float)
        strategy_counts: Counter = Counter()
        for e in candidates:
            w = {"success": 1.0, "partial": 0.25, "failure": -0.5}.get(e.outcome, 0.0)
            strategy_scores[e.recovery_strategy_used] += w
            strategy_counts[e.recovery_strategy_used] += 1

        best_strategy = max(strategy_scores, key=lambda s: strategy_scores[s])
        best_score = strategy_scores[best_strategy]
        total = strategy_counts[best_strategy]

        # Confidence: normalise score relative to count (1.0 = all successes)
        confidence = max(0.0, min(1.0, best_score / total)) if total > 0 else 0.0

        return Evidence(
            agent_name=self.name,
            finding=f"Historical advice for '{failure_signature}': use '{best_strategy}' (score={best_score:.2f} over {total} cases).",
            confidence_score=confidence,
            supporting_data={
                "failure_signature": failure_signature,
                "recommended_strategy": best_strategy,
                "historical_score": best_score,
                "cases_examined": total,
                "all_strategy_scores": dict(strategy_scores),
            },
            recommended_action=f"Apply '{best_strategy}' based on {total} historical recovery records.",
            predicted_impact=f"Success rate for this strategy: {confidence:.0%}.",
            estimated_cost="Refer to strategy documentation for cost estimate.",
        )

    def summary(self) -> Evidence:
        """Return a summary of all recorded knowledge as an Evidence object."""
        total = len(self._entries)
        if total == 0:
            return Evidence(
                agent_name=self.name,
                finding="Knowledge repository is empty. No historical data recorded yet.",
                confidence_score=1.0,
                supporting_data={"total_entries": 0},
            )

        outcome_counts = Counter(e.outcome for e in self._entries)
        cause_counts = Counter(e.root_cause_type for e in self._entries)
        strategy_counts = Counter(e.recovery_strategy_used for e in self._entries)
        success_rate = outcome_counts.get("success", 0) / total

        top_causes = cause_counts.most_common(3)
        top_strategies = strategy_counts.most_common(3)

        return Evidence(
            agent_name=self.name,
            finding=(
                f"Knowledge repository: {total} entries, "
                f"success rate={success_rate:.0%}, "
                f"top cause='{top_causes[0][0]}', "
                f"top strategy='{top_strategies[0][0]}'."
            ),
            confidence_score=1.0,
            supporting_data={
                "total_entries": total,
                "outcome_distribution": dict(outcome_counts),
                "success_rate": success_rate,
                "top_root_causes": top_causes,
                "top_strategies": top_strategies,
            },
        )

    def __len__(self) -> int:
        return len(self._entries)
