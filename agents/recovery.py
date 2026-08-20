"""
Recovery Planning Agent
=======================
Role: Select the best recovery strategy given ranked Evidence from upstream agents,
then simulate/log its execution and produce a RecoveryAction record.

Strategy Selection Logic (ordered by priority):
    1. OOM / GPU memory critical        -> reduce_batch_size
    2. Exploding gradients              -> add_gradient_clipping
    3. Vanishing gradients              -> switch_activation_function
    4. Training divergence (loss NaN)   -> reduce_learning_rate
    5. Overfitting                      -> apply_regularisation
    6. Underfitting                     -> increase_model_capacity
    7. Loss plateau                     -> apply_lr_schedule
    8. Schema drift / data quality      -> retrigger_preprocessing
    9. Ingestion / data quality failure -> reingestion
    10. Resource warning (non-OOM)      -> scale_resources
    11. Fallback                        -> retrigger_last_stage

Each strategy is simulated: the agent returns a `RecoveryAction` and a refreshed
`Evidence` describing the planned action.

Input:
    evidence_list: list[Evidence]   -- ranked evidence from all upstream agents

Output:
    Tuple[RecoveryAction, Evidence]
        RecoveryAction -- structured record of the planned/simulated action
        Evidence        -- rich evidence suitable for the decision engine
"""

import uuid
from datetime import datetime
from agents.base import BaseAgent
from agents.schemas import Evidence, RecoveryAction

# Ordered priority rules: (check_substring_in_finding, strategy_name, description, confidence)
_RULES: list[tuple[str, str, str, float]] = [
    # Resource rules (highest priority -- prevent OOM crash first)
    ("CRITICAL GPU Memory",       "reduce_batch_size",            "Halve the batch size to free GPU memory and prevent CUDA OOM.", 0.95),
    ("CRITICAL RAM",              "reduce_batch_size",            "Halve the batch size to reduce peak memory pressure and prevent OOM kill.", 0.92),
    # Gradient rules
    ("Exploding gradients",       "add_gradient_clipping",        "Add gradient clipping (max_norm=1.0) to prevent weight divergence.", 0.93),
    ("Vanishing gradients",       "switch_activation_function",   "Replace Sigmoid/Tanh with ReLU/GELU and add batch normalisation.", 0.88),
    # Loss rules
    ("Training divergence",       "reduce_learning_rate",         "Reduce learning rate by 10× and restart training from last checkpoint.", 0.90),
    # Fit rules
    ("Overfitting",               "apply_regularisation",         "Increase dropout rate by 0.1 and add L2 weight decay (lambda=1e-4).", 0.88),
    ("Underfitting",              "increase_model_capacity",      "Double the number of hidden units in the penultimate layer and retrain.", 0.80),
    ("Loss plateau",              "apply_lr_schedule",            "Enable cosine annealing LR schedule and continue training for 10 more epochs.", 0.82),
    # Data rules
    ("Schema Drift",              "retrigger_preprocessing",      "Retrigger the preprocessing stage with the updated reference schema.", 0.87),
    ("data quality",              "retrigger_preprocessing",      "Retrigger preprocessing with data quality fixes applied.", 0.85),
    ("ingestion",                 "reingestion",                  "Retrigger the data ingestion stage with validation checks enabled.", 0.88),
    # Resource warning (non-OOM)
    ("WARNING RAM",               "scale_resources",              "Request additional memory allocation or spin up a larger instance.", 0.75),
    ("WARNING CPU",               "scale_resources",              "Increase data-loading worker count or request additional CPU cores.", 0.72),
    ("WARNING Disk",              "scale_resources",              "Archive or delete old checkpoints to free disk space.", 0.78),
]

_FALLBACK_STRATEGY = ("retrigger_last_stage", "Retrigger the last failed pipeline stage with identical configuration.", 0.55)


class RecoveryAgent(BaseAgent):
    """Selects and simulates the best recovery strategy given upstream evidence."""

    def __init__(self, name: str = "RecoveryAgent"):
        super().__init__(name)

    # ------------------------------------------------------------------
    def plan(self, evidence_list: list[Evidence]) -> tuple[RecoveryAction, Evidence]:
        """Select a recovery strategy and return (RecoveryAction, Evidence)."""
        strategy_name, description, confidence = self._select_strategy(evidence_list)

        action_id = str(uuid.uuid4())[:8]
        action = RecoveryAction(
            action_id=action_id,
            strategy_name=strategy_name,
            description=description,
            executed=True,        # simulated execution
            success=True,
            execution_log=f"[SIMULATED] Strategy '{strategy_name}' applied at {datetime.now().isoformat()}. Action ID: {action_id}.",
        )

        evidence = Evidence(
            agent_name=self.name,
            finding=f"Recovery strategy selected: '{strategy_name}'. {description}",
            confidence_score=confidence,
            supporting_data={
                "strategy_name": strategy_name,
                "action_id": action_id,
                "triggering_findings": [e.finding[:80] for e in evidence_list[:3]],
            },
            recommended_action=description,
            predicted_impact=f"Applying '{strategy_name}' is expected to resolve the identified failure.",
            estimated_cost="Depends on strategy; see description for details.",
        )

        return action, evidence

    # ------------------------------------------------------------------
    def _select_strategy(self, evidence_list: list[Evidence]) -> tuple[str, str, float]:
        """Match evidence findings to strategy rules in priority order."""
        all_findings = " ".join(e.finding for e in evidence_list).lower()
        # Priority match: iterate rules in order, return first match
        for keyword, strategy, description, confidence in _RULES:
            if keyword.lower() in all_findings:
                return strategy, description, confidence
        # Fallback
        return _FALLBACK_STRATEGY
