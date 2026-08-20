"""
Verification Agent
==================
Role: After a recovery action is executed (simulated), verify that the pipeline
health metrics have improved and that no new failures were introduced.

Checks performed (each produces a VerificationResult):
    1. loss_improvement  -- val_loss decreased after recovery
    2. accuracy_recovery -- val_accuracy improved or stayed stable
    3. no_new_nan        -- no NaN/Inf in post-recovery metrics
    4. resource_ok       -- critical resource metrics are below their thresholds
    5. schema_valid      -- post-recovery schema matches the reference schema

Inputs:
    pre_metrics:  dict  -- metrics snapshot BEFORE recovery
    post_metrics: dict  -- metrics snapshot AFTER recovery
    reference_schema: dict | None  -- e.g. {"col": "dtype", …}  (for schema check)

Output: list[VerificationResult]
"""

import math
from agents.base import BaseAgent
from agents.schemas import VerificationResult


class VerificationAgent(BaseAgent):
    """Verifies that a recovery action resolved the reported failure."""

    def __init__(
        self,
        name: str = "VerificationAgent",
        loss_improvement_min: float = 0.001,      # must drop by at least this much
        accuracy_drop_max: float = 0.02,           # val_accuracy must not fall by more
        memory_critical_percent: float = 95.0,
        cpu_critical_percent: float = 97.0,
    ):
        super().__init__(name)
        self.loss_improvement_min = loss_improvement_min
        self.accuracy_drop_max = accuracy_drop_max
        self.memory_critical_percent = memory_critical_percent
        self.cpu_critical_percent = cpu_critical_percent

    # ------------------------------------------------------------------
    def verify(
        self,
        pre_metrics: dict,
        post_metrics: dict,
        reference_schema: dict | None = None,
    ) -> list[VerificationResult]:
        """Run all verification checks and return a list of VerificationResult."""
        results: list[VerificationResult] = []
        results.extend(self._check_loss_improvement(pre_metrics, post_metrics))
        results.extend(self._check_accuracy_recovery(pre_metrics, post_metrics))
        results.extend(self._check_no_new_nan(post_metrics))
        results.extend(self._check_resources(post_metrics))
        if reference_schema:
            results.extend(self._check_schema(post_metrics, reference_schema))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_loss_improvement(self, pre: dict, post: dict) -> list[VerificationResult]:
        pre_loss = pre.get("val_loss") or pre.get("loss")
        post_loss = post.get("val_loss") or post.get("loss")
        if pre_loss is None or post_loss is None:
            return [VerificationResult(
                check_name="loss_improvement",
                passed=True,
                details="No loss metrics available to compare; skipping check.",
            )]
        improvement = float(pre_loss) - float(post_loss)
        passed = improvement >= self.loss_improvement_min
        return [VerificationResult(
            check_name="loss_improvement",
            passed=passed,
            details=(
                f"Loss improved by {improvement:.4f} (pre={pre_loss:.4f}, post={post_loss:.4f})."
                if passed
                else f"Loss did NOT improve sufficiently (improvement={improvement:.4f}, required>={self.loss_improvement_min})."
            ),
            metric_before=float(pre_loss),
            metric_after=float(post_loss),
        )]

    def _check_accuracy_recovery(self, pre: dict, post: dict) -> list[VerificationResult]:
        pre_acc = pre.get("val_accuracy") or pre.get("accuracy")
        post_acc = post.get("val_accuracy") or post.get("accuracy")
        if pre_acc is None or post_acc is None:
            return [VerificationResult(
                check_name="accuracy_recovery",
                passed=True,
                details="No accuracy metrics available; skipping check.",
            )]
        drop = float(pre_acc) - float(post_acc)
        passed = drop <= self.accuracy_drop_max
        return [VerificationResult(
            check_name="accuracy_recovery",
            passed=passed,
            details=(
                f"Accuracy maintained or improved (pre={pre_acc:.4f}, post={post_acc:.4f}, drop={drop:.4f})."
                if passed
                else f"Accuracy dropped too much: {drop:.4f} (max allowed={self.accuracy_drop_max})."
            ),
            metric_before=float(pre_acc),
            metric_after=float(post_acc),
        )]

    def _check_no_new_nan(self, post: dict) -> list[VerificationResult]:
        nan_keys = [k for k, v in post.items() if v is not None and _is_nan_or_inf(v)]
        passed = len(nan_keys) == 0
        return [VerificationResult(
            check_name="no_new_nan",
            passed=passed,
            details=(
                "No NaN/Inf values detected in post-recovery metrics."
                if passed
                else f"NaN/Inf values found in post-recovery metrics for keys: {nan_keys}."
            ),
        )]

    def _check_resources(self, post: dict) -> list[VerificationResult]:
        results = []
        checks = [
            ("memory_percent", self.memory_critical_percent, "RAM"),
            ("cpu_percent",    self.cpu_critical_percent,    "CPU"),
        ]
        for key, threshold, label in checks:
            val = post.get(key)
            if val is None:
                continue
            val = float(val)
            passed = val < threshold
            results.append(VerificationResult(
                check_name=f"resource_ok_{key}",
                passed=passed,
                details=(
                    f"{label} is within safe limits: {val:.1f}% < {threshold}%."
                    if passed
                    else f"{label} is still critical after recovery: {val:.1f}% >= {threshold}%."
                ),
                metric_after=val,
            ))
        return results

    def _check_schema(self, post: dict, reference_schema: dict) -> list[VerificationResult]:
        """Compare post_metrics['data_schema'] against the reference schema."""
        post_schema = post.get("data_schema")
        if post_schema is None:
            return [VerificationResult(
                check_name="schema_valid",
                passed=False,
                details="Post-recovery metrics do not include 'data_schema'; cannot validate.",
            )]

        missing = [col for col in reference_schema if col not in post_schema]
        type_mismatches = [
            col for col, dtype in reference_schema.items()
            if col in post_schema and post_schema[col] != dtype
        ]

        passed = not missing and not type_mismatches
        details_parts = []
        if missing:
            details_parts.append(f"Missing columns: {missing}.")
        if type_mismatches:
            details_parts.append(f"Type mismatches: {type_mismatches}.")
        if passed:
            details_parts.append("Schema matches reference exactly.")

        return [VerificationResult(
            check_name="schema_valid",
            passed=passed,
            details=" ".join(details_parts),
        )]


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _is_nan_or_inf(value) -> bool:
    try:
        f = float(value)
        return math.isnan(f) or math.isinf(f)
    except (TypeError, ValueError):
        return False
