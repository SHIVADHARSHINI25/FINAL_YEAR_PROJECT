"""
Model Analysis Agent
====================
Role: Deep-dive analysis on model/training-specific problems. Called when Root Cause
Analysis flags an algorithmic/training cause, but also runnable standalone on a
training metrics history list and optional hyperparameter dict.

Checks:
- Loss curve behaviour: divergence, plateauing, oscillation
- Overfitting vs underfitting (train/validation metric gap)
- Learning rate issues (loss spikes)
- Gradient health: vanishing / exploding gradients
- Hyperparameter sanity (absurd LR, batch > dataset size, etc.)

Input:
    training_metrics_history: list[dict]
        Each dict is one epoch/step, expected keys (all optional):
            loss, val_loss, train_accuracy, val_accuracy,
            learning_rate, grad_norm
    hyperparameters: dict | None
        Expected keys (all optional):
            learning_rate, batch_size, dataset_size, epochs,
            optimizer, dropout_rate

Output: list[Evidence]  (empty if no issues found)
"""

import math
from datetime import datetime
from agents.base import BaseAgent
from agents.schemas import Evidence


class ModelAnalysisAgent(BaseAgent):
    """Analyses training metrics history and hyperparameters for model/training issues."""

    def __init__(
        self,
        name: str = "ModelAnalysisAgent",
        plateau_std_threshold: float = 0.005,
        plateau_min_epochs: int = 5,
        overfit_gap_threshold: float = 0.10,
        grad_vanish_threshold: float = 1e-5,
        grad_explode_threshold: float = 100.0,
        max_reasonable_lr: float = 1.0,
    ):
        super().__init__(name)
        self.plateau_std_threshold = plateau_std_threshold
        self.plateau_min_epochs = plateau_min_epochs
        self.overfit_gap_threshold = overfit_gap_threshold
        self.grad_vanish_threshold = grad_vanish_threshold
        self.grad_explode_threshold = grad_explode_threshold
        self.max_reasonable_lr = max_reasonable_lr

    # ------------------------------------------------------------------
    def analyze(
        self,
        training_metrics_history: list[dict],
        hyperparameters: dict | None = None,
    ) -> list[Evidence]:
        """Inspect training metrics over time and hyperparameters; return Evidence."""
        evidences: list[Evidence] = []
        hp = hyperparameters or {}

        if training_metrics_history:
            evidences.extend(self._check_loss_curve(training_metrics_history))
            evidences.extend(self._check_overfit_underfit(training_metrics_history))
            evidences.extend(self._check_lr_spikes(training_metrics_history))
            evidences.extend(self._check_gradients(training_metrics_history))

        if hp:
            evidences.extend(self._check_hyperparameters(hp))

        return evidences

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_series(self, history: list[dict], key: str) -> list[float]:
        return [
            float(h[key])
            for h in history
            if key in h and h[key] is not None and _is_finite(h[key])
        ]

    def _check_loss_curve(self, history: list[dict]) -> list[Evidence]:
        evidences = []

        # 1. NaN / Inf in loss -> training divergence  (checked before filtering)
        bad = [h.get("loss") for h in history if "loss" in h and not _is_finite(h.get("loss"))]
        if bad:
            evidences.append(Evidence(
                agent_name=self.name,
                finding=f"Training divergence: Loss contains NaN/Inf values ({len(bad)} epoch(s)).",
                confidence_score=1.0,
                supporting_data={
                    "check_type": "loss_divergence",
                    "bad_values": [str(v) for v in bad[:5]],
                },
                recommended_action="Lower learning rate, add gradient clipping, or check for un-scaled features.",
                predicted_impact="Model weights are corrupted; training cannot continue.",
                estimated_cost="High compute (requires full retrain).",
            ))

        losses = self._extract_series(history, "loss")
        if len(losses) < 2:
            return evidences

        # 2. Loss consistently increasing -> diverging
        if len(losses) >= 3 and all(losses[i] < losses[i + 1] for i in range(len(losses) - 1)):
            evidences.append(Evidence(
                agent_name=self.name,
                finding=f"Training divergence: Loss increased monotonically over {len(losses)} epochs ({losses[0]:.4f} -> {losses[-1]:.4f}).",
                confidence_score=0.90,
                supporting_data={
                    "check_type": "loss_divergence",
                    "loss_start": losses[0],
                    "loss_end": losses[-1],
                    "epochs_checked": len(losses),
                },
                recommended_action="Reduce learning rate by 10×; verify label encodings and feature scales.",
                predicted_impact="Model will not converge under current configuration.",
                estimated_cost="High compute (requires retrain with lower LR).",
            ))

        # 3. Loss plateau (std dev of last N epochs < threshold)
        if len(losses) >= self.plateau_min_epochs:
            tail = losses[-self.plateau_min_epochs:]
            std = _std(tail)
            mean = sum(tail) / len(tail)
            if std < self.plateau_std_threshold and mean > 0.01:
                evidences.append(Evidence(
                    agent_name=self.name,
                    finding=f"Loss plateau detected: std={std:.6f} over last {self.plateau_min_epochs} epochs (mean loss={mean:.4f}).",
                    confidence_score=0.80,
                    supporting_data={
                        "check_type": "loss_plateau",
                        "tail_std": std,
                        "tail_mean": mean,
                        "epochs_checked": self.plateau_min_epochs,
                    },
                    recommended_action="Apply learning rate decay, use a cosine schedule, or increase model capacity.",
                    predicted_impact="Model has stopped learning; accuracy will not improve further.",
                    estimated_cost="Medium compute (LR schedule change + retrain).",
                ))

        # 4. Loss oscillation (frequent sign-changes in delta)
        if len(losses) >= 4:
            deltas = [losses[i + 1] - losses[i] for i in range(len(losses) - 1)]
            sign_changes = sum(1 for i in range(len(deltas) - 1) if deltas[i] * deltas[i + 1] < 0)
            oscillation_rate = sign_changes / (len(deltas) - 1)
            if oscillation_rate > 0.6:
                evidences.append(Evidence(
                    agent_name=self.name,
                    finding=f"Loss oscillation detected: {sign_changes}/{len(deltas)-1} consecutive epochs reversed direction ({oscillation_rate:.0%} rate).",
                    confidence_score=0.75,
                    supporting_data={
                        "check_type": "loss_oscillation",
                        "sign_change_rate": oscillation_rate,
                    },
                    recommended_action="Reduce learning rate or increase batch size to stabilise gradient estimates.",
                    predicted_impact="Training is unstable; final model quality is unpredictable.",
                    estimated_cost="Medium compute (LR reduction + retrain).",
                ))

        return evidences

    def _check_overfit_underfit(self, history: list[dict]) -> list[Evidence]:
        evidences = []
        train_acc = self._extract_series(history, "train_accuracy")
        val_acc = self._extract_series(history, "val_accuracy")

        if not train_acc or not val_acc:
            return evidences

        latest_train = train_acc[-1]
        latest_val = val_acc[-1]
        gap = latest_train - latest_val

        if gap > self.overfit_gap_threshold:
            evidences.append(Evidence(
                agent_name=self.name,
                finding=f"Overfitting detected: train_accuracy={latest_train:.2f}, val_accuracy={latest_val:.2f}, gap={gap:.2f} (threshold={self.overfit_gap_threshold}).",
                confidence_score=0.85,
                supporting_data={
                    "check_type": "overfitting",
                    "train_accuracy": latest_train,
                    "val_accuracy": latest_val,
                    "gap": gap,
                    "threshold": self.overfit_gap_threshold,
                },
                recommended_action="Increase dropout rate, apply L2 regularisation, or reduce model capacity.",
                predicted_impact="Model will underperform on unseen data despite high training accuracy.",
                estimated_cost="Medium compute (regularisation + retrain).",
            ))
        elif latest_train < 0.60 and latest_val < 0.60:
            evidences.append(Evidence(
                agent_name=self.name,
                finding=f"Underfitting detected: train_accuracy={latest_train:.2f}, val_accuracy={latest_val:.2f} -- both below 0.60.",
                confidence_score=0.80,
                supporting_data={
                    "check_type": "underfitting",
                    "train_accuracy": latest_train,
                    "val_accuracy": latest_val,
                },
                recommended_action="Increase model capacity, train for more epochs, or reduce regularisation strength.",
                predicted_impact="Model cannot capture the underlying data patterns.",
                estimated_cost="High compute (architecture change + retrain).",
            ))

        return evidences

    def _check_lr_spikes(self, history: list[dict]) -> list[Evidence]:
        """Detect epochs where LR increased and loss spiked simultaneously."""
        evidences = []
        lrs = self._extract_series(history, "learning_rate")
        losses = self._extract_series(history, "loss")
        if len(lrs) < 2 or len(losses) < 2 or len(lrs) != len(losses):
            return evidences

        spike_epochs = []
        for i in range(1, len(lrs)):
            lr_increased = lrs[i] > lrs[i - 1] * 1.5
            loss_spiked = losses[i] > losses[i - 1] * 1.3
            if lr_increased and loss_spiked:
                spike_epochs.append(i)

        if spike_epochs:
            evidences.append(Evidence(
                agent_name=self.name,
                finding=f"Learning rate spike(s) caused loss spikes at epoch(s) {spike_epochs}.",
                confidence_score=0.85,
                supporting_data={
                    "check_type": "lr_spike",
                    "spike_epochs": spike_epochs,
                    "lr_at_spikes": [lrs[i] for i in spike_epochs],
                    "loss_at_spikes": [losses[i] for i in spike_epochs],
                },
                recommended_action="Use a gentler LR schedule (e.g. cosine annealing) or warm restarts with a smaller max LR.",
                predicted_impact="Loss instability during training; risk of divergence at each LR increase.",
                estimated_cost="Low compute (scheduler change only).",
            ))

        return evidences

    def _check_gradients(self, history: list[dict]) -> list[Evidence]:
        evidences = []
        grad_norms = [h.get("grad_norm") for h in history if "grad_norm" in h]
        if not grad_norms:
            return evidences

        nan_inf = [g for g in grad_norms if g is None or not _is_finite(g)]
        valid = [float(g) for g in grad_norms if g is not None and _is_finite(g)]

        if nan_inf:
            evidences.append(Evidence(
                agent_name=self.name,
                finding=f"Exploding gradients: {len(nan_inf)} epoch(s) reported NaN/Inf gradient norm.",
                confidence_score=1.0,
                supporting_data={
                    "check_type": "exploding_gradients",
                    "nan_inf_count": len(nan_inf),
                },
                recommended_action="Apply gradient clipping (e.g. clip_grad_norm=1.0) immediately.",
                predicted_impact="Model weights will become NaN; training is broken.",
                estimated_cost="Low compute (add one line of gradient clipping).",
            ))

        if valid:
            max_norm = max(valid)
            min_norm = min(valid)
            if max_norm > self.grad_explode_threshold:
                evidences.append(Evidence(
                    agent_name=self.name,
                    finding=f"Exploding gradients: max gradient norm = {max_norm:.2f} (threshold={self.grad_explode_threshold}).",
                    confidence_score=0.90,
                    supporting_data={
                        "check_type": "exploding_gradients",
                        "max_grad_norm": max_norm,
                        "threshold": self.grad_explode_threshold,
                    },
                    recommended_action="Apply gradient clipping (clip_grad_norm <= 1.0) and reduce learning rate.",
                    predicted_impact="Training instability; weights diverge over time.",
                    estimated_cost="Low compute (clipping config change).",
                ))
            if min_norm < self.grad_vanish_threshold:
                evidences.append(Evidence(
                    agent_name=self.name,
                    finding=f"Vanishing gradients: min gradient norm = {min_norm:.2e} (threshold={self.grad_vanish_threshold:.2e}).",
                    confidence_score=0.85,
                    supporting_data={
                        "check_type": "vanishing_gradients",
                        "min_grad_norm": min_norm,
                        "threshold": self.grad_vanish_threshold,
                    },
                    recommended_action="Use batch normalisation, residual connections, or switch to ReLU/GELU activations.",
                    predicted_impact="Early layers stop learning; deep model fails to capture complex patterns.",
                    estimated_cost="Medium compute (architecture change + retrain).",
                ))

        return evidences

    def _check_hyperparameters(self, hp: dict) -> list[Evidence]:
        evidences = []

        lr = hp.get("learning_rate")
        if lr is not None:
            if lr > self.max_reasonable_lr:
                evidences.append(Evidence(
                    agent_name=self.name,
                    finding=f"Hyperparameter sanity: learning_rate={lr} is dangerously high (max reasonable={self.max_reasonable_lr}).",
                    confidence_score=0.95,
                    supporting_data={"check_type": "hyperparameter_sanity", "param": "learning_rate", "value": lr},
                    recommended_action=f"Reduce learning_rate to <={self.max_reasonable_lr / 100:.4f} and re-run training.",
                    predicted_impact="Training will diverge immediately.",
                    estimated_cost="Low compute (config change only).",
                ))
            if lr <= 0:
                evidences.append(Evidence(
                    agent_name=self.name,
                    finding=f"Hyperparameter sanity: learning_rate={lr} is non-positive -- training cannot proceed.",
                    confidence_score=1.0,
                    supporting_data={"check_type": "hyperparameter_sanity", "param": "learning_rate", "value": lr},
                    recommended_action="Set learning_rate to a positive value (e.g. 1e-3).",
                    predicted_impact="Training will not update any weights.",
                    estimated_cost="Low compute (config change only).",
                ))

        batch_size = hp.get("batch_size")
        dataset_size = hp.get("dataset_size")
        if batch_size is not None and dataset_size is not None and batch_size > dataset_size:
            evidences.append(Evidence(
                agent_name=self.name,
                finding=f"Hyperparameter sanity: batch_size={batch_size} > dataset_size={dataset_size}.",
                confidence_score=0.95,
                supporting_data={"check_type": "hyperparameter_sanity", "param": "batch_size", "batch_size": batch_size, "dataset_size": dataset_size},
                recommended_action=f"Reduce batch_size to at most {dataset_size // 4} (<=25% of dataset).",
                predicted_impact="DataLoader will raise an error or produce single-batch training, causing severe overfitting.",
                estimated_cost="Low compute (config change only).",
            ))

        epochs = hp.get("epochs")
        if epochs is not None and epochs <= 0:
            evidences.append(Evidence(
                agent_name=self.name,
                finding=f"Hyperparameter sanity: epochs={epochs} is non-positive.",
                confidence_score=1.0,
                supporting_data={"check_type": "hyperparameter_sanity", "param": "epochs", "value": epochs},
                recommended_action="Set epochs to a positive integer (e.g. 20).",
                predicted_impact="Training loop will not execute.",
                estimated_cost="Low compute (config change only).",
            ))

        return evidences


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def _is_finite(value) -> bool:
    try:
        f = float(value)
        return not (math.isnan(f) or math.isinf(f))
    except (TypeError, ValueError):
        return False


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)
