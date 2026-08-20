"""
Resource Monitoring Agent
=========================
Role: Assess system resource state and flag resource-related failures or risks.
Operates on a snapshot dict of system metrics (memory, CPU, disk, GPU). Also accepts
an optional recent history list of such snapshots to detect trends.

Checks:
- Memory pressure (RAM high / OOM risk)
- CPU saturation (sustained high usage)
- Disk space (insufficient space for checkpoints / dataset)
- GPU memory pressure (if GPU keys present)
- Resource trend: memory / CPU climbing towards limits

Input:
    current_metrics: dict
        Expected keys (all optional):
            memory_used_mb, memory_total_mb, memory_percent,
            cpu_percent,
            disk_used_gb, disk_total_gb, disk_percent,
            gpu_memory_used_mb, gpu_memory_total_mb, gpu_memory_percent
    history: list[dict] | None
        List of older metric snapshots (same key structure as current_metrics)

Output: list[Evidence]  (empty if no resource issues found)
"""

import math
from agents.base import BaseAgent
from agents.schemas import Evidence


class ResourceMonitoringAgent(BaseAgent):
    """Monitors system resource utilisation and flags resource bottlenecks."""

    def __init__(
        self,
        name: str = "ResourceMonitoringAgent",
        memory_warn_percent: float = 80.0,
        memory_critical_percent: float = 95.0,
        cpu_warn_percent: float = 85.0,
        cpu_critical_percent: float = 97.0,
        disk_warn_percent: float = 85.0,
        disk_critical_percent: float = 95.0,
        gpu_warn_percent: float = 85.0,
        gpu_critical_percent: float = 97.0,
        trend_min_points: int = 3,
        trend_slope_threshold: float = 3.0,   # % per snapshot
    ):
        super().__init__(name)
        self.memory_warn_percent = memory_warn_percent
        self.memory_critical_percent = memory_critical_percent
        self.cpu_warn_percent = cpu_warn_percent
        self.cpu_critical_percent = cpu_critical_percent
        self.disk_warn_percent = disk_warn_percent
        self.disk_critical_percent = disk_critical_percent
        self.gpu_warn_percent = gpu_warn_percent
        self.gpu_critical_percent = gpu_critical_percent
        self.trend_min_points = trend_min_points
        self.trend_slope_threshold = trend_slope_threshold

    # ------------------------------------------------------------------
    def monitor(
        self,
        current_metrics: dict,
        history: list[dict] | None = None,
    ) -> list[Evidence]:
        """Assess resource health from a metrics snapshot and optional history."""
        evidences: list[Evidence] = []
        evidences.extend(self._check_memory(current_metrics))
        evidences.extend(self._check_cpu(current_metrics))
        evidences.extend(self._check_disk(current_metrics))
        evidences.extend(self._check_gpu(current_metrics))

        if history and len(history) >= self.trend_min_points:
            evidences.extend(self._check_trends(history + [current_metrics]))

        return evidences

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pct(self, metrics: dict, used_key: str, total_key: str, pct_key: str) -> float | None:
        """Return percentage: explicit pct_key takes priority; else compute from used/total."""
        if pct_key in metrics and metrics[pct_key] is not None:
            return float(metrics[pct_key])
        used = metrics.get(used_key)
        total = metrics.get(total_key)
        if used is not None and total is not None and float(total) > 0:
            return 100.0 * float(used) / float(total)
        return None

    def _threshold_evidence(
        self,
        value: float,
        label: str,
        unit: str,
        warn: float,
        critical: float,
        check_type: str,
        extra: dict | None = None,
    ) -> Evidence | None:
        if value >= critical:
            severity = "CRITICAL"
            confidence = 0.98
        elif value >= warn:
            severity = "WARNING"
            confidence = 0.80
        else:
            return None

        finding = f"{severity} {label}: {value:.1f}{unit} utilisation (warn>={warn}%, critical>={critical}%)."
        action_map = {
            "memory": "Free memory by reducing batch size, enabling gradient checkpointing, or adding RAM.",
            "cpu": "Reduce data-loading workers, simplify preprocessing, or distribute training.",
            "disk": "Free disk space: delete old checkpoints or move dataset to larger volume.",
            "gpu_memory": "Reduce batch size, use mixed-precision (fp16), or enable gradient checkpointing.",
        }
        impact_map = {
            "memory": "OOM kill risk -- training process will crash." if severity == "CRITICAL" else "Swapping to disk will drastically slow training.",
            "cpu": "Training throughput bottlenecked; high risk of NaN timeouts." if severity == "CRITICAL" else "Data loading becomes the training bottleneck.",
            "disk": "Checkpoint saves will fail, losing all training progress." if severity == "CRITICAL" else "Risk of mid-training checkpoint save failure.",
            "gpu_memory": "CUDA OOM error will crash training." if severity == "CRITICAL" else "Training will be slower due to GPU memory pressure.",
        }

        data = {
            "check_type": check_type,
            "severity": severity,
            f"{check_type}_percent": value,
        }
        if extra:
            data.update(extra)

        return Evidence(
            agent_name=self.name,
            finding=finding,
            confidence_score=confidence,
            supporting_data=data,
            recommended_action=action_map.get(check_type, "Investigate resource usage."),
            predicted_impact=impact_map.get(check_type, "Performance degradation."),
            estimated_cost="Low compute (config / infra change).",
        )

    def _check_memory(self, metrics: dict) -> list[Evidence]:
        pct = self._pct(metrics, "memory_used_mb", "memory_total_mb", "memory_percent")
        if pct is None:
            return []
        ev = self._threshold_evidence(
            pct, "RAM", "%", self.memory_warn_percent, self.memory_critical_percent,
            "memory",
            extra={
                "memory_used_mb": metrics.get("memory_used_mb"),
                "memory_total_mb": metrics.get("memory_total_mb"),
            },
        )
        return [ev] if ev else []

    def _check_cpu(self, metrics: dict) -> list[Evidence]:
        pct = metrics.get("cpu_percent")
        if pct is None:
            return []
        ev = self._threshold_evidence(
            float(pct), "CPU", "%", self.cpu_warn_percent, self.cpu_critical_percent, "cpu"
        )
        return [ev] if ev else []

    def _check_disk(self, metrics: dict) -> list[Evidence]:
        pct = self._pct(metrics, "disk_used_gb", "disk_total_gb", "disk_percent")
        if pct is None:
            return []
        ev = self._threshold_evidence(
            pct, "Disk", "%", self.disk_warn_percent, self.disk_critical_percent,
            "disk",
            extra={
                "disk_used_gb": metrics.get("disk_used_gb"),
                "disk_total_gb": metrics.get("disk_total_gb"),
            },
        )
        return [ev] if ev else []

    def _check_gpu(self, metrics: dict) -> list[Evidence]:
        pct = self._pct(metrics, "gpu_memory_used_mb", "gpu_memory_total_mb", "gpu_memory_percent")
        if pct is None:
            return []
        ev = self._threshold_evidence(
            pct, "GPU Memory", "%", self.gpu_warn_percent, self.gpu_critical_percent,
            "gpu_memory",
            extra={
                "gpu_memory_used_mb": metrics.get("gpu_memory_used_mb"),
                "gpu_memory_total_mb": metrics.get("gpu_memory_total_mb"),
            },
        )
        return [ev] if ev else []

    def _check_trends(self, snapshots: list[dict]) -> list[Evidence]:
        """Detect upward resource trends that will imminently hit critical levels."""
        evidences = []

        trend_targets = [
            ("memory_percent", "RAM", self.memory_critical_percent, "memory"),
            ("cpu_percent", "CPU", self.cpu_critical_percent, "cpu"),
            ("disk_percent", "Disk", self.disk_critical_percent, "disk"),
            ("gpu_memory_percent", "GPU Memory", self.gpu_critical_percent, "gpu_memory"),
        ]

        for key, label, critical, check_type in trend_targets:
            series = [float(s[key]) for s in snapshots if key in s and s[key] is not None]
            if len(series) < self.trend_min_points:
                continue

            slope = _linear_slope(series)
            if slope >= self.trend_slope_threshold:
                steps_to_critical = None
                if slope > 0 and series[-1] < critical:
                    steps_to_critical = int(math.ceil((critical - series[-1]) / slope))

                evidences.append(Evidence(
                    agent_name=self.name,
                    finding=f"{label} trend alert: increasing at {slope:.2f}%/snapshot. Current={series[-1]:.1f}%.",
                    confidence_score=0.75,
                    supporting_data={
                        "check_type": f"{check_type}_trend",
                        "slope_per_snapshot": slope,
                        "current_percent": series[-1],
                        "critical_percent": critical,
                        "estimated_snapshots_to_critical": steps_to_critical,
                    },
                    recommended_action=f"Proactively free {label} resources before reaching the critical threshold.",
                    predicted_impact=f"{label} will hit critical levels in ~{steps_to_critical} snapshots if unchecked." if steps_to_critical else f"{label} is climbing rapidly.",
                    estimated_cost="Low compute (proactive intervention now vs full crash later).",
                ))

        return evidences


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _linear_slope(values: list[float]) -> float:
    """Least-squares slope of a 1D series (x = 0,1,2,…)."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((xs[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0
