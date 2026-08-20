"""Unit tests for ResourceMonitoringAgent."""
import pytest
from agents.resource_monitoring import ResourceMonitoringAgent


@pytest.fixture
def agent():
    return ResourceMonitoringAgent()


# ------------------------------------------------------------------
# Memory tests
# ------------------------------------------------------------------

def test_memory_critical_detected(agent):
    ev = agent.monitor({"memory_percent": 97.0})
    assert len(ev) == 1
    assert "CRITICAL" in ev[0].finding
    assert ev[0].supporting_data["check_type"] == "memory"


def test_memory_warning_detected(agent):
    ev = agent.monitor({"memory_percent": 85.0})
    assert len(ev) == 1
    assert "WARNING" in ev[0].finding


def test_memory_computed_from_used_total(agent):
    # 16 384 / 16 384 = 100% → critical
    ev = agent.monitor({"memory_used_mb": 16384, "memory_total_mb": 16384})
    assert len(ev) == 1
    assert "CRITICAL" in ev[0].finding


def test_memory_healthy_no_alert(agent):
    ev = agent.monitor({"memory_percent": 60.0})
    assert ev == []


# ------------------------------------------------------------------
# CPU tests
# ------------------------------------------------------------------

def test_cpu_critical_detected(agent):
    ev = agent.monitor({"cpu_percent": 99.0})
    assert any("CPU" in e.finding for e in ev)
    assert any("CRITICAL" in e.finding for e in ev)


def test_cpu_warning_detected(agent):
    ev = agent.monitor({"cpu_percent": 90.0})
    assert any("WARNING" in e.finding for e in ev)


def test_cpu_healthy_no_alert(agent):
    ev = agent.monitor({"cpu_percent": 50.0})
    assert ev == []


# ------------------------------------------------------------------
# Disk tests
# ------------------------------------------------------------------

def test_disk_critical_detected(agent):
    ev = agent.monitor({"disk_percent": 96.0})
    assert any("CRITICAL" in e.finding for e in ev)


def test_disk_computed_from_used_total(agent):
    # 950 GB of 1000 GB → 95% → critical
    ev = agent.monitor({"disk_used_gb": 950, "disk_total_gb": 1000})
    assert any("CRITICAL" in e.finding for e in ev)


def test_disk_healthy_no_alert(agent):
    ev = agent.monitor({"disk_percent": 50.0})
    assert ev == []


# ------------------------------------------------------------------
# GPU memory tests
# ------------------------------------------------------------------

def test_gpu_critical_detected(agent):
    ev = agent.monitor({"gpu_memory_percent": 98.0})
    assert any("GPU" in e.finding for e in ev)
    assert any("CRITICAL" in e.finding for e in ev)


def test_gpu_computed_from_used_total(agent):
    ev = agent.monitor({"gpu_memory_used_mb": 11000, "gpu_memory_total_mb": 12000})
    # 11000/12000 ≈ 91.7% → warning
    assert any("GPU" in e.finding for e in ev)


def test_gpu_healthy_no_alert(agent):
    ev = agent.monitor({"gpu_memory_percent": 70.0})
    assert ev == []


# ------------------------------------------------------------------
# Trend tests
# ------------------------------------------------------------------

def test_memory_trend_detected(agent):
    history = [
        {"memory_percent": 60.0},
        {"memory_percent": 65.0},
        {"memory_percent": 70.0},
    ]
    current = {"memory_percent": 75.0}
    ev = agent.monitor(current, history=history)
    trend_ev = [e for e in ev if "trend" in e.supporting_data.get("check_type", "")]
    assert len(trend_ev) >= 1
    assert trend_ev[0].supporting_data["slope_per_snapshot"] > 0


def test_stable_metrics_no_trend_alert(agent):
    history = [{"memory_percent": 60.0}] * 5
    current = {"memory_percent": 60.0}
    ev = agent.monitor(current, history=history)
    assert ev == []


def test_trend_not_triggered_with_too_few_history_points(agent):
    # Only 1 history point — trend needs at least 3
    history = [{"memory_percent": 60.0}]
    current = {"memory_percent": 90.0}
    ev = agent.monitor(current, history=history)
    trend_ev = [e for e in ev if "trend" in e.supporting_data.get("check_type", "")]
    assert len(trend_ev) == 0


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_empty_metrics_no_crash(agent):
    ev = agent.monitor({})
    assert ev == []


def test_multiple_resource_alerts_in_one_snapshot(agent):
    ev = agent.monitor({"memory_percent": 97.0, "cpu_percent": 99.0, "disk_percent": 97.0})
    assert len(ev) == 3
