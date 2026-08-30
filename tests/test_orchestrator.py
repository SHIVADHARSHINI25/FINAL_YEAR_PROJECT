import pytest
import asyncio
from orchestrator.pipeline_runner import PipelineOrchestrator

@pytest.mark.asyncio
async def test_pipeline_runner_healthy():
    events = []
    
    async def emit_cb(event_type, data):
        events.append(event_type)
        
    runner = PipelineOrchestrator("run_1", emit_cb)
    await runner.run_pipeline({"dataset": "dummy"}, inject_failure=False)
    
    assert "run_started" in events
    assert "stage_started" in events
    assert "stage_completed" in events
    assert "run_completed" in events
    assert runner.status == "success"

@pytest.mark.asyncio
async def test_pipeline_runner_failure():
    events = []
    
    async def emit_cb(event_type, data):
        events.append(event_type)
        
    runner = PipelineOrchestrator("run_2", emit_cb)
    await runner.run_pipeline({"dataset": "dummy"}, inject_failure=True)
    
    assert "failure_detected" in events
    assert "handling_failure" in events
    assert "rca_completed" in events
    assert "decision_engine_ranked" in events
    assert "recovery_executed" in events
    assert runner.status == "recovered"
