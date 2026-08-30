# Autonomous Self-Healing ML Pipelines

This is a research prototype for a **Context-Aware, Evidence-Based Multi-Agent Framework for Autonomous Self-Healing of Machine Learning Pipelines**. 

The system simulates a fully autonomous pipeline where multiple specialized agents collaboratively monitor, diagnose, and recover from failures in machine learning workflows.

## Architecture

The framework consists of 8 specialized agents, each inheriting from a common `BaseAgent` and producing standardized `Evidence` objects that are fed into the system:

1. **Failure Detection Agent** (`agents/failure_detection.py`)
   - Monitors pipeline execution logs and metric snapshots.
   - Detects failures (crashes, NaNs, schema drift, resource exhaustion, etc.).
   
2. **Root Cause Analysis Agent** (`agents/root_cause_analysis.py`)
   - Takes failure evidence and historical pipeline states.
   - Diagnoses the root cause (e.g., cascading data drift, algorithmic instability, OOM).
   
3. **Data Quality Agent** (`agents/data_quality.py`)
   - Inspects the raw dataset (e.g., Pandas DataFrames).
   - Flags missing values, zero variance columns, high cardinality, and outlier anomalies.
   
4. **Model Analysis Agent** (`agents/model_analysis.py`)
   - Analyzes training metrics history (loss, accuracy, gradients, learning rates).
   - Detects overfitting, underfitting, diverging losses, and exploding/vanishing gradients.
   
5. **Resource Monitoring Agent** (`agents/resource_monitoring.py`)
   - Monitors CPU, RAM, Disk, and GPU usage.
   - Flags critical bottlenecks and warns about upward resource consumption trends.

6. **Recovery Planning Agent** (`agents/recovery.py`)
   - Acts as the Decision Engine.
   - Selects the best recovery strategy based on prioritized rules and the evidence provided by upstream agents.
   
7. **Verification Agent** (`agents/verification.py`)
   - Compares pre- and post-recovery metrics.
   - Ensures the chosen recovery action actually improved the pipeline health (e.g., loss decreased, no OOMs).
   
8. **Knowledge Repository Agent** (`agents/knowledge_repository.py`)
   - In-memory store for historical failure and recovery cases.
   - Learns from past recoveries and advises the system on the historically most successful strategy for a given failure signature.

## Installation

Ensure you have Python 3.10+ installed.

```bash
pip install -r requirements.txt
```

## Running the Dashboard

This project includes a full React frontend and FastAPI backend orchestrator.

### 1. Start the Backend API
```bash
uvicorn api.main:app --reload
```
This runs on `http://localhost:8000` and creates `pipeline.db` automatically.

### 2. Start the Frontend Dashboard
Open a new terminal:
```bash
cd frontend
npm run dev
```
Navigate to the local URL (usually `http://localhost:5173`). You can upload a CSV, trigger a mock failure, and watch the agents and Decision Engine recover the pipeline live.

## Running the CLI Demo (Legacy)

A text-based end-to-end demonstration is provided in `demo.py`.

```bash
python demo.py
```

## Testing

The framework is thoroughly tested with `pytest`. Each agent has a dedicated test suite verifying its specific rules and edge cases.

To run the full test suite:

```bash
python -m pytest tests/ -v
```

Total tests: 117
Coverage: Focuses on unit-level validation of rule triggers and edge cases.
