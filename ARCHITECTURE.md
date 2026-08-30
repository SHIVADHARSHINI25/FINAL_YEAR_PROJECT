# NeuroHeal Architecture

## Core Components

The framework consists of 8 specialized agents, a `DecisionEngine`, and a `PipelineOrchestrator` exposed via a FastAPI REST/WebSocket backend.

### The Pipeline Orchestrator
The orchestrator manages the lifecycle of the ML pipeline execution (ingestion → preprocessing → training → evaluation). 

It calls the `FailureDetectionAgent` after each step. If a failure occurs, the orchestrator triggers the self-healing flow:
1. **Root Cause Analysis**: `RootCauseAnalysisAgent` diagnoses the failure.
2. **Deep Dives**: Additional agents (`DataQualityAgent`, `ModelAnalysisAgent`, `ResourceMonitoringAgent`) gather context-specific evidence.
3. **Recovery Options**: `RecoveryAgent` proposes candidate recovery actions.
4. **Decision Engine**: Options are ranked.
5. **Execution**: The top strategy is executed.
6. **Verification**: `VerificationAgent` confirms the fix.
7. **Knowledge Persistence**: `KnowledgeRepositoryAgent` persists the outcome.

### Decision Engine Scoring Formula
The `DecisionEngine` ranks recovery strategies based on a weighted sum of normalized indicators. 
For each candidate strategy `S`:

```
Score(S) = (0.3 × Confidence_Upstream) + 
           (0.3 × Historical_Success_Rate) + 
           (0.2 × Expected_Accuracy_Preservation) + 
           (0.1 × Recovery_Cost) + 
           (0.1 × Risk_Level)
```

- **Confidence_Upstream (0.0-1.0)**: Confidence score provided by the proposing agent.
- **Historical_Success_Rate (0.0-1.0)**: Queried from `KnowledgeRepositoryAgent`. `(successes + 0.5 * partials) / total_historical_cases`.
- **Expected_Accuracy_Preservation (0.0-1.0)**: Derived from the agent's `predicted_impact`. 
- **Recovery_Cost (0.0-1.0)**: Derived from `estimated_cost` (Higher score = lower cost/faster recovery time).
- **Risk_Level (0.0-1.0)**: Derived from `predicted_impact` (Higher score = lower risk).

This formula is entirely interpretable and displayed directly to the user in the Dashboard's Failure Detail View, contributing to the explainability of the autonomous recovery process.
