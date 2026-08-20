from datetime import datetime
from typing import Any
from agents.base import BaseAgent
from agents.schemas import Evidence, PipelineState

class RootCauseAnalysisAgent(BaseAgent):
    """
    Analyzes failure evidence and historical pipeline states to determine the
    probable root causes, cascading failures, and ranks diagnostic hypotheses.
    """
    def __init__(self, name: str = "RootCauseAnalysisAgent"):
        super().__init__(name)

    def diagnose(
        self,
        failure_evidence: list[Evidence],
        pipeline_history: list[PipelineState] | None = None
    ) -> list[Evidence]:
        """
        Analyze failure evidence and history, and return ranked hypotheses as Evidence.
        """
        if not failure_evidence:
            return []

        hypotheses = []
        history = pipeline_history or []

        # Extract failure types present in the current failure evidence
        failure_types = {e.supporting_data.get("failure_type") for e in failure_evidence if "failure_type" in e.supporting_data}
        
        # Check historical states for failures or anomalies as well
        historical_failures = []
        for state in history:
            if state.status == "failed":
                historical_failures.append(state)

        # Helper flags for reasoning
        has_ingestion_fail = "ingestion_failure" in failure_types
        has_schema_drift = "schema_drift" in failure_types
        has_training_divergence = "training_divergence" in failure_types
        has_resource_exhaustion = "resource_exhaustion" in failure_types
        has_eval_collapse = "evaluation_collapse" in failure_types

        # Let's inspect the historical pipeline logs/states to enrich the context
        has_historical_schema_drift = any(
            state.stage == "preprocessing" and state.data_summary and state.data_summary.get("schema_drift_detected")
            for state in history
        )
        has_historical_dq_issues = any(
            state.data_summary and any(pct > 0.15 for pct in state.data_summary.get("missing_percentages", {}).values())
            for state in history
        )

        # ----------------------------------------------------------------------
        # Rule 1: Cascading Failure - Ingestion / Schema Drift causing Training crash
        # ----------------------------------------------------------------------
        if (has_training_divergence or has_eval_collapse) and (has_schema_drift or has_historical_schema_drift or has_ingestion_fail):
            drift_cols = []
            for e in failure_evidence:
                if e.supporting_data.get("failure_type") == "schema_drift":
                    drift_cols.extend(e.supporting_data.get("missing_columns", []))
                    drift_cols.extend(list(e.supporting_data.get("type_mismatches", {}).keys()))

            finding = "Root Cause: Upstream schema drift or ingestion failure caused training/evaluation degradation."
            hypotheses.append(
                Evidence(
                    agent_name=self.name,
                    finding=finding,
                    confidence_score=0.90,
                    supporting_data={
                        "root_cause_type": "data_schema_cascade",
                        "reasoning_steps": [
                            "1. Detected training divergence or evaluation collapse symptom in current stage.",
                            "2. Located upstream ingestion failure or schema drift in the history/evidence.",
                            "3. Mapped columns containing drift/missing values to model training inputs.",
                            "4. Concluded that model crashed or degraded due to missing/incompatible features."
                        ],
                        "primary_symptom": "training_divergence" if has_training_divergence else "evaluation_collapse",
                        "upstream_trigger": "schema_drift" if (has_schema_drift or has_historical_schema_drift) else "ingestion_failure",
                        "drifted_columns": list(set(drift_cols))
                    },
                    recommended_action="Resolve the upstream schema drift or ingestion error, cast columns correctly, and run the pipeline again.",
                    predicted_impact="Accuracy will remain zero or training will continue to crash until schema is fixed.",
                    estimated_cost="Low compute, ~15 min manual script update."
                )
            )

        # ----------------------------------------------------------------------
        # Rule 2: Resource Exhaustion causing Training/Pipeline Crash
        # ----------------------------------------------------------------------
        if has_resource_exhaustion:
            res_ev = next(e for e in failure_evidence if e.supporting_data.get("failure_type") == "resource_exhaustion")
            resource_type = res_ev.supporting_data.get("resource", "memory")
            
            finding = f"Root Cause: Hardware resource exhaustion ({resource_type.upper()}) during pipeline execution."
            hypotheses.append(
                Evidence(
                    agent_name=self.name,
                    finding=finding,
                    confidence_score=0.95,
                    supporting_data={
                        "root_cause_type": "resource_constraint",
                        "reasoning_steps": [
                            f"1. Detected stage failure or crash associated with {resource_type}.",
                            "2. OOM signature verified in errors/logs or threshold limits exceeded.",
                            "3. Confirmed infrastructure limits reached."
                        ],
                        "resource_type": resource_type,
                        "limit_exceeded": res_ev.supporting_data.get("limit_mb") or res_ev.supporting_data.get("limit_sec")
                    },
                    recommended_action="Scale memory resource limit, optimize code, or decrease model batch size.",
                    predicted_impact="Pipeline cannot run successfully under current hardware configuration and data volume.",
                    estimated_cost="Medium infrastructure cost, low developer time."
                )
            )

        # ----------------------------------------------------------------------
        # Rule 3: Training Divergence (NaN / Non-convergence) - Algorithmic Issue
        # ----------------------------------------------------------------------
        if has_training_divergence and not (has_schema_drift or has_historical_schema_drift or has_ingestion_fail or has_resource_exhaustion):
            # No data or resource issues flagged, but training diverged
            finding = "Root Cause: Model training instability or hyperparameter divergence (algorithmic)."
            hypotheses.append(
                Evidence(
                    agent_name=self.name,
                    finding=finding,
                    confidence_score=0.85,
                    supporting_data={
                        "root_cause_type": "algorithmic_instability",
                        "reasoning_steps": [
                            "1. Training loss reported as NaN/Infinity, or train accuracy did not converge.",
                            "2. No resource limit violations (OOM or timeouts) detected.",
                            "3. No upstream schema drift or data quality issues found.",
                            "4. Concluded that the model weights diverged due to learning rate, gradient explosion, or initialization."
                        ]
                    },
                    recommended_action="Reduce the learning rate, apply gradient clipping, or check weight initialization.",
                    predicted_impact="Training will continue to produce NaN weights and fail.",
                    estimated_cost="High compute (requires full retrain), low code changes."
                )
            )

        # ----------------------------------------------------------------------
        # Rule 4: Data Quality / Missing Values causing Training/Evaluation failure
        # ----------------------------------------------------------------------
        if (has_training_divergence or has_eval_collapse) and has_historical_dq_issues:
            finding = "Root Cause: High proportion of missing values in dataset causing model divergence or performance degradation."
            hypotheses.append(
                Evidence(
                    agent_name=self.name,
                    finding=finding,
                    confidence_score=0.80,
                    supporting_data={
                        "root_cause_type": "data_quality_cascade",
                        "reasoning_steps": [
                            "1. Training failed or validation metric collapsed.",
                            "2. Upstream history indicates data quality issues (e.g. >15% missing values).",
                            "3. Concluded that missing features led to null/NaN entries in model inputs or sub-optimal predictions."
                        ]
                    },
                    recommended_action="Invoke Data Quality Agent to analyze and apply appropriate imputation strategies.",
                    predicted_impact="Model will crash or yield random/poor predictions due to incomplete inputs.",
                    estimated_cost="Low compute to impute, high compute to retrain."
                )
            )

        # ----------------------------------------------------------------------
        # Rule 5: Evaluation collapse with no data drift -> Algorithmic Underfitting/Overfitting
        # ----------------------------------------------------------------------
        if has_eval_collapse and not (has_schema_drift or has_historical_schema_drift or has_historical_dq_issues):
            finding = "Root Cause: Model underfitting or overfitting to training dataset."
            hypotheses.append(
                Evidence(
                    agent_name=self.name,
                    finding=finding,
                    confidence_score=0.75,
                    supporting_data={
                        "root_cause_type": "model_underfitting_overfitting",
                        "reasoning_steps": [
                            "1. Evaluation stage reported metrics below configured thresholds.",
                            "2. Pipeline executed successfully with no crashes or resource issues.",
                            "3. Upstream data schemas and quality scores are healthy.",
                            "4. Concluded that model capacity is either too low or too high relative to task difficulty."
                        ]
                    },
                    recommended_action="Tune hyperparameters, optimize features, or select a different model architecture.",
                    predicted_impact="Model performance remains sub-par.",
                    estimated_cost="High compute for hyperparameter search."
                )
            )

        # ----------------------------------------------------------------------
        # Rule 6 (NEW): Direct Ingestion Failure - no downstream symptom needed
        # ----------------------------------------------------------------------
        # Rules 1-5 above all require a downstream symptom (training divergence or
        # eval collapse) before they attribute anything to ingestion/schema issues.
        # If the pipeline fails at ingestion, training never runs, so none of those
        # symptoms exist yet -- without this rule the agent would fall through to
        # the generic "uncaught exception" hypothesis despite Failure Detection
        # having already reported the exact cause at 1.0 confidence.
        if has_ingestion_fail and not hypotheses:
            ing_ev = next(e for e in failure_evidence if e.supporting_data.get("failure_type") == "ingestion_failure")
            hypotheses.append(
                Evidence(
                    agent_name=self.name,
                    finding="Root Cause: Data ingestion failure at source (no downstream training/evaluation symptom yet observed).",
                    confidence_score=0.95,
                    supporting_data={
                        "root_cause_type": "ingestion_failure_direct",
                        "reasoning_steps": [
                            "1. Failure Detection Agent reported an ingestion-stage failure or empty dataset.",
                            "2. Pipeline never reached training/evaluation, so no downstream symptom is available.",
                            "3. Attributed the root cause directly to the ingestion-stage evidence rather than falling back to a generic hypothesis."
                        ],
                        "source_error": ing_ev.supporting_data.get("error_message"),
                    },
                    recommended_action="Verify data source path, access permissions, query filters, or file integrity before retrying.",
                    predicted_impact="Pipeline cannot proceed past ingestion until the source issue is resolved.",
                    estimated_cost="Low compute, ~5-10 min to check source/paths."
                )
            )

        # ----------------------------------------------------------------------
        # Rule 7 (NEW): Direct Schema Drift - no downstream symptom needed
        # ----------------------------------------------------------------------
        if has_schema_drift and not hypotheses:
            drift_cols = []
            for e in failure_evidence:
                if e.supporting_data.get("failure_type") == "schema_drift":
                    drift_cols.extend(e.supporting_data.get("missing_columns", []))
                    drift_cols.extend(list(e.supporting_data.get("type_mismatches", {}).keys()))

            hypotheses.append(
                Evidence(
                    agent_name=self.name,
                    finding="Root Cause: Upstream schema drift detected (no downstream training/evaluation symptom yet observed).",
                    confidence_score=0.85,
                    supporting_data={
                        "root_cause_type": "schema_drift_direct",
                        "reasoning_steps": [
                            "1. Failure Detection Agent reported schema drift (missing columns or type mismatches).",
                            "2. Pipeline has not yet reached training/evaluation, so no downstream symptom is available.",
                            "3. Attributed the root cause directly to the schema drift evidence rather than falling back to a generic hypothesis."
                        ],
                        "drifted_columns": list(set(drift_cols))
                    },
                    recommended_action="Check upstream data pipeline schema or migration scripts and cast/restore columns before retrying.",
                    predicted_impact="Feature computation or training will fail once the pipeline advances further.",
                    estimated_cost="Low-medium manual effort, ~10-30 min."
                )
            )

        # ----------------------------------------------------------------------
        # Rule 8: Ambiguous Failures -> Generate ranked hypotheses (fallback)
        # ----------------------------------------------------------------------
        if not hypotheses:
            # Ambiguous crash during training or preprocessing stage
            finding_1 = "Hypothesis 1 (Probable): Uncaught codebase exception or parameter mismatch."
            hypotheses.append(
                Evidence(
                    agent_name=self.name,
                    finding=finding_1,
                    confidence_score=0.60,
                    supporting_data={
                        "root_cause_type": "uncaught_exception",
                        "reasoning_steps": [
                            "1. Pipeline stage crashed without specific resource or data alarms.",
                            "2. Generating default software exception hypothesis."
                        ]
                    },
                    recommended_action="Inspect the full traceback in python execution logs.",
                    predicted_impact="Pipeline execution block is halted.",
                    estimated_cost="Manual debugging required."
                )
            )
            finding_2 = "Hypothesis 2 (Possible): Intermittent environment or network failure."
            hypotheses.append(
                Evidence(
                    agent_name=self.name,
                    finding=finding_2,
                    confidence_score=0.40,
                    supporting_data={
                        "root_cause_type": "environment_issue",
                        "reasoning_steps": [
                            "1. Pipeline stage crashed.",
                            "2. Generating secondary environment/network hypothesis."
                        ]
                    },
                    recommended_action="Verify connection to model registries, database, or compute cluster.",
                    predicted_impact="Pipeline cannot download data or upload artifacts.",
                    estimated_cost="Low resource cost to retry."
                )
            )

        # Sort hypotheses by confidence score descending
        hypotheses.sort(key=lambda x: x.confidence_score, reverse=True)
        return hypotheses
