import math
from datetime import datetime
from typing import Any
from agents.base import BaseAgent
from agents.schemas import Evidence, PipelineState

class FailureDetectionAgent(BaseAgent):
    """
    Monitors pipeline state and execution logs to detect data ingestion,
    schema drift, training, resource, and evaluation failures.
    """
    def __init__(
        self,
        name: str = "FailureDetectionAgent",
        reference_schema: dict[str, str] | None = None,
        min_accuracy: float = 0.75,
        min_f1: float = 0.70,
        max_execution_time_sec: float = 1800.0,
        max_memory_mb: float = 4096.0
    ):
        super().__init__(name)
        self.reference_schema = reference_schema
        self.min_accuracy = min_accuracy
        self.min_f1 = min_f1
        self.max_execution_time_sec = max_execution_time_sec
        self.max_memory_mb = max_memory_mb

    def monitor(self, pipeline_state: PipelineState) -> list[Evidence]:
        """
        Inspect pipeline_state and return Evidence for any failures found.
        Returns an empty list if the pipeline is healthy.
        """
        evidences = []

        # 1. Ingestion Failures
        ingestion_evidence = self._check_ingestion_failures(pipeline_state)
        if ingestion_evidence:
            evidences.append(ingestion_evidence)

        # 2. Schema Drift (only check if reference schema is provided or defined)
        schema_evidence = self._check_schema_drift(pipeline_state)
        if schema_evidence:
            evidences.extend(schema_evidence)

        # 3. Training Failures (NaN loss, non-convergence)
        training_evidence = self._check_training_failures(pipeline_state)
        if training_evidence:
            evidences.extend(training_evidence)

        # 4. Resource Failures (OOM, timeout)
        resource_evidence = self._check_resource_failures(pipeline_state)
        if resource_evidence:
            evidences.extend(resource_evidence)

        # 5. Evaluation Collapse
        eval_evidence = self._check_evaluation_collapse(pipeline_state)
        if eval_evidence:
            evidences.append(eval_evidence)

        return evidences

    def _check_ingestion_failures(self, state: PipelineState) -> Evidence | None:
        # Check if the ingestion stage explicitly failed or dataset is empty
        if state.stage == "data_ingestion":
            if state.status == "failed":
                return Evidence(
                    agent_name=self.name,
                    finding=f"Ingestion stage failed: {state.error_message or 'Unknown ingestion error'}",
                    confidence_score=1.0,
                    supporting_data={
                        "failure_type": "ingestion_failure",
                        "error_message": state.error_message,
                        "stage": state.stage
                    },
                    recommended_action="Verify data source path, access permissions, or file integrity.",
                    predicted_impact="Downstream pipeline stages cannot run.",
                    estimated_cost="Low compute, ~5 min to check paths."
                )
            
            # Check for empty datasets
            if state.data_summary:
                row_count = state.data_summary.get("row_count")
                if row_count == 0:
                    return Evidence(
                        agent_name=self.name,
                        finding="Ingestion succeeded but dataset is empty (0 rows).",
                        confidence_score=1.0,
                        supporting_data={
                            "failure_type": "ingestion_failure",
                            "row_count": 0,
                            "stage": state.stage
                        },
                        recommended_action="Verify query or filter conditions on data ingestion source.",
                        predicted_impact="Downstream model training or evaluation will fail.",
                        estimated_cost="Low compute, ~2 min."
                    )
        return None

    def _check_schema_drift(self, state: PipelineState) -> list[Evidence]:
        evidences = []
        if not state.data_schema:
            return evidences

        current_cols = state.data_schema.get("columns", {})
        ref_schema = self.reference_schema

        if not ref_schema:
            return evidences

        missing_cols = []
        type_mismatches = {}

        def clean_type(t):
            t = str(t).lower()
            if "int" in t: return "integer"
            if "float" in t or "double" in t: return "float"
            if "str" in t or "object" in t or "string" in t: return "string"
            if "bool" in t: return "boolean"
            return t

        for col, expected_type in ref_schema.items():
            if col not in current_cols:
                missing_cols.append(col)
            elif clean_type(current_cols[col]) != clean_type(expected_type):
                type_mismatches[col] = {
                    "expected": expected_type,
                    "actual": current_cols[col]
                }

        # Extra columns are drift but less critical than missing columns
        extra_cols = [c for c in current_cols if c not in ref_schema]

        if missing_cols:
            evidences.append(
                Evidence(
                    agent_name=self.name,
                    finding=f"Critical Schema Drift: Missing expected columns: {missing_cols}",
                    confidence_score=1.0,
                    supporting_data={
                        "failure_type": "schema_drift",
                        "severity": "critical",
                        "missing_columns": missing_cols,
                        "current_columns": list(current_cols.keys())
                    },
                    recommended_action="Check upstream data pipeline schemas or database migration scripts.",
                    predicted_impact="Features cannot be computed; training/inference will crash.",
                    estimated_cost="Medium manual effort, ~30 min."
                )
            )

        if type_mismatches:
            evidences.append(
                Evidence(
                    agent_name=self.name,
                    finding=f"Schema Drift: Data type mismatch detected: {type_mismatches}",
                    confidence_score=0.9,
                    supporting_data={
                        "failure_type": "schema_drift",
                        "severity": "high",
                        "type_mismatches": type_mismatches
                    },
                    recommended_action="Apply data type casting in ingestion or check source table schema.",
                    predicted_impact="Feature engineering steps may fail or produce invalid values.",
                    estimated_cost="Low compute, ~10 min."
                )
            )

        if extra_cols and len(extra_cols) > 3: # threshold of extra cols to alert
            evidences.append(
                Evidence(
                    agent_name=self.name,
                    finding=f"Schema Drift: Extra columns detected: {extra_cols}",
                    confidence_score=0.6,
                    supporting_data={
                        "failure_type": "schema_drift",
                        "severity": "low",
                        "extra_columns": extra_cols
                    },
                    recommended_action="Update the pipeline's reference schema or filter out extra columns.",
                    predicted_impact="May increase storage/memory usage slightly, but pipeline should survive.",
                    estimated_cost="Low compute, ~5 min."
                )
            )

        return evidences

    def _check_training_failures(self, state: PipelineState) -> list[Evidence]:
        evidences = []
        if state.stage != "training":
            return evidences

        # Check explicit crash/error
        if state.status == "failed":
            evidences.append(
                Evidence(
                    agent_name=self.name,
                    finding=f"Training crashed: {state.error_message or 'Unknown error'}",
                    confidence_score=1.0,
                    supporting_data={
                        "failure_type": "training_divergence",
                        "error_message": state.error_message
                    },
                    recommended_action="Inspect execution logs and code traceback to identify code exceptions.",
                    predicted_impact="No model weights generated.",
                    estimated_cost="Manual debugging required."
                )
            )
            return evidences

        loss = state.metrics.get("loss")
        if loss is not None:
            # Check for NaN / Infinite loss
            if (isinstance(loss, float) and (math.isnan(loss) or math.isinf(loss))) or str(loss).lower() in ("nan", "inf", "-inf"):
                evidences.append(
                    Evidence(
                        agent_name=self.name,
                        finding="Training divergence detected: Loss value is NaN or Infinity.",
                        confidence_score=1.0,
                        supporting_data={
                            "failure_type": "training_divergence",
                            "metric": "loss",
                            "value": str(loss)
                        },
                        recommended_action="Lower the learning rate, check features for un-scaled values, or add gradient clipping.",
                        predicted_impact="Model learning process failed; model weights corrupted.",
                        estimated_cost="High compute, requires retraining model."
                    )
                )

        # Check for exploding/non-converging training metrics
        # For this prototype, if training accuracy is extremely low (e.g. < 0.1 on a classification task) after many epochs
        epochs = state.metrics.get("epochs")
        accuracy = state.metrics.get("train_accuracy")
        if epochs is not None and epochs > 5 and accuracy is not None and accuracy < 0.2:
            evidences.append(
                Evidence(
                    agent_name=self.name,
                    finding=f"Non-convergence detected: Train accuracy remains at {accuracy:.2f} after {epochs} epochs.",
                    confidence_score=0.8,
                    supporting_data={
                        "failure_type": "training_divergence",
                        "metric": "train_accuracy",
                        "value": accuracy,
                        "epochs": epochs
                    },
                    recommended_action="Check if the label class distribution is balanced or verify learning rate/optimisation settings.",
                    predicted_impact="Model fails to learn useful representations.",
                    estimated_cost="High compute, requires retraining."
                )
            )

        return evidences

    def _check_resource_failures(self, state: PipelineState) -> list[Evidence]:
        evidences = []
        
        # Check logs for OOM signals
        oom_indicators = ["out of memory", "oom", "exit code 137", "killed"]
        log_text = " ".join(state.logs).lower() if state.logs else ""
        error_text = state.error_message.lower() if state.error_message else ""
        
        has_oom_signal = any(ind in log_text or ind in error_text for ind in oom_indicators)

        # Check explicit metric thresholds
        memory_used = state.metrics.get("memory_mb")
        execution_time = state.metrics.get("execution_time_sec")

        if has_oom_signal or (memory_used and memory_used > self.max_memory_mb):
            evidences.append(
                Evidence(
                    agent_name=self.name,
                    finding=f"Resource exhaustion (OOM): Memory limit exceeded. Used: {memory_used or 'Unknown'} MB, Limit: {self.max_memory_mb} MB.",
                    confidence_score=1.0 if has_oom_signal else 0.9,
                    supporting_data={
                        "failure_type": "resource_exhaustion",
                        "resource": "memory",
                        "memory_mb": memory_used,
                        "limit_mb": self.max_memory_mb,
                        "oom_logged": has_oom_signal
                    },
                    recommended_action="Scale vertical memory resources, reduce batch size, or optimize data loaders.",
                    predicted_impact="Pipeline crashes and execution is aborted.",
                    estimated_cost="Medium infrastructure cost to scale memory, low cost if reducing batch size."
                )
            )

        if execution_time and execution_time > self.max_execution_time_sec:
            evidences.append(
                Evidence(
                    agent_name=self.name,
                    finding=f"Timeout limit exceeded: Stage '{state.stage}' ran for {execution_time:.1f}s, Limit: {self.max_execution_time_sec}s.",
                    confidence_score=0.95,
                    supporting_data={
                        "failure_type": "resource_exhaustion",
                        "resource": "time",
                        "execution_time_sec": execution_time,
                        "limit_sec": self.max_execution_time_sec
                    },
                    recommended_action="Profile execution time, enable early stopping, optimize computations, or increase timeout thresholds.",
                    predicted_impact="Pipeline execution is terminated due to timeout constraint.",
                    estimated_cost="Medium tuning cost."
                )
            )

        return evidences

    def _check_evaluation_collapse(self, state: PipelineState) -> Evidence | None:
        if state.stage != "evaluation":
            return None

        accuracy = state.metrics.get("accuracy")
        f1 = state.metrics.get("f1")

        findings = []
        supporting = {}
        
        if accuracy is not None and accuracy < self.min_accuracy:
            findings.append(f"Accuracy {accuracy:.2f} < Threshold {self.min_accuracy:.2f}")
            supporting["accuracy"] = accuracy
            supporting["min_accuracy"] = self.min_accuracy

        if f1 is not None and f1 < self.min_f1:
            findings.append(f"F1 Score {f1:.2f} < Threshold {self.min_f1:.2f}")
            supporting["f1"] = f1
            supporting["min_f1"] = self.min_f1

        if findings:
            supporting["failure_type"] = "evaluation_collapse"
            return Evidence(
                agent_name=self.name,
                finding=f"Evaluation metric collapse: {'; '.join(findings)}",
                confidence_score=0.9,
                supporting_data=supporting,
                recommended_action="Invoke Root Cause Analysis to identify if degradation is due to schema drift, class imbalance, or model tuning issues.",
                predicted_impact="Model accuracy is insufficient for production deployment.",
                estimated_cost="Low compute to diagnose, potentially high compute to retrain."
            )

        return None
