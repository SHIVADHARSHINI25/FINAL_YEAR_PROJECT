import asyncio
import uuid
import pandas as pd
from typing import Callable, Awaitable, Any, Dict

from agents import (
    FailureDetectionAgent,
    RootCauseAnalysisAgent,
    DataQualityAgent,
    ModelAnalysisAgent,
    ResourceMonitoringAgent,
    RecoveryAgent,
    VerificationAgent,
    KnowledgeRepositoryAgent,
    PipelineState,
    KnowledgeEntry,
)
from agents.schemas import Evidence, RecoveryAction
from decision_engine.engine import DecisionEngine


class PipelineOrchestrator:
    def __init__(self, run_id: str, emit_callback: Callable[[str, Dict[str, Any]], Awaitable[None]]):
        self.run_id = run_id
        self.emit_callback = emit_callback
        
        self.fda = FailureDetectionAgent()
        self.rca = RootCauseAnalysisAgent()
        self.dqa = DataQualityAgent()
        self.maa = ModelAnalysisAgent()
        self.rma = ResourceMonitoringAgent()
        self.ra = RecoveryAgent()
        self.va = VerificationAgent()
        self.kra = KnowledgeRepositoryAgent()
        self.decision_engine = DecisionEngine()
        
        self.pipeline_history: list[PipelineState] = []
        self.status = "initialized"
        self.failures = {} # To store history of the failure handling for API
        
    async def emit(self, event_type: str, data: Dict[str, Any]):
        if self.emit_callback:
            await self.emit_callback(event_type, data)

    async def run_pipeline(self, config: Dict[str, Any], inject_failure: bool = False):
        """
        Runs the mock pipeline through stages.
        """
        self.status = "running"
        try:
            await self.emit("run_started", {"run_id": self.run_id, "config": config})
            
            import random
            failure_mode = random.choice(["oom", "schema_drift", "divergence"]) if inject_failure else "none"
            
            # Stage 1: Ingestion / Preprocessing
            await self.emit("stage_started", {"stage": "preprocessing"})
            for i in range(5):
                await self.emit("resource_metrics", {"cpu_percent": 15 + random.random()*10, "gpu_memory_percent": 20 + random.random()*5, "time": f"t+{i}s"})
                await asyncio.sleep(0.4)
            
            ref_schema = {"age": "int64", "income": "float64", "label": "int64"}
            self.fda.reference_schema = ref_schema
            
            if failure_mode == "schema_drift":
                prep_state = PipelineState(
                    stage="preprocessing",
                    status="success",
                    metrics={"execution_time_sec": 2.5},
                    data_schema={"columns": {"age": "string", "income": "float64"}}, # label missing, age type wrong
                    data_summary={"row_count": 1000},
                    logs=["Data loaded with schema drift"]
                )
            else:
                prep_state = PipelineState(
                    stage="preprocessing",
                    status="success",
                    metrics={"execution_time_sec": 2.5},
                    data_schema={"columns": ref_schema},
                    data_summary={"row_count": 1000},
                    logs=["Preprocessing OK"]
                )
                
            self.pipeline_history.append(prep_state)
            await self.emit("stage_completed", {"stage": "preprocessing", "state": prep_state.model_dump(mode='json')})
            
            # Check FDA
            failure_evidence = self.fda.monitor(prep_state)
            if failure_evidence:
                await self.emit("failure_detected", {"evidence": [e.model_dump(mode='json') for e in failure_evidence]})
                await self.handle_failure(prep_state, failure_evidence, config)
                return
                
            # Stage 2: Training
            await self.emit("stage_started", {"stage": "training"})
            for epoch in range(1, 6):
                cpu = 40 + random.random()*20
                if failure_mode == "oom" and epoch >= 4:
                    gpu = 95 + random.random()*5 # OOM Spike
                else:
                    gpu = 30 + random.random()*10
                await self.emit("resource_metrics", {"cpu_percent": cpu, "gpu_memory_percent": gpu, "time": f"epoch {epoch}"})
                
                if failure_mode == "divergence" and epoch >= 4:
                    loss = 100.0 * (epoch**2) # Spiking loss
                else:
                    loss = 0.5 * (0.8 ** epoch) + random.random()*0.1
                await self.emit("training_metrics", {"epoch": epoch, "loss": loss})
                await asyncio.sleep(0.5)
            
            if failure_mode == "oom":
                train_state = PipelineState(
                    stage="training",
                    status="failed",
                    metrics={"execution_time_sec": 15.0, "memory_mb": 11800},
                    logs=["Epoch 1 OK", "CUDA out of memory"],
                    error_message="CUDA out of memory."
                )
            elif failure_mode == "divergence":
                train_state = PipelineState(
                    stage="training",
                    status="failed",
                    metrics={"execution_time_sec": 12.0, "loss": "NaN"},
                    logs=["Epoch 1: loss=0.5", "Epoch 2: loss=NaN"],
                    error_message="Loss value became NaN"
                )
            else:
                train_state = PipelineState(
                    stage="training",
                    status="success",
                    metrics={"execution_time_sec": 15.0, "memory_mb": 2048},
                    logs=["Epoch 1 OK", "Epoch 2 OK", "Training complete"]
                )
                
            self.pipeline_history.append(train_state)
            await self.emit("stage_completed", {"stage": "training", "state": train_state.model_dump(mode='json')})
            
            failure_evidence = self.fda.monitor(train_state)
            
            if failure_mode == "oom":
                # Manually inject evidence to simulate OOM for demo purposes
                failure_evidence.append(Evidence(
                    agent_name="FailureDetectionAgent",
                    finding="CRITICAL GPU Memory: 98.3% utilisation -- CUDA OOM during training epoch 3.",
                    confidence_score=1.0,
                    supporting_data={
                        "failure_type": "resource_exhaustion",
                        "resource": "gpu_memory",
                    },
                    recommended_action="Reduce batch size or enable mixed-precision (fp16) training.",
                    predicted_impact="Training process was killed; no checkpoint saved.",
                    estimated_cost="Low compute -- config change only."
                ))
            
            if failure_evidence:
                await self.emit("failure_detected", {"evidence": [e.model_dump(mode='json') for e in failure_evidence]})
                await self.handle_failure(train_state, failure_evidence, config)
                return
                
            # Stage 3: Evaluation
            await self.emit("stage_started", {"stage": "evaluation"})
            await asyncio.sleep(1)
            eval_state = PipelineState(
                stage="evaluation",
                status="success",
                metrics={"accuracy": 0.95},
                logs=["Eval OK"]
            )
            self.pipeline_history.append(eval_state)
            await self.emit("stage_completed", {"stage": "evaluation", "state": eval_state.model_dump(mode='json')})
            
            self.status = "success"
            
            # Generate mock final output
            is_regression = "regression" in config.get("algorithm", "") or "regressor" in config.get("algorithm", "")
            target_col = config.get("target_column", "label")
            dataset_name = config.get("dataset_name", "")
            
            if "iris" in dataset_name.lower():
                perf_metrics = {"accuracy": 0.96, "f1_score": 0.95, "precision": 0.96, "recall": 0.94}
                sample_predictions = [
                    {"input": {"sepal_len": 5.1, "sepal_wid": 3.5, "petal_len": 1.4, "petal_wid": 0.2}, "prediction": 0, "actual": 0},
                    {"input": {"sepal_len": 6.7, "sepal_wid": 3.0, "petal_len": 5.2, "petal_wid": 2.3}, "prediction": 2, "actual": 2},
                    {"input": {"sepal_len": 5.9, "sepal_wid": 3.0, "petal_len": 4.2, "petal_wid": 1.5}, "prediction": 1, "actual": 1}
                ]
            elif is_regression:
                perf_metrics = {"mse": 120.5, "rmse": 10.97, "r2_score": 0.89, "mae": 8.4}
                sample_predictions = [
                    {"input": {"age": 25, "income": 50000.0}, "prediction": 45.2, "actual": 42.0},
                    {"input": {"age": 45, "income": 120000.0}, "prediction": 88.7, "actual": 90.0},
                    {"input": {"age": 34, "income": 85000.0}, "prediction": 70.1, "actual": 72.5}
                ]
            else:
                perf_metrics = {"accuracy": 0.95, "f1_score": 0.94, "precision": 0.96, "recall": 0.92}
                sample_predictions = [
                    {"input": {"age": 25, "income": 50000.0}, "prediction": 0, "actual": 0},
                    {"input": {"age": 45, "income": 120000.0}, "prediction": 1, "actual": 1},
                    {"input": {"age": 34, "income": 85000.0}, "prediction": 1, "actual": 1}
                ]
                
            await self.emit("run_completed", {"status": "success", "metrics": perf_metrics, "samples": sample_predictions, "target_col": target_col})
            
        except Exception as e:
            await self.emit("run_error", {"error": str(e)})
            await self.emit("run_completed", {"status": "error"})

    async def handle_failure(self, failed_state: PipelineState, failure_evidence: list[Evidence], config: Dict[str, Any]):
        """
        Invokes Root Cause Analysis and Decision Engine to heal the pipeline.
        """
        await self.emit("handling_failure", {"stage": failed_state.stage})
        
        # --- FAST PATH CHECK ---
        finding = failure_evidence[0].finding if failure_evidence else ""
        failure_sig = "gpu_oom_training" if "OOM" in finding else "unknown_failure"
        advice = self.kra.advise(failure_sig)
        
        cases_examined = advice.supporting_data.get("cases_examined", 0) if advice.supporting_data else 0
        
        if advice.confidence_score >= 0.8 and cases_examined > 0:
            await self.emit("fast_path_activated", {"advice": advice.model_dump(mode='json')})
            from decision_engine.engine import RankedStrategy
            top_strategy = RankedStrategy(
                strategy_name=advice.supporting_data.get("recommended_strategy", "unknown"),
                description=advice.recommended_action or "Fast-path recovery",
                total_score=1.0,
                confidence_score=advice.confidence_score,
                historical_success_rate=advice.confidence_score,
                expected_accuracy_preservation=0.9,
                recovery_time_cost=1.0,
                risk_level=0.8
            )
            hypotheses = []
            all_deep_dive = []
            ranked_strategies = [top_strategy]
        else:
            # 1. Root Cause Analysis
            hypotheses = self.rca.diagnose(failure_evidence, self.pipeline_history)
        await self.emit("rca_completed", {"hypotheses": [h.model_dump(mode='json') for h in hypotheses]})
        
        # 2. Deep Dive Agents
        df = pd.DataFrame({"age": [25, 30], "income": [50000.0, 60000.0], "label": [0, 1]})
        dq_evidence = self.dqa.assess(df, reference_schema={"age": "int64", "income": "float64", "label": "int64"})
        
        training_history = [{"loss": 0.5, "train_accuracy": 0.8}]
        model_evidence = self.maa.analyze(training_history, {"learning_rate": 0.01})
        
        resource_snapshot = {"gpu_memory_used_mb": 11800, "gpu_memory_total_mb": 12288}
        resource_evidence = self.rma.monitor(resource_snapshot, [])
        
        all_deep_dive = dq_evidence + model_evidence + resource_evidence
        await self.emit("deep_dive_completed", {"evidence": [e.model_dump(mode='json') for e in all_deep_dive]})
        
        # 3. Recovery Planning & Decision Engine
        all_evidence = failure_evidence + hypotheses + all_deep_dive
        
        recovery_options = []
        from agents.recovery import _RULES, _FALLBACK_STRATEGY
        all_findings = " ".join(e.finding for e in all_evidence).lower()
        for keyword, strategy, description, confidence in _RULES:
            if keyword.lower() in all_findings:
                recovery_options.append(Evidence(
                    agent_name="RecoveryAgent",
                    finding=f"Candidate strategy: {strategy}",
                    confidence_score=confidence,
                    supporting_data={"strategy_name": strategy},
                    recommended_action=description,
                    predicted_impact=f"Resolves failure using {strategy}",
                    estimated_cost="moderate"
                ))
        if not recovery_options:
            recovery_options.append(Evidence(
                agent_name="RecoveryAgent",
                finding="Fallback candidate",
                confidence_score=_FALLBACK_STRATEGY[2],
                supporting_data={"strategy_name": _FALLBACK_STRATEGY[0]},
                recommended_action=_FALLBACK_STRATEGY[1]
            ))
            
        ranked_strategies = self.decision_engine.rank_strategies(hypotheses, recovery_options, self.kra)
        await self.emit("decision_engine_ranked", {"ranked": [r.model_dump(mode='json') for r in ranked_strategies]})
        
        top_strategy = ranked_strategies[0]
        
        # Execute top strategy
        action_id = str(uuid.uuid4())[:8]
        recovery_action = RecoveryAction(
            action_id=action_id,
            strategy_name=top_strategy.strategy_name,
            description=top_strategy.description,
            executed=True,
            success=True,
            execution_log=f"Simulated execution of {top_strategy.strategy_name}"
        )
        await self.emit("recovery_executed", {"action": recovery_action.model_dump(mode='json')})
        
        # 4. Verification
        pre_metrics = {"gpu_memory_used_mb": 11800, "gpu_memory_percent": 98.3, "memory_percent": 70.0, "cpu_percent": 50.0}
        post_metrics = {"gpu_memory_used_mb": 5000, "gpu_memory_percent": 40.0, "memory_percent": 50.0, "cpu_percent": 45.0, "data_schema": {"age": "int64", "income": "float64", "label": "int64"}}
        
        verification_results = self.va.verify(pre_metrics, post_metrics, reference_schema={"age": "int64", "income": "float64", "label": "int64"})
        all_passed = all(r.passed for r in verification_results)
        await self.emit("verification_completed", {"results": [r.model_dump(mode='json') for r in verification_results], "all_passed": all_passed})
        
        # 5. Knowledge Repository
        primary_cause_type = hypotheses[0].supporting_data.get("root_cause_type", "unknown") if hypotheses and hypotheses[0].supporting_data else "unknown"
        failure_sig = "gpu_oom_training" if "OOM" in failure_evidence[0].finding else "unknown_failure"
        
        self.kra.record(KnowledgeEntry(
            failure_signature=failure_sig,
            root_cause_type=primary_cause_type,
            recovery_strategy_used=top_strategy.strategy_name,
            outcome="success" if all_passed else "partial",
            notes="Recovered via orchestrator."
        ))
        
        self.failures = {
            "detection": [e.model_dump(mode='json') for e in failure_evidence],
            "hypotheses": [h.model_dump(mode='json') for h in hypotheses],
            "deep_dive": [e.model_dump(mode='json') for e in all_deep_dive],
            "ranked_strategies": [r.model_dump(mode='json') for r in ranked_strategies],
            "recovery_action": recovery_action.model_dump(mode='json'),
            "verification_results": [r.model_dump(mode='json') for r in verification_results]
        }
        
        if all_passed:
            self.status = "recovered"
            is_regression = "regression" in config.get("algorithm", "") or "regressor" in config.get("algorithm", "")
            target_col = config.get("target_column", "label")
            dataset_name = config.get("dataset_name", "")
            
            if "iris" in dataset_name.lower():
                perf_metrics = {"accuracy": 0.94, "f1_score": 0.93, "precision": 0.95, "recall": 0.93}
                sample_predictions = [
                    {"input": {"sepal_len": 5.1, "sepal_wid": 3.5, "petal_len": 1.4, "petal_wid": 0.2}, "prediction": 0, "actual": 0},
                    {"input": {"sepal_len": 6.7, "sepal_wid": 3.0, "petal_len": 5.2, "petal_wid": 2.3}, "prediction": 2, "actual": 2},
                    {"input": {"sepal_len": 5.9, "sepal_wid": 3.0, "petal_len": 4.2, "petal_wid": 1.5}, "prediction": 1, "actual": 1}
                ]
            elif is_regression:
                perf_metrics = {"mse": 125.0, "rmse": 11.18, "r2_score": 0.88, "mae": 8.7}
                sample_predictions = [
                    {"input": {"age": 25, "income": 50000.0}, "prediction": 44.5, "actual": 42.0},
                    {"input": {"age": 45, "income": 120000.0}, "prediction": 87.2, "actual": 90.0},
                    {"input": {"age": 34, "income": 85000.0}, "prediction": 69.4, "actual": 72.5}
                ]
            else:
                perf_metrics = {"accuracy": 0.93, "f1_score": 0.92, "precision": 0.95, "recall": 0.90}
                sample_predictions = [
                    {"input": {"age": 25, "income": 50000.0}, "prediction": 0, "actual": 0},
                    {"input": {"age": 45, "income": 120000.0}, "prediction": 1, "actual": 1},
                    {"input": {"age": 34, "income": 85000.0}, "prediction": 1, "actual": 1}
                ]
            await self.emit("run_completed", {"status": "recovered", "metrics": perf_metrics, "samples": sample_predictions, "target_col": target_col})
        else:
            self.status = "failed"
            await self.emit("run_completed", {"status": "failed"})
