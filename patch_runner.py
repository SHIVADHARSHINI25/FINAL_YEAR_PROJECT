import sys

target = '''    async def handle_failure(self, failed_state: PipelineState, failure_evidence: list[Evidence], config: Dict[str, Any]):
        """
        Invokes Root Cause Analysis and Decision Engine to heal the pipeline.
        """
        await self.emit("handling_failure", {"stage": failed_state.stage})
        
        # 1. Root Cause Analysis
        hypotheses = self.rca.diagnose(failure_evidence, self.pipeline_history)'''

replacement = '''    async def handle_failure(self, failed_state: PipelineState, failure_evidence: list[Evidence], config: Dict[str, Any]):
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
            hypotheses = self.rca.diagnose(failure_evidence, self.pipeline_history)'''

with open('orchestrator/pipeline_runner.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_content = content.replace(target, replacement)
if new_content == content:
    print('Failed to replace.')
    sys.exit(1)
    
with open('orchestrator/pipeline_runner.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
print('Replaced successfully.')
