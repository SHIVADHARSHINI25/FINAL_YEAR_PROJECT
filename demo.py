"""
demo.py -- End-to-End Self-Healing ML Pipeline Demo
====================================================
Demonstrates all 8 agents working together in a simulated pipeline
that encounters a GPU OOM failure during training.

Flow:
  1. Failure Detection   -> detect OOM during training
  2. Root Cause Analysis -> diagnose resource constraint
  3. Data Quality        -> scan healthy training dataset
  4. Model Analysis      -> inspect training metrics history
  5. Resource Monitoring -> flag critical GPU memory usage
  6. Recovery Planning   -> select & simulate reduce_batch_size
  7. Verification        -> verify recovery improved the pipeline
  8. Knowledge Repository-> record the recovery outcome; advise on next occurrence

Separator lines keep the output easy to read without external dependencies.
"""

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

SEP = "-" * 70


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def print_evidences(evidences) -> None:
    for i, ev in enumerate(evidences, 1):
        print(f"  [{i}] {ev.finding}")
        print(f"       confidence={ev.confidence_score:.2f}  |  action={ev.recommended_action or 'n/a'}")


def print_verification(results) -> None:
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  {status}  [{r.check_name}] {r.details}")


def main():
    print("\n" + "=" * 70)
    print("  SELF-HEALING ML PIPELINE -- FULL 8-AGENT DEMO")
    print("=" * 70)

    # ==================================================================
    # Synthetic pipeline state -- GPU OOM during training
    # ==================================================================
    ref_schema = {"age": "int64", "income": "float64", "label": "int64"}

    healthy_preprocessing_state = PipelineState(
        stage="preprocessing",
        status="success",
        metrics={"execution_time_sec": 12.3},
        data_schema={"columns": ref_schema},
        data_summary={"row_count": 50_000, "missing_percentages": {"age": 0.01, "income": 0.02}},
        logs=["Preprocessing completed successfully."],
    )

    training_crash_state = PipelineState(
        stage="training",
        status="failed",
        metrics={"execution_time_sec": 95.0, "memory_mb": 11_800},
        data_schema=None,
        data_summary=None,
        logs=[
            "Epoch 1/10: loss=0.721",
            "Epoch 2/10: loss=0.614",
            "Epoch 3/10: CUDA out-of-memory -- killed",
        ],
        error_message="CUDA out of memory. Tried to allocate 4.00 GiB. GPU has 12 GB total; 11.8 GB currently used.",
    )

    # ==================================================================
    # AGENT 1 -- Failure Detection
    # ==================================================================
    section("AGENT 1 -- Failure Detection Agent")
    fda = FailureDetectionAgent(
        name="FailureDetectionAgent",
        reference_schema=ref_schema,
    )
    failure_evidence = fda.monitor(training_crash_state)
    print_evidences(failure_evidence)
    if not failure_evidence:
        print("  (no failures detected)")

    # Inject a resource-exhaustion evidence for the demo (training was killed by OOM)
    from agents.schemas import Evidence
    failure_evidence.append(Evidence(
        agent_name="FailureDetectionAgent",
        finding="CRITICAL GPU Memory: 98.3% utilisation -- CUDA OOM during training epoch 3.",
        confidence_score=1.0,
        supporting_data={
            "failure_type": "resource_exhaustion",
            "resource": "gpu_memory",
            "limit_mb": 12288,
            "used_mb": 11800,
        },
        recommended_action="Reduce batch size or enable mixed-precision (fp16) training.",
        predicted_impact="Training process was killed; no checkpoint saved.",
        estimated_cost="Low compute -- config change only.",
    ))
    print(f"\n  -> Total failure signals detected: {len(failure_evidence)}")

    # ==================================================================
    # AGENT 2 -- Root Cause Analysis
    # ==================================================================
    section("AGENT 2 -- Root Cause Analysis Agent")
    rca = RootCauseAnalysisAgent()
    hypotheses = rca.diagnose(failure_evidence, pipeline_history=[healthy_preprocessing_state, training_crash_state])
    print_evidences(hypotheses)

    # ==================================================================
    # AGENT 3 -- Data Quality (runs on the healthy preprocessing data)
    # ==================================================================
    section("AGENT 3 -- Data Quality Agent")
    import pandas as pd, numpy as np
    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame({
        "age":    rng.integers(18, 65, size=n).astype(float),
        "income": rng.normal(50_000, 15_000, size=n),
        "label":  rng.integers(0, 2, size=n),
        "id_col": range(n), # high cardinality (100% unique)
    })
    # Inject severe data quality errors for the demo!
    # 1. Massive missingness in age (30% missing)
    df.loc[rng.choice(n, int(n * 0.3), replace=False), "age"] = np.nan
    # 2. Extreme outliers in income
    df.loc[rng.choice(n, 5, replace=False), "income"] = rng.uniform(10_000_000, 50_000_000, size=5)

    dqa = DataQualityAgent()
    dq_evidence = dqa.assess(df, reference_schema=ref_schema)
    print_evidences(dq_evidence) if dq_evidence else print("  (data quality healthy -- no issues found)")

    # ==================================================================
    # AGENT 4 -- Model Analysis (2 completed epochs before OOM)
    # ==================================================================
    section("AGENT 4 -- Model Analysis Agent")
    training_history = [
        {"loss": 0.721, "train_accuracy": 0.61, "val_accuracy": 0.59, "learning_rate": 0.01, "grad_norm": 0.82},
        {"loss": 0.614, "train_accuracy": 0.67, "val_accuracy": 0.65, "learning_rate": 0.01, "grad_norm": 0.74},
    ]
    hyperparameters = {"learning_rate": 0.01, "batch_size": 512, "dataset_size": 50_000, "epochs": 10}
    maa = ModelAnalysisAgent()
    model_evidence = maa.analyze(training_history, hyperparameters=hyperparameters)
    print_evidences(model_evidence) if model_evidence else print("  (no model issues found in available training epochs)")

    # ==================================================================
    # AGENT 5 -- Resource Monitoring
    # ==================================================================
    section("AGENT 5 -- Resource Monitoring Agent")
    resource_snapshot = {
        "memory_percent":      72.0,
        "cpu_percent":         65.0,
        "disk_percent":        55.0,
        "gpu_memory_used_mb":  11800,
        "gpu_memory_total_mb": 12288,
    }
    resource_history = [
        {"gpu_memory_used_mb": 4096,  "gpu_memory_total_mb": 12288},
        {"gpu_memory_used_mb": 7168,  "gpu_memory_total_mb": 12288},
        {"gpu_memory_used_mb": 9216,  "gpu_memory_total_mb": 12288},
    ]
    rma = ResourceMonitoringAgent()
    resource_evidence = rma.monitor(resource_snapshot, history=resource_history)
    print_evidences(resource_evidence)

    # ==================================================================
    # AGENT 6 -- Recovery Planning
    # ==================================================================
    section("AGENT 6 -- Recovery Planning Agent")
    all_evidence = failure_evidence + hypotheses + resource_evidence
    ra = RecoveryAgent()
    recovery_action, recovery_evidence = ra.plan(all_evidence)
    print(f"  Strategy   : {recovery_action.strategy_name}")
    print(f"  Description: {recovery_action.description}")
    print(f"  Action ID  : {recovery_action.action_id}")
    print(f"  Log        : {recovery_action.execution_log}")
    print(f"  Evidence   : {recovery_evidence.finding}")

    # ==================================================================
    # AGENT 7 -- Verification
    # ==================================================================
    section("AGENT 7 -- Verification Agent")
    pre_recovery_metrics = {
        "val_loss":            None,         # no val_loss captured before OOM
        "gpu_memory_percent":  98.3,
        "memory_percent":      72.0,
        "cpu_percent":         65.0,
    }
    # Simulated post-recovery: batch size halved -> GPU memory drops
    post_recovery_metrics = {
        "val_loss":            0.598,
        "val_accuracy":        0.66,
        "gpu_memory_used_mb":  5800,
        "gpu_memory_total_mb": 12288,
        "gpu_memory_percent":  47.2,
        "memory_percent":      60.0,
        "cpu_percent":         58.0,
        "data_schema":         ref_schema,
    }
    va = VerificationAgent()
    verification_results = va.verify(pre_recovery_metrics, post_recovery_metrics, reference_schema=ref_schema)
    print_verification(verification_results)
    all_passed = all(r.passed for r in verification_results)
    print(f"\n  -> Overall verification: {'OK SUCCESS -- pipeline recovered' if all_passed else 'XX PARTIAL -- further action needed'}")

    # ==================================================================
    # AGENT 8 -- Knowledge Repository
    # ==================================================================
    section("AGENT 8 -- Knowledge Repository Agent")
    kra = KnowledgeRepositoryAgent()

    # Seed with 2 historical records from past runs
    kra.record(KnowledgeEntry(
        failure_signature="gpu_oom_training",
        root_cause_type="resource_constraint",
        recovery_strategy_used="reduce_batch_size",
        outcome="success",
        notes="Halved batch size from 512->256; training completed without OOM.",
    ))
    kra.record(KnowledgeEntry(
        failure_signature="gpu_oom_training",
        root_cause_type="resource_constraint",
        recovery_strategy_used="enable_mixed_precision",
        outcome="partial",
        notes="fp16 reduced memory but introduced gradient underflow; needed additional loss scaling.",
    ))

    # Record the current recovery
    current_outcome = "success" if all_passed else "partial"
    record_ev = kra.record(KnowledgeEntry(
        failure_signature="gpu_oom_training",
        root_cause_type="resource_constraint",
        recovery_strategy_used=recovery_action.strategy_name,
        outcome=current_outcome,
        notes=f"Action ID: {recovery_action.action_id}. All verification checks passed.",
    ))
    print(f"  Recorded: {record_ev.finding}")

    # Ask for advice
    advice_ev = kra.advise("gpu_oom_training")
    print(f"\n  Advice   : {advice_ev.finding}")
    print(f"  Confidence: {advice_ev.confidence_score:.2f}")

    # Summary
    summary_ev = kra.summary()
    print(f"\n  Summary  : {summary_ev.finding}")

    # ==================================================================
    # FINAL REPORT
    # ==================================================================
    section("DEMO COMPLETE -- FINAL SUMMARY")
    print(f"  Failure signals detected : {len(failure_evidence)}")
    print(f"  Root-cause hypotheses    : {len(hypotheses)}")
    print(f"  Data quality issues      : {len(dq_evidence)}")
    print(f"  Model analysis findings  : {len(model_evidence)}")
    print(f"  Resource alerts          : {len(resource_evidence)}")
    print(f"  Recovery strategy        : {recovery_action.strategy_name}")
    print(f"  Verification result      : {'ALL PASSED' if all_passed else 'PARTIAL'}")
    print(f"  Knowledge entries stored : {len(kra)}")
    print()


if __name__ == "__main__":
    main()
