import pytest
from pipeline.ensemble import EnsembleDAG
from pipeline.bls import BLSOrchestrator


def test_stage_failure_triggers_fallback():
    dag = EnsembleDAG()
    dag.add_stage("preprocess", lambda x: x.strip())
    dag.add_stage(
        "model",
        lambda x: 1 / 0 if x == "error" else x.upper(),
        dependencies=["preprocess"],
    )
    dag.add_stage("postprocess", lambda x: f"out:{x}", dependencies=["model"])

    orchestrator = BLSOrchestrator(dag)
    fallbacks = {"model": "FALLBACK_MODEL_OUTPUT"}

    result = orchestrator.execute_with_fault_tolerance(
        "error", fallback_responses=fallbacks
    )
    assert result == "out:FALLBACK_MODEL_OUTPUT"


def test_unhandled_failure_raises_runtime_error():
    dag = EnsembleDAG()
    dag.add_stage("preprocess", lambda x: x.strip())
    dag.add_stage("model", lambda x: [][0], dependencies=["preprocess"])

    orchestrator = BLSOrchestrator(dag)
    with pytest.raises(RuntimeError):
        orchestrator.execute_with_fault_tolerance("test")
