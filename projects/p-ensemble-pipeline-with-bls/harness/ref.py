import numpy as np
from pipeline.ensemble import EnsembleDAG
from pipeline.bls import BLSOrchestrator


def build_sample_pipeline():
    dag = EnsembleDAG()

    def preprocess_fn(data):
        return [t.lower().strip() for t in data]

    def model_fn(tokens):
        return np.array([len(t) * 0.5 for t in tokens], dtype=np.float32)

    def postprocess_fn(scores):
        return {
            "mean_score": float(np.mean(scores)),
            "passed": bool(np.mean(scores) > 1.0),
        }

    dag.add_stage("preprocess", preprocess_fn)
    dag.add_stage("model", model_fn, dependencies=["preprocess"])
    dag.add_stage("postprocess", postprocess_fn, dependencies=["model"])

    return dag


def run_oracle_bls(input_data):
    dag = build_sample_pipeline()
    orch = BLSOrchestrator(dag)
    return orch.execute_in_process(input_data)
