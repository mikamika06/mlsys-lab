import sys


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"dag_valid": 0.0, "stages_count": 0.0}

    try:
        from pipeline.ensemble import EnsembleDAG

        dag = EnsembleDAG()
        dag.add_stage("preprocess", lambda x: x)
        dag.add_stage("model", lambda x: x, dependencies=["preprocess"])
        dag.add_stage("postprocess", lambda x: x, dependencies=["model"])

        if hasattr(dag, "stages") and len(dag.stages) == 3:
            out["stages_count"] = 3.0

        if dag.validate():
            out["dag_valid"] = 1.0

        invalid_dag = EnsembleDAG()
        invalid_dag.add_stage("preprocess", lambda x: x, dependencies=["missing"])
        if invalid_dag.validate():
            out["dag_valid"] = 0.0
    except Exception:
        pass

    return out
