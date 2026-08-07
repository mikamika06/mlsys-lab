import sys


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"outputs_identical": 0.0}

    try:
        import ref
        from pipeline.bls import BLSOrchestrator

        dag = ref.build_sample_pipeline()
        orch = BLSOrchestrator(dag)

        sample_input = ["  Test ", " Ensemble ", " BLS Pipeline  "]

        remote_res = dag.execute_remote(sample_input)
        bls_res = orch.execute_in_process(sample_input)

        if remote_res["output"] == bls_res["output"]:
            out["outputs_identical"] = 1.0
    except Exception:
        pass

    return out
