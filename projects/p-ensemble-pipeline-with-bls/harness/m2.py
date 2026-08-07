import sys


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"bls_execution_ok": 0.0}

    try:
        import ref
        from pipeline.bls import BLSOrchestrator

        dag = ref.build_sample_pipeline()
        orch = BLSOrchestrator(dag)

        sample_input = [" Hello ", "  WORLD  ", "MLSys "]
        res = orch.execute_in_process(sample_input)

        if isinstance(res, dict) and "output" in res:
            val = res["output"]
            if isinstance(val, dict) and "mean_score" in val:
                out["bls_execution_ok"] = 1.0
    except Exception:
        pass

    return out
