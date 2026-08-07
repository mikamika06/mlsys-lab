import sys


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"latency_speedup": 0.0}

    try:
        import ref
        from pipeline.bls import BLSOrchestrator

        dag = ref.build_sample_pipeline()
        orch = BLSOrchestrator(dag)

        sample_input = ["data"]
        stats = orch.measure_overhead(sample_input, remote_latency_ms=10.0)

        out["latency_speedup"] = float(stats.get("speedup", 0.0))
    except Exception:
        pass

    return out
