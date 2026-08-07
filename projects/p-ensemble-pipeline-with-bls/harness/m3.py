import sys


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"bytes_saved_ratio": 0.0}

    try:
        import ref
        from pipeline.bls import BLSOrchestrator

        dag = ref.build_sample_pipeline()
        orch = BLSOrchestrator(dag)

        sample_input = ["large_payload_token_" * 10 for _ in range(50)]
        stats = orch.measure_overhead(sample_input, remote_latency_ms=10.0)

        out["bytes_saved_ratio"] = float(stats.get("bytes_saved_ratio", 0.0))
    except Exception:
        pass

    return out
