import ref


def check(workdir):
    from hashbench.hashing import measure_throughput

    dummy_data = b"0123456789abcdef" * 100000
    ref_res = ref.measure_throughput(dummy_data, iterations=2)
    try:
        learner_res = measure_throughput(dummy_data, iterations=2)
    except Exception as e:
        return {"latency_ratio": 0.0, "_note": f"Execution failed: {e}"}

    ratio = learner_res.get("ratio", 0.0)
    return {
        "latency_ratio": float(ratio),
        "_note": f"Measured ratio xxhash/sha256: {ratio:.2f}"
    }
