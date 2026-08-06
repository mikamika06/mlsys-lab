import os
import tempfile
import numpy as np
import ref


def check(workdir):
    out = {
        "latency_ratio": 0.0,
        "p95_matches_reference": 0.0,
    }
    from edge_cache.compiler import compile_and_run
    from edge_cache.metrics import compute_population_p95

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.bin")
        cache_dir = os.path.join(tmpdir, "cache")
        with open(model_path, "wb") as f:
            f.write(b"model_weights_payload_data_for_test")

        inp = [1, 2, 3, 4, 5]
        res_cold = compile_and_run(model_path, cache_dir, inp)
        res_warm = compile_and_run(model_path, cache_dir, inp)

        if not res_cold.get("is_cold", False):
            out["_note"] = "First run was not flagged as cold start"
            return out
        if res_warm.get("is_cold", True):
            out["_note"] = "Second run was flagged as cold start instead of warm"
            return out

        cold_lat = res_cold.get("latency_ms", 0.0)
        warm_lat = res_warm.get("latency_ms", 0.0)

        if warm_lat > 0:
            out["latency_ratio"] = float(cold_lat / warm_lat)

        cohort_samples = [
            [10.0, 12.0, 15.0, 100.0, 120.0],
            [5.0, 6.0, 7.0, 8.0, 90.0],
            [20.0, 22.0, 25.0, 30.0, 35.0],
        ]
        cohort_weights = [0.5, 0.3, 0.2]

        ref_p95 = ref.compute_population_p95(cohort_samples, cohort_weights)
        got_p95 = compute_population_p95(cohort_samples, cohort_weights)

        if abs(ref_p95 - got_p95) < 1e-4:
            out["p95_matches_reference"] = 1.0
        else:
            out["_note"] = f"p95 mismatch: got {got_p95}, expected {ref_p95}"

    return out
