import sys
import os


def check(workdir):
    sys.path.insert(0, workdir)
    res = {"benchmark_completed": 0.0, "latency_ratio_ok": 0.0}

    try:
        from exporter.runtime_runner import StandaloneAOTRunner, benchmark_aot_runner

        so_path = os.path.join(workdir, "build", "model_aot.so")
        os.makedirs(os.path.dirname(so_path), exist_ok=True)
        with open(so_path, "wb") as f:
            f.write(b"\x7fELF_MOCK_DATA")

        runner = StandaloneAOTRunner(so_path)
        metrics = benchmark_aot_runner(runner, num_runs=20)

        if "cold_start_ms" in metrics and "mean_latency_ms" in metrics and "p99_latency_ms" in metrics:
            res["benchmark_completed"] = 1.0

        if metrics["mean_latency_ms"] >= 0 and metrics["p99_latency_ms"] >= metrics["mean_latency_ms"]:
            res["latency_ratio_ok"] = 1.0

    except Exception:
        pass

    return res
