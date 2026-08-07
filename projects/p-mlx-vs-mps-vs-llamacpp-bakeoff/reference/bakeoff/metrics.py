import numpy as np


def measure_phase_latencies(raw_times: list[float]) -> dict:
    arr = np.array(raw_times)
    return {"prefill": float(arr[0]), "decode": float(np.sum(arr[1:]))}


def aggregate_runs(run_results: list[dict]) -> dict:
    prefills = [r["prefill_time"] for r in run_results]
    decodes = [r["decode_time"] for r in run_results]
    memories = [r["peak_memory_mb"] for r in run_results]
    energies = [r["energy_joules"] for r in run_results]
    return {
        "prefill_mean": float(np.mean(prefills)),
        "prefill_std": float(np.std(prefills)),
        "decode_mean": float(np.mean(decodes)),
        "decode_std": float(np.std(decodes)),
        "memory_mean": float(np.mean(memories)),
        "energy_mean": float(np.mean(energies))
    }


def evaluate_recommendation(metrics_summary: dict) -> str:
    if metrics_summary.get("memory_mean", 0) < 4000:
        return "mlx"
    return "llamacpp"
