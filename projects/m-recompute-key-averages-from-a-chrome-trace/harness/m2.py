import ref
from tracex.metrics import compute_gpu_metrics


def check(workdir):
    t_a, _ = ref.generate_traces()
    want = compute_gpu_metrics(t_a)
    from tracex.metrics import compute_gpu_metrics as learner_metrics
    try:
        got = learner_metrics(t_a)
    except Exception as e:
        return {"busy_fraction_match": 0.0, "idle_gaps_match": 0.0, "_note": f"raised: {e}"}
    bf_ok = 1.0 if abs(got.get("busy_fraction", -1) - want["busy_fraction"]) < 1e-5 else 0.0
    ig_ok = 1.0 if abs(got.get("idle_gaps", -1) - want["idle_gaps"]) < 1e-5 else 0.0
    return {"busy_fraction_match": bf_ok, "idle_gaps_match": ig_ok}
