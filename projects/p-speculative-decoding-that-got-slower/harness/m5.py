import numpy as np


def check(workdir):
    from specdec.tracker import AcceptanceTracker
    from specdec.model import SpeculativeModel
    from specdec.policy import AdaptivePolicy
    import ref

    out = {
        "p95_latency_bounded": 0.0,
        "throughput_maintained": 0.0
    }

    traffic = ref.generate_synthetic_traffic(n=120, seed=123)

    tracker = AcceptanceTracker(window_size=50)
    model = SpeculativeModel(target_step_cost=10.0, draft_step_cost=1.0, overhead_per_draft=0.1)
    policy = AdaptivePolicy(model, tracker, min_speedup=1.02)

    res = policy.evaluate_p95_and_throughput(traffic)

    baseline_latencies = []
    baseline_tokens = 0
    baseline_time = 0.0
    for req in traffic:
        b_time = req.get("base_step_time", 10.0) * (1.0 + 0.05 * (req["batch_size"] - 1))
        baseline_latencies.append(b_time)
        baseline_tokens += 1
        baseline_time += b_time

    base_p95 = float(np.percentile(baseline_latencies, 95))
    base_throughput = float(baseline_tokens) / float(baseline_time)

    if res["p95_latency"] <= base_p95 * 1.05:
        out["p95_latency_bounded"] = 1.0

    if res["throughput"] >= base_throughput * 1.05:
        out["throughput_maintained"] = 1.0

    return out
