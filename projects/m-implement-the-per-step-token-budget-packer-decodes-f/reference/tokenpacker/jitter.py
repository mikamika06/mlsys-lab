from tokenpacker.steps import compute_steps


def predict_itl_jitter(prefill_lens, budget, decodes_per_step, unchunked=False):
    if unchunked:
        latencies = [float(p + decodes_per_step) for p in prefill_lens]
        return max(latencies) if latencies else 0.0
    else:
        total_latencies = [float(compute_steps(p, budget, decodes_per_step)) for p in prefill_lens]
        return max(total_latencies) if total_latencies else 0.0
