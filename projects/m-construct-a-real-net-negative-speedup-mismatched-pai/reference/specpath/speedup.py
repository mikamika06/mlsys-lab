def compute_speculative_speedup(draft_latency, target_verify_latency, target_step_latency, K, acceptance_rate):
    r = float(acceptance_rate)
    k = float(K)
    expected_accepted_tokens = (1.0 - (r ** (k + 1.0))) / (1.0 - r) if r != 1.0 else (k + 1.0)
    speculative_step_time = (k * float(draft_latency)) + float(target_verify_latency)
    if speculative_step_time <= 0:
        return 0.0
    speculative_rate = expected_accepted_tokens / speculative_step_time
    baseline_rate = 1.0 / float(target_step_latency)
    return float(speculative_rate / baseline_rate)


def find_net_negative_pairs(configs):
    net_negatives = []
    for cfg in configs:
        speedup = compute_speculative_speedup(
            cfg["draft_latency"],
            cfg["target_verify_latency"],
            cfg["target_step_latency"],
            cfg["K"],
            cfg["acceptance_rate"]
        )
        if speedup < 1.0:
            net_negatives.append({
                "id": cfg["id"],
                "speedup": speedup,
                "acceptance_rate": cfg["acceptance_rate"]
            })
    return net_negatives
