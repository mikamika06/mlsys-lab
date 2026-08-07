from cascade.measure import measure_single_stage_latency, measure_stage_latencies


def compute_latency_ratio(cascade_latency, single_stage_latency):
    if single_stage_latency <= 0:
        return float("inf")
    return cascade_latency / single_stage_latency


def run_cascade_benchmark(config):
    s1_model = config["stage1_model"]
    s2_model = config["stage2_model"]
    target_model = config["target_model"]
    inputs = config["inputs"]
    draft_steps = config["draft_steps"]

    cascade_res = measure_stage_latencies(s1_model, s2_model, inputs, draft_steps)
    single_res = measure_single_stage_latency(target_model, inputs, cascade_res["accepted_count"])

    ratio = compute_latency_ratio(cascade_res["total_latency"], single_res["total_latency"])

    return {
        "cascade_latency": cascade_res["total_latency"],
        "single_stage_latency": single_res["total_latency"],
        "latency_ratio": ratio,
        "accepted_count": cascade_res["accepted_count"],
        "draft_count": cascade_res["draft_count"]
    }
