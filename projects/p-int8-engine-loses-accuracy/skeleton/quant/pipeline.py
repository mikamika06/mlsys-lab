def optimize_engine_fallback(
    layer_names,
    sensitive_layers,
    fp16_latencies,
    int8_latencies,
    target_accuracy_drop=0.01,
    min_speedup_retention=0.70,
):
    raise NotImplementedError


def generate_sensitivity_report(
    metrics, ranked_layers, fallback_layers, stats
):
    raise NotImplementedError
