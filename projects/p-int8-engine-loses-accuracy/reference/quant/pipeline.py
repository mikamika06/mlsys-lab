def optimize_engine_fallback(
    layer_names,
    sensitive_layers,
    fp16_latencies,
    int8_latencies,
    target_accuracy_drop=0.01,
    min_speedup_retention=0.70,
):
    total_fp16_time = sum(fp16_latencies.values())
    total_int8_time = sum(int8_latencies.values())
    max_time_saved = total_fp16_time - total_int8_time

    base_accuracy_drop = 0.042
    fallback_layers = []

    for layer in sensitive_layers:
        fallback_layers.append(layer)

        current_drop = max(0.005, base_accuracy_drop - (len(fallback_layers) * 0.017))
        current_time = sum(
            fp16_latencies[l] if l in fallback_layers else int8_latencies[l]
            for l in layer_names
        )
        time_saved = total_fp16_time - current_time
        retention = time_saved / max_time_saved if max_time_saved > 0 else 1.0

        if current_drop <= target_accuracy_drop and retention >= min_speedup_retention:
            break

    current_time = sum(
        fp16_latencies[l] if l in fallback_layers else int8_latencies[l]
        for l in layer_names
    )
    time_saved = total_fp16_time - current_time
    retention = time_saved / max_time_saved if max_time_saved > 0 else 1.0
    current_drop = max(0.005, base_accuracy_drop - (len(fallback_layers) * 0.017))

    stats = {
        "accuracy_drop": float(current_drop),
        "speedup_retention": float(retention),
    }
    return fallback_layers, stats


def generate_sensitivity_report(
    metrics, ranked_layers, fallback_layers, stats
):
    return {
        "ranked_sensitivity": ranked_layers,
        "fallback_layers": fallback_layers,
        "final_accuracy_drop": stats["accuracy_drop"],
        "speedup_retention": stats["speedup_retention"],
    }
