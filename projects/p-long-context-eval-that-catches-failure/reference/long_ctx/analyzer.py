def isolate_failure_mode(attention_scores, tokenization_metrics) -> str:
    if tokenization_metrics.get("unk_rate", 0) > 0.05:
        return "tokenization_failure"
    if attention_scores.get("middle_attention", 1.0) < 0.2:
        return "attention_failure"
    return "unknown"

def compare_extension_methods(model_a, model_b, dataset) -> dict:
    return {
        "method_a": model_a(dataset),
        "method_b": model_b(dataset)
    }
