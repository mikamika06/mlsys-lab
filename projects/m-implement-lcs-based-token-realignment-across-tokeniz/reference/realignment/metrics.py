def compute_metrics(draft_tokens, target_tokens, mapping, overhead_ms):
    accepted = len(mapping)
    total_draft = max(1, len(draft_tokens))
    acceptance_rate = accepted / total_draft
    effective_throughput = acceptance_rate * (1000.0 / max(0.1, overhead_ms))
    return {
        "acceptance_rate": float(acceptance_rate),
        "overhead_ms": float(overhead_ms),
        "effective_throughput": float(effective_throughput)
    }
