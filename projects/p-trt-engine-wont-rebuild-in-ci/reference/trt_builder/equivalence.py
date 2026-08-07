from trt_builder.inspect import diff_engine_graphs


def verify_equivalence(graph_a, graph_b, max_tactic_drift=0.0):
    diff = diff_engine_graphs(graph_a, graph_b)

    if diff["added"] or diff["removed"] or diff["precision_mismatches"]:
        return False, "Structural or precision mismatch detected"

    total_layers = len(graph_a)
    if total_layers == 0:
        return True, "Both graphs are empty"

    num_tactic_diffs = len(diff["tactic_mismatches"])
    drift_ratio = num_tactic_diffs / float(total_layers)

    if drift_ratio > max_tactic_drift:
        return False, f"Tactic drift ratio {drift_ratio:.3f} exceeds maximum allowed {max_tactic_drift:.3f}"

    return True, "Equivalent"
