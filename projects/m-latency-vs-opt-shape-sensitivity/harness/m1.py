import ref

def check(workdir):
    from trtopt.tensors import classify_tensors
    from trtopt.sensitivity import compute_shape_sensitivity

    out = {"classified_correctly": 0.0, "sensitivity_scores_matched": 0.0}

    got_class = classify_tensors(ref.GRAPH_SPEC)
    want_class = ref.classify_tensors_oracle(ref.GRAPH_SPEC)
    if got_class == want_class:
        out["classified_correctly"] = 1.0
    else:
        out["_note"] = f"Classification mismatch: got {got_class}, want {want_class}"

    got_sens = compute_shape_sensitivity(ref.WIDE_PROFILE, ref.cost_function)
    want_sens = ref.compute_sensitivity_oracle(ref.WIDE_PROFILE, ref.cost_function)

    min_diff = abs(got_sens.get("sens_min", 0.0) - want_sens["sens_min"])
    max_diff = abs(got_sens.get("sens_max", 0.0) - want_sens["sens_max"])
    tot_diff = abs(got_sens.get("total_sensitivity", 0.0) - want_sens["total_sensitivity"])

    if min_diff < 1e-4 and max_diff < 1e-4 and tot_diff < 1e-4:
        out["sensitivity_scores_matched"] = 1.0
    elif "_note" not in out:
        out["_note"] = f"Sensitivity mismatch: got {got_sens}, want {want_sens}"

    return out
