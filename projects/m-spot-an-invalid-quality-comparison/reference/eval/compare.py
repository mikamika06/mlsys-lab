import math


def check_comparison_validity(run_a, run_b):
    reasons = []
    if run_a.get("tokenizer_id") != run_b.get("tokenizer_id"):
        reasons.append("tokenizer_mismatch")
    if run_a.get("context_length") != run_b.get("context_length"):
        reasons.append("context_length_mismatch")
    if run_a.get("stride") != run_b.get("stride"):
        reasons.append("stride_mismatch")
    if run_a.get("dataset_hash") != run_b.get("dataset_hash"):
        reasons.append("dataset_mismatch")
    return {"valid": len(reasons) == 0, "reasons": reasons}


def is_statistically_significant(score_a, stderr_a, score_b, stderr_b, z_threshold=1.96):
    diff = abs(score_a - score_b)
    combined_err = math.sqrt(stderr_a ** 2 + stderr_b ** 2)
    if combined_err == 0.0:
        return diff > 0.0
    z_score = diff / combined_err
    return z_score >= z_threshold
