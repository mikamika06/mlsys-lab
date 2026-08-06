def classify_guard_failure(change_description):
    structural_keywords = ["shape", "stride", "dtype", "device", "global_variable", "none_to_tensor"]
    for kw in structural_keywords:
        if kw in change_description:
            return "guard_failure"
    return "safe"
