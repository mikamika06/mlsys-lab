def classify_output_diff(baseline_output, candidate_output):
    if baseline_output == candidate_output:
        return "numeric drift"
    if "BUG" in candidate_output:
        return "bug"
    if "TEMPLATE" in candidate_output:
        return "template change"
    if "SAMPLE" in candidate_output:
        return "sampling difference"
    return "numeric drift"
