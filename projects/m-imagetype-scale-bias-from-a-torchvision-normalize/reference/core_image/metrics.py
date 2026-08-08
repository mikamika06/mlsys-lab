def compute_drift_and_ratio(fp32_outputs, fp16_outputs, fp32_size, fp16_size):
    max_diff = max(abs(a - b) for a, b in zip(fp32_outputs, fp16_outputs))
    ratio = fp16_size / float(fp32_size)
    return max_diff, ratio
