def estimate_package_ratio(fp32_weights_bytes, overhead_bytes=1024):
    fp32_total = fp32_weights_bytes + overhead_bytes
    fp16_weights_bytes = fp32_weights_bytes // 2
    fp16_total = fp16_weights_bytes + overhead_bytes
    return float(fp16_total) / float(fp32_total)
