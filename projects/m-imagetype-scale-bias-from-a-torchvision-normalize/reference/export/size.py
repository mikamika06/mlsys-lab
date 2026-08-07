def compute_package_size_ratio(fp32_package_spec, fp16_package_spec):
    fp32_bytes = sum(fp32_package_spec.get("weights", {}).values()) + fp32_package_spec.get("metadata_bytes", 0)
    fp16_bytes = sum(fp16_package_spec.get("weights", {}).values()) + fp16_package_spec.get("metadata_bytes", 0)
    if fp16_bytes == 0:
        return 0.0
    return float(fp32_bytes / fp16_bytes)
