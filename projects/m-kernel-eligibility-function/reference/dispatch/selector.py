def is_eligible(kernel_spec, layer_config):
    """Checks whether a layer configuration satisfies all requirements for a kernel."""
    if layer_config.get("in_dtype") not in kernel_spec.get("allowed_in_dtypes", []):
        return False
    if layer_config.get("out_dtype") not in kernel_spec.get("allowed_out_dtypes", []):
        return False
    if layer_config.get("quant_scheme") != kernel_spec.get("quant_scheme"):
        return False

    req_group_size = kernel_spec.get("group_size")
    if req_group_size is not None and layer_config.get("group_size") != req_group_size:
        return False

    min_k = kernel_spec.get("min_k", 0)
    if layer_config.get("k", 0) < min_k:
        return False

    align_k = kernel_spec.get("align_k", 1)
    if layer_config.get("k", 0) % align_k != 0:
        return False

    align_n = kernel_spec.get("align_n", 1)
    if layer_config.get("n", 0) % align_n != 0:
        return False

    req_align_bytes = kernel_spec.get("req_align_bytes")
    if req_align_bytes is not None:
        ptr_align = layer_config.get("ptr_align_bytes", 1)
        if ptr_align % req_align_bytes != 0:
            return False

    return True


def dispatch_kernel(available_kernels, layer_config):
    """Returns the name of the highest-priority eligible kernel, or 'fallback_gemm'."""
    eligible = [
        k for k in available_kernels
        if is_eligible(k, layer_config)
    ]
    if not eligible:
        return "fallback_gemm"
    eligible.sort(key=lambda x: x.get("priority", 0), reverse=True)
    return eligible[0]["name"]
