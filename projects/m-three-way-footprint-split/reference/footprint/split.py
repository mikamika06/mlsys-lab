def analyze_three_way_footprint(binary_info, runtime_config, model_tensors):
    """Analyze memory footprint split into binary, runtime, and tensor dynamic components."""
    text_size = binary_info.get("text_bytes", 0)
    rodata_size = binary_info.get("rodata_bytes", 0)
    data_size = binary_info.get("data_bytes", 0)
    bss_size = binary_info.get("bss_bytes", 0)

    static_binary = text_size + rodata_size + data_size + bss_size

    base_heap = runtime_config.get("base_heap_bytes", 0)
    ctx_structs = runtime_config.get("context_struct_bytes", 0)
    arena_overhead = runtime_config.get("arena_metadata_bytes", 0)

    runtime_infra = base_heap + ctx_structs + arena_overhead

    weights_bytes = 0
    activations_bytes = 0
    workspace_bytes = 0

    for tensor in model_tensors:
        numel = tensor.get("numel", 0)
        elem_bytes = tensor.get("elem_bytes", 1)
        total_bytes = numel * elem_bytes
        category = tensor.get("category", "activation")
        if category == "weight":
            weights_bytes += total_bytes
        elif category == "workspace":
            workspace_bytes += total_bytes
        else:
            activations_bytes += total_bytes

    dynamic_tensors = weights_bytes + activations_bytes + workspace_bytes
    total_footprint = static_binary + runtime_infra + dynamic_tensors

    return {
        "static_binary_bytes": static_binary,
        "runtime_infra_bytes": runtime_infra,
        "dynamic_tensors_bytes": dynamic_tensors,
        "total_footprint_bytes": total_footprint,
        "weight_bytes": weights_bytes,
        "activation_bytes": activations_bytes,
        "workspace_bytes": workspace_bytes
    }
