def check_tool_compatibility(metadata: dict, tool_profile: dict) -> dict:
    reasons = []
    arch = metadata.get("general.architecture", "")
    supported_archs = tool_profile.get("supported_architectures", [])
    if supported_archs and arch not in supported_archs:
        reasons.append(f"Unsupported architecture: {arch}")

    req_keys = tool_profile.get("required_keys", [])
    for key in req_keys:
        formatted_key = key.format(arch=arch) if "{arch}" in key else key
        if formatted_key not in metadata:
            reasons.append(f"Missing required key: {formatted_key}")

    max_ctx = tool_profile.get("max_context_length")
    if max_ctx is not None:
        ctx_key = f"{arch}.context_length"
        model_ctx = metadata.get(ctx_key, metadata.get("general.context_length", 0))
        if model_ctx > max_ctx:
            reasons.append(f"Model context length ({model_ctx}) exceeds maximum supported ({max_ctx})")

    max_mem = tool_profile.get("max_memory_bytes")
    if max_mem is not None:
        est_mem = metadata.get("general.estimated_memory_bytes", 0)
        if est_mem > max_mem:
            reasons.append(f"Estimated memory ({est_mem}) exceeds tool limit ({max_mem})")

    return {
        "compatible": len(reasons) == 0,
        "reasons": reasons
    }
