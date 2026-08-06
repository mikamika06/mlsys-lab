"""Version skew diagnostic engine."""

def diagnose_version_skew(model_manifest: dict, runtime_spec: dict) -> dict:
    skews = []

    if not model_manifest.get("valid_magic", False):
        skews.append("INVALID_MAGIC")

    c_ver = model_manifest.get("container_version", 0)
    min_c = runtime_spec.get("min_container_version", 1)
    max_c = runtime_spec.get("max_container_version", 3)
    if c_ver < min_c or c_ver > max_c:
        skews.append(f"UNSUPPORTED_CONTAINER_VERSION_{c_ver}")

    supported_quants = set(runtime_spec.get("supported_quant_types", []))
    used_quants = set(model_manifest.get("tensor_types", []))
    unsupported = sorted(list(used_quants - supported_quants))
    if unsupported:
        skews.append(f"UNSUPPORTED_QUANT_TYPES:{','.join(map(str, unsupported))}")

    req_keys = set(runtime_spec.get("required_metadata_keys", []))
    present_keys = set(model_manifest.get("metadata_keys", []))
    missing_keys = sorted(list(req_keys - present_keys))
    if missing_keys:
        skews.append(f"MISSING_METADATA_KEYS:{','.join(missing_keys)}")

    align = model_manifest.get("alignment", 32)
    min_align = runtime_spec.get("required_alignment", 32)
    if align < min_align:
        skews.append(f"ALIGNMENT_MISMATCH_{align}_VS_{min_align}")

    return {
        "compatible": len(skews) == 0,
        "skews": skews
    }
