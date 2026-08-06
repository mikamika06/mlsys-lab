def check_header_compatibility(header_dict):
    issues = []
    magic = header_dict.get("magic")
    if magic != "GGUF":
        issues.append(f"Invalid magic: expected 'GGUF', got '{magic}'")

    version = header_dict.get("version", 0)
    min_ver = header_dict.get("min_supported_version", 2)
    max_ver = header_dict.get("max_supported_version", 3)

    if version < min_ver or version > max_ver:
        issues.append(f"Unsupported header version {version} (expected between {min_ver} and {max_ver})")

    return {
        "valid": len(issues) == 0,
        "issues": issues
    }


def detect_version_skew(converter_meta, runtime_meta):
    mismatches = []
    missing_required_keys = []
    unsupported_quants = []

    c_ver = converter_meta.get("version")
    r_ver = runtime_meta.get("version")
    if c_ver != r_ver:
        mismatches.append(f"Version mismatch: converter {c_ver} vs runtime {r_ver}")

    c_build = converter_meta.get("build_id")
    r_build = runtime_meta.get("build_id")
    if c_build != r_build:
        mismatches.append(f"Build ID mismatch: converter {c_build} vs runtime {r_build}")

    req_keys = runtime_meta.get("required_keys", [])
    c_keys = set(converter_meta.get("metadata_keys", []))
    for k in req_keys:
        if k not in c_keys:
            missing_required_keys.append(k)

    r_quants = set(runtime_meta.get("supported_quant_types", []))
    c_quants = converter_meta.get("quant_types", [])
    for q in c_quants:
        if q not in r_quants:
            unsupported_quants.append(q)

    has_skew = len(mismatches) > 0 or len(missing_required_keys) > 0 or len(unsupported_quants) > 0

    return {
        "has_skew": has_skew,
        "version_mismatches": mismatches,
        "missing_required_keys": missing_required_keys,
        "unsupported_quant_types": unsupported_quants
    }
