def audit_dependencies(manifest, registry):
    selected = manifest.get("selected", [])
    deprecated_count = 0
    warnings = []
    remediations = {}
    for item in selected:
        lib = item["lib"]
        fmt = item["format"]
        dep_map = registry.get(lib, {})
        if fmt in dep_map:
            deprecated_count += 1
            replacement = dep_map[fmt]
            warnings.append(f"{lib}.{fmt} is deprecated")
            remediations[f"{lib}.{fmt}"] = replacement

    valid = (deprecated_count == 0)
    return {
        "valid": valid,
        "deprecated_count": deprecated_count,
        "warnings": warnings,
        "remediations": remediations,
    }
