"""Container patch mismatch resolution."""


def parse_version(ver_str):
    parts = [int(x) for x in ver_str.split(".")]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def resolve_container_patch(container_version, engine_version, patch_policy):
    c_maj, c_min, c_patch = parse_version(container_version)
    e_maj, e_min, e_patch = parse_version(engine_version)

    if (c_maj, c_min) != (e_maj, e_min):
        return {
            "compatible": False,
            "resolved_version": engine_version,
            "action": "reject_major_minor_mismatch",
        }

    if c_patch == e_patch:
        return {
            "compatible": True,
            "resolved_version": engine_version,
            "action": "exact_match",
        }

    if patch_policy == "strict":
        return {
            "compatible": False,
            "resolved_version": engine_version,
            "action": "reject_patch_mismatch",
        }
    elif patch_policy == "allow_patch_drift":
        return {
            "compatible": True,
            "resolved_version": engine_version,
            "action": "allow_drift",
        }
    elif patch_policy == "auto_patch_alias":
        resolved = f"{c_maj}.{c_min}.{c_patch}"
        return {
            "compatible": True,
            "resolved_version": resolved,
            "action": "aliased_to_container_patch",
        }
    else,
        raise ValueError(f"Unknown patch policy: {patch_policy}")
