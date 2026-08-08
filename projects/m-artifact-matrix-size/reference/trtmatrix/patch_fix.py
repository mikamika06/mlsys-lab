def plan_container_patch_fixes(build_version, containers, options):
    """Categorize containers and assign lifecycle actions for TensorRT engine compatibility."""
    def parse_ver(v_str):
        parts = [int(p) for p in v_str.split(".")]
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])

    build_v = parse_ver(build_version)
    allow_patch = options.get("allow_patch_mismatch", False)
    vc_enabled = options.get("vc_enabled", False)

    exact_matches = []
    patch_compatible = []
    vc_compatible = []
    rebuild_required = []
    actions = {}

    for c in sorted(containers, key=lambda x: x["id"]):
        cid = c["id"]
        cv = parse_ver(c["trt_version"])

        if cv == build_v:
            exact_matches.append(cid)
            actions[cid] = "reuse"
        elif cv[0] == build_v[0] and cv[1] == build_v[1]:
            if allow_patch:
                patch_compatible.append(cid)
                actions[cid] = "patch_alias"
            else:
                rebuild_required.append(cid)
                actions[cid] = "rebuild"
        elif cv[0] == build_v[0]:
            if vc_enabled and cv >= build_v:
                vc_compatible.append(cid)
                actions[cid] = "vc_load"
            else:
                rebuild_required.append(cid)
                actions[cid] = "rebuild"
        else:
            rebuild_required.append(cid)
            actions[cid] = "rebuild"

    return {
        "exact_matches": sorted(exact_matches),
        "patch_compatible": sorted(patch_compatible),
        "vc_compatible": sorted(vc_compatible),
        "rebuild_required": sorted(rebuild_required),
        "actions": actions,
    }
