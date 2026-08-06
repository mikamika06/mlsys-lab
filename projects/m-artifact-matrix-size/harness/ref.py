"""Reference generator for harness checks."""

MATRIX_INPUTS = [
    {
        "architectures": ["sm_80", "sm_89", "sm_90"],
        "trt_versions": ["10.0.1", "10.2.0"],
        "runtime_modes": ["standard", "vc_enabled"],
        "engine_base_mb": 450.0,
        "compatibility_bloat_factors": {"standard": 1.0, "vc_enabled": 1.35},
    },
    {
        "architectures": ["sm_86"],
        "trt_versions": ["8.6.1"],
        "runtime_modes": ["standard", "vc_enabled", "lean"],
        "engine_base_mb": 120.0,
        "compatibility_bloat_factors": {"standard": 1.0, "vc_enabled": 1.4, "lean": 0.85},
    },
]

COST_INPUTS = [
    {"base_engine_mb": 500.0, "enable_vc": True, "refit_enabled": True, "lean_runtime": False},
    {"base_engine_mb": 250.0, "enable_vc": False, "refit_enabled": False, "lean_runtime": True},
    {"base_engine_mb": 1000.0, "enable_vc": True, "refit_enabled": False, "lean_runtime": True},
]

PATCH_INPUTS = [
    ("10.2.1", "10.2.1", "strict"),
    ("10.2.1", "10.2.4", "strict"),
    ("10.2.1", "10.2.4", "allow_patch_drift"),
    ("10.2.1", "10.2.4", "auto_patch_alias"),
    ("10.2.1", "10.3.0", "auto_patch_alias"),
]


def ref_calculate_matrix_size(architectures, trt_versions, runtime_modes, engine_base_mb, bloat_factors):
    total = sum(
        engine_base_mb * bloat_factors.get(mode, 1.0)
        for _ in architectures
        for _ in trt_versions
        for mode in runtime_modes
    )
    return round(total, 4)


def ref_estimate_vc_engine_cost(base_engine_mb, enable_vc, refit_enabled, lean_runtime):
    size = float(base_engine_mb)
    if enable_vc:
        size *= 1.35
    if refit_enabled:
        size *= 1.15
    if lean_runtime:
        size *= 0.85
    size = round(size, 4)
    return {
        "final_size_mb": size,
        "delta_mb": round(size - base_engine_mb, 4),
        "ratio": round(size / base_engine_mb, 4),
    }


def ref_resolve_container_patch(c_ver, e_ver, policy):
    c_p = [int(x) for x in c_ver.split(".")]
    e_p = [int(x) for x in e_ver.split(".")]
    if c_p[:2] != e_p[:2]:
        return {"compatible": False, "resolved_version": e_ver, "action": "reject_major_minor_mismatch"}
    if c_p[2] == e_p[2]:
        return {"compatible": True, "resolved_version": e_ver, "action": "exact_match"}
    if policy == "strict":
        return {"compatible": False, "resolved_version": e_ver, "action": "reject_patch_mismatch"}
    if policy == "allow_patch_drift":
        return {"compatible": True, "resolved_version": e_ver, "action": "allow_drift"}
    if policy == "auto_patch_alias":
        return {"compatible": True, "resolved_version": c_ver, "action": "aliased_to_container_patch"}
    raise ValueError(f"Unknown patch policy: {policy}")
