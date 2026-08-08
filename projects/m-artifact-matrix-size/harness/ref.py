def compute_matrix_footprint(models, gpu_architectures, trt_versions, precisions, vc_config):
    total_standard_bytes = 0
    total_standard_artifacts = 0
    total_vc_bytes = 0
    total_vc_artifacts = 0

    vc_enabled_models = set(vc_config.get("enabled_models", []))
    vc_overhead = vc_config.get("vc_overhead_bytes", 0)

    def get_major(v_str):
        return v_str.split(".")[0]

    unique_majors = sorted(list({get_major(v) for v in trt_versions}))

    for model in models:
        base_b = model["base_bytes"]
        tactics_b = model["tactics_bytes"]
        name = model["name"]
        is_vc = name in vc_enabled_models

        for gpu in gpu_architectures:
            for prec, scale in precisions.items():
                engine_std_size = int(base_b * scale + tactics_b)

                n_std = len(trt_versions)
                total_standard_artifacts += n_std
                total_standard_bytes += engine_std_size * n_std

                if is_vc:
                    engine_vc_size = engine_std_size + vc_overhead
                    n_vc = len(unique_majors)
                    total_vc_artifacts += n_vc
                    total_vc_bytes += engine_vc_size * n_vc
                else:
                    total_vc_artifacts += n_std
                    total_vc_bytes += engine_std_size * n_std

    savings_bytes = total_standard_bytes - total_vc_bytes

    return {
        "total_standard_bytes": total_standard_bytes,
        "total_standard_artifacts": total_standard_artifacts,
        "total_vc_bytes": total_vc_bytes,
        "total_vc_artifacts": total_vc_artifacts,
        "savings_bytes": savings_bytes,
    }


def plan_container_patch_fixes(build_version, containers, options):
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


def analyze_vc_cost_tradeoff(models, trt_version_count, vc_overhead_bytes, refit_overhead_bytes):
    per_model = {}
    total_net_bytes_saved = 0
    recommended = []

    for m in sorted(models, key=lambda x: x["name"]):
        name = m["name"]
        scale = m.get("precision_scale", 1.0)
        std_bytes = int(m["base_bytes"] * scale + m["tactics_bytes"])

        refit_extra = refit_overhead_bytes if m.get("enable_refit", False) else 0
        vc_bytes = std_bytes + vc_overhead_bytes + refit_extra

        total_std = std_bytes * trt_version_count
        total_vc = vc_bytes * 1
        net_saved = total_std - total_vc

        breakeven = (vc_bytes // std_bytes) + 1 if std_bytes > 0 else 1
        is_beneficial = net_saved > 0

        per_model[name] = {
            "std_engine_bytes": std_bytes,
            "vc_engine_bytes": vc_bytes,
            "total_std_storage_bytes": total_std,
            "total_vc_storage_bytes": total_vc,
            "net_bytes_saved": net_saved,
            "breakeven_versions": breakeven,
            "is_vc_beneficial": is_beneficial,
        }

        total_net_bytes_saved += net_saved
        if is_beneficial:
            recommended.append(name)

    return {
        "per_model": per_model,
        "total_net_bytes_saved": total_net_bytes_saved,
        "recommended_vc_models": sorted(recommended),
    }


MODELS_M1 = [
    {"name": "resnet50", "base_bytes": 100_000_000, "tactics_bytes": 5_000_000},
    {"name": "bert_base", "base_bytes": 400_000_000, "tactics_bytes": 20_000_000},
    {"name": "llama_7b", "base_bytes": 7_000_000_000, "tactics_bytes": 150_000_000},
]

GPU_ARCHS_M1 = ["sm_80", "sm_86", "sm_90"]

TRT_VERSIONS_M1 = ["10.0.1", "10.0.2", "10.1.0", "10.2.0", "11.0.0", "11.1.0"]

PRECISONS_M1 = {"fp16": 1.0, "int8": 0.5}

VC_CONFIG_M1 = {
    "enabled_models": ["bert_base", "llama_7b"],
    "vc_overhead_bytes": 12_000_000,
}

CONTAINERS_M2 = [
    {"id": "node-01", "trt_version": "10.0.1"},
    {"id": "node-02", "trt_version": "10.0.3"},
    {"id": "node-03", "trt_version": "10.1.0"},
    {"id": "node-04", "trt_version": "10.2.0"},
    {"id": "node-05", "trt_version": "9.2.0"},
    {"id": "node-06", "trt_version": "10.0.1"},
]

BUILD_VERSION_M2 = "10.0.1"

PATCH_OPTIONS_M2 = {
    "allow_patch_mismatch": True,
    "vc_enabled": True,
}

MODELS_M2 = [
    {
        "name": "resnet50",
        "base_bytes": 100_000_000,
        "tactics_bytes": 5_000_000,
        "precision_scale": 0.5,
        "enable_refit": False,
    },
    {
        "name": "bert_base",
        "base_bytes": 400_000_000,
        "tactics_bytes": 20_000_000,
        "precision_scale": 1.0,
        "enable_refit": True,
    },
    {
        "name": "llama_7b",
        "base_bytes": 7_000_000_000,
        "tactics_bytes": 150_000_000,
        "precision_scale": 0.5,
        "enable_refit": True,
    },
]

TRT_VERSION_COUNT_M2 = 5
VC_OVERHEAD_M2 = 15_000_000
REFIT_OVERHEAD_M2 = 8_000_000
