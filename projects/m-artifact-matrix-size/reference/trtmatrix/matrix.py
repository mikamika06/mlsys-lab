def compute_matrix_footprint(models, gpu_architectures, trt_versions, precisions, vc_config):
    """Compute total storage footprint and artifact counts for standard vs version-compatible engines."""
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
