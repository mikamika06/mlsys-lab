"""Artifact matrix size calculation."""


def calculate_matrix_size(architectures, trt_versions, runtime_modes, engine_base_mb, compatibility_bloat_factors):
    total_mb = 0.0
    for arch in architectures:
        for ver in trt_versions:
            for mode in runtime_modes:
                factor = compatibility_bloat_factors.get(mode, 1.0)
                total_mb += float(engine_base_mb) * float(factor)
    return round(total_mb, 4)
