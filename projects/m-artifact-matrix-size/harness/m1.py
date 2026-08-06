import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    from artifact_matrix.planner import calculate_matrix_size
    from artifact_matrix.cost_model import estimate_vc_engine_cost

    out = {"matrix_sizes_matched": 1.0, "costs_matched": 1.0}

    for idx, inp in enumerate(ref.MATRIX_INPUTS):
        want = ref.ref_calculate_matrix_size(
            inp["architectures"],
            inp["trt_versions"],
            inp["runtime_modes"],
            inp["engine_base_mb"],
            inp["compatibility_bloat_factors"],
        )
        got = calculate_matrix_size(
            inp["architectures"],
            inp["trt_versions"],
            inp["runtime_modes"],
            inp["engine_base_mb"],
            inp["compatibility_bloat_factors"],
        )
        if got != want:
            out["matrix_sizes_matched"] = 0.0
            out["_note"] = f"matrix size case {idx}: got {got}, want {want}"
            break

    if out["matrix_sizes_matched"] == 1.0:
        for idx, inp in enumerate(ref.COST_INPUTS):
            want = ref.ref_estimate_vc_engine_cost(**inp)
            got = estimate_vc_engine_cost(**inp)
            if got != want:
                out["costs_matched"] = 0.0
                out["_note"] = f"cost model case {idx}: got {got}, want {want}"
                break

    return out
