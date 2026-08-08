import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    from trtmatrix.patch_fix import plan_container_patch_fixes
    from trtmatrix.vc_cost import analyze_vc_cost_tradeoff

    want_patch = ref.plan_container_patch_fixes(
        ref.BUILD_VERSION_M2,
        ref.CONTAINERS_M2,
        ref.PATCH_OPTIONS_M2,
    )
    want_cost = ref.analyze_vc_cost_tradeoff(
        ref.MODELS_M2,
        ref.TRT_VERSION_COUNT_M2,
        ref.VC_OVERHEAD_M2,
        ref.REFIT_OVERHEAD_M2,
    )

    out = {"patch_fixes_matched": 0.0, "cost_analysis_matched": 0.0}

    try:
        got_patch = plan_container_patch_fixes(
            ref.BUILD_VERSION_M2,
            ref.CONTAINERS_M2,
            ref.PATCH_OPTIONS_M2,
        )
        if got_patch == want_patch:
            out["patch_fixes_matched"] = 1.0
        else:
            out["_note"] = (
                f"patch plan mismatch: got {got_patch}, want {want_patch}"
            )
    except Exception as e:
        out["_note"] = f"plan_container_patch_fixes raised {type(e).__name__}: {e}"

    try:
        got_cost = analyze_vc_cost_tradeoff(
            ref.MODELS_M2,
            ref.TRT_VERSION_COUNT_M2,
            ref.VC_OVERHEAD_M2,
            ref.REFIT_OVERHEAD_M2,
        )
        if got_cost == want_cost:
            out["cost_analysis_matched"] = 1.0
        elif "_note" not in out:
            out["_note"] = (
                f"cost analysis mismatch: got {got_cost}, want {want_cost}"
            )
    except Exception as e:
        if "_note" not in out:
            out["_note"] = (
                f"analyze_vc_cost_tradeoff raised {type(e).__name__}: {e}"
            )

    return out
