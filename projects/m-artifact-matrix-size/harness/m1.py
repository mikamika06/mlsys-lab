import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    from trtmatrix.matrix import compute_matrix_footprint

    want = ref.compute_matrix_footprint(
        ref.MODELS_M1,
        ref.GPU_ARCHS_M1,
        ref.TRT_VERSIONS_M1,
        ref.PRECISONS_M1,
        ref.VC_CONFIG_M1,
    )
    try:
        got = compute_matrix_footprint(
            ref.MODELS_M1,
            ref.GPU_ARCHS_M1,
            ref.TRT_VERSIONS_M1,
            ref.PRECISONS_M1,
            ref.VC_CONFIG_M1,
        )
    except Exception as e:
        return {
            "matrix_matched": 0.0,
            "_note": f"compute_matrix_footprint raised {type(e).__name__}: {e}",
        }

    out = {"matrix_matched": 0.0}
    if got == want:
        out["matrix_matched"] = 1.0
    else:
        out["_note"] = f"Expected {want}, got {got}"

    return out
