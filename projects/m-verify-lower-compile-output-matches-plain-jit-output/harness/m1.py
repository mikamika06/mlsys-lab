import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from jaxinspect.verify import verify_compile_vs_jit

    out = {"max_abs_err": 1.0, "all_matched": 0.0}

    res_match = verify_compile_vs_jit(ref.AOT_OUTPUTS, ref.JIT_OUTPUTS_MATCH)
    res_drift = verify_compile_vs_jit(ref.AOT_OUTPUTS, ref.JIT_OUTPUTS_DRIFT)

    ref_match = ref.verify_compile_vs_jit(ref.AOT_OUTPUTS, ref.JIT_OUTPUTS_MATCH)
    ref_drift = ref.verify_compile_vs_jit(ref.AOT_OUTPUTS, ref.JIT_OUTPUTS_DRIFT)

    if res_match["is_close"] and not res_drift["is_close"]:
        if abs(res_match["max_abs_err"] - ref_match["max_abs_err"]) < 1e-6:
            out["max_abs_err"] = res_match["max_abs_err"]
            out["all_matched"] = 1.0
        else:
            out["_note"] = f"max_abs_err discrepancy: got {res_match['max_abs_err']}, expected {ref_match['max_abs_err']}"
    else:
        out["_note"] = f"is_close flag evaluation failed: match={res_match}, drift={res_drift}"

    return out
