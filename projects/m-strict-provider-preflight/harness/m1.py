import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"compat_checks_matched": 0.0, "preflight_matched": 0.0}
    try:
        from ortpreflight.oracle import check_cuda_cudnn_compat
        from ortpreflight.preflight import validate_preflight
    except Exception as e:
        out["_note"] = f"failed to import learner module: {e}"
        return out

    ok_compat = 0
    for i, (ort_v, cuda_v, cudnn_v) in enumerate(ref.COMPAT_CASES):
        want = ref.check_cuda_cudnn_compat(ort_v, cuda_v, cudnn_v)
        try:
            got = check_cuda_cudnn_compat(ort_v, cuda_v, cudnn_v)
            if got == want:
                ok_compat += 1
            elif "_note" not in out:
                out["_note"] = f"compat case {i}: got {got}, reference {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"compat case {i} raised {e}"
    out["compat_checks_matched"] = float(ok_compat)

    ok_pf = 0
    for i, (req, avail, env, strict) in enumerate(ref.PREFLIGHT_CASES):
        want = ref.validate_preflight(req, avail, env, strict)
        try:
            got = validate_preflight(req, avail, env, strict)
            if got == want:
                ok_pf += 1
            elif "_note" not in out:
                out["_note"] = f"preflight case {i}: got {got}, reference {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"preflight case {i} raised {e}"
    out["preflight_matched"] = float(ok_pf)
    return out
