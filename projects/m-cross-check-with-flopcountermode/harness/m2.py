import ref


def check(workdir):
    out = {"forward_shape": 0.0, "rel_err_is_zero": 0.0, "analytical_matches": 0.0}

    try:
        from attention.cross_check import attention_forward, analytical_flops, empirical_flops, rel_err
    except ImportError:
        out["_note"] = "ImportError"
        return out

    try:
        shape = attention_forward(2, 4, 128, 64)
        if shape == (2, 4, 128, 64):
            out["forward_shape"] = 1.0
    except Exception:
        pass

    try:
        err = rel_err(2, 4, 128, 64)
        if err == 0.0:
            out["rel_err_is_zero"] = 1.0
    except Exception:
        pass

    try:
        ana = analytical_flops(2, 4, 128, 64)
        if ana == ref.ref_analytical(2, 4, 128, 64):
            out["analytical_matches"] = 1.0
    except Exception:
        pass

    return out
