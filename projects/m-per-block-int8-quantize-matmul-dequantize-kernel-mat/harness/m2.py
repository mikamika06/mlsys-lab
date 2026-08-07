import ref

def check(workdir):
    out = {"e4m3_err_diff": 1.0, "scaling_diff": 1.0}
    try:
        import qmat.quant as quant
        import qmat.scaling as scaling

        x = ref.generate_quant_fixtures()
        want_err = ref.e4m3_max_rel_error(x)
        got_err = quant.e4m3_max_rel_error(x)
        out["e4m3_err_diff"] = float(abs(want_err - got_err))

        y = ref.generate_scaling_fixtures()
        w_t, w_b = ref.per_tensor_vs_block(y, 16)
        g_t, g_b = scaling.per_tensor_vs_block(y, 16)
        out["scaling_diff"] = float(abs(w_t - g_t) + abs(w_b - g_b))
    except Exception as e:
        out["_note"] = f"m2 failed: {type(e).__name__}: {str(e)[:120]}"

    return out
