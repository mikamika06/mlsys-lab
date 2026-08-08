import ref

def check(workdir):
    from quant_observe.observer import mse_observer

    out = {"mse_rel_err": 0.0}
    max_err = 0.0
    for t in ref.TENSORS:
        for s in ref.SCHEMES:
            args = ref.parse_scheme(s)
            w_scale, w_zp = ref.mse_observer(t, args)
            try:
                g_scale, g_zp = mse_observer(t, args)
                if g_zp != w_zp:
                    out["_note"] = f"zp mismatch for {s}: got {g_zp}, want {w_zp}"
                    max_err = 1.0
                    break
                err = abs(g_scale - w_scale) / (w_scale + 1e-9)
                max_err = max(max_err, err)
            except Exception as e:
                out["_note"] = f"mse_observer raised {e}"
                max_err = 1.0
                break

    out["mse_rel_err"] = float(max_err)
    return out
