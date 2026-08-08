import ref

def check(workdir):
    from quant_observe.scheme import parse_scheme
    from quant_observe.observer import minmax_observer

    out = {"parse_ok": 0.0, "minmax_rel_err": 0.0}

    p_ok = True
    for s in ref.SCHEMES:
        want = ref.parse_scheme(s)
        try:
            got = parse_scheme(s)
            if want != got:
                p_ok = False
                out["_note"] = f"parse_scheme({s}) got {got}, want {want}"
                break
        except Exception as e:
            p_ok = False
            out["_note"] = f"parse_scheme({s}) raised {e}"
            break
    if p_ok:
        out["parse_ok"] = 1.0

    max_err = 0.0
    for t in ref.TENSORS:
        for s in ref.SCHEMES:
            args = ref.parse_scheme(s)
            w_scale, w_zp = ref.minmax_observer(t, args)
            try:
                g_scale, g_zp = minmax_observer(t, args)
                if g_zp != w_zp:
                    out["_note"] = f"zp mismatch for {s}: got {g_zp}, want {w_zp}"
                    max_err = 1.0
                    break
                err = abs(g_scale - w_scale) / (w_scale + 1e-9)
                max_err = max(max_err, err)
            except Exception as e:
                out["_note"] = f"minmax_observer raised {e}"
                max_err = 1.0
                break

    out["minmax_rel_err"] = float(max_err)
    return out
