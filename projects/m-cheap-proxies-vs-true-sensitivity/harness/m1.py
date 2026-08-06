import ref


def check(workdir):
    try:
        from sens import proxy
    except ImportError:
        return {"proxy_matched": 0.0}

    out = {"proxy_matched": 0.0}
    ok = 0
    for i, layers in enumerate(ref.CONFIGS):
        for layer in layers:
            want = ref.compute_proxy(layer)
            try:
                got = proxy.compute_proxy(layer)
                rel_err = abs(want - got) / (abs(want) + 1e-9)
                if rel_err < 1e-3:
                    ok += 1
            except Exception:
                pass

    out["proxy_matched"] = float(ok)
    return out
