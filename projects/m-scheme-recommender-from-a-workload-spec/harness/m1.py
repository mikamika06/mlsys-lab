import ref


def check(workdir):
    try:
        import sys
        sys.path.insert(0, workdir)
        from quantrec.bytes import bytes_per_token
    except ImportError:
        return {"bytes_rel_err": 1.0}

    schemes = ["FP16", "W8A8", "W4A16"]
    param_counts = [1e9, 7e9, 13e9, 70e9]
    max_err = 0.0
    for p in param_counts:
        for s in schemes:
            want = ref.bytes_per_token(p, s)
            try:
                got = float(bytes_per_token(p, s))
            except Exception:
                return {"bytes_rel_err": 1.0}
            if want > 0:
                err = abs(got - want) / want
            else:
                err = abs(got - want)
            if err > max_err:
                max_err = err
    return {"bytes_rel_err": float(max_err)}
