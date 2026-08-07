import ref


def check(workdir):
    from edgeexport.checker import check_symbolic_propagation
    out = {"checks_matched": 0.0}
    ok = 0
    for g in ref.GRAPHS:
        try:
            res = check_symbolic_propagation(g, g["constraints"])
            if res is True:
                ok += 1
        except Exception:
            pass
    out["checks_matched"] = float(ok)
    return out
