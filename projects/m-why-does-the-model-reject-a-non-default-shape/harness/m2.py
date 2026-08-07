import ref


def check(workdir):
    from edgeexport.compiler import compile_with_shapes
    out = {"compile_matches": 0.0}
    ok = 0

    res_ok = compile_with_shapes(None, ref.SHAPES_LIST, 100)
    if res_ok.get("status") == "compiled" and res_ok.get("shapes_count") == len(ref.SHAPES_LIST):
        ok += 1

    res_fail = compile_with_shapes(None, ref.SHAPES_LIST, 10)
    if res_fail.get("status") == "rejected":
        ok += 1

    out["compile_matches"] = float(ok)
    return out
