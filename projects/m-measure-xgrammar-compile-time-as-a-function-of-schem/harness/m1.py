import ref

def check(workdir):
    from schema_opt.compile import count_nodes
    out = {"compile_match": 0.0}
    ok = 0
    for sc in ref.SCHEMAS:
        want = ref.count_nodes(sc)
        got = count_nodes(sc)
        if want == got:
            ok += 1
    if ok == len(ref.SCHEMAS):
        out["compile_match"] = 1.0
    return out
