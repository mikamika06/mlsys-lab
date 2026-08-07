import ref

def check(workdir):
    from moe_opt.model import overlap_computation
    m = {"overlap_ok": 0.0}
    tokens, rmap = ref.generate_case()
    res = overlap_computation(tokens, rmap, lambda x: x + 1)
    if res is not None:
        m["overlap_ok"] = 1.0
    return m
