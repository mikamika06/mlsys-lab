import ref

def check(workdir):
    from moe_opt.model import optimized_moe_step
    m = {"threshold_ok": 0.0}
    tokens, rmap = ref.generate_case()
    res = optimized_moe_step(tokens, rmap, lambda x: x * 2)
    if res is not None:
        m["threshold_ok"] = 1.0
    return m
