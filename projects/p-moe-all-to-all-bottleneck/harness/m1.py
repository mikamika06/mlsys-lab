import ref

def check(workdir):
    from moe_opt.model import measure_communication
    m = {"measure_ok": 0.0}
    tokens, rmap = ref.generate_case()
    val = measure_communication(tokens, rmap)
    if isinstance(val, (int, float)) and val >= 0:
        m["measure_ok"] = 1.0
    return m
