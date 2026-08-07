import ref

def check(workdir):
    from moe_opt.model import analyze_imbalance
    m = {"imbalance_ok": 0.0}
    tokens, rmap = ref.generate_case()
    val = analyze_imbalance(tokens, rmap)
    if isinstance(val, (int, float)) and val > 0:
        m["imbalance_ok"] = 1.0
    return m
