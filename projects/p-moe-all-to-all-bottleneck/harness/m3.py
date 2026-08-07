import ref

def check(workdir):
    from moe_opt.model import group_tokens
    m = {"group_ok": 0.0}
    tokens, rmap = ref.generate_case()
    grouped = group_tokens(tokens, rmap)
    if isinstance(grouped, list) and len(grouped) == rmap.shape[1]:
        m["group_ok"] = 1.0
    return m
