import ref


def check(workdir):
    from modfilter import rules

    out = {"priority_correct": 0.0}
    mods = ["layer.0.mlp", "layer.0.mlp_proj", "layer.1.attn"]
    rlist = [
        {"action": "include", "patterns": [r"layer\.\d+\.mlp"]},
    ]
    got = rules.apply_rules(mods, rlist)
    want = ["layer.0.mlp"]
    if got == want:
        out["priority_correct"] = 1.0
    else:
        out["_note"] = f"apply_rules got {got}, want {want}"
    return out
