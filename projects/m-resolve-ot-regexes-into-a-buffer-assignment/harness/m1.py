import ref

def check(workdir):
    from moeoff.resolver import resolve_ot_regexes
    out = {"matches": 0.0}
    got = resolve_ot_regexes(ref.TENSORS, ref.OVERRIDES, "GPU")
    want = {
        "blk.0.attn_q.weight": "CPU",
        "blk.0.ffn_gate.weight": "GPU",
        "blk.1.attn_q.weight": "GPU",
        "blk.1.ffn_gate.weight": "GPU",
    }
    if got == want:
        out["matches"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
