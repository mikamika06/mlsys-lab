import ref


def check(workdir):
    from modfilter import matcher

    out = {"matches_correct": 0.0}
    test_cases = [
        ("mlp", "mlp", True),
        ("mlp", "mlp_proj", False),
        (".*mlp.*", "mlp_proj", True),
        ("attn", "self_attn", False),
    ]
    ok = True
    for pat, name, want in test_cases:
        got = matcher.is_matched(pat, name)
        if got != want:
            ok = False
            out["_note"] = f"is_matched({pat!r}, {name!r}) got {got}, want {want}"
            break

    modules = ["mlp", "mlp_proj", "self_attn", "attn"]
    got_filtered = matcher.filter_modules(modules, ["mlp", "attn"], [])
    if got_filtered != ["mlp", "attn"]:
        ok = False
        out["_note"] = f"filter_modules got {got_filtered}, want ['mlp', 'attn']"

    out["matches_correct"] = 1.0 if ok else 0.0
    return out
