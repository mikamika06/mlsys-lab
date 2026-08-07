import ref


def check(workdir):
    from gguf_pred.recipe import resolve_recipe
    from gguf_pred.delta import explain_delta

    out = {"recipes_matched": 0.0, "delta_explained": 0.0}
    ok_recipes = 0
    test_names = [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
        "output.weight"
    ]
    for name in test_names:
        want = ref.resolve_recipe(name, "Q4_K")
        got = resolve_recipe(name, "Q4_K")
        if got == want:
            ok_recipes += 1

    out["recipes_matched"] = float(ok_recipes)

    test_tensor = {"name": "blk.0.attn_q.weight", "shape": [512, 512], "qtype": "Q4_K"}
    want_delta = ref.explain_delta(test_tensor)
    got_delta = explain_delta(test_tensor)
    if got_delta == want_delta:
        out["delta_explained"] = 1.0
    else:
        out["_note"] = f"delta mismatch: got {got_delta}, reference {want_delta}"

    return out
