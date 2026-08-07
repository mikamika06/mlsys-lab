import ref


def check(workdir):
    from kvquant.recipe import make_recipe

    out = {"recipes_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.make_recipe(cfg, scheme="fp8", block_size=16)
        got = make_recipe(cfg, scheme="fp8", block_size=16)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["recipes_matched"] = float(ok)
    return out
