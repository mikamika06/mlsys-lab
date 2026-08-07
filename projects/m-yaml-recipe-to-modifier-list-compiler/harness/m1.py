import ref

def check(workdir):
    from compiler.recipe import compile_yaml_recipe
    out = {"recipes_matched": 0.0, "total": float(len(ref.RECIPES))}
    ok = 0
    for i, r in enumerate(ref.RECIPES):
        want = r["expected"]
        try:
            got = compile_yaml_recipe(r["yaml_str"])
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"recipe {i} raised {type(e).__name__}: {e}"
            continue
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"recipe {i}: got {got}, want {want}"
    out["recipes_matched"] = float(ok)
    return out
